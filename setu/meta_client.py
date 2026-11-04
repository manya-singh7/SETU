"""
Meta Model API client wrapper for Project SETU.

Wraps calls to Meta's OpenAI-compatible Model API (Muse Spark 1.3 and hosted SAM).
Supports three execution modes:
  - 'mock': Returns offline fixtures from tests/fixtures/ with zero network calls.
  - 'cached': Checks local disk cache in cache/meta/ keyed by SHA-256 of model + inputs.
  - 'live': Direct API calls with token accounting, usage logging, and hard budget cap enforcement.
"""

from datetime import datetime, timezone
import hashlib
import base64
import json
import logging
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Union
import requests

from setu.config import settings

logger = logging.getLogger(__name__)


class MetaAPIError(Exception):
    """Raised when a Meta Model API call fails."""
    pass


class BudgetExceededError(MetaAPIError):
    """Raised when total API token spend exceeds the hard budget cap."""
    pass


# Default token pricing estimation for Muse Spark 1.3 ($ per 1,000 tokens)
INPUT_COST_PER_1K_TOKENS = 0.0005
OUTPUT_COST_PER_1K_TOKENS = 0.0015

# SAM 3.1 pricing: $2.50 per 1,000 images ($0.0025 per image)
SAM_COST_PER_IMAGE = 0.0025
SAM_VIDEO_COST_PER_FRAME = 0.20 / 1000.0
MUSE_VOICE_COST_PER_HOUR = 0.18


def get_running_total_cost(usage_path: str = "usage.jsonl") -> float:
    """Read usage.jsonl and calculate the cumulative cost of live calls."""
    if not os.path.exists(usage_path):
        return 0.0
    total = 0.0
    try:
        with open(usage_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    total += float(entry.get("estimated_cost", 0.0))
                except (json.JSONDecodeError, ValueError):
                    continue
    except Exception as exc:
        logger.warning(f"Error reading usage log {usage_path}: {exc}")
    return round(total, 6)


class MetaClient:
    """Client wrapper for Meta Model API (Muse Spark and SAM)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        mode: Optional[str] = None,
        budget_cap: Optional[float] = None,
        usage_log_path: Optional[str] = None,
        cache_dir: Optional[str] = None,
        fixtures_dir: Optional[str] = None,
        timeout: int = 30,
    ) -> None:
        self.api_key = api_key or settings.META_API_KEY
        self.base_url = base_url or settings.META_API_BASE_URL
        self.model = model or settings.MUSE_MODEL
        if mode is not None:
            self.mode = mode.lower()
        elif "PYTEST_CURRENT_TEST" in os.environ and not os.environ.get("SETU_TEST_REAL_API"):
            self.mode = "mock"
        else:
            self.mode = (settings.SETU_META_MODE or "mock").lower()
        self.budget_cap = budget_cap if budget_cap is not None else settings.SETU_BUDGET_CAP
        self.usage_log_path = usage_log_path or settings.USAGE_LOG_PATH
        self.cache_dir = cache_dir or settings.CACHE_META_DIR
        self.fixtures_dir = fixtures_dir or os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "tests", "fixtures"
        )
        self.timeout = timeout

        if self.mode not in ("mock", "cached", "live"):
            logger.warning(f"Unknown SETU_META_MODE '{self.mode}'; defaulting to 'mock'.")
            self.mode = "mock"

    def _compute_cache_key(self, model: str, messages: Any, extra: Optional[Dict[str, Any]] = None) -> str:
        """Compute deterministic SHA-256 hash of model + prompt + inputs."""
        canonical = {
            "model": model,
            "messages": messages,
            "extra": extra or {},
        }
        serialized = json.dumps(canonical, sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _load_fixture(self, fixture_name: str) -> Dict[str, Any]:
        """Load fixture file from tests/fixtures/."""
        fixture_path = os.path.join(self.fixtures_dir, fixture_name)
        if not os.path.exists(fixture_path):
            # Fallback path if tests directory structure varies
            alt_path = os.path.join("tests", "fixtures", fixture_name)
            if os.path.exists(alt_path):
                fixture_path = alt_path
            else:
                raise FileNotFoundError(f"Fixture '{fixture_name}' not found at {fixture_path}")

        with open(fixture_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _record_live_usage(self, model: str, usage: Dict[str, Any]) -> float:
        """Log live API token usage to usage.jsonl and print running total."""
        input_tokens = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        output_tokens = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)

        estimated_cost = (
            (input_tokens / 1000.0) * INPUT_COST_PER_1K_TOKENS
            + (output_tokens / 1000.0) * OUTPUT_COST_PER_1K_TOKENS
        )
        estimated_cost = round(estimated_cost, 6)

        current_total = get_running_total_cost(self.usage_log_path)
        new_total = round(current_total + estimated_cost, 6)

        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "estimated_cost": estimated_cost,
        }

        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.usage_log_path)), exist_ok=True)
            with open(self.usage_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as exc:
            logger.warning(f"Could not write to usage log {self.usage_log_path}: {exc}")

        # Print running total cost and budget status
        print(
            f"[Meta API Usage] Model: {model} | In: {input_tokens} tok, Out: {output_tokens} tok | "
            f"Call: ${estimated_cost:.6f} | Running Total: ${new_total:.6f} / Cap: ${self.budget_cap:.2f}"
        )
        return estimated_cost

    def _execute_live_call(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Execute HTTP request against Meta Model API with budget check and usage accounting."""
        # 1. Enforce hard budget cap
        current_total = get_running_total_cost(self.usage_log_path)
        if current_total >= self.budget_cap:
            raise BudgetExceededError(
                f"SETU budget cap of ${self.budget_cap:.2f} reached (Current spend: ${current_total:.4f}). "
                f"Refusing live call."
            )

        if not self.api_key:
            raise MetaAPIError("META_API_KEY is not set. Cannot execute live call.")

        url = f"{self.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "SETU-Civil-Telemetry/0.2.0",
        }

        try:
            resp = self._post_with_retries(url, headers=headers, json=payload, timeout=self.timeout)
            if resp.status_code != 200:
                raise MetaAPIError(
                    f"Meta API responded with HTTP {resp.status_code}: {resp.text[:300]}"
                )
            data = resp.json()
        except requests.exceptions.RequestException as exc:
            raise MetaAPIError(f"Meta API network request failed: {exc}") from exc

        # 2. Log usage and print running total
        usage = data.get("usage") or {}
        self._record_live_usage(payload.get("model", self.model), usage)

        return data

    @staticmethod
    def _is_retryable_network_error(exc: requests.exceptions.RequestException) -> bool:
        """Only retry transient connection reset and timeout failures."""
        return isinstance(exc, (requests.exceptions.ConnectionError, requests.exceptions.Timeout))

    def _post_with_retries(self, url: str, **kwargs: Any) -> requests.Response:
        """POST with an initial attempt and up to three transient-failure retries."""
        for attempt in range(4):
            try:
                return requests.post(url, **kwargs)
            except requests.exceptions.RequestException as exc:
                if not self._is_retryable_network_error(exc) or attempt == 3:
                    raise
                delay = 0.25 * (2 ** attempt)
                logger.warning(
                    "Meta API connection failed (%s); retrying attempt %d/3 in %.2fs.",
                    exc, attempt + 1, delay,
                )
                time.sleep(delay)
        raise AssertionError("unreachable")

    def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: float = 0.2,
        response_format: Optional[Dict[str, Any]] = None,
        fixture_name: Optional[str] = None,
        extra_inputs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Perform chat completion with mock/cached/live dispatch.

        Args:
            messages: List of chat messages [{"role": ..., "content": ...}].
            model: Model name override (defaults to MUSE_MODEL).
            temperature: Sampling temperature.
            response_format: Optional schema enforcement (e.g. {"type": "json_object"}).
            fixture_name: Fixture file name for mock mode.
            extra_inputs: Additional inputs to include in cache key hashing.

        Returns:
            OpenAI-compatible chat completion response dictionary.
        """
        active_model = model or self.model

        # MODE 1: MOCK
        if self.mode == "mock":
            fname = fixture_name or "muse_spark_assess.json"
            fixture_data = self._load_fixture(fname)
            # If fixture is already raw OpenAI response
            if "choices" in fixture_data:
                return fixture_data
            # Otherwise wrap raw fixture payload in standard OpenAI completion envelope
            content_str = (
                json.dumps(fixture_data)
                if isinstance(fixture_data, (dict, list))
                else str(fixture_data)
            )
            return {
                "id": "mock-muse-spark-001",
                "object": "chat.completion",
                "model": active_model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content_str},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 150, "completion_tokens": 80, "total_tokens": 230},
                "_mock_data": fixture_data,
            }

        payload: Dict[str, Any] = {
            "model": active_model,
            "messages": messages,
            "temperature": temperature,
        }
        if response_format:
            payload["response_format"] = response_format

        # MODE 2: CACHED
        if self.mode == "cached":
            cache_key = self._compute_cache_key(active_model, messages, extra_inputs)
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, "r", encoding="utf-8") as f:
                        return json.load(f)
                except Exception as exc:
                    logger.warning(f"Error reading Meta cache {cache_file}: {exc}")

            # Cache miss: execute live call and cache result
            result = self._execute_live_call(payload)
            try:
                os.makedirs(self.cache_dir, exist_ok=True)
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2)
            except Exception as exc:
                logger.warning(f"Could not write Meta cache to {cache_file}: {exc}")
            return result

        # MODE 3: LIVE
        return self._execute_live_call(payload)

    def vision_completion(
        self,
        image_base64: str,
        prompt: str,
        model: Optional[str] = None,
        fixture_name: Optional[str] = "muse_spark_vision.json",
    ) -> Dict[str, Any]:
        """Send an image frame + prompt to Muse Spark vision."""
        active_model = model or self.model

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                    },
                ],
            }
        ]

        # Use image hash in extra_inputs for cache key computation
        image_hash = hashlib.sha256(image_base64.encode("utf-8")).hexdigest()
        return self.chat_completion(
            messages=messages,
            model=active_model,
            response_format={"type": "json_object"},
            fixture_name=fixture_name,
            extra_inputs={"image_hash": image_hash, "prompt": prompt},
        )

    def video_completion(
        self,
        video_path: Union[str, Path],
        prompt: str,
        model: Optional[str] = None,
        fps: float = 5.0,
        fixture_name: Optional[str] = "muse_spark_director.json",
    ) -> Dict[str, Any]:
        """Send an inline video data URL to Muse Spark through Chat Completions."""
        video_bytes = Path(video_path).read_bytes()
        video_b64 = base64.b64encode(video_bytes).decode("ascii")
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "video_url", "video_url": {
                    "url": f"data:video/mp4;base64,{video_b64}", "fps": fps,
                }},
            ],
        }]
        video_hash = hashlib.sha256(video_bytes).hexdigest()
        return self.chat_completion(
            messages=messages,
            model=model,
            response_format={"type": "json_object"},
            fixture_name=fixture_name,
            extra_inputs={"video_sha256": video_hash, "prompt": prompt, "fps": fps},
        )

    def transcribe_audio(self, wav_path: str, language: str, duration_s: float) -> str:
        """Transcribe a mono PCM WAV through Muse Voice Transcribe."""
        model = "muse-voice-transcribe-1.0"
        audio_hash = hashlib.sha256(Path(wav_path).read_bytes()).hexdigest()
        cache_file = os.path.join(self.cache_dir, f"voice-{audio_hash}-{language}.json")
        if self.mode == "mock":
            return "The concrete has a crack and rust near the water."
        if self.mode == "cached" and os.path.exists(cache_file):
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)["transcript"]
        cost = round(max(0.0, duration_s) / 3600 * MUSE_VOICE_COST_PER_HOUR, 6)
        total = get_running_total_cost(self.usage_log_path)
        if total + cost > self.budget_cap:
            raise BudgetExceededError(f"SETU budget cap of ${self.budget_cap:.2f} reached; refusing live call.")
        if not self.api_key:
            raise MetaAPIError("META_API_KEY is not set. Cannot execute live call.")
        language_names = {
            "ar": "Arabic", "bn": "Bengali", "nl": "Dutch", "en": "English",
            "fr": "French", "de": "German", "he": "Hebrew", "hi": "Hindi",
            "id": "Indonesian", "it": "Italian", "ja": "Japanese", "kn": "Kannada",
            "ko": "Korean", "ms": "Malay", "zh": "Mandarin Chinese", "mr": "Marathi",
            "pl": "Polish", "pt": "Portuguese", "es": "Spanish", "tl": "Tagalog",
            "ta": "Tamil", "te": "Telugu", "th": "Thai", "tr": "Turkish", "vi": "Vietnamese",
        }
        language_hint = language_names.get(language.lower(), language)
        try:
            with open(wav_path, "rb") as audio:
                response = requests.post(
                    f"{self.base_url.rstrip('/')}/asr/transcribe",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    files={
                        "request": (None, json.dumps({"model": model, "audioEncoding": "WAV", "languageBias": [language_hint]}), "application/json"),
                        "audio": ("audio.wav", audio, "audio/wav"),
                    }, timeout=self.timeout,
                )
            if response.status_code != 200:
                raise MetaAPIError(f"Meta Voice API responded with HTTP {response.status_code}: {response.text[:300]}")
            data = response.json()
        except requests.exceptions.RequestException as exc:
            raise MetaAPIError(f"Meta Voice API network request failed: {exc}") from exc
        transcript = str(data.get("transcript") or data.get("text") or "")
        if not transcript:
            raise MetaAPIError("Meta Voice API response did not contain a transcript.")
        log_entry = {"timestamp": datetime.now(timezone.utc).isoformat(), "model": model,
                     "audio_seconds": duration_s, "estimated_cost": cost}
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.usage_log_path)), exist_ok=True)
            with open(self.usage_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as exc:
            logger.warning("Could not write voice usage log: %s", exc)
        print(
            f"[Meta API Usage] Model: {model} | Audio: {duration_s:.2f}s | "
            f"Call: ${cost:.6f} | Running Total: ${total + cost:.6f} / Cap: ${self.budget_cap:.2f}"
        )
        if self.mode == "cached":
            os.makedirs(self.cache_dir, exist_ok=True)
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump({"transcript": transcript}, f)
        return transcript

    def _record_live_sam_usage(self, model: str, images_count: int = 1) -> float:
        """Log live SAM API image usage to usage.jsonl and print running total."""
        estimated_cost = round(images_count * SAM_COST_PER_IMAGE, 6)
        current_total = get_running_total_cost(self.usage_log_path)
        new_total = round(current_total + estimated_cost, 6)

        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "images": images_count,
            "estimated_cost": estimated_cost,
        }

        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.usage_log_path)), exist_ok=True)
            with open(self.usage_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as exc:
            logger.warning(f"Could not write to usage log {self.usage_log_path}: {exc}")

        print(
            f"[Meta API Usage] Model: {model} | Images: {images_count} | "
            f"Call: ${estimated_cost:.6f} | Running Total: ${new_total:.6f} / Cap: ${self.budget_cap:.2f}"
        )
        return estimated_cost

    def segment_image(
        self,
        image_base64: str,
        prompt: str,
        model: Optional[str] = None,
        fixture_name: Optional[str] = "sam_3_1_segment.json",
    ) -> Dict[str, Any]:
        """Call Meta Model API SAM 3.1 segmentation endpoint.

        Args:
            image_base64: Base64-encoded JPEG image string of Frame 0.
            prompt: Short text prompt (max 32 tokens, e.g. "concrete pillar").
            model: SAM model identifier (defaults to settings.SAM_MODEL, "sam-3.1").
            fixture_name: Fixture file name for mock mode.

        Returns:
            Dictionary containing API response with output_text segmentation grammar.
        """
        active_model = model or getattr(settings, "SAM_MODEL", "sam-3.1")
        prompt_trimmed = " ".join(prompt.strip().split()[:32])

        # MODE 1: MOCK
        if self.mode == "mock":
            try:
                return self._load_fixture(fixture_name or "sam_3_1_segment.json")
            except FileNotFoundError:
                return {
                    "id": "mock-sam-3-1-001",
                    "object": "response",
                    "model": active_model,
                    "status": "completed",
                    "output": [
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": "<0f>0<|box;x1=256;y1=168;x2=384;y2=336;w=640;h=480|><|mask;x=256;y=168;data=mock|>",
                                }
                            ],
                        }
                    ],
                    "confidence": 0.94,
                    "_mock_data": True,
                }

        # Cache key hashing based on model + prompt + image hash
        image_hash = hashlib.sha256(image_base64.encode("utf-8")).hexdigest()
        cache_key = self._compute_cache_key(
            active_model,
            [{"prompt": prompt_trimmed}],
            {"image_hash": image_hash},
        )
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")

        # MODE 2: CACHED
        if self.mode == "cached" and os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as exc:
                logger.warning(f"Error reading Meta SAM cache {cache_file}: {exc}")

        # Enforce hard budget cap before live network call
        current_total = get_running_total_cost(self.usage_log_path)
        if current_total + SAM_COST_PER_IMAGE > self.budget_cap:
            raise BudgetExceededError(
                f"SETU budget cap of ${self.budget_cap:.2f} reached (Current spend: ${current_total:.4f}). "
                f"Refusing live call."
            )

        if not self.api_key:
            raise MetaAPIError("META_API_KEY is not set. Cannot execute live call.")

        url = f"{self.base_url.rstrip('/')}/responses"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "SETU-Civil-Telemetry/0.2.0",
        }
        payload = {
            "model": active_model,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt_trimmed},
                        {"type": "input_image", "image_url": f"data:image/jpeg;base64,{image_base64}"},
                    ],
                }
            ],
        }

        try:
            resp = self._post_with_retries(url, headers=headers, json=payload, timeout=self.timeout)
            if resp.status_code != 200:
                raise MetaAPIError(
                    f"Meta SAM API responded with HTTP {resp.status_code}: {resp.text[:300]}"
                )
            data = resp.json()
        except requests.exceptions.RequestException as exc:
            raise MetaAPIError(f"Meta SAM API network request failed: {exc}") from exc

        # Log SAM usage ($0.0025 per image)
        self._record_live_sam_usage(active_model, images_count=1)

        # Cache on disk if in cached mode
        if self.mode == "cached":
            try:
                os.makedirs(self.cache_dir, exist_ok=True)
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
            except Exception as exc:
                logger.warning(f"Could not write Meta SAM cache to {cache_file}: {exc}")

        return data

    def track_video(
        self,
        video_path: Union[str, Path],
        prompt: str,
        model: Optional[str] = None,
        fps: float = 10.0,
        frame_count: Optional[int] = None,
        fixture_name: Optional[str] = "sam_3_1_video_response.json",
    ) -> Optional[Dict[str, Any]]:
        """Track one prompted concept in a video with SAM 3.1 Responses API.

        Video is supplied as a base64 data URL. ``frame_count`` is the number
        of frames sent and is used for the documented per-frame cost estimate.
        Public demo and missing-key calls return None without making a request.
        """
        from setu.config import settings

        active_model = model or getattr(settings, "SAM_MODEL", "sam-3.1")
        prompt = " ".join(prompt.strip().split()[:32])
        path = Path(video_path)
        video_bytes = path.read_bytes()
        video_hash = hashlib.sha256(video_bytes).hexdigest()
        nframes = max(0, int(frame_count or 0))
        cache_key = self._compute_cache_key(
            active_model, [{"prompt": prompt}],
            {"video_sha256": video_hash, "fps": float(fps), "frame_count": nframes},
        )
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")

        if self.mode == "mock":
            try:
                return self._load_fixture(fixture_name or "sam_3_1_video_response.json")
            except FileNotFoundError:
                return {"id": "mock-sam-video-empty", "status": "completed", "output": [], "_mock_data": True}

        if self.mode == "cached" and os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as exc:
                logger.warning("Error reading Meta SAM video cache %s: %s", cache_file, exc)

        # Demo deployments may read existing cache entries, but never call live.
        public_demo = os.environ.get("SETU_PUBLIC_DEMO", "").strip().lower() in ("true", "1", "yes")
        if public_demo or not self.api_key:
            logger.info("SAM video tracking skipped: %s", "public demo mode" if public_demo else "META_API_KEY is not set")
            return None

        cost = round(nframes * SAM_VIDEO_COST_PER_FRAME, 6)
        current_total = get_running_total_cost(self.usage_log_path)
        if current_total + cost > self.budget_cap:
            raise BudgetExceededError(
                f"SETU budget cap of ${self.budget_cap:.2f} reached (Current spend: ${current_total:.4f}). Refusing live call."
            )
        payload = {
            "model": active_model,
            "input": [{"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_video", "video_url": f"data:video/mp4;base64,{base64.b64encode(video_bytes).decode('ascii')}"},
            ]}],
            "stream": False,
            "metadata": {"propagation_direction": "both", "mask_encoding": "one_bit"},
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json", "User-Agent": "SETU-Civil-Telemetry/0.2.0"}
        try:
            resp = self._post_with_retries(f"{self.base_url.rstrip('/')}/responses", headers=headers, json=payload, timeout=self.timeout)
            if resp.status_code != 200:
                raise MetaAPIError(f"Meta SAM video API responded with HTTP {resp.status_code}: {resp.text[:300]}")
            data = resp.json()
        except requests.exceptions.RequestException as exc:
            raise MetaAPIError(f"Meta SAM video API network request failed: {exc}") from exc

        total = round(current_total + cost, 6)
        entry = {"timestamp": datetime.now(timezone.utc).isoformat(), "model": active_model,
                 "frames": nframes, "estimated_cost": cost}
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.usage_log_path)), exist_ok=True)
            with open(self.usage_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as exc:
            logger.warning("Could not write Meta SAM video usage log: %s", exc)
        print(f"[Meta API Usage] Model: {active_model} | Video frames: {nframes} | Call: ${cost:.6f} | Running Total: ${total:.6f} / Cap: ${self.budget_cap:.2f}")
        if self.mode == "cached":
            try:
                os.makedirs(self.cache_dir, exist_ok=True)
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
            except Exception as exc:
                logger.warning("Could not write Meta SAM video cache: %s", exc)
        return data


_default_client: Optional[MetaClient] = None


def get_meta_client() -> MetaClient:
    """Singleton getter for the configured MetaClient."""
    global _default_client
    if _default_client is None:
        _default_client = MetaClient()
    return _default_client
