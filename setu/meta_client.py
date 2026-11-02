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
import json
import logging
import os
from pathlib import Path
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
        self.mode = (mode or settings.SETU_META_MODE or "mock").lower()
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
            resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
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


_default_client: Optional[MetaClient] = None


def get_meta_client() -> MetaClient:
    """Singleton getter for the configured MetaClient."""
    global _default_client
    if _default_client is None:
        _default_client = MetaClient()
    return _default_client
