"""Video-directed scan setup using Muse Spark vision."""

from __future__ import annotations

import json
import logging
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import cv2

from setu.meta_client import MetaClient, get_meta_client

logger = logging.getLogger(__name__)

STRUCTURE_TYPES = {"pillar", "deck", "cable", "railing", "truss", "arch", "other", "none"}
LANGUAGE_NAMES = {
    "en": "English", "hi": "Hindi", "ta": "Tamil", "bn": "Bengali", "te": "Telugu",
    "mr": "Marathi", "kn": "Kannada", "gu": "Gujarati", "pa": "Punjabi", "ml": "Malayalam",
}


def _small_h264_copy(video_path: str | Path, output_path: str | Path) -> None:
    """Write at most 30 seconds at 5 fps, scaled to at most 480p, as H.264 MP4."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError("Could not open scan video")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 0:
            raise ValueError("Could not determine scan video frame rate")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            raise ValueError("Could not determine scan video dimensions")
        scale = min(1.0, 854.0 / width, 480.0 / height)
        out_width = max(2, int(width * scale) // 2 * 2)
        out_height = max(2, int(height * scale) // 2 * 2)
        writer = cv2.VideoWriter(
            str(output_path), cv2.VideoWriter_fourcc(*"avc1"), 5.0, (out_width, out_height)
        )
        if not writer.isOpened():
            raise ValueError("H.264 video encoding is unavailable")
        frame_interval = max(1, int(round(fps / 5.0)))
        max_frames = int(fps * 30)
        frame_index = 0
        while frame_index < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % frame_interval == 0:
                if (frame.shape[1], frame.shape[0]) != (out_width, out_height):
                    frame = cv2.resize(frame, (out_width, out_height), interpolation=cv2.INTER_AREA)
                writer.write(frame)
            frame_index += 1
        writer.release()
        if frame_index == 0 or not Path(output_path).is_file() or Path(output_path).stat().st_size == 0:
            raise ValueError("Scan video contained no readable frames")
    finally:
        capture.release()


def _message_content(response: Dict[str, Any]) -> str:
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("Muse response did not contain a completion") from exc
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    raise ValueError("Muse response content was not text")


def _validate_director_payload(payload: Any) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Director output must be a JSON object")
    required = {
        "structure_visible", "structure_type", "target_prompt", "background_prompt",
        "is_cable", "recording_issues", "advice",
    }
    if set(payload) != required:
        raise ValueError("Director output did not match the required schema")
    if type(payload["structure_visible"]) is not bool or type(payload["is_cable"]) is not bool:
        raise ValueError("Visibility and cable flags must be booleans")
    if payload["structure_type"] not in STRUCTURE_TYPES:
        raise ValueError("Unknown structure type")
    target = payload["target_prompt"]
    background = payload["background_prompt"]
    if not isinstance(target, str) or not target.strip():
        raise ValueError("Target prompt must be a non-empty noun phrase")
    if background is not None and (not isinstance(background, str) or not background.strip()):
        raise ValueError("Background prompt must be a noun phrase or null")
    if not isinstance(payload["recording_issues"], list) or not all(
        isinstance(issue, str) and issue.strip() for issue in payload["recording_issues"]
    ):
        raise ValueError("Recording issues must be plain-language strings")
    advice = payload["advice"]
    if not isinstance(advice, str) or not advice.strip() or len(re.findall(r"[.!?](?:\s|$)", advice.strip())) != 1:
        raise ValueError("Advice must be one plain sentence")
    payload["target_prompt"] = " ".join(target.split()[:5])
    if background is not None:
        payload["background_prompt"] = " ".join(background.split()[:5])
    if not payload["structure_visible"]:
        payload["structure_type"] = "none"
    return payload


def direct_scan(
    video_path: str | Path,
    language: str,
    meta_client: Optional[MetaClient] = None,
) -> Optional[Dict[str, Any]]:
    """Ask Muse Spark to identify scan prompts and recording issues from a clip.

    Returns a validated JSON-ready dict, or ``None`` after two failed attempts.
    """
    client = meta_client or get_meta_client()
    prompt = (
        "Watch this bridge scan video and prepare scan setup. Return only a JSON object with exactly these keys: "
        'structure_visible (boolean), structure_type (one of pillar, deck, cable, railing, truss, arch, other, none), '
        'target_prompt (SAM noun phrase, at most five words), background_prompt (stationary textured item noun phrase or null), '
        'is_cable (boolean), recording_issues (array of plain-language strings), advice (one plain sentence in '
        f"{LANGUAGE_NAMES.get((language or '').lower(), language or 'the user')} language. "
        "Set structure_type to none if no bridge structure is visible. "
        "Do not make any statement about structural safety. Do not infer unseen details."
    )
    try:
        with tempfile.TemporaryDirectory(prefix="setu-director-") as temp_dir:
            small_video = Path(temp_dir) / "scan-small.mp4"
            _small_h264_copy(video_path, small_video)
            for attempt in range(2):
                attempt_prompt = prompt if attempt == 0 else prompt + " Return valid JSON matching the schema exactly."
                try:
                    response = client.video_completion(
                        small_video, attempt_prompt, model="muse-spark-1.3", fps=5.0,
                        fixture_name="muse_spark_director.json",
                    )
                    content = _message_content(response)
                    return _validate_director_payload(json.loads(content))
                except Exception as exc:
                    logger.warning("Muse Spark scan setup attempt %d failed: %s", attempt + 1, exc)
        return None
    except Exception as exc:
        logger.warning("Could not prepare scan video for Muse Spark: %s", exc)
        return None
