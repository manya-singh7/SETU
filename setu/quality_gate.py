"""
Video frame quality gate module for Project SETU.

Step 4 in SETU Pipeline:
Evaluates an uploaded recording frame using Meta Muse Spark Vision via MetaClient
to verify structural element visibility, static background availability for ego-motion compensation,
and overall visual recording stability.
"""

import base64
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import cv2
import numpy as np

from setu.config import settings
from setu.meta_client import MetaClient, get_meta_client

logger = logging.getLogger(__name__)

QUALITY_GATE_PROMPT = """You are an expert structural vision inspection model for Project SETU.
Analyze this video frame recorded for bridge vibration monitoring.
Evaluate:
1. is_structural_element: true if the target in the frame is a structural infrastructure element (e.g., bridge beam, pier, girder, stay cable, truss node, abutment, deck).
2. has_static_background: true if there is a stationary background reference (e.g., ground, bedrock, buildings, skyline, distant terrain) visible to subtract camera micro-jitter.
3. appears_stable: true if the frame is sufficiently sharp and stable without severe motion blur or gross camera shake.
4. issues: list of specific visual quality warnings (e.g., "target obscured by vegetation", "extreme motion blur", "lack of stationary background").

Return ONLY valid JSON strictly matching this schema:
{
  "is_structural_element": true,
  "has_static_background": true,
  "appears_stable": true,
  "issues": []
}"""


def resize_frame_to_768(frame: np.ndarray) -> np.ndarray:
    """Resize an image frame so its maximum dimension is at most 768px preserving aspect ratio."""
    h, w = frame.shape[:2]
    max_dim = max(h, w)
    if max_dim <= 768:
        return frame

    scale = 768.0 / max_dim
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def encode_frame_to_base64(frame: np.ndarray) -> str:
    """Encode numpy frame to base64 JPEG string."""
    success, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not success:
        raise ValueError("Failed to encode frame to JPEG buffer.")
    return base64.b64encode(buffer).decode("utf-8")


class QualityGate:
    """Visual recording quality gate using Muse Spark Vision."""

    def __init__(self, client: Optional[MetaClient] = None) -> None:
        self.client = client or get_meta_client()

    def check_frame(
        self,
        frame_input: Union[np.ndarray, bytes, str, Path],
    ) -> Dict[str, Any]:
        """Send one frame (resized to 768px) to Muse Spark vision via MetaClient.

        Args:
            frame_input: Image as numpy ndarray, JPEG/PNG bytes, or file path.

        Returns:
            Dictionary matching:
            {
                "is_structural_element": bool,
                "has_static_background": bool,
                "appears_stable": bool,
                "issues": List[str]
            }
        """
        # In mock mode, immediately load the mock fixture without processing image
        if getattr(self.client, "mode", "mock") == "mock":
            fixture_data = self.client._load_fixture("muse_spark_vision.json")
            return {
                "is_structural_element": bool(fixture_data.get("is_structural_element", True)),
                "has_static_background": bool(fixture_data.get("has_static_background", True)),
                "appears_stable": bool(fixture_data.get("appears_stable", True)),
                "issues": list(fixture_data.get("issues", [])),
            }

        # Load / decode frame into numpy array
        frame: Optional[np.ndarray] = None
        if isinstance(frame_input, np.ndarray):
            frame = frame_input
        elif isinstance(frame_input, (str, Path)):
            path_str = str(frame_input)
            if not os.path.exists(path_str):
                raise FileNotFoundError(f"Frame image file not found: {path_str}")
            frame = cv2.imread(path_str)
            if frame is None:
                raise ValueError(f"Could not decode image file from {path_str}")
        elif isinstance(frame_input, (bytes, bytearray)):
            np_arr = np.frombuffer(frame_input, dtype=np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError("Could not decode image from provided bytes.")
        else:
            raise TypeError(f"Unsupported frame input type: {type(frame_input)}")

        # Resize so maximum dimension is 768px
        resized_frame = resize_frame_to_768(frame)
        image_base64 = encode_frame_to_base64(resized_frame)

        # Call Muse Spark Vision via MetaClient
        try:
            resp = self.client.vision_completion(
                image_base64=image_base64,
                prompt=QUALITY_GATE_PROMPT,
                fixture_name="muse_spark_vision.json",
            )

            # Check mock data payload
            if "_mock_data" in resp:
                data = resp["_mock_data"]
            else:
                choices = resp.get("choices", [])
                content = choices[0].get("message", {}).get("content", "{}") if choices else "{}"
                # Strip markdown code fencing if present
                clean = content.strip()
                if clean.startswith("```"):
                    lines = clean.splitlines()
                    if lines[0].startswith("```"):
                        lines = lines[1:]
                    if lines and lines[-1].startswith("```"):
                        lines = lines[:-1]
                    clean = "\n".join(lines).strip()
                data = json.loads(clean)

            return {
                "is_structural_element": bool(data.get("is_structural_element", False)),
                "has_static_background": bool(data.get("has_static_background", False)),
                "appears_stable": bool(data.get("appears_stable", False)),
                "issues": [str(i) for i in data.get("issues", [])],
            }

        except Exception as exc:
            logger.warning(f"QualityGate vision call failed: {exc}. Defaulting to conservative inspection.")
            return {
                "is_structural_element": False,
                "has_static_background": False,
                "appears_stable": False,
                "issues": [f"Vision quality check error: {exc}"],
            }


def check_frame_quality(frame: Union[np.ndarray, bytes, str, Path]) -> Dict[str, Any]:
    """Convenience function inspecting a single recording frame."""
    gate = QualityGate()
    return gate.check_frame(frame)
