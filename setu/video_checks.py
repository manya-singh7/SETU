"""
Video validation and integrity checks for Project SETU.

Step 1 in SETU Pipeline:
Verifies that the uploaded or recorded video meets technical requirements:
- File is readable and opens cleanly with cv2.VideoCapture.
- Video stream contains decodable frames (frame 0 decoded successfully).
- Duration is between 5.0 and 60.0 seconds.
- Frame rate (fps) is between 15.0 and 240.0 Hz.
- Spatial resolution is at least 320x240 pixels (width >= 320, height >= 240).

Returns a structured check result with pass/fail status, diagnostic reasons,
and video metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import numpy.typing as npt

from setu.result import VideoInfo

MIN_DURATION_S: float = 5.0
MAX_DURATION_S: float = 60.0
MIN_FPS: float = 15.0
MAX_FPS: float = 240.0
MIN_WIDTH: int = 320
MIN_HEIGHT: int = 240


@dataclass
class VideoCheckResult:
    """Result of video integrity and compatibility checks.

    Attributes:
        passed: True if video meets all requirements, False otherwise.
        reasons: List of reasons if the check failed, empty if passed.
        video_info: Populated VideoInfo metadata if video could be inspected.
        frame0: Decoded reference frame 0 as (H, W, 3) BGR array, or None.
    """

    passed: bool
    reasons: List[str] = field(default_factory=list)
    video_info: Optional[VideoInfo] = None
    frame0: Optional[npt.NDArray[np.uint8]] = None

    def __bool__(self) -> bool:
        return self.passed


def check_video(video_path: Union[str, Path]) -> VideoCheckResult:
    """Inspect and validate input video file against SETU requirements.

    Args:
        video_path: Path to video file.

    Returns:
        VideoCheckResult indicating pass/fail status and reasons.
    """
    path_str = str(video_path)
    filename = os.path.basename(path_str)

    if not os.path.exists(path_str):
        return VideoCheckResult(
            passed=False,
            reasons=[f"Video file does not exist: {path_str}"],
            video_info=VideoInfo(filename=filename),
            frame0=None,
        )

    if not os.path.isfile(path_str):
        return VideoCheckResult(
            passed=False,
            reasons=[f"Path is not a regular file: {path_str}"],
            video_info=VideoInfo(filename=filename),
            frame0=None,
        )

    cap = cv2.VideoCapture(path_str)
    try:
        if not cap.isOpened():
            return VideoCheckResult(
                passed=False,
                reasons=[f"Unable to open video stream at '{path_str}' (unreadable or unsupported codec)"],
                video_info=VideoInfo(filename=filename),
                frame0=None,
            )

        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        ret, frame0 = cap.read()
        if not ret or frame0 is None:
            return VideoCheckResult(
                passed=False,
                reasons=["Failed to decode video frames (stream has no decodable frames)"],
                video_info=VideoInfo(
                    filename=filename,
                    fps=fps,
                    duration_s=0.0,
                    width=width,
                    height=height,
                    frame_count=frame_count,
                ),
                frame0=None,
            )

        # Update dimensions from actual decoded frame if needed
        actual_h, actual_w = frame0.shape[:2]
        if actual_w > 0 and actual_h > 0:
            width = actual_w
            height = actual_h

        # Compute duration
        duration_s = float(frame_count / fps) if fps > 0.0 else 0.0

        reasons: List[str] = []

        # Check duration
        if duration_s < MIN_DURATION_S or duration_s > MAX_DURATION_S:
            reasons.append(
                f"Video duration {duration_s:.1f} s outside allowed range "
                f"[{MIN_DURATION_S:.1f}, {MAX_DURATION_S:.1f}] s"
            )

        # Check FPS
        if fps < MIN_FPS or fps > MAX_FPS or np.isnan(fps):
            reasons.append(
                f"Video frame rate {fps:.1f} fps outside allowed range "
                f"[{MIN_FPS:.1f}, {MAX_FPS:.1f}] fps"
            )

        # Check resolution
        if width < MIN_WIDTH or height < MIN_HEIGHT:
            reasons.append(
                f"Video resolution {width}x{height} below minimum required "
                f"{MIN_WIDTH}x{MIN_HEIGHT}"
            )

        video_info = VideoInfo(
            filename=filename,
            fps=fps,
            duration_s=round(duration_s, 2),
            width=width,
            height=height,
            frame_count=frame_count,
        )

        passed = len(reasons) == 0
        return VideoCheckResult(
            passed=passed,
            reasons=reasons,
            video_info=video_info,
            frame0=frame0,
        )
    finally:
        cap.release()
