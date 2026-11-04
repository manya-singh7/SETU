"""Shared PyAV video decoding with display-matrix orientation applied."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Union

import av
import cv2
import numpy as np


def _rotation(frame: av.VideoFrame) -> int:
    """Return clockwise display rotation in degrees from the MP4 matrix."""
    for index, kind in enumerate(frame.side_data):
        if "DISPLAYMATRIX" in str(kind):
            matrix = np.frombuffer(frame.side_data[index], dtype=np.int32)
            if matrix.size >= 9:
                angle = int(round(np.degrees(np.arctan2(matrix[1], matrix[0])) / 90) * 90) % 360
                return angle
    return 0


def iter_video_frames(path: Union[str, Path], format: str = "bgr24") -> Iterator[np.ndarray]:
    """Decode frames in display orientation (including portrait phone footage)."""
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        rotation = None
        for frame in container.decode(stream):
            if rotation is None:
                rotation = _rotation(frame)
            image = frame.to_ndarray(format=format)
            if rotation == 90:
                image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            elif rotation == 180:
                image = cv2.rotate(image, cv2.ROTATE_180)
            elif rotation == 270:
                image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            yield image


def read_video_frame0(path: Union[str, Path], format: str = "bgr24") -> np.ndarray:
    try:
        return next(iter_video_frames(path, format=format))
    except StopIteration as exc:
        raise ValueError(f"Failed to read frame 0 from video: {path}") from exc
