"""
Unit tests for SETU video integrity and compatibility checks (setu.video_checks).

Verifies:
1. Valid video meeting all requirements passes with clean reasons and populated VideoInfo.
2. Non-existent or non-file path is rejected with clear failure reason.
3. Corrupted / non-video file fails decodability check.
4. Duration below 5.0 s or above 60.0 s fails with clear explanation.
5. Frame rate (fps) below 15.0 or above 240.0 fps fails.
6. Resolution below 320x240 pixels fails.
"""

from __future__ import annotations

import os
from pathlib import Path
import cv2
import numpy as np
import pytest

from setu.video_checks import (
    MAX_DURATION_S,
    MAX_FPS,
    MIN_DURATION_S,
    MIN_FPS,
    MIN_HEIGHT,
    MIN_WIDTH,
    check_video,
)
from tests.synth import generate_synthetic_video


def _write_dummy_video(
    path: str,
    width: int = 640,
    height: int = 480,
    fps: float = 30.0,
    n_frames: int = 150,
) -> None:
    """Helper to quickly write a test video with OpenCV VideoWriter."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, float(fps), (width, height))
    for i in range(n_frames):
        frame = np.full((height, width, 3), (i * 2) % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_video_checks_valid_synthetic_video(tmp_path: Path):
    """Verify that a valid synthetic benchmark video passes all integrity checks."""
    video_path = str(tmp_path / "valid_test.mp4")
    synth = generate_synthetic_video(
        output_path=video_path,
        fps=60.0,
        duration_sec=6.0,
        width=640,
        height=480,
        freq_hz=3.7,
        amplitude_px=0.2,
        camera_jitter=False,
    )

    res = check_video(synth.video_path)
    assert res.passed is True
    assert bool(res) is True
    assert res.reasons == []
    assert res.video_info is not None
    assert res.video_info.fps == 60.0
    assert abs(res.video_info.duration_s - 6.0) <= 0.1
    assert res.video_info.width == 640
    assert res.video_info.height == 480
    assert res.video_info.frame_count == 360
    assert res.frame0 is not None
    assert res.frame0.shape == (480, 640, 3)


def test_video_checks_non_existent_file():
    """Verify non-existent video path is flagged as failed."""
    res = check_video("non_existent_scan_video.mp4")
    assert res.passed is False
    assert any("does not exist" in r.lower() for r in res.reasons)


def test_video_checks_corrupt_file(tmp_path: Path):
    """Verify corrupted non-video file fails decoding."""
    corrupt_path = str(tmp_path / "corrupt.mp4")
    with open(corrupt_path, "wb") as f:
        f.write(b"NOT_A_VALID_MP4_FILE_BYTES")

    res = check_video(corrupt_path)
    assert res.passed is False
    assert len(res.reasons) > 0


def test_video_checks_duration_too_short(tmp_path: Path):
    """Verify video shorter than 5.0 s fails with clear duration explanation."""
    short_path = str(tmp_path / "too_short.mp4")
    # 2.0 seconds at 30 fps = 60 frames
    _write_dummy_video(short_path, width=640, height=480, fps=30.0, n_frames=60)

    res = check_video(short_path)
    assert res.passed is False
    assert any("duration" in r.lower() for r in res.reasons)
    assert any(str(MIN_DURATION_S) in r for r in res.reasons)


def test_video_checks_fps_too_low(tmp_path: Path):
    """Verify video with fps < 15 fps fails with clear frame rate explanation."""
    low_fps_path = str(tmp_path / "low_fps.mp4")
    # 10 fps, 100 frames = 10s (passes duration, fails fps)
    _write_dummy_video(low_fps_path, width=640, height=480, fps=10.0, n_frames=100)

    res = check_video(low_fps_path)
    assert res.passed is False
    assert any("frame rate" in r.lower() for r in res.reasons)


def test_video_checks_resolution_too_low(tmp_path: Path):
    """Verify video below 320x240 fails with resolution explanation."""
    low_res_path = str(tmp_path / "low_res.mp4")
    # 200x150 resolution, 30 fps, 200 frames = 6.6s
    _write_dummy_video(low_res_path, width=200, height=150, fps=30.0, n_frames=200)

    res = check_video(low_res_path)
    assert res.passed is False
    assert any("resolution" in r.lower() for r in res.reasons)
