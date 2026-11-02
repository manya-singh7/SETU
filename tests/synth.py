"""
Synthetic video generation module for Project SETU testing.

Generates realistic benchmark video sequences (60 fps, 30 s, 640x480) simulating:
- A textured vertical rectangle "pillar" oscillating at a known subpixel frequency (default 3.7 Hz, amp 0.2 px).
- A textured static background representing civil infrastructure / bedrock.
- Optional global camera micro-jitter (random walk + 9 Hz tremor).
- Optional additive Gaussian sensor noise.
- H.264 / MP4 compression encoding with configurable CRF.
- Ground-truth target and background bounding box annotations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
import tempfile
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import numpy.typing as npt


@dataclass
class SyntheticVideoResult:
    """Telemetry and ground-truth metadata from synthetic video generation.

    Attributes:
        video_path: Absolute or relative filepath to generated MP4 video.
        target_bbox: Ground-truth bounding box (x, y, w, h) for the vibrating pillar.
        background_bbox: Ground-truth bounding box (x, y, w, h) for stationary background.
        fps: Video frame rate.
        duration_sec: Video length in seconds.
        frame_count: Total frames written to video.
        freq_hz: Injected fundamental vibration frequency in Hz.
        amplitude_px: Injected vibration amplitude in pixels.
        target_displacement: 1D array of injected pure structural displacement.
        camera_jitter: 1D array of injected camera micro-jitter motion.
        total_target_displacement: Target motion observed by camera (structural + jitter).
        total_background_displacement: Background motion observed by camera (jitter only).
    """

    video_path: str
    target_bbox: Tuple[int, int, int, int]
    background_bbox: Tuple[int, int, int, int]
    fps: float
    duration_sec: float
    frame_count: int
    freq_hz: float
    amplitude_px: float
    target_displacement: npt.NDArray[np.float64]
    camera_jitter: npt.NDArray[np.float64]
    total_target_displacement: npt.NDArray[np.float64]
    total_background_displacement: npt.NDArray[np.float64]


def _create_textured_background(width: int, height: int, rng: np.random.Generator) -> npt.NDArray[np.uint8]:
    """Create a realistic textured civil background with horizontal and vertical gradients."""
    # Base low-frequency noise
    base = rng.integers(50, 90, size=(height // 4, width // 4, 3), dtype=np.uint8)
    base = cv2.resize(base, (width, height), interpolation=cv2.INTER_CUBIC)

    # Add high-frequency speckle texture (concrete/rock grain)
    grain = rng.normal(0, 8, size=(height, width, 3)).astype(np.float32)
    bg = np.clip(base.astype(np.float32) + grain, 0, 255).astype(np.uint8)

    # Add stationary horizontal deck/ground horizon band
    horizon_y = int(height * 0.75)
    bg[horizon_y : horizon_y + 8, :] = np.clip(bg[horizon_y : horizon_y + 8, :].astype(int) - 30, 0, 255).astype(np.uint8)
    return bg


def _create_textured_pillar(
    width: int,
    height: int,
    margin: int,
    rng: np.random.Generator,
) -> npt.NDArray[np.uint8]:
    """Create a textured rectangular pillar patch with margins for subpixel warping."""
    full_w = width + 2 * margin
    full_h = height + 2 * margin

    # Concrete-like texture
    base = rng.integers(130, 190, size=(full_h // 2, full_w // 2, 3), dtype=np.uint8)
    base = cv2.resize(base, (full_w, full_h), interpolation=cv2.INTER_CUBIC)

    # Vertical formwork seams / rebar ridges
    for x_seam in range(margin + 15, full_w - margin, 25):
        base[:, x_seam : x_seam + 2] = np.clip(base[:, x_seam : x_seam + 2].astype(int) - 35, 0, 255).astype(np.uint8)

    # High-contrast edge chamfers
    base[:, margin : margin + 2] = 230  # bright left highlight
    base[:, full_w - margin - 2 : full_w - margin] = 60  # dark right shadow

    return base


def generate_synthetic_video(
    output_path: Optional[str] = None,
    fps: float = 60.0,
    duration_sec: float = 30.0,
    width: int = 640,
    height: int = 480,
    freq_hz: float = 3.7,
    amplitude_px: float = 0.2,
    camera_jitter: bool = False,
    jitter_tremor_freq: float = 9.0,
    jitter_tremor_amp: float = 0.08,
    jitter_walk_scale: float = 0.015,
    gaussian_noise_std: float = 0.0,
    crf: int = 23,
    target_bbox: Optional[Tuple[int, int, int, int]] = None,
    background_bbox: Optional[Tuple[int, int, int, int]] = None,
    seed: int = 42,
) -> SyntheticVideoResult:
    """Generate a synthetic civil infrastructure video simulating subpixel vibration.

    Args:
        output_path: Destination MP4 file path. If None, a temporary file is created.
        fps: Frame rate in Hz (default: 60.0).
        duration_sec: Recording duration in seconds (default: 30.0).
        width: Video frame width in pixels (default: 640).
        height: Video frame height in pixels (default: 480).
        freq_hz: Structural resonant oscillation frequency in Hz (default: 3.7).
        amplitude_px: Structural oscillation amplitude in pixels (default: 0.2).
        camera_jitter: If True, injects random walk + 9 Hz tremor camera motion.
        jitter_tremor_freq: Frequency of the tripod/wind tremor component in Hz (default: 9.0).
        jitter_tremor_amp: Amplitude of the tremor jitter in pixels (default: 0.08).
        jitter_walk_scale: Standard deviation of frame-to-frame random walk drift.
        gaussian_noise_std: Standard deviation of additive sensor Gaussian noise.
        crf: Constant Rate Factor for H.264 compression (default: 23).
        target_bbox: Optional (x, y, w, h) for the pillar. Defaults to (270, 90, 100, 280).
        background_bbox: Optional (x, y, w, h) for stationary anchor. Defaults to (40, 360, 120, 80).
        seed: Random generator seed for repeatable benchmark runs.

    Returns:
        SyntheticVideoResult containing the video path, ground truth bounding boxes,
        and injected displacement time series.
    """
    rng = np.random.default_rng(seed)

    if output_path is None:
        fd, output_path = tempfile.mkstemp(prefix="setu_synth_", suffix=".mp4")
        os.close(fd)

    # Bounding boxes
    t_bbox = target_bbox or (270, 90, 100, 280)
    b_bbox = background_bbox or (40, 360, 120, 80)
    tx, ty, tw, th = t_bbox

    # Time series
    n_frames = int(round(fps * duration_sec))
    time_series = np.arange(n_frames, dtype=np.float64) / fps

    # 1. Structural target oscillation (pure 3.7 Hz sinusoidal motion)
    target_disp = amplitude_px * np.sin(2.0 * np.pi * freq_hz * time_series)

    # 2. Camera micro-jitter (random walk + 9 Hz tremor)
    if camera_jitter:
        # Random walk drift with zero-mean detrending
        walk_steps = rng.normal(0, jitter_walk_scale, size=n_frames)
        random_walk = np.cumsum(walk_steps)
        random_walk -= np.linspace(random_walk[0], random_walk[-1], n_frames)

        # 9 Hz high-frequency camera tremor
        tremor = jitter_tremor_amp * np.sin(2.0 * np.pi * jitter_tremor_freq * time_series + 0.45)
        camera_jitter_disp = random_walk + tremor
    else:
        camera_jitter_disp = np.zeros(n_frames, dtype=np.float64)

    total_target_disp = target_disp + camera_jitter_disp
    total_bg_disp = camera_jitter_disp.copy()

    # Pre-generate background and pillar texture patches
    margin = 20
    bg_base = _create_textured_background(width, height, rng)
    pillar_base = _create_textured_pillar(tw, th, margin, rng)

    # Video Writer configuration
    # Set FFMPEG options for CRF if OpenCV uses ffmpeg backend
    os.environ["OPENCV_FFMPEG_WRITER_OPTIONS"] = f"crf={crf}"

    # Try H.264 / avc1 codec first, fallback to mp4v
    fourcc_options = ["avc1", "H264", "X264", "mp4v"]
    writer: Optional[cv2.VideoWriter] = None

    for codec in fourcc_options:
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            candidate = cv2.VideoWriter(output_path, fourcc, float(fps), (width, height))
            if candidate.isOpened():
                writer = candidate
                break
        except Exception:
            continue

    if writer is None or not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, float(fps), (width, height))

    if not writer.isOpened():
        raise RuntimeError(f"Failed to initialize cv2.VideoWriter for path '{output_path}'")

    # Render frames
    try:
        for i in range(n_frames):
            dx_target = total_target_disp[i]
            dx_camera = camera_jitter_disp[i]

            # If camera jitter exists, translate background by subpixel jitter
            if camera_jitter and abs(dx_camera) > 1e-4:
                m_bg = np.float32([[1.0, 0.0, dx_camera], [0.0, 1.0, 0.0]])
                frame = cv2.warpAffine(bg_base, m_bg, (width, height), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)
            else:
                frame = bg_base.copy()

            # Translate pillar by total target displacement with subpixel accuracy
            # Warp the oversized pillar patch (+margin) and composite into pillar coordinates
            m_pillar = np.float32([[1.0, 0.0, dx_target + margin], [0.0, 1.0, float(margin)]])
            warped_pillar = cv2.warpAffine(
                pillar_base,
                m_pillar,
                (tw + 2 * margin, th + 2 * margin),
                flags=cv2.INTER_LANCZOS4,
                borderMode=cv2.BORDER_REFLECT,
            )

            # Insert pillar into frame
            frame[ty : ty + th, tx : tx + tw] = warped_pillar[margin : margin + th, margin : margin + tw]

            # Add optional Gaussian noise
            if gaussian_noise_std > 0.0:
                noise = rng.normal(0, gaussian_noise_std, size=(height, width, 3))
                frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)

            writer.write(frame)
    finally:
        writer.release()

    return SyntheticVideoResult(
        video_path=output_path,
        target_bbox=t_bbox,
        background_bbox=b_bbox,
        fps=fps,
        duration_sec=duration_sec,
        frame_count=n_frames,
        freq_hz=freq_hz,
        amplitude_px=amplitude_px,
        target_displacement=target_disp,
        camera_jitter=camera_jitter_disp,
        total_target_displacement=total_target_disp,
        total_background_displacement=total_bg_disp,
    )
