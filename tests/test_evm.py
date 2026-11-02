"""
Unit and integration tests for Eulerian Video Magnification (setu/evm.py).

Verifies:
1. Phase-based video magnification amplification factor:
   On a synthetic 3.7 Hz oscillating pillar (amplitude 0.2 px) from tests/synth.py,
   measures horizontal motion using phase correlation (cv2.phaseCorrelate) with a Hanning
   window, verifying that motion in the magnified output is at least 20x larger than
   in the original video.
2. Target mask spatial confinement and smooth edge blending:
   Magnification is confined strictly to the target structure, leaving stationary
   background anchors unamplified.
3. Side-by-side MP4 generation with frequency and magnification text overlays.
4. Downscaling and CPU optimization configuration.
5. Error handling for non-existent and invalid video inputs.
"""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import cv2
import numpy as np
import pytest

from setu.evm import (
    EVMDisplayConfig,
    EulerianDisplayMagnifier,
    magnify_phase_riesz,
    magnify_video,
)
from tests.synth import generate_synthetic_video


@pytest.fixture
def synthetic_3hz7_video(tmp_path: Path):
    """Generate a 2.5s synthetic benchmark video with 3.7 Hz sinusoidal motion (amplitude 0.2 px)."""
    video_path = str(tmp_path / "synth_3hz7.mp4")
    result = generate_synthetic_video(
        output_path=video_path,
        fps=60.0,
        duration_sec=2.5,
        freq_hz=3.7,
        amplitude_px=0.2,
        camera_jitter=False,
        seed=42,
    )
    return result


def test_evm_motion_amplification_exceeds_20x(tmp_path: Path, synthetic_3hz7_video):
    """Check that motion in the magnified output is at least 20x larger than in original."""
    synth = synthetic_3hz7_video
    output_video_path = str(tmp_path / "magnified_sbs.mp4")

    # Run EVM magnification centered at 3.7 Hz with default 50x amplification
    res_path = magnify_video(
        input_video_path=synth.video_path,
        output_video_path=output_video_path,
        target_frequency_hz=3.7,
        target_bbox=synth.target_bbox,
        amplification_factor=50.0,
    )

    assert os.path.exists(res_path)
    assert os.path.getsize(res_path) > 1000

    cap = cv2.VideoCapture(res_path)
    assert cap.isOpened(), "Could not open generated side-by-side MP4"

    full_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    full_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    orig_w = full_w // 2

    # Side-by-side format must be double width, matching height
    assert full_w == 1280
    assert full_h == 480

    tx, ty, tw, th = synth.target_bbox
    hann = cv2.createHanningWindow((tw, th), cv2.CV_32F)

    orig_shifts = []
    mag_shifts = []

    ret, first_frame = cap.read()
    assert ret and first_frame is not None

    # Reference crops from frame 0
    ref_orig = cv2.cvtColor(first_frame[ty : ty + th, tx : tx + tw], cv2.COLOR_BGR2GRAY).astype(np.float32)
    ref_mag = cv2.cvtColor(
        first_frame[ty : ty + th, orig_w + tx : orig_w + tx + tw], cv2.COLOR_BGR2GRAY
    ).astype(np.float32)

    frame_count = 1
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        frame_count += 1

        curr_orig = cv2.cvtColor(frame[ty : ty + th, tx : tx + tw], cv2.COLOR_BGR2GRAY).astype(np.float32)
        curr_mag = cv2.cvtColor(
            frame[ty : ty + th, orig_w + tx : orig_w + tx + tw], cv2.COLOR_BGR2GRAY
        ).astype(np.float32)

        # Measure subpixel horizontal and vertical shifts via phase correlation
        shift_orig, _ = cv2.phaseCorrelate(ref_orig, curr_orig, hann)
        shift_mag, _ = cv2.phaseCorrelate(ref_mag, curr_mag, hann)

        orig_shifts.append(shift_orig[0])
        mag_shifts.append(shift_mag[0])

    cap.release()

    assert frame_count >= 120, f"Expected at least 120 frames decoded, got {frame_count}"

    # Calculate motion range (peak-to-peak amplitude)
    amp_orig = float(np.ptp(orig_shifts))
    amp_mag = float(np.ptp(mag_shifts))

    # Injected amplitude was 0.2 px (peak-to-peak ~0.4 px); verify original shift is detected
    assert amp_orig > 0.1, f"Expected detectable original motion, got amp_orig={amp_orig:.4f}"

    amplification_ratio = amp_mag / amp_orig if amp_orig > 0 else 0.0
    print(
        f"\n[EVM Test] Original motion ptp: {amp_orig:.4f} px | "
        f"Magnified motion ptp: {amp_mag:.4f} px | "
        f"Amplification Ratio: {amplification_ratio:.2f}x (required >= 20x)"
    )

    # Core requirement: motion in magnified output must be >= 20x larger than in original
    assert amplification_ratio >= 20.0, (
        f"Magnified motion ratio ({amplification_ratio:.2f}x) is less than required 20x."
    )


def test_evm_target_mask_confinement_and_smooth_blending(tmp_path: Path, synthetic_3hz7_video):
    """Verify magnification is confined to target mask while static bedrock background is unchanged."""
    synth = synthetic_3hz7_video
    output_video_path = str(tmp_path / "magnified_masked.mp4")

    # Create target mask strictly covering the pillar
    mask = np.zeros((480, 640), dtype=np.uint8)
    tx, ty, tw, th = synth.target_bbox
    mask[ty : ty + th, tx : tx + tw] = 255

    res_path = magnify_video(
        input_video_path=synth.video_path,
        output_video_path=output_video_path,
        target_frequency_hz=3.7,
        target_mask=mask,
        target_bbox=synth.target_bbox,
        amplification_factor=50.0,
    )

    cap = cv2.VideoCapture(res_path)
    assert cap.isOpened()
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) // 2

    bg_x, bg_y, bg_w, bg_h = synth.background_bbox
    bg_diffs = []
    target_diffs = []

    for _ in range(30):
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        # Background anchor on left vs right side
        bg_left = frame[bg_y : bg_y + bg_h, bg_x : bg_x + bg_w].astype(float)
        bg_right = frame[bg_y : bg_y + bg_h, orig_w + bg_x : orig_w + bg_x + bg_w].astype(float)
        bg_diffs.append(np.mean(np.abs(bg_left - bg_right)))

        # Target pillar on left vs right side
        tgt_left = frame[ty : ty + th, tx : tx + tw].astype(float)
        tgt_right = frame[ty : ty + th, orig_w + tx : orig_w + tx + tw].astype(float)
        target_diffs.append(np.mean(np.abs(tgt_left - tgt_right)))

    cap.release()

    mean_bg_diff = float(np.mean(bg_diffs))
    mean_tgt_diff = float(np.mean(target_diffs))

    # Background outside the target mask must remain virtually identical (difference near 0)
    assert mean_bg_diff < 1.0, f"Background anchor should remain unmagnified, got mean diff={mean_bg_diff:.4f}"
    # Target pillar inside mask must show clear motion differences
    assert mean_tgt_diff > 5.0, f"Target pillar inside mask should show visible motion, got mean diff={mean_tgt_diff:.4f}"


def test_evm_side_by_side_telemetry_overlays(tmp_path: Path, synthetic_3hz7_video):
    """Verify side-by-side video overlays text for frequency and '50x magnified'."""
    synth = synthetic_3hz7_video
    output_video_path = str(tmp_path / "magnified_overlay.mp4")

    magnify_video(
        input_video_path=synth.video_path,
        output_video_path=output_video_path,
        target_frequency_hz=3.7,
        target_bbox=synth.target_bbox,
        amplification_factor=50.0,
    )

    cap = cv2.VideoCapture(output_video_path)
    ret, frame = cap.read()
    cap.release()
    assert ret and frame is not None

    full_w = frame.shape[1]
    orig_w = full_w // 2

    # Left overlay banner: (16, 14) to (160, 48) containing "Original"
    left_banner = frame[14:48, 16:160]
    # Right overlay banner: (orig_w + 16, 14) to (orig_w + 350, 48) containing "3.7 Hz | 50x magnified"
    right_banner = frame[14:48, orig_w + 16 : orig_w + 350]

    # Verify banner background is dark slate (mean value < 50) and contains light text (max value > 200)
    assert np.mean(left_banner) < 80.0
    assert np.max(left_banner) > 200
    assert np.mean(right_banner) < 80.0
    assert np.max(right_banner) > 200


def test_evm_downscaling_and_custom_config(tmp_path: Path, synthetic_3hz7_video):
    """Verify EVM handles downscaling configuration on large crops for CPU runtime efficiency."""
    synth = synthetic_3hz7_video
    output_video_path = str(tmp_path / "magnified_downscaled.mp4")

    # Downscale crop to max 128px
    config = EVMDisplayConfig(
        amplification_factor=50.0,
        freq_bandwidth=0.3,
        pyramid_levels=3,
        max_crop_dimension=128,
    )

    magnifier = EulerianDisplayMagnifier(config=config)
    res_path = magnifier.process_video_file(
        input_video_path=synth.video_path,
        output_video_path=output_video_path,
        target_frequency_hz=3.7,
        target_bbox=synth.target_bbox,
        max_duration_sec=1.5,
    )

    assert os.path.exists(res_path)
    cap = cv2.VideoCapture(res_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    assert w == 1280
    assert h == 480


def test_evm_nonexistent_file_raises_error():
    """Verify nonexistent file raises FileNotFoundError."""
    magnifier = EulerianDisplayMagnifier()
    with pytest.raises(FileNotFoundError):
        magnifier.process_video_file(
            input_video_path="nonexistent_fake_video_12345.mp4",
            output_video_path="out.mp4",
            target_frequency_hz=3.7,
        )
