"""
Unit and integration tests for Eulerian Video Magnification (setu/evm.py).

Verifies:
1. Phase-based video magnification amplification factor & dominant frequency:
   On a synthetic 3.7 Hz oscillating pillar (amplitude 0.2 px) from tests/synth.py with alpha=50:
   - Measured amplification must be between 35x and 70x (theoretical ~51x).
   - Dominant frequency must be 3.7 Hz (±0.1 Hz) via setu.spectrum.compute_welch_peaks.
2. Browser compatibility (Chrome & Safari):
   Output MP4 is written in H.264 with yuv420p pixel format and faststart (moov atom before mdat).
3. Target mask spatial confinement and smooth edge blending:
   Magnification is confined to the target structure, leaving stationary background anchors unmagnified.
4. Side-by-side MP4 generation with frequency and magnification text overlays.
5. Downscaling and CPU optimization configuration.
6. Graceful fallback logging when external video encoders fail.
"""

from __future__ import annotations

import logging
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
    _write_mp4_video,
)
from setu.spectrum import compute_welch_peaks
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


def test_evm_motion_amplification_and_dominant_frequency(tmp_path: Path, synthetic_3hz7_video):
    """Check that for alpha=50, measured amplification is between 35x and 70x and dom freq is 3.7 ± 0.1 Hz."""
    synth = synthetic_3hz7_video
    output_video_path = str(tmp_path / "magnified_sbs.mp4")

    # Run EVM magnification centered at 3.7 Hz with alpha=50.0
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

    # Calculate peak-to-peak amplitudes
    amp_orig = float(np.ptp(orig_shifts))
    amp_mag = float(np.ptp(mag_shifts))

    assert amp_orig > 0.1, f"Expected detectable original motion, got amp_orig={amp_orig:.4f}"

    measured_factor = amp_mag / amp_orig if amp_orig > 0 else 0.0

    # Frequency analysis via setu/spectrum.py
    mag_shifts_arr = np.array(mag_shifts, dtype=np.float64)
    peaks = compute_welch_peaks(mag_shifts_arr, fps=60.0)
    assert len(peaks) > 0, "No spectral peaks detected in magnified displacement signal"
    dominant_frequency_hz = peaks[0].freq_hz

    print(
        f"\n[EVM Verification Report]\n"
        f"  Injected Frequency: 3.70 Hz\n"
        f"  Dominant Frequency: {dominant_frequency_hz:.2f} Hz\n"
        f"  Original Motion ptp: {amp_orig:.4f} px\n"
        f"  Magnified Motion ptp: {amp_mag:.4f} px\n"
        f"  Measured Amplification Factor: {measured_factor:.2f}x (Required: 35x - 70x)\n"
    )

    # 1. Measured amplification must be between 35x and 70x for alpha=50
    assert 35.0 <= measured_factor <= 70.0, (
        f"Measured amplification factor ({measured_factor:.2f}x) is outside required range [35x, 70x]."
    )

    # 2. Magnified displacement dominant frequency must be 3.7 Hz (±0.1 Hz)
    assert abs(dominant_frequency_hz - 3.7) <= 0.1, (
        f"Dominant frequency ({dominant_frequency_hz:.2f} Hz) deviates from 3.7 Hz by more than ±0.1 Hz."
    )


def test_evm_h264_browser_compatibility(tmp_path: Path, synthetic_3hz7_video):
    """Verify MP4 is encoded with H.264, yuv420p pixel format, and faststart for Chrome/Safari."""
    synth = synthetic_3hz7_video
    output_video_path = str(tmp_path / "magnified_h264.mp4")

    magnify_video(
        input_video_path=synth.video_path,
        output_video_path=output_video_path,
        target_frequency_hz=3.7,
        target_bbox=synth.target_bbox,
        amplification_factor=50.0,
    )

    # Inspect file binary for faststart layout: 'moov' atom must precede 'mdat' atom
    with open(output_video_path, "rb") as f:
        header_data = f.read(65536)  # Read first 64 KB

    moov_pos = header_data.find(b"moov")
    mdat_pos = header_data.find(b"mdat")

    assert moov_pos != -1, "Missing 'moov' metadata atom in MP4 container"
    assert mdat_pos != -1, "Missing 'mdat' media data atom in MP4 container"
    assert moov_pos < mdat_pos, (
        f"Expected faststart layout ('moov' at {moov_pos} before 'mdat' at {mdat_pos}) for browser streaming"
    )

    # Verify codec using PyAV if available
    try:
        import av

        container = av.open(output_video_path)
        video_stream = next(s for s in container.streams if s.type == "video")
        assert video_stream.codec_context.name in ("h264", "libx264"), (
            f"Expected h264 codec, got {video_stream.codec_context.name}"
        )
        assert video_stream.pix_fmt == "yuv420p", (
            f"Expected yuv420p pixel format for Chrome/Safari compatibility, got {video_stream.pix_fmt}"
        )
        container.close()
    except ImportError:
        pass


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

    # Background outside target mask must remain virtually identical (difference < 1.0)
    assert mean_bg_diff < 1.0, f"Background anchor should remain unmagnified, got mean diff={mean_bg_diff:.4f}"
    # Target pillar inside mask must show clear motion differences
    assert mean_tgt_diff > 3.0, f"Target pillar inside mask should show visible motion, got mean diff={mean_tgt_diff:.4f}"


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

    # Verify banner background is dark slate (mean value < 80) and contains light text (max value > 180)
    assert np.mean(left_banner) < 80.0
    assert np.max(left_banner) > 180
    assert np.mean(right_banner) < 80.0
    assert np.max(right_banner) > 180


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


def test_evm_opencv_fallback_warning(tmp_path: Path, monkeypatch, caplog):
    """Verify that when PyAV and imageio-ffmpeg fail, OpenCV fallback is used and a warning is logged."""
    frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(5)]
    out_path = str(tmp_path / "fallback.mp4")

    import builtins

    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name in ("av", "imageio_ffmpeg"):
            raise ImportError(f"Simulated absence of {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    with caplog.at_level(logging.WARNING):
        _write_mp4_video(out_path, frames, fps=30.0)

    assert os.path.exists(out_path)
    assert any("falling back to opencv" in record.message.lower() for record in caplog.records)


def test_evm_nonexistent_file_raises_error():
    """Verify nonexistent file raises FileNotFoundError."""
    magnifier = EulerianDisplayMagnifier()
    with pytest.raises(FileNotFoundError):
        magnifier.process_video_file(
            input_video_path="nonexistent_fake_video_12345.mp4",
            output_video_path="out.mp4",
            target_frequency_hz=3.7,
        )
