"""
Tests for spectral frequency estimation and synthetic video generation.

Verifies:
1. Pure 3.7 Hz sinusoidal displacement is identified within ±0.05 Hz.
2. 3.7 Hz signal with additive Gaussian noise and a weaker 9 Hz component
   still reports 3.7 Hz as the top dominant peak.
3. tests/synth.py produces a playable video file with the expected frame count (1800 frames).
"""

from __future__ import annotations

import os
import tempfile
import cv2
import numpy as np
import pytest

from setu.spectrum import compute_welch_peaks
from setu.types import Peak
from tests.synth import generate_synthetic_video


def test_pure_sine_frequency_accuracy():
    """Verify that a pure 3.7 Hz sine comes back within ±0.05 Hz."""
    fps = 60.0
    duration_sec = 30.0
    freq_target = 3.7
    amplitude = 0.2

    n_samples = int(fps * duration_sec)
    t = np.arange(n_samples, dtype=np.float64) / fps
    signal = amplitude * np.sin(2.0 * np.pi * freq_target * t)

    peaks = compute_welch_peaks(signal, fps=fps, min_freq_hz=0.5, top_k=3)

    assert len(peaks) >= 1, "Expected at least 1 peak detected from pure sine"
    top_peak = peaks[0]

    assert isinstance(top_peak, Peak), f"Expected Peak instance, got {type(top_peak)}"
    assert abs(top_peak.freq_hz - freq_target) <= 0.05, (
        f"Peak frequency {top_peak.freq_hz:.4f} Hz deviates more than ±0.05 Hz from target {freq_target} Hz"
    )
    assert top_peak.rel_power > 0.05, "Relative power should be significant for pure sine"
    assert top_peak.snr > 10.0, "SNR should be high for pure sine signal"


def test_sine_with_noise_and_9hz_component_reports_3_7hz_top():
    """Verify that 3.7 Hz sine plus noise and a weaker 9 Hz component reports 3.7 Hz as top peak."""
    fps = 60.0
    duration_sec = 30.0
    freq_primary = 3.7
    freq_secondary = 9.0
    amp_primary = 0.2
    amp_secondary = 0.05  # 4x weaker component

    rng = np.random.default_rng(12345)
    n_samples = int(fps * duration_sec)
    t = np.arange(n_samples, dtype=np.float64) / fps

    # Primary 3.7 Hz + secondary 9.0 Hz + Gaussian noise + linear drift
    primary_sine = amp_primary * np.sin(2.0 * np.pi * freq_primary * t)
    secondary_tremor = amp_secondary * np.sin(2.0 * np.pi * freq_secondary * t + 0.3)
    noise = rng.normal(0.0, 0.02, size=n_samples)
    linear_drift = 0.005 * t

    combined_signal = primary_sine + secondary_tremor + noise + linear_drift

    peaks = compute_welch_peaks(combined_signal, fps=fps, min_freq_hz=0.5, top_k=3)

    assert len(peaks) >= 2, "Expected at least 2 peaks detected"
    top_peak = peaks[0]

    # Top peak must be 3.7 Hz within ±0.05 Hz
    assert abs(top_peak.freq_hz - freq_primary) <= 0.05, (
        f"Top peak frequency {top_peak.freq_hz:.4f} Hz should be 3.7 Hz ± 0.05 Hz"
    )

    # 9.0 Hz component should be detected among top peaks
    detected_freqs = [p.freq_hz for p in peaks]
    has_9hz_peak = any(abs(f - freq_secondary) <= 0.2 for f in detected_freqs)
    assert has_9hz_peak, f"Expected 9 Hz component in detected peaks: {detected_freqs}"

    # Top peak must have higher relative power than secondary peaks
    assert peaks[0].rel_power > peaks[1].rel_power, "Primary peak must have higher relative power"


def test_synthetic_video_generation_playable_and_frame_count():
    """Verify tests/synth.py produces a playable video file with the expected frame count."""
    fps = 60.0
    duration_sec = 30.0
    expected_frames = int(fps * duration_sec)  # 1800 frames
    width = 640
    height = 480

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        temp_video_path = tmp.name

    try:
        result = generate_synthetic_video(
            output_path=temp_video_path,
            fps=fps,
            duration_sec=duration_sec,
            width=width,
            height=height,
            freq_hz=3.7,
            amplitude_px=0.2,
            camera_jitter=True,
            gaussian_noise_std=1.0,
            crf=23,
            seed=42,
        )

        # 1. Output file exists and has valid size
        assert os.path.exists(result.video_path), "Generated video file does not exist"
        file_size = os.path.getsize(result.video_path)
        assert file_size > 1000, f"Generated video is too small ({file_size} bytes)"

        # 2. Check metadata
        assert result.frame_count == expected_frames
        assert result.fps == fps
        assert len(result.target_displacement) == expected_frames
        assert len(result.target_bbox) == 4
        assert len(result.background_bbox) == 4

        # 3. Open video and verify it is playable
        cap = cv2.VideoCapture(result.video_path)
        assert cap.isOpened(), "cv2.VideoCapture failed to open the generated video"

        actual_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        assert actual_frame_count == expected_frames, (
            f"Frame count mismatch: expected {expected_frames}, got {actual_frame_count}"
        )

        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        assert actual_width == width, f"Width mismatch: expected {width}, got {actual_width}"
        assert actual_height == height, f"Height mismatch: expected {height}, got {actual_height}"

        # 4. Read frames to verify stream is decodeable and playable
        ret_first, frame_first = cap.read()
        assert ret_first is True, "Failed to read first frame from synthetic video"
        assert frame_first.shape == (height, width, 3)

        # Seek and read a middle frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, expected_frames // 2)
        ret_mid, frame_mid = cap.read()
        assert ret_mid is True, "Failed to read middle frame from synthetic video"
        assert frame_mid.shape == (height, width, 3)

        cap.release()

    finally:
        if os.path.exists(temp_video_path):
            try:
                os.remove(temp_video_path)
            except OSError:
                pass
