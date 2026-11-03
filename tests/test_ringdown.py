"""
Tests for transient ring-down vibration analysis (flicked ruler demo scenario).

Verifies:
1. tests/synth.py ring-down mode generates decaying oscillations every 5 seconds.
2. Spectral analysis on ring-down signals correctly isolates active bursts,
   preventing quiescent dilution and comb line sideband interference.
3. Dominant frequency is recovered within +/- 0.05 Hz of 12.0 Hz with verdict "usable"
   and confidence interval narrower than 0.05 Hz.
4. Amplitude sensitivity: evaluates recovery across [2.0, 1.0, 0.5, 0.25] px
   and verifies lowest usable amplitude is 1.0 px.
"""

from __future__ import annotations

import os
import tempfile
import cv2
import numpy as np
import pytest

from setu.motion import track_motion
from setu.spectrum import (
    compute_welch_peaks,
    detect_active_segments,
    estimate_frequency_uncertainty,
    is_peak_reliable,
)
from setu.types import Peak
from tests.synth import generate_synthetic_video


def test_ringdown_synthetic_video_generation():
    """Verify generate_synthetic_video in ringdown mode produces expected oscillation and playable video."""
    fps = 60.0
    duration_sec = 12.0
    expected_frames = int(fps * duration_sec)

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        temp_path = tmp.name

    try:
        result = generate_synthetic_video(
            output_path=temp_path,
            fps=fps,
            duration_sec=duration_sec,
            ringdown=True,
            ringdown_freq_hz=12.0,
            ringdown_amplitude_px=2.0,
            ringdown_damping=0.02,
            ringdown_start_sec=5.0,
            ringdown_interval_sec=5.0,
            camera_jitter=False,
            crf=18,
            seed=42,
        )

        assert os.path.exists(result.video_path)
        assert result.frame_count == expected_frames
        assert result.freq_hz == 12.0
        assert result.amplitude_px == 2.0

        disp = result.target_displacement
        assert len(disp) == expected_frames

        # Before 5.0s, displacement should be exactly 0
        still_frames = int(4.9 * fps)
        assert np.max(np.abs(disp[:still_frames])) == 0.0

        # At 5.0s, flick initiates decaying oscillation
        flick_frame = int(5.0 * fps)
        flick_burst = disp[flick_frame : flick_frame + int(1.5 * fps)]
        assert np.max(np.abs(flick_burst)) > 1.5, "Expected initial oscillation amplitude close to 2 px"

        # Check decay: amplitude after 2.5s should have damped significantly
        tail_frame = flick_frame + int(2.5 * fps)
        assert np.max(np.abs(disp[tail_frame : tail_frame + 30])) < 0.2

        # Verify video is readable
        cap = cv2.VideoCapture(result.video_path)
        assert cap.isOpened()
        ret, frame = cap.read()
        assert ret is True
        assert frame.shape == (480, 640, 3)
        cap.release()

    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


def test_ringdown_active_segment_detection():
    """Verify detect_active_segments correctly isolates intermittent ring-down bursts."""
    fps = 60.0
    duration_sec = 30.0
    n_frames = int(fps * duration_sec)
    t = np.arange(n_frames, dtype=np.float64) / fps

    # Still until 5s, then flicked every 5s (flicks at 5s, 10s, 15s, 20s, 25s)
    disp = np.zeros(n_frames, dtype=np.float64)
    omega_n = 2.0 * np.pi * 12.0
    zeta = 0.02
    omega_d = omega_n * np.sqrt(1.0 - zeta**2)
    for i, t_val in enumerate(t):
        if t_val >= 5.0:
            dt = (t_val - 5.0) % 5.0
            disp[i] = 2.0 * np.exp(-zeta * omega_n * dt) * np.sin(omega_d * dt)

    segments = detect_active_segments(disp, fps=fps)
    assert len(segments) >= 4, f"Expected at least 4 active flick segments, got {len(segments)}"

    # Check each segment begins near a 5s flick
    for start_idx, end_idx in segments:
        start_t = start_idx / fps
        expected_flicks = [5.0, 10.0, 15.0, 20.0, 25.0]
        nearest_flick = min(abs(start_t - f) for f in expected_flicks)
        assert nearest_flick <= 0.25, f"Segment start {start_t:.2f}s is not near a flick event"


def test_ringdown_ground_truth_spectrum_recovery():
    """Verify spectral analysis detects 12 Hz with usable verdict and narrow confidence interval."""
    fps = 60.0
    duration_sec = 30.0
    n_frames = int(fps * duration_sec)
    t = np.arange(n_frames, dtype=np.float64) / fps

    disp = np.zeros(n_frames, dtype=np.float64)
    omega_n = 2.0 * np.pi * 12.0
    zeta = 0.02
    omega_d = omega_n * np.sqrt(1.0 - zeta**2)
    for i, t_val in enumerate(t):
        if t_val >= 5.0:
            dt = (t_val - 5.0) % 5.0
            disp[i] = 2.0 * np.exp(-zeta * omega_n * dt) * np.sin(omega_d * dt)

    bg_disp = np.zeros(n_frames, dtype=np.float64)
    peaks = compute_welch_peaks(disp, fps=fps, min_freq_hz=0.5, top_k=3, background_displacement=bg_disp)

    assert len(peaks) >= 1, "Expected at least 1 peak detected from ring-down"
    top_peak = peaks[0]

    # Frequency recovery within +/- 0.05 Hz of 12.0 Hz
    assert abs(top_peak.freq_hz - 12.0) <= 0.05, f"Expected 12.0 Hz +/- 0.05 Hz, got {top_peak.freq_hz} Hz"

    # Three-level verdict must be 'usable'
    assert top_peak.verdict == "usable", f"Expected verdict 'usable', got '{top_peak.verdict}' (reasons: {top_peak.reasons})"
    assert is_peak_reliable(top_peak) is True
    assert top_peak.reasons == []

    # Uncertainty confidence interval:
    # Physical damped natural frequency f_d = 12 * sqrt(1 - 0.02^2) = 11.9976 Hz
    f_damped = 12.0 * np.sqrt(1.0 - zeta**2)
    assert abs(top_peak.freq_ci_lower - f_damped) <= 0.05
    assert abs(top_peak.freq_ci_upper - f_damped) <= 0.05
    assert top_peak.freq_ci_width < 0.05, f"CI width {top_peak.freq_ci_width} Hz should be < 0.05 Hz"


def test_ringdown_video_motion_tracking_and_modal_recovery():
    """Verify end-to-end motion tracking + spectrum analysis on synthetic ring-down video."""
    fps = 60.0
    duration_sec = 15.0  # 15 seconds contains flicks at 5s and 10s

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        temp_path = tmp.name

    try:
        synth_res = generate_synthetic_video(
            output_path=temp_path,
            fps=fps,
            duration_sec=duration_sec,
            ringdown=True,
            ringdown_freq_hz=12.0,
            ringdown_amplitude_px=2.0,
            ringdown_damping=0.02,
            ringdown_start_sec=5.0,
            ringdown_interval_sec=5.0,
            camera_jitter=False,
            crf=18,
            seed=100,
        )

        # Track motion using phase correlation
        motion_res = track_motion(
            video_path=synth_res.video_path,
            target_bbox=synth_res.target_bbox,
            background_bbox=synth_res.background_bbox,
            method="phase_correlation",
        )

        assert motion_res.verdict != "failed"

        # Spectral analysis
        peaks = compute_welch_peaks(
            motion_res.displacement,
            fps=fps,
            min_freq_hz=0.5,
            top_k=3,
            background_displacement=synth_res.total_background_displacement,
        )

        assert len(peaks) >= 1
        top_peak = peaks[0]

        # Check detected frequency
        assert abs(top_peak.freq_hz - 12.0) <= 0.05, (
            f"Detected frequency {top_peak.freq_hz:.4f} Hz deviates > 0.05 Hz from 12.0 Hz"
        )
        assert top_peak.verdict == "usable"
        assert top_peak.reasons == []
        assert top_peak.freq_ci_lower <= 12.0 <= top_peak.freq_ci_upper
        assert top_peak.freq_ci_width < 0.15

    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


def test_ringdown_amplitude_sensitivity():
    """Verify amplitude sensitivity across 2.0, 1.0, 0.5, and 0.25 px.

    Confirms:
    - 2.0 px: usable
    - 1.0 px: usable
    - 0.5 px: weak (subpixel displacement drops below sensor noise floor / drift)
    - 0.25 px: weak
    Lowest usable amplitude is 1.0 px.
    """
    fps = 60.0
    duration_sec = 30.0
    n_frames = int(fps * duration_sec)
    t = np.arange(n_frames, dtype=np.float64) / fps

    amplitudes = [2.0, 1.0, 0.5, 0.25]
    verdicts = {}

    rng = np.random.default_rng(42)
    # Typical sensor tracking drift noise floor (standard deviation 0.015 per frame walk)
    drift = np.cumsum(rng.normal(0, 0.015, size=n_frames))
    drift -= np.linspace(drift[0], drift[-1], n_frames)

    for amp in amplitudes:
        omega_n = 2.0 * np.pi * 12.0
        zeta = 0.02
        omega_d = omega_n * np.sqrt(1.0 - zeta**2)
        disp = np.zeros(n_frames, dtype=np.float64)
        for i, t_val in enumerate(t):
            if t_val >= 5.0:
                dt = (t_val - 5.0) % 5.0
                disp[i] = amp * np.exp(-zeta * omega_n * dt) * np.sin(omega_d * dt)

        # Total displacement including tracking drift
        tot_disp = disp + drift
        bg_disp = drift.copy()

        peaks = compute_welch_peaks(tot_disp, fps=fps, min_freq_hz=0.5, top_k=3, background_displacement=bg_disp)
        if peaks:
            top = peaks[0]
            # Check whether 12 Hz is the top peak and usable
            is_12hz = abs(top.freq_hz - 12.0) <= 0.1
            verdicts[amp] = top.verdict if is_12hz else "weak"
        else:
            verdicts[amp] = "failed"

    assert verdicts[2.0] == "usable", f"2.0 px should be usable, got {verdicts[2.0]}"
    assert verdicts[1.0] == "usable", f"1.0 px should be usable, got {verdicts[1.0]}"
    assert verdicts[0.5] == "weak", f"0.5 px should be weak, got {verdicts[0.5]}"
    assert verdicts[0.25] == "weak", f"0.25 px should be weak, got {verdicts[0.25]}"
