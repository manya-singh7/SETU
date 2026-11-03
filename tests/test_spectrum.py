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

from setu.spectrum import (
    LOCAL_PROMINENCE_THRESHOLD_DB,
    LOWER_BAND_EDGE_MARGIN_HZ,
    MIN_BG_RATIO,
    RELIABILITY_THRESHOLD_SNR_DB,
    SpectrumOutput,
    WelchSpectrumAnalyzer,
    compute_welch_peaks,
    determine_spectrum_verdict,
    estimate_frequency_uncertainty,
    is_peak_reliable,
)
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
    assert top_peak.rel_power > 0.80, f"Clean sine must have relative power > 80%, got {top_peak.rel_power * 100:.2f}%"
    assert top_peak.snr >= 10.0, f"SNR in dB should be >= 10.0 dB for pure sine signal, got {top_peak.snr} dB"
    assert is_peak_reliable(top_peak), "Clean sine peak must meet reliability threshold"


def test_clean_sine_relative_power_above_80_percent():
    """Explicitly verify that a clean sinusoidal tone concentrates > 80% relative power in peak band."""
    fps = 60.0
    duration_sec = 30.0
    freq_target = 3.7
    amplitude = 0.2

    n_samples = int(fps * duration_sec)
    t = np.arange(n_samples, dtype=np.float64) / fps
    signal = amplitude * np.sin(2.0 * np.pi * freq_target * t)

    peaks = compute_welch_peaks(signal, fps=fps, min_freq_hz=0.5, top_k=3)

    assert len(peaks) >= 1
    top_peak = peaks[0]
    # Dominant peak power lobe must contain > 80% (0.80) of total 0.5Hz-Nyquist band power
    assert top_peak.rel_power > 0.80, (
        f"Expected relative power > 80% (0.80) for clean sine, got {top_peak.rel_power * 100:.2f}%"
    )
    # Peak SNR in dB must exceed 10.0 dB reliability threshold
    assert top_peak.snr >= RELIABILITY_THRESHOLD_SNR_DB, (
        f"Expected SNR >= {RELIABILITY_THRESHOLD_SNR_DB} dB for clean sine, got {top_peak.snr} dB"
    )
    assert is_peak_reliable(top_peak) is True


def test_low_confidence_flag_on_noise():
    """Verify that pure white noise yields low SNR and is marked low confidence."""
    fps = 60.0
    duration_sec = 10.0
    n_samples = int(fps * duration_sec)
    rng = np.random.default_rng(999)
    noise_signal = rng.normal(0.0, 0.05, size=n_samples)

    peaks = compute_welch_peaks(noise_signal, fps=fps, min_freq_hz=0.5, top_k=3)
    if peaks:
        top_peak = peaks[0]
        # Pure noise should not produce high relative power
        assert top_peak.rel_power < 0.50
        # Peak SNR on random noise should be below 10 dB or fail reliability check
        if top_peak.snr < RELIABILITY_THRESHOLD_SNR_DB:
            assert is_peak_reliable(top_peak) is False


def test_drift_flag_near_lower_band_edge():
    """Verify peaks within 0.3 Hz of lower band edge (0.5 Hz) are flagged as possible drift and marked unreliable."""
    fps = 60.0
    duration_sec = 20.0
    n_samples = int(fps * duration_sec)
    t = np.arange(n_samples, dtype=np.float64) / fps

    # 0.65 Hz is within 0.3 Hz of 0.5 Hz (0.5 <= f <= 0.8 Hz)
    drift_freq = 0.65
    sig = 0.2 * np.sin(2.0 * np.pi * drift_freq * t)

    peaks = compute_welch_peaks(sig, fps=fps, min_freq_hz=0.5, top_k=3)
    assert len(peaks) >= 1
    top_peak = peaks[0]

    assert top_peak.is_drift is True, "Peak <= 0.8 Hz must have is_drift=True"
    assert any("possible drift" in f for f in top_peak.flags), f"Expected 'possible drift' flag in {top_peak.flags}"
    assert top_peak.is_reliable is False, "Drift peak must be flagged unreliable"
    assert is_peak_reliable(top_peak) is False


def test_local_prominence_check():
    """Verify that sharp resonant peaks have high local prominence while smooth noise lacks prominence."""
    fps = 60.0
    duration_sec = 20.0
    n_samples = int(fps * duration_sec)
    t = np.arange(n_samples, dtype=np.float64) / fps

    # A sharp resonant sine tone at 3.7 Hz
    sharp_sine = 0.2 * np.sin(2.0 * np.pi * 3.7 * t)
    peaks_sharp = compute_welch_peaks(sharp_sine, fps=fps, min_freq_hz=0.5, top_k=1)
    assert len(peaks_sharp) >= 1
    assert peaks_sharp[0].local_prom_db >= LOCAL_PROMINENCE_THRESHOLD_DB, (
        f"Sharp sine must have local prominence >= {LOCAL_PROMINENCE_THRESHOLD_DB} dB, got {peaks_sharp[0].local_prom_db} dB"
    )

    # A smooth broad noise drift slope (e.g. cumulative random walk without distinct resonance)
    rng = np.random.default_rng(42)
    broad_drift = np.cumsum(rng.normal(0, 0.02, size=n_samples))
    peaks_drift = compute_welch_peaks(broad_drift, fps=fps, min_freq_hz=0.5, top_k=1)
    if peaks_drift:
        assert peaks_drift[0].is_reliable is False, "Broad drift signal must be marked unreliable"


def test_background_referenced_ratio_check():
    """Verify that high background residual at peak frequency marks the peak unreliable."""
    fps = 60.0
    duration_sec = 20.0
    n_samples = int(fps * duration_sec)
    t = np.arange(n_samples, dtype=np.float64) / fps

    # Case A: Target has 3.7 Hz vibration, background is completely stationary (zero motion)
    target_clean = 0.2 * np.sin(2.0 * np.pi * 3.7 * t)
    bg_clean = np.zeros(n_samples, dtype=np.float64)

    peaks_clean = compute_welch_peaks(target_clean, fps=fps, min_freq_hz=0.5, top_k=1, background_displacement=bg_clean)
    assert len(peaks_clean) >= 1
    assert peaks_clean[0].bg_ratio >= MIN_BG_RATIO
    assert peaks_clean[0].is_reliable is True

    # Case B: Target has 3.7 Hz vibration, but background also has strong 3.7 Hz motion (camera jitter leakage)
    target_contaminated = 0.2 * np.sin(2.0 * np.pi * 3.7 * t)
    bg_contaminated = 0.25 * np.sin(2.0 * np.pi * 3.7 * t)  # stronger background residual

    peaks_bad = compute_welch_peaks(
        target_contaminated,
        fps=fps,
        min_freq_hz=0.5,
        top_k=1,
        background_displacement=bg_contaminated,
    )
    assert len(peaks_bad) >= 1
    assert peaks_bad[0].bg_ratio < MIN_BG_RATIO, f"Expected BG ratio < {MIN_BG_RATIO}, got {peaks_bad[0].bg_ratio}"
    assert any("low target-to-background ratio" in f for f in peaks_bad[0].flags)
    assert peaks_bad[0].is_reliable is False
    assert is_peak_reliable(peaks_bad[0]) is False


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


def test_no_background_marks_peaks_unverified_and_never_reliable():
    """Verify that when has_background=False, peaks are marked weak (unverified) and never usable,
    and the background-ratio check is skipped.
    """
    fps = 60.0
    duration_sec = 20.0
    n_samples = int(fps * duration_sec)
    t = np.arange(n_samples, dtype=np.float64) / fps

    # Even with a perfectly clean 3.7 Hz sine and high SNR
    sig = 0.2 * np.sin(2.0 * np.pi * 3.7 * t)

    peaks = compute_welch_peaks(sig, fps=fps, min_freq_hz=0.5, top_k=3, has_background=False)
    assert len(peaks) >= 1
    top_peak = peaks[0]

    # Must be marked weak / unverified and never usable
    assert top_peak.is_reliable is False, "Result must never be labelled Reliable/usable without a background"
    assert is_peak_reliable(top_peak) is False
    assert top_peak.verdict == "weak", f"Expected verdict 'weak' without background, got {top_peak.verdict}"
    assert any("no stationary background" in r.lower() for r in top_peak.reasons)

    # Background-ratio check must be skipped (bg_ratio is None)
    assert top_peak.bg_ratio is None, "Background ratio check should be skipped when has_background=False"
    assert top_peak.bg_ratio_db is None


def test_input_guards_flat_non_finite_top_k():
    """Verify input guards in compute_welch_peaks:
    1. Returns no peaks for non-finite input (NaN or Inf).
    2. Returns no peaks for top_k <= 0.
    3. Returns no peaks for a flat signal after detrending (max abs <= 1e-10).
    4. A still recording must never produce a peak.
    """
    fps = 60.0
    n_samples = 600

    # A. Non-finite inputs
    nan_sig = np.zeros(n_samples, dtype=np.float64)
    nan_sig[100] = np.nan
    assert compute_welch_peaks(nan_sig, fps=fps) == []

    inf_sig = np.zeros(n_samples, dtype=np.float64)
    inf_sig[50] = np.inf
    assert compute_welch_peaks(inf_sig, fps=fps) == []

    # B. top_k <= 0
    t = np.arange(n_samples) / fps
    valid_sine = 0.2 * np.sin(2.0 * np.pi * 3.7 * t)
    assert compute_welch_peaks(valid_sine, fps=fps, top_k=0) == []
    assert compute_welch_peaks(valid_sine, fps=fps, top_k=-2) == []

    # C. Flat signal / still recording (max abs <= 1e-10 after detrending)
    flat_zero = np.zeros(n_samples, dtype=np.float64)
    assert compute_welch_peaks(flat_zero, fps=fps) == [], "Still recording (zeros) must never produce a peak"

    flat_const = np.ones(n_samples, dtype=np.float64) * 42.123
    assert compute_welch_peaks(flat_const, fps=fps) == [], "Still recording (constant offset) must never produce a peak"

    flat_linear_drift = 0.05 * t  # Linear detrending reduces this to ~0 <= 1e-10
    assert compute_welch_peaks(flat_linear_drift, fps=fps) == [], "Pure linear drift with no vibration must produce no peaks"

    # D. determine_spectrum_verdict returns 'failed' for flat or non-finite inputs
    v_flat, r_flat = determine_spectrum_verdict([], is_flat=True)
    assert v_flat == "failed"
    assert any("flat signal" in r for r in r_flat)

    v_nan, r_nan = determine_spectrum_verdict([], is_non_finite=True)
    assert v_nan == "failed"
    assert any("non-finite" in r for r in r_nan)


def test_frequency_uncertainty_overlapping_segments():
    """Verify frequency uncertainty estimation via overlapping segments:
    On synthetic 3.7 Hz vibration, the confidence interval contains 3.7 and is narrower than 0.1 Hz.
    """
    fps = 60.0
    duration_sec = 10.0
    freq_nominal = 3.70
    n_samples = int(fps * duration_sec)
    t = np.arange(n_samples, dtype=np.float64) / fps

    sig = 0.2 * np.sin(2.0 * np.pi * freq_nominal * t)

    # 1. Direct uncertainty estimation call
    uncert = estimate_frequency_uncertainty(
        displacement=sig,
        fps=fps,
        center_freq_hz=freq_nominal,
        n_segments=5,
        overlap=0.5,
        confidence_level=0.95,
    )

    assert uncert["n_segments"] == 5
    assert uncert["ci_lower"] <= freq_nominal <= uncert["ci_upper"], (
        f"Confidence interval [{uncert['ci_lower']}, {uncert['ci_upper']}] does not contain target {freq_nominal}"
    )
    assert uncert["ci_width"] < 0.10, (
        f"Confidence interval width {uncert['ci_width']:.4f} Hz is not narrower than 0.1 Hz"
    )
    assert abs(uncert["mean_hz"] - freq_nominal) <= 0.01

    # 2. Uncertainty attributes attached to Peak objects
    peaks = compute_welch_peaks(sig, fps=fps, min_freq_hz=0.5, top_k=1)
    assert len(peaks) >= 1
    top_peak = peaks[0]

    assert hasattr(top_peak, "freq_mean")
    assert hasattr(top_peak, "freq_uncertainty")
    assert hasattr(top_peak, "freq_ci")
    assert top_peak.freq_ci_lower <= freq_nominal <= top_peak.freq_ci_upper
    assert top_peak.freq_ci_width < 0.10


def test_three_level_verdict_usable_weak_failed():
    """Verify three-level verdict system ('usable', 'weak', 'failed') and reason tracking.
    Weak input must never be described as an anomaly.
    """
    fps = 60.0
    t = np.arange(600) / fps

    # Case A: Usable (clean 3.7 Hz sine)
    sig_usable = 0.2 * np.sin(2.0 * np.pi * 3.7 * t)
    peaks_usable = compute_welch_peaks(sig_usable, fps=fps, min_freq_hz=0.5, top_k=1)
    assert len(peaks_usable) >= 1
    assert peaks_usable[0].verdict == "usable"
    assert peaks_usable[0].is_reliable is True
    assert peaks_usable[0].reasons == []

    # Case B: Weak due to drift (0.65 Hz near 0.5 Hz lower band edge)
    sig_drift = 0.2 * np.sin(2.0 * np.pi * 0.65 * t)
    peaks_drift = compute_welch_peaks(sig_drift, fps=fps, min_freq_hz=0.5, top_k=1)
    assert len(peaks_drift) >= 1
    assert peaks_drift[0].verdict == "weak"
    assert peaks_drift[0].is_reliable is False
    assert any("drift" in r.lower() for r in peaks_drift[0].reasons)
    # Weak input must never be described as an anomaly
    for r in peaks_drift[0].reasons:
        assert "anomaly" not in r.lower()

    # Case C: Weak due to missing background
    peaks_no_bg = compute_welch_peaks(sig_usable, fps=fps, min_freq_hz=0.5, top_k=1, has_background=False)
    assert len(peaks_no_bg) >= 1
    assert peaks_no_bg[0].verdict == "weak"
    assert any("no stationary background" in r.lower() for r in peaks_no_bg[0].reasons)

    # Case D: Failed on flat signal or non-finite
    analyzer = WelchSpectrumAnalyzer(fps=fps)
    out_flat = analyzer.compute_welch_psd(np.zeros(600))
    assert out_flat.verdict == "failed"
    assert out_flat.is_reliable is False
    assert any("flat" in r.lower() for r in out_flat.reasons)

    nan_arr = np.zeros(600)
    nan_arr[5] = np.nan
    out_nan = analyzer.compute_welch_psd(nan_arr)
    assert out_nan.verdict == "failed"
    assert out_nan.is_reliable is False
    assert any("non-finite" in r.lower() for r in out_nan.reasons)


def test_spectrum_units_and_scale():
    """Verify displacement and spectral peaks maintain explicit units='px' and scale=None."""
    fps = 60.0
    t = np.arange(600) / fps
    sig = 0.2 * np.sin(2.0 * np.pi * 3.7 * t)

    peaks = compute_welch_peaks(sig, fps=fps, min_freq_hz=0.5, top_k=1)
    assert len(peaks) >= 1
    assert peaks[0].units == "px"
    assert peaks[0].scale is None

    analyzer = WelchSpectrumAnalyzer(fps=fps)
    out = analyzer.compute_welch_psd(sig)
    assert out.units == "px"
    assert out.scale is None
