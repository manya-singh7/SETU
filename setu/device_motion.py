"""Analyze phone accelerometer readings and compare against camera frequency."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import signal, stats


MIN_MODAL_HZ = 0.5
MAX_MODAL_HZ = 30.0
STEADY_DT_REL_TOLERANCE = 0.05
HANDHELD_LOW_FREQUENCY_RMS_MS2 = 0.5


def _peak_and_ci(values: np.ndarray, fs: float) -> Tuple[float, Tuple[float, float], np.ndarray, np.ndarray]:
    frequencies, psd = signal.welch(
        signal.detrend(values, type="linear"), fs=fs, window="hann",
        nperseg=min(len(values), 1024), noverlap=min(len(values), 1024) // 2,
        nfft=max(min(len(values), 1024), 4096), scaling="density",
    )
    band = (frequencies >= MIN_MODAL_HZ) & (frequencies <= min(MAX_MODAL_HZ, fs / 2.0))
    if not np.any(band):
        raise ValueError("sampling rate has no bins in the structural frequency band")
    indices = np.flatnonzero(band)
    peak_index = int(indices[np.argmax(psd[band])])
    peak_hz = float(frequencies[peak_index])

    # Match validation/compare.py: five overlapping (50%) Hann-windowed
    # segments, peak frequency per segment, then a two-sided Student's t CI.
    count = 5
    seg_len = max(int(len(values) / (1 + (count - 1) * 0.5)), 32)
    step = max(int(seg_len * 0.5), 1)
    segment_peaks: List[float] = []
    for start in range(count):
        segment = values[start * step : min(start * step + seg_len, len(values))]
        if len(segment) < 32:
            continue
        f_seg, p_seg = signal.welch(
            signal.detrend(segment, type="linear"), fs=fs, window="hann",
            nperseg=len(segment), noverlap=len(segment) // 2,
            nfft=max(len(segment), 4096),
        )
        local = (f_seg >= max(MIN_MODAL_HZ, peak_hz - 1.0)) & (f_seg <= peak_hz + 1.0)
        if np.any(local):
            local_indices = np.flatnonzero(local)
            k = int(local_indices[np.argmax(p_seg[local])])
            refined = float(f_seg[k])
            if 0 < k < len(p_seg) - 1 and np.all(p_seg[k - 1:k + 2] > 0):
                y1, y2, y3 = np.log(p_seg[k - 1:k + 2])
                denominator = y1 - 2.0 * y2 + y3
                if abs(denominator) > 1e-12:
                    delta = float(np.clip(0.5 * (y1 - y3) / denominator, -0.5, 0.5))
                    refined += delta * float(f_seg[1] - f_seg[0])
            segment_peaks.append(refined)
    if len(segment_peaks) >= 2:
        arr = np.asarray(segment_peaks)
        mean = float(np.mean(arr))
        half = float(stats.t.ppf(0.975, df=len(arr) - 1) * np.std(arr, ddof=1) / np.sqrt(len(arr)))
    else:
        mean, half = peak_hz, 0.0
    return peak_hz, (mean - half, mean + half), frequencies, psd


def analyze_device_motion(
    samples: Sequence[Dict[str, float]],
    camera_frequency_hz: Optional[float] = None,
    camera_ci_hz: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """Compute per-axis Welch spectra and optionally compare camera frequency.

    A CI is estimated from overlapping segments with a two-sided Student's t
    interval. Low-frequency movement above the stated threshold marks the phone
    as handheld; no sensor agreement is claimed in that case.
    """
    if len(samples) < 64:
        raise ValueError("at least 64 accelerometer samples are required")
    try:
        times = np.asarray([float(sample["t"]) for sample in samples], dtype=np.float64)
        axes = {axis: np.asarray([float(sample[f"a{axis}"]) for sample in samples], dtype=np.float64)
                for axis in ("x", "y", "z")}
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("samples must contain finite t, ax, ay, and az values") from exc
    if not np.all(np.isfinite(times)) or any(not np.all(np.isfinite(v)) for v in axes.values()):
        raise ValueError("samples must contain finite t, ax, ay, and az values")
    dt = np.diff(times)
    if np.any(dt <= 0):
        raise ValueError("sample timestamps must be strictly increasing")
    median_dt = float(np.median(dt))
    fs = 1.0 / median_dt
    steady = bool(np.max(np.abs(dt - median_dt)) / median_dt <= STEADY_DT_REL_TOLERANCE)

    axis_results: Dict[str, Dict[str, Any]] = {}
    for axis, values in axes.items():
        peak, ci, frequencies, psd = _peak_and_ci(values, fs)
        axis_results[axis] = {
            "dominant_frequency_hz": peak,
            "confidence_interval_hz": [float(ci[0]), float(ci[1])],
            "frequencies_hz": frequencies.tolist(),
            "psd": psd.tolist(),
        }
    dominant_axis = max(axis_results, key=lambda key: float(np.max(axis_results[key]["psd"])))
    dominant = axis_results[dominant_axis]

    # Remove gravity/linear drift, then measure sub-0.5 Hz motion. This is a
    # conservative operational check, not a structural safety determination.
    low_rms_by_axis: Dict[str, float] = {}
    for axis, values in axes.items():
        detrended = signal.detrend(values, type="linear")
        if fs > 1.0:
            sos = signal.butter(4, min(0.45, fs * 0.2), btype="lowpass", fs=fs, output="sos")
            low = signal.sosfiltfilt(sos, detrended)
            low_rms_by_axis[axis] = float(np.sqrt(np.mean(low**2)))
        else:
            low_rms_by_axis[axis] = 0.0
    low_motion_rms = float(np.sqrt(sum(value**2 for value in low_rms_by_axis.values())))
    resting = low_motion_rms <= HANDHELD_LOW_FREQUENCY_RMS_MS2
    result: Dict[str, Any] = {
        "sampling_rate_hz": fs,
        "sampling_rate_steady": steady,
        "median_sample_interval_s": median_dt,
        "axes": axis_results,
        "dominant_axis": dominant_axis,
        "dominant_frequency_hz": dominant["dominant_frequency_hz"],
        "confidence_interval_hz": dominant["confidence_interval_hz"],
        "low_frequency_motion_rms_ms2": low_motion_rms,
        "resting_on_structure": resting,
        "comparison": None,
        "message": None,
    }
    if not resting:
        result["message"] = "not resting on the structure; comparison not valid"
        return result
    if camera_frequency_hz is not None:
        camera_frequency = float(camera_frequency_hz)
        if not np.isfinite(camera_frequency) or camera_frequency <= 0:
            raise ValueError("camera_frequency_hz must be positive and finite")
        accel_ci = dominant["confidence_interval_hz"]
        accel_half = abs(float(accel_ci[1]) - float(accel_ci[0])) / 2.0
        camera_half = 0.0
        camera_interval = None
        if camera_ci_hz is not None:
            if len(camera_ci_hz) != 2:
                raise ValueError("camera_ci_hz must be a [low, high] pair")
            camera_interval = [float(camera_ci_hz[0]), float(camera_ci_hz[1])]
            if not np.all(np.isfinite(camera_interval)) or camera_interval[0] > camera_interval[1]:
                raise ValueError("camera_ci_hz must be a finite increasing interval")
            camera_half = (camera_interval[1] - camera_interval[0]) / 2.0
        combined = float(np.sqrt(accel_half**2 + camera_half**2))
        difference = abs(camera_frequency - float(dominant["dominant_frequency_hz"]))
        agrees = difference <= combined
        result["comparison"] = {
            "camera_frequency_hz": camera_frequency,
            "camera_confidence_interval_hz": camera_interval,
            "difference_hz": difference,
            "combined_uncertainty_hz": combined,
            "agreement": agrees,
            "verdict": "PASS" if agrees else "FAIL",
        }
    return result
