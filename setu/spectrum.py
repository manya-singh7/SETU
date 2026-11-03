"""
Spectral analysis and dominant modal peak detection module for Project SETU.

Computes the Power Spectral Density (PSD) using Welch's method (linear detrending,
Hann windowing, and segment averaging) from a 1D subpixel structural displacement signal.
Extracts top dominant frequency peaks as Peak objects with sub-bin quadratic interpolation,
relative spectral power, peak-prominence signal-to-noise ratio (SNR), and frequency
uncertainty confidence intervals, ignoring low-frequency drift (< 0.5 Hz).

Spectral Metrics & Multi-Tier Reliability Definitions:
1. Relative Power: Defined as the peak band's integrated power (sum of PSD energy across
   the peak's main lobe bounded by local minima / troughs or a +/- 0.4 Hz band) divided by
   the total power in the analysis band (0.5 Hz to Nyquist). For a clean sinusoidal tone,
   this metric concentrates > 80% of total spectral power in the resonant lobe.
2. Peak SNR (dB): Defined as the peak power spectral density (P_peak) over the median noise
   floor (P_noise) across the analysis band (0.5 Hz to Nyquist), reported in decibels:
       SNR_dB = 10 * log10(P_peak / P_noise)
3. Background-Referenced Ratio Check: Compares target PSD to the stationary background
   residual PSD at the peak frequency: P_target(f0) / P_background(f0).
   Threshold: MIN_BG_RATIO = 500.0 (approx 27.0 dB). If the target does not exceed background
   residual motion by at least this factor, the peak is marked unreliable.
4. Local Prominence Check: Peak power versus neighbouring bins within +/- 1.0 Hz (excluding
   the peak's own main lobe): 10 * log10(P_peak / median(P_neighbours)).
   Threshold: LOCAL_PROMINENCE_THRESHOLD_DB = 20.0 dB. Real structural resonance produces sharp,
   narrow spectral spikes; broad 1/f noise drift produces low local prominence (< 20 dB).
5. Lower Band Edge Drift Check: Peaks within 0.3 Hz of the lower analysis boundary
   (f0 <= min_freq_hz + 0.3 = 0.8 Hz) are flagged as "possible drift" and marked unreliable.

Frequency Uncertainty Estimation:
   The displacement recording is split into K overlapping Hann-windowed segments (default K=5,
   50% overlap). The resonant peak frequency f_i is extracted per segment using sub-bin parabolic
   interpolation on the segment PSD. Frequency is reported as sample mean +/- two-tailed Student's
   t confidence interval at 95% confidence (df = K - 1).

Three-Level Verdict System:
   Per the SETU team plan, evaluation produces one of three standardized verdicts:
   - "usable": All reliability checks pass (stationary background present, global SNR >= 10 dB,
     not in lower drift margin, local prominence >= 20 dB, BG ratio >= 500x).
   - "weak": A candidate peak exists but any reliability check fails, or there is no stationary
     background (unverified), or target sensor saturation/clipping is detected.
   - "failed": Invalid or non-finite input, flat signal (max abs <= 1e-10), no peaks detected,
     or motion tracking failure.
   Note on Classification:
   Weak input must NEVER be described as an anomaly. A "weak" verdict reflects measurement,
   sensor dynamic range, or environmental reference limitations requiring engineer triage.

Physical Units Note:
   Displacement time series and spectral quantities are expressed in pixel units (units="px",
   scale=None). Converting pixel displacement to physical millimetres (mm) requires a known
   physical reference length in the camera plane at the bridge structure (e.g. pillar diameter,
   bolt spacing, or surveyor scale bar) to calibrate pixels per millimetre.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import scipy.signal as signal
import scipy.stats as stats

from setu.types import Peak

# ===========================================================================
# Reliability Thresholds (Calibrated on Synthetic Benchmarks)
# NOTE: These thresholds were set on synthetic data only and must be
# recalibrated on real recordings (physical accelerometers / vibrometers).
# ===========================================================================

RELIABILITY_THRESHOLD_SNR_DB: float = 10.0
"""Global peak-to-median-noise SNR threshold in decibels (dB)."""

LOCAL_PROMINENCE_THRESHOLD_DB: float = 20.0
"""Local prominence threshold in dB (peak power over median of neighbouring bins within +/- 1.0 Hz).

Justification:
True mechanical structural resonances have narrow spectral bandwidths (Lorentzian/Hann lobes),
yielding high local prominence (>= 20.0 dB) above immediate neighbouring spectral bins.
Conversely, broad 1/f noise slopes and low-frequency drift lack sharp spectral confinement,
reporting low local prominence (< 20.0 dB) despite elevated broadband energy.
"""

MIN_BG_RATIO: float = 500.0
"""Minimum target-to-background spectral power ratio at the peak frequency (approx 27.0 dB).

Justification:
Structural vibration of a bridge pillar or truss must substantially exceed residual motion
observed in the stationary reference background after camera jitter cancellation. When the
target PSD at the peak frequency does not exceed the background residual PSD by at least 500x
(27 dB), the detected oscillation cannot be reliably differentiated from camera micro-jitter
tracking leakage or mount vibration.
"""

LOWER_BAND_EDGE_MARGIN_HZ: float = 0.3
"""Margin in Hz above min_freq_hz below which peaks are flagged as 'possible drift'.

Justification:
In mobile camera recordings, camera tilt drift, thermal drift, and 1/f sensor noise dominate
the region immediately adjacent to the highpass cutoff (0.5 Hz - 0.8 Hz). Peaks within 0.3 Hz
of the lower band edge are suspect and flagged as 'possible drift'.
"""


def is_peak_reliable(
    peak: Peak,
    threshold_snr_db: float = RELIABILITY_THRESHOLD_SNR_DB,
    min_bg_ratio: float = MIN_BG_RATIO,
    min_local_prom_db: float = LOCAL_PROMINENCE_THRESHOLD_DB,
    min_freq_hz: float = 0.5,
) -> bool:
    """Assess whether a detected peak meets all reliability criteria.

    Returns True if verdict is 'usable' (all checks pass), False if 'weak' or 'failed'.
    """
    verdict = getattr(peak, "verdict", None)
    if verdict is not None:
        return verdict == "usable"

    if not getattr(peak, "has_background", True):
        return False
    if not getattr(peak, "is_reliable", True):
        return False

    if peak.snr < threshold_snr_db:
        return False

    is_drift = getattr(peak, "is_drift", None)
    if is_drift is None:
        is_drift = (peak.freq_hz - min_freq_hz) <= LOWER_BAND_EDGE_MARGIN_HZ
    if is_drift:
        return False

    local_prom = getattr(peak, "local_prom_db", None)
    if local_prom is not None and local_prom < min_local_prom_db:
        return False

    bg_ratio = getattr(peak, "bg_ratio", None)
    if bg_ratio is not None and bg_ratio < min_bg_ratio:
        return False

    return True


@dataclass
class SpectrumOutput:
    """Spectral analysis telemetry container preserved for backward compatibility.

    Physical Units Note:
        Displacement is in pixels (units='px', scale=None). Converting to millimetres (mm)
        requires a known physical reference length in the camera plane.
    """

    frequencies_hz: npt.NDArray[np.float64]
    psd: npt.NDArray[np.float64]
    dominant_peaks: List[Peak]
    fundamental_frequency_hz: Optional[float]
    snr_db: float = 0.0
    is_reliable: bool = True
    confidence: str = "usable"
    verdict: Literal["usable", "weak", "failed"] = "usable"
    reasons: List[str] = field(default_factory=list)
    units: str = "px"
    scale: Optional[float] = None
    freq_uncertainty: Optional[float] = None
    freq_ci: Optional[Tuple[float, float]] = None


def extract_background_residual_motion(
    video_path: str,
    background_bbox: Tuple[int, int, int, int],
    max_corners: int = 40,
) -> npt.NDArray[np.float64]:
    """Track subpixel background residual motion across frames after camera stabilization.

    Args:
        video_path: Path to input MP4 video file.
        background_bbox: Bounding box tuple (x, y, w, h) for stationary background anchor.
        max_corners: Maximum corner features to track in background anchor.

    Returns:
        1D numpy array of residual background displacement across frames.
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return np.zeros(0, dtype=np.float64)

    ret, frame0 = cap.read()
    if not ret or frame0 is None:
        cap.release()
        return np.zeros(0, dtype=np.float64)

    gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
    h, w = gray0.shape
    bx, by, bw, bh = background_bbox
    bg_mask = np.zeros(gray0.shape, dtype=np.uint8)
    bg_mask[by : by + bh, bx : bx + bw] = 255

    p_bg0 = cv2.goodFeaturesToTrack(gray0, maxCorners=max_corners, qualityLevel=0.02, minDistance=8, mask=bg_mask)
    if p_bg0 is None or len(p_bg0) < 4:
        ys, xs = np.mgrid[by + 10 : by + bh - 10 : 12, bx + 10 : bx + bw - 10 : 12]
        p_bg0 = np.vstack((xs.flatten(), ys.flatten())).T.astype(np.float32)[:, np.newaxis, :]

    disp_x = [0.0]
    M_prev = np.float32([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        p_curr, st, _ = cv2.calcOpticalFlowPyrLK(gray0, gray, p_bg0, None, winSize=(15, 15), maxLevel=2)
        st_flat = (st.ravel() == 1) if st is not None else np.zeros(len(p_bg0), bool)
        good0 = p_bg0[st_flat]
        good_curr = p_curr[st_flat]

        if len(good_curr) >= 4:
            M, _ = cv2.estimateAffinePartial2D(good_curr, good0, method=cv2.RANSAC, ransacReprojThreshold=2.0)
            if M is None:
                M = M_prev.copy()
            else:
                M_prev = M.copy()
        else:
            M = M_prev.copy()

        warped = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        p_res, st_res, _ = cv2.calcOpticalFlowPyrLK(gray0, warped, p_bg0, None, winSize=(11, 11), maxLevel=1)
        if st_res is not None and np.sum(st_res) > 0:
            diffs = p_res[st_res.ravel() == 1] - p_bg0[st_res.ravel() == 1]
            disp_x.append(float(np.median(diffs[:, :, 0])))
        else:
            disp_x.append(0.0)

    cap.release()
    return np.array(disp_x, dtype=np.float64)


def detect_active_segments(
    displacement: Union[npt.NDArray[np.float64], List[float]],
    fps: float,
    threshold_ratio: float = 4.0,
    min_quiet_fraction: float = 0.35,
) -> List[Tuple[int, int]]:
    """Detect active transient vibration bursts (e.g. ring-down events or hammer strikes).

    For continuous harmonic vibration (such as steady sinusoidal excitation or ambient wind),
    the short-time energy envelope remains relatively uniform across time (p95 / p50 < 4.0
    or quiet fraction < 0.35), and the full signal [(0, N)] is returned.

    For transient ring-down events (such as a flicked ruler demo), vibration consists of
    intermittent decaying bursts separated by quiescent stillness. This function isolates
    each active burst interval from onset through decay, excluding quiet background intervals.

    Args:
        displacement: 1D array of subpixel displacement values.
        fps: Sampling frame rate in Hz.
        threshold_ratio: Ratio of 95th percentile to median RMS energy above which
            a signal is classified as transient / burst-like (default 4.0).
        min_quiet_fraction: Minimum fraction of time the signal must be in quiescence
            (< 25% of p95 RMS) to qualify as intermittent transient ring-downs (default 0.35).

    Returns:
        List of (start_index, end_index) frame tuples for active vibration segments.
    """
    sig = np.asarray(displacement, dtype=np.float64).ravel()
    N = len(sig)
    if N < int(fps * 1.5) or fps <= 0.0 or not np.all(np.isfinite(sig)):
        return [(0, N)]

    sig_detrended = signal.detrend(sig, type="linear")
    win = max(int(fps * 0.15), 4)
    rms = np.sqrt(np.convolve(sig_detrended**2, np.ones(win) / win, mode="same"))

    p95 = float(np.percentile(rms, 95))
    p50 = float(np.percentile(rms, 50))
    quiet_frac = float(np.mean(rms < 0.25 * p95))

    if p95 / max(p50, 1e-9) < threshold_ratio or quiet_frac < min_quiet_fraction:
        return [(0, N)]

    diff_rms = np.diff(rms)
    min_dist = int(fps * 2.5)
    max_diff = float(np.max(diff_rms)) if len(diff_rms) > 0 else 0.0
    height = 0.05 * max_diff if max_diff > 0 else 1e-6
    onsets, _ = signal.find_peaks(diff_rms, height=height, distance=min_dist)

    if len(onsets) == 0:
        thresh = max(0.10 * p95, 2.5 * p50)
        act = rms >= thresh
        pad = int(fps * 0.3)
        act_p = np.convolve(act.astype(float), np.ones(pad), mode="same") > 0
        diff_a = np.diff(act_p.astype(int))
        st = np.where(diff_a == 1)[0] + 1
        if act_p[0]:
            st = np.r_[0, st]
        en = np.where(diff_a == -1)[0] + 1
        if act_p[-1]:
            en = np.r_[en, len(act_p)]
        regs = [(int(s), int(e)) for s, e in zip(st, en) if e - s >= int(fps * 0.5)]
        return regs if len(regs) >= 1 else [(0, N)]

    segments: List[Tuple[int, int]] = []
    max_dur = int(fps * 5.0)
    for i, ons in enumerate(onsets):
        start = max(0, int(ons) - int(fps * 0.05))
        if i + 1 < len(onsets):
            end = min(int(onsets[i + 1]), start + max_dur, N)
        else:
            end = min(start + max_dur, N)
        if end - start >= int(fps * 0.5):
            segments.append((start, end))

    return segments if segments else [(0, N)]


def estimate_frequency_uncertainty(
    displacement: Union[npt.NDArray[np.float64], List[float]],
    fps: float,
    center_freq_hz: float,
    n_segments: int = 5,
    overlap: float = 0.5,
    confidence_level: float = 0.95,
    search_half_width_hz: float = 0.8,
    active_segments: Optional[List[Tuple[int, int]]] = None,
) -> Dict[str, Any]:
    """Estimate modal frequency and confidence interval using overlapping Hann-windowed segments.

    Methodology:
    1. Splits the detrended displacement signal of length N into K overlapping segments (default K=5,
       50% overlap).
    2. For each segment, applies a Hann window and computes Welch PSD with high FFT resolution (nfft>=4096).
    3. Identifies the resonant peak nearest center_freq_hz (within +/- search_half_width_hz) and refines
       it with parabolic interpolation on the log-PSD for sub-bin precision.
    4. Computes the sample mean and sample standard deviation across segment estimates:
           mean_f = (1 / K) * sum(f_i)
           s = sqrt((1 / (K - 1)) * sum((f_i - mean_f)^2))
    5. Computes a two-tailed Student's t confidence interval:
           CI = mean_f +/- t_crit * (s / sqrt(K))
       where t_crit = t_{1 - alpha/2, K-1}. For K=5 and 95% confidence, t_{0.975, 4} ~ 2.776.

    Args:
        displacement: 1D array of subpixel displacement values in pixels.
        fps: Video frame rate in frames per second.
        center_freq_hz: Dominant candidate peak frequency from full-signal analysis in Hz.
        n_segments: Number of overlapping segments (default 5, typically 4 to 6).
        overlap: Overlap fraction between consecutive segments (default 0.5 = 50%).
        confidence_level: Confidence level for Student's t interval (default 0.95 = 95%).
        search_half_width_hz: Spectral search window half-width around center_freq_hz (default 0.8 Hz).

    Returns:
        Dictionary containing mean_hz, std_hz, ci_lower, ci_upper, ci_half_width, ci_width,
        confidence_level, n_segments, segment_freqs, and method description.
    """
    sig = np.asarray(displacement, dtype=np.float64).ravel()
    if len(sig) < 8 or fps <= 0.0 or not np.all(np.isfinite(sig)):
        return {
            "mean_hz": center_freq_hz,
            "std_hz": 0.0,
            "ci_lower": center_freq_hz,
            "ci_upper": center_freq_hz,
            "ci_half_width": 0.0,
            "ci_width": 0.0,
            "confidence_level": confidence_level,
            "n_segments": 0,
            "segment_freqs": [],
            "method": "insufficient_data",
        }

    sig_detrended = signal.detrend(sig, type="linear")
    if np.max(np.abs(sig_detrended)) <= 1e-10:
        return {
            "mean_hz": center_freq_hz,
            "std_hz": 0.0,
            "ci_lower": center_freq_hz,
            "ci_upper": center_freq_hz,
            "ci_half_width": 0.0,
            "ci_width": 0.0,
            "confidence_level": confidence_level,
            "n_segments": 0,
            "segment_freqs": [],
            "method": "flat_signal",
        }

    N = len(sig_detrended)
    if active_segments is None:
        active_segments = detect_active_segments(sig_detrended, fps)

    use_active_segments = len(active_segments) >= 2
    segment_freqs: List[float] = []

    if use_active_segments:
        for start_idx, end_idx in active_segments:
            seg = sig_detrended[start_idx:end_idx]
            if len(seg) < 32:
                continue

            seg_det = signal.detrend(seg, type="linear")
            nfft = max(len(seg_det), 4096)
            f_axis, psd = signal.welch(
                seg_det,
                fs=float(fps),
                window="hann",
                nperseg=len(seg_det),
                nfft=nfft,
                detrend=False,
            )

            f_low = max(0.5, center_freq_hz - search_half_width_hz)
            f_high = center_freq_hz + search_half_width_hz
            mask = (f_axis >= f_low) & (f_axis <= f_high)
            if not np.any(mask):
                mask = f_axis >= 0.5
            if not np.any(mask):
                continue

            f_band = f_axis[mask]
            p_band = psd[mask]
            k = int(np.argmax(p_band))
            f0 = float(f_band[k])
            df = float(f_band[1] - f_band[0]) if len(f_band) > 1 else 1.0

            if 0 < k < len(p_band) - 1 and p_band[k - 1] > 0 and p_band[k] > 0 and p_band[k + 1] > 0:
                y1 = np.log(p_band[k - 1])
                y2 = np.log(p_band[k])
                y3 = np.log(p_band[k + 1])
                denom = y1 - 2.0 * y2 + y3
                delta = 0.5 * (y1 - y3) / denom if abs(denom) > 1e-12 else 0.0
                delta = float(np.clip(delta, -0.5, 0.5))
                f_ref = f_band[k] + delta * df
            else:
                f_ref = f0

            segment_freqs.append(float(f_ref))
    else:
        K = max(int(n_segments), 2)
        step_factor = 1.0 + (K - 1) * (1.0 - overlap)
        seg_len = int(N / step_factor)
        seg_len = max(seg_len, 32)
        step_size = max(int(seg_len * (1.0 - overlap)), 1)

        for i in range(K):
            start_idx = i * step_size
            end_idx = min(start_idx + seg_len, N)
            seg = sig_detrended[start_idx:end_idx]
            if len(seg) < 32:
                continue

            seg_det = signal.detrend(seg, type="linear")
            nfft = max(len(seg_det), 4096)
            f_axis, psd = signal.welch(
                seg_det,
                fs=float(fps),
                window="hann",
                nperseg=len(seg_det),
                nfft=nfft,
                detrend=False,
            )

            f_low = max(0.5, center_freq_hz - search_half_width_hz)
            f_high = center_freq_hz + search_half_width_hz
            mask = (f_axis >= f_low) & (f_axis <= f_high)
            if not np.any(mask):
                mask = f_axis >= 0.5
            if not np.any(mask):
                continue

            f_band = f_axis[mask]
            p_band = psd[mask]
            k = int(np.argmax(p_band))
            f0 = float(f_band[k])
            df = float(f_band[1] - f_band[0]) if len(f_band) > 1 else 1.0

            if 0 < k < len(p_band) - 1 and p_band[k - 1] > 0 and p_band[k] > 0 and p_band[k + 1] > 0:
                y1 = np.log(p_band[k - 1])
                y2 = np.log(p_band[k])
                y3 = np.log(p_band[k + 1])
                denom = y1 - 2.0 * y2 + y3
                delta = 0.5 * (y1 - y3) / denom if abs(denom) > 1e-12 else 0.0
                delta = float(np.clip(delta, -0.5, 0.5))
                f_ref = f_band[k] + delta * df
            else:
                f_ref = f0

            segment_freqs.append(float(f_ref))

    if len(segment_freqs) < 2:
        mean_hz = float(np.mean(segment_freqs)) if segment_freqs else center_freq_hz
        return {
            "mean_hz": round(mean_hz, 4),
            "std_hz": 0.0,
            "ci_lower": round(mean_hz, 4),
            "ci_upper": round(mean_hz, 4),
            "ci_half_width": 0.0,
            "ci_width": 0.0,
            "confidence_level": confidence_level,
            "n_segments": len(segment_freqs),
            "segment_freqs": [round(f, 4) for f in segment_freqs],
            "method": "single_segment_fallback",
        }

    freqs_arr = np.array(segment_freqs, dtype=np.float64)
    mean_hz = float(np.mean(freqs_arr))
    std_hz = float(np.std(freqs_arr, ddof=1))

    alpha = 1.0 - confidence_level
    dof = len(freqs_arr) - 1
    t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=dof))

    ci_half_width = float(t_crit * std_hz / np.sqrt(len(freqs_arr)))
    ci_lower = float(mean_hz - ci_half_width)
    ci_upper = float(mean_hz + ci_half_width)
    ci_width = float(ci_upper - ci_lower)

    if use_active_segments:
        method_desc = (
            f"Active transient burst ensemble ({len(freqs_arr)} ring-down segments). "
            f"Two-tailed Student's t-interval at {int(confidence_level * 100)}% confidence (df={dof}, t_crit={t_crit:.3f})."
        )
    else:
        method_desc = (
            f"Overlapping Hann-windowed Welch ensemble ({len(freqs_arr)} segments, {int(overlap * 100)}% overlap). "
            f"Two-tailed Student's t-interval at {int(confidence_level * 100)}% confidence (df={dof}, t_crit={t_crit:.3f})."
        )

    return {
        "mean_hz": round(mean_hz, 4),
        "std_hz": round(std_hz, 5),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "ci_half_width": round(ci_half_width, 4),
        "ci_width": round(ci_width, 4),
        "confidence_level": confidence_level,
        "n_segments": len(segment_freqs),
        "segment_freqs": [round(f, 4) for f in segment_freqs],
        "method": method_desc,
    }


def determine_spectrum_verdict(
    peaks: List[Peak],
    has_background: bool = True,
    is_flat: bool = False,
    is_non_finite: bool = False,
    tracking_failed: bool = False,
    clipping_detected: bool = False,
    tracking_error: Optional[str] = None,
) -> Tuple[Literal["usable", "weak", "failed"], List[str]]:
    """Determine three-level quality verdict per SETU team plan.

    Verdicts:
    - failed: invalid or non-finite input, flat signal, no peaks, or motion tracking failure.
    - weak: a candidate peak exists but any reliability check fails, or there is no stationary
      background (unverified), or sensor saturation/clipping is detected.
    - usable: all reliability checks pass.

    Note on Classification:
    Weak input must NEVER be described as an anomaly. It reflects measurement,
    sensor dynamic range, or environmental reference limitations requiring engineer triage.
    """
    reasons: List[str] = []

    if is_non_finite:
        return "failed", ["non-finite input (NaN or Inf in displacement signal)"]

    if is_flat:
        return "failed", ["flat signal after detrending (max abs <= 1e-10)"]

    if tracking_failed:
        msg = f"motion tracking failure: {tracking_error}" if tracking_error else "motion tracking failure"
        return "failed", [msg]

    if not peaks:
        return "failed", ["no dominant spectral peaks detected above cutoff frequency"]

    top_peak = peaks[0]
    if not has_background:
        reasons.append("unverified: no stationary background in frame")

    if clipping_detected:
        reasons.append("target mask pixel saturation/clipping detected (saturation reduces sub-pixel accuracy)")

    if hasattr(top_peak, "reasons") and top_peak.reasons:
        for r in top_peak.reasons:
            if r not in reasons:
                reasons.append(r)
    elif hasattr(top_peak, "flags") and top_peak.flags:
        for f in top_peak.flags:
            if f not in reasons:
                reasons.append(f)

    if len(reasons) > 0:
        return "weak", reasons

    return "usable", []


def compute_welch_peaks(
    displacement: Union[npt.NDArray[np.float64], List[float]],
    fps: float,
    min_freq_hz: float = 0.5,
    top_k: int = 3,
    nperseg: Optional[int] = None,
    nfft: Optional[int] = None,
    background_displacement: Optional[Union[npt.NDArray[np.float64], List[float]]] = None,
    has_background: bool = True,
    active_segments_only: bool = True,
) -> List[Peak]:
    """Given a 1D displacement signal and fps, compute Welch PSD and extract top peaks.

    Multi-Tier Reliability Checks:
    1. Frequency uncertainty: Computes confidence interval via overlapping Hann-windowed segments
       or active transient burst ensemble.
    2. Three-level verdict: "usable" if all reliability checks pass, "weak" if any check fails or no BG.
    3. Physical units: Displacement remains in pixels (units="px", scale=None).

    Args:
        displacement: 1D array of subpixel structural displacements (in pixels).
        fps: Sampling rate (frames per second).
        min_freq_hz: Frequency cutoff below which spectral content is ignored (default 0.5 Hz).
        top_k: Number of dominant peaks to return (default 3).
        nperseg: Length of each segment for Welch averaging. Defaults to min(len(signal), 1024).
        nfft: FFT length. Defaults to max(nperseg, 4096) for high resolution.
        background_displacement: Optional 1D displacement of stationary background after stabilization.
        has_background: Whether a stationary background anchor is present in frame (default True).
        active_segments_only: If True, detects active transient ring-down bursts when present
            and computes PSD across active vibration intervals (default True).

    Returns:
        List of up to top_k Peak objects sorted by prominence/power.
    """
    sig = np.asarray(displacement, dtype=np.float64).ravel()
    if len(sig) < 8 or fps <= 0.0:
        return []

    # 1. Linear detrending
    sig_detrended = signal.detrend(sig, type="linear")

    # 2. Windowing and Welch PSD computation
    n_samples = len(sig_detrended)
    active_segments = (
        detect_active_segments(sig_detrended, fps)
        if active_segments_only and nperseg is None
        else [(0, n_samples)]
    )
    is_transient = len(active_segments) >= 2 or (
        len(active_segments) == 1 and (active_segments[0][1] - active_segments[0][0]) < n_samples * 0.8
    )

    if is_transient and len(active_segments) >= 1:
        fft_len = nfft if nfft is not None else 4096
        psd_list: List[npt.NDArray[np.float64]] = []
        bg_psd_list: List[npt.NDArray[np.float64]] = []

        for s_idx, e_idx in active_segments:
            seg = sig_detrended[s_idx:e_idx]
            if len(seg) < 32:
                continue
            seg_det = signal.detrend(seg, type="linear")
            freqs, p_seg = signal.welch(
                seg_det,
                fs=float(fps),
                window="hann",
                nperseg=len(seg_det),
                nfft=fft_len,
                detrend=False,
            )
            psd_list.append(p_seg)

            if has_background and background_displacement is not None:
                bg_sig = np.asarray(background_displacement, dtype=np.float64).ravel()
                if len(bg_sig) >= e_idx and np.all(np.isfinite(bg_sig[s_idx:e_idx])):
                    bg_seg = signal.detrend(bg_sig[s_idx:e_idx], type="linear")
                    _, p_bg_seg = signal.welch(
                        bg_seg,
                        fs=float(fps),
                        window="hann",
                        nperseg=len(bg_seg),
                        nfft=fft_len,
                        detrend=False,
                    )
                    bg_psd_list.append(p_bg_seg)

        if psd_list:
            psd = np.mean(psd_list, axis=0)
            f_bg = freqs if bg_psd_list else None
            p_bg = np.mean(bg_psd_list, axis=0) if bg_psd_list else None
        else:
            seg_len = min(n_samples, 1024)
            fft_len = nfft if nfft is not None else max(seg_len, 4096)
            freqs, psd = signal.welch(
                sig_detrended,
                fs=float(fps),
                window="hann",
                nperseg=seg_len,
                noverlap=seg_len // 2,
                nfft=fft_len,
                detrend=False,
            )
            f_bg, p_bg = None, None
    else:
        seg_len = nperseg if nperseg is not None else min(n_samples, 1024)
        fft_len = nfft if nfft is not None else max(seg_len, 4096)

        freqs, psd = signal.welch(
            sig_detrended,
            fs=float(fps),
            window="hann",
            nperseg=seg_len,
            noverlap=seg_len // 2,
            nfft=fft_len,
            detrend=False,
        )

        # 4. Optional background PSD computation (only if stationary background is present)
        f_bg, p_bg = None, None
        if has_background and background_displacement is not None:
            bg_sig = np.asarray(background_displacement, dtype=np.float64).ravel()
            if len(bg_sig) >= 8 and np.all(np.isfinite(bg_sig)):
                bg_detrended = signal.detrend(bg_sig, type="linear")
                f_bg, p_bg = signal.welch(
                    bg_detrended,
                    fs=float(fps),
                    window="hann",
                    nperseg=seg_len,
                    noverlap=seg_len // 2,
                    nfft=fft_len,
                    detrend=False,
                )

    # 3. Filter band (ignore content below min_freq_hz)
    valid_mask = freqs >= min_freq_hz
    if not np.any(valid_mask):
        return []

    f_band = freqs[valid_mask]
    p_band = psd[valid_mask]

    total_band_power = float(np.sum(p_band))
    if total_band_power <= 0.0:
        total_band_power = 1e-12

    # Noise floor estimated via median PSD in valid band
    noise_floor = float(np.median(p_band))
    if noise_floor <= 0.0:
        noise_floor = float(np.mean(p_band) * 0.1) + 1e-12

    # 5. Peak detection using prominence
    peak_indices, properties = signal.find_peaks(p_band, prominence=0.0)

    if len(peak_indices) == 0:
        peak_indices = np.array([int(np.argmax(p_band))])
        prominences = np.array([float(p_band[peak_indices[0]])])
    else:
        prominences = properties["prominences"]

    # Order peaks descending by prominence
    order = np.argsort(prominences)[::-1][:top_k]

    df = float(f_band[1] - f_band[0]) if len(f_band) > 1 else 1.0
    peaks: List[Peak] = []

    for idx in order:
        k = int(peak_indices[idx])
        peak_psd = float(p_band[k])
        f0 = float(f_band[k])

        # Parabolic interpolation around peak bin for sub-bin frequency accuracy
        if 0 < k < len(p_band) - 1 and p_band[k - 1] > 0 and p_band[k] > 0 and p_band[k + 1] > 0:
            y1 = np.log(p_band[k - 1])
            y2 = np.log(p_band[k])
            y3 = np.log(p_band[k + 1])
            denom = y1 - 2.0 * y2 + y3
            delta = 0.5 * (y1 - y3) / denom if abs(denom) > 1e-12 else 0.0
            delta = float(np.clip(delta, -0.5, 0.5))
            f_refined = f_band[k] + delta * df
        else:
            f_refined = f_band[k]

        # Calculate peak band power: integrate PSD across peak's main lobe
        if is_transient:
            max_lobe_half_width = max(0.4, min(1.2, 0.15 * f0))
            neigh_search_half_width = max(1.0, max_lobe_half_width + 0.6)
        else:
            max_lobe_half_width = 0.4
            neigh_search_half_width = 1.0

        left_idx = k
        while left_idx > 0 and p_band[left_idx - 1] <= p_band[left_idx] and (f0 - f_band[left_idx - 1]) <= max_lobe_half_width:
            left_idx -= 1

        right_idx = k
        while right_idx < len(p_band) - 1 and p_band[right_idx + 1] <= p_band[right_idx] and (f_band[right_idx + 1] - f0) <= max_lobe_half_width:
            right_idx += 1

        peak_band_power = float(np.sum(p_band[left_idx : right_idx + 1]))
        rel_power = peak_band_power / total_band_power
        rel_power = float(np.clip(rel_power, 0.0, 1.0))

        # Peak SNR in dB: 10 * log10(P_peak / P_noise_median)
        snr_ratio = max(peak_psd, 1e-12) / max(noise_floor, 1e-12)
        snr_db = 10.0 * np.log10(snr_ratio)

        # 6. Lower band edge drift check: within 0.3 Hz of min_freq_hz
        is_drift = bool((f_refined - min_freq_hz) <= LOWER_BAND_EDGE_MARGIN_HZ)

        # 7. Local prominence check: neighbouring bins excluding lobe
        neigh_mask = (f_band >= f0 - neigh_search_half_width) & (f_band <= f0 + neigh_search_half_width) & ((f_band < f_band[left_idx]) | (f_band > f_band[right_idx]))
        p_neigh = float(np.median(p_band[neigh_mask])) if np.any(neigh_mask) else noise_floor
        local_prom_db = float(round(10.0 * np.log10(max(peak_psd, 1e-12) / max(p_neigh, 1e-12)), 2))

        # 8. Background-referenced check: target PSD / background PSD at peak frequency
        if has_background:
            if f_bg is not None and p_bg is not None:
                p_bg_at_peak = float(np.interp(f_refined, f_bg, p_bg))
                bg_ratio = float(peak_psd / max(p_bg_at_peak, 1e-12))
                bg_ratio_db = float(round(10.0 * np.log10(bg_ratio), 2))
            else:
                bg_ratio = float("inf")
                bg_ratio_db = float("inf")
        else:
            # Skip background-ratio check when no stationary background is present
            bg_ratio = None
            bg_ratio_db = None

        # 9. Diagnostic reasons and three-level verdict
        reasons: List[str] = []
        if not has_background:
            reasons.append("unverified: no stationary background in frame")
        if is_drift:
            reasons.append(f"possible drift (within {LOWER_BAND_EDGE_MARGIN_HZ:.1f} Hz of lower band edge {min_freq_hz:.1f} Hz)")
        if local_prom_db < LOCAL_PROMINENCE_THRESHOLD_DB:
            reasons.append(f"low local prominence ({local_prom_db:.1f} dB < {LOCAL_PROMINENCE_THRESHOLD_DB:.1f} dB)")
        if has_background and bg_ratio is not None and bg_ratio < MIN_BG_RATIO:
            reasons.append(f"low target-to-background ratio ({bg_ratio:.1f} < {MIN_BG_RATIO:.1f})")
        if snr_db < RELIABILITY_THRESHOLD_SNR_DB:
            reasons.append(f"low global SNR ({snr_db:.1f} dB < {RELIABILITY_THRESHOLD_SNR_DB:.1f} dB)")

        if len(reasons) == 0:
            verdict: Literal["usable", "weak", "failed"] = "usable"
            is_reliable = True
        else:
            verdict = "weak"
            is_reliable = False

        # Frequency uncertainty estimation using overlapping segments or active transient bursts
        uncert = estimate_frequency_uncertainty(
            displacement=sig,
            fps=fps,
            center_freq_hz=f_refined,
            n_segments=5,
            overlap=0.5,
            confidence_level=0.95,
            active_segments=active_segments if is_transient else None,
        )

        p_obj = Peak(
            freq_hz=float(round(f_refined, 4)),
            rel_power=float(round(rel_power, 6)),
            snr=float(round(snr_db, 2)),
            verdict=verdict,
            reasons=reasons,
            units="px",
            scale=None,
            freq_uncertainty=uncert["ci_half_width"],
            freq_ci=(uncert["ci_lower"], uncert["ci_upper"]),
            is_reliable=is_reliable,
            bg_ratio=bg_ratio,
            bg_ratio_db=bg_ratio_db,
        )

        # Additional diagnostic attributes
        p_obj.has_background = has_background
        p_obj.is_drift = is_drift
        p_obj.local_prom_db = local_prom_db
        p_obj.freq_mean = uncert["mean_hz"]
        p_obj.freq_ci_lower = uncert["ci_lower"]
        p_obj.freq_ci_upper = uncert["ci_upper"]
        p_obj.freq_ci_width = uncert["ci_width"]
        p_obj.freq_ci_method = uncert["method"]

        peaks.append(p_obj)

    return peaks


# Convenient aliases
analyze_spectrum = compute_welch_peaks
compute_peaks = compute_welch_peaks


class WelchSpectrumAnalyzer:
    """Class wrapper for computing Welch PSD and dominant resonant peaks."""

    def __init__(self, fps: float = 60.0, min_freq_hz: float = 0.5, top_k: int = 3) -> None:
        self.fps = fps
        self.min_freq_hz = min_freq_hz
        self.top_k = top_k

    def compute_welch_psd(
        self, signal_arr: npt.NDArray[np.float64], has_background: bool = True
    ) -> SpectrumOutput:
        """Compute PSD and return SpectrumOutput containing top peaks."""
        sig_in = np.asarray(signal_arr, dtype=np.float64).ravel()
        sig_detrended = signal.detrend(sig_in, type="linear")

        peaks = compute_welch_peaks(
            signal_arr,
            fps=self.fps,
            min_freq_hz=self.min_freq_hz,
            top_k=self.top_k,
            has_background=has_background,
        )

        seg_len = min(len(sig_detrended), 1024)
        fft_len = max(seg_len, 4096)
        freqs, psd = signal.welch(
            sig_detrended,
            fs=self.fps,
            window="hann",
            nperseg=seg_len,
            nfft=fft_len,
        )

        verdict, reasons = determine_spectrum_verdict(
            peaks=peaks,
            has_background=has_background,
        )

        fund_hz = peaks[0].freq_hz if peaks else None
        snr_val = peaks[0].snr if peaks else 0.0
        reliable = (verdict == "usable")
        uncert_val = getattr(peaks[0], "freq_uncertainty", None) if peaks else None
        ci_val = getattr(peaks[0], "freq_ci", None) if peaks else None

        return SpectrumOutput(
            frequencies_hz=freqs,
            psd=psd,
            dominant_peaks=peaks,
            fundamental_frequency_hz=fund_hz,
            snr_db=snr_val,
            is_reliable=reliable,
            confidence=verdict,
            verdict=verdict,
            reasons=reasons,
            units="px",
            scale=None,
            freq_uncertainty=uncert_val,
            freq_ci=ci_val,
        )

    def analyze(
        self, signal_arr: npt.NDArray[np.float64], has_background: bool = True
    ) -> List[Peak]:
        """Analyze displacement array and return top 3 peaks."""
        return compute_welch_peaks(
            signal_arr,
            fps=self.fps,
            min_freq_hz=self.min_freq_hz,
            top_k=self.top_k,
            has_background=has_background,
        )
