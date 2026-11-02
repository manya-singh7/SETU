"""
Spectral analysis and dominant modal peak detection module for Project SETU.

Computes the Power Spectral Density (PSD) using Welch's method (linear detrending,
Hann windowing, and segment averaging) from a 1D subpixel structural displacement signal.
Extracts top dominant frequency peaks as Peak objects with sub-bin quadratic interpolation,
relative spectral power, and peak-prominence signal-to-noise ratio (SNR), ignoring low-frequency
drift (< 0.5 Hz).

Spectral Metrics Definitions:
- Relative Power: Defined as the peak band's integrated power (sum of PSD energy across
  the peak's main lobe bounded by local minima / troughs or a +/- 0.4 Hz band) divided by
  the total power in the analysis band (0.5 Hz to Nyquist). For a clean sinusoidal tone,
  this metric concentrates > 80% of total spectral power in the resonant lobe.
- Peak SNR (dB): Defined as the peak power spectral density (P_peak) over the median noise
  floor (P_noise) across the analysis band (0.5 Hz to Nyquist), reported in decibels:
      SNR_dB = 10 * log10(P_peak / P_noise)
- Reliability Threshold: Set to 10.0 dB (a 10:1 power ratio over the median noise floor).
  In ambient vibration testing of civil infrastructure, spectral peaks below 10.0 dB
  cannot be reliably distinguished from colored 1/f ambient noise, camera jitter residuals,
  or thermal noise. Peaks below 10.0 dB are flagged as "low confidence".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import scipy.signal as signal

from setu.types import Peak

RELIABILITY_THRESHOLD_SNR_DB: float = 10.0
"""Reliability threshold in decibels (dB) for modal peak detection.

Justification:
A 10.0 dB peak SNR corresponds to a 10:1 power spectral density ratio relative to
the median background noise floor. In vision-based civil infrastructure screening,
ambient wind turbulence, pedestrian footfalls, and camera tripod micro-drift create
an elevated 1/f noise floor. Peaks with SNR < 10.0 dB risk false identification
of noise artifacts as structural resonant modes and are flagged as 'low confidence'.
"""


def is_peak_reliable(peak: Peak, threshold_db: float = RELIABILITY_THRESHOLD_SNR_DB) -> bool:
    """Assess whether a detected peak meets the minimum SNR reliability threshold (default: 10.0 dB)."""
    return peak.snr >= threshold_db


@dataclass
class SpectrumOutput:
    """Spectral analysis telemetry container preserved for backward compatibility."""

    frequencies_hz: npt.NDArray[np.float64]
    psd: npt.NDArray[np.float64]
    dominant_peaks: List[Peak]
    fundamental_frequency_hz: Optional[float]
    snr_db: float = 0.0
    is_reliable: bool = True
    confidence: str = "high"


def compute_welch_peaks(
    displacement: Union[npt.NDArray[np.float64], List[float]],
    fps: float,
    min_freq_hz: float = 0.5,
    top_k: int = 3,
    nperseg: Optional[int] = None,
    nfft: Optional[int] = None,
) -> List[Peak]:
    """Given a 1D displacement signal and fps, compute Welch PSD and extract top peaks.

    Metrics:
    - Relative power: The peak band's power (integrated over the resonant main lobe) divided
      by total power in the analysis band (min_freq_hz to Nyquist).
    - Peak SNR: Peak spectral power density over median noise floor, reported in dB:
      10 * log10(P_peak / median(P_band)).

    Args:
        displacement: 1D array of subpixel structural displacements (in pixels).
        fps: Sampling rate (frames per second).
        min_freq_hz: Frequency cutoff below which spectral content is ignored (default 0.5 Hz).
        top_k: Number of dominant peaks to return (default 3).
        nperseg: Length of each segment for Welch averaging. Defaults to min(len(signal), 1024).
        nfft: FFT length. Defaults to max(nperseg, 4096) for high resolution.

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

    # 4. Peak detection using prominence
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
        # bounded by local minima / troughs or a maximum half-width of 0.4 Hz
        left_idx = k
        while left_idx > 0 and p_band[left_idx - 1] <= p_band[left_idx] and (f0 - f_band[left_idx - 1]) <= 0.4:
            left_idx -= 1

        right_idx = k
        while right_idx < len(p_band) - 1 and p_band[right_idx + 1] <= p_band[right_idx] and (f_band[right_idx + 1] - f0) <= 0.4:
            right_idx += 1

        peak_band_power = float(np.sum(p_band[left_idx : right_idx + 1]))
        rel_power = peak_band_power / total_band_power
        rel_power = float(np.clip(rel_power, 0.0, 1.0))

        # Peak SNR in dB: 10 * log10(P_peak / P_noise_median)
        snr_ratio = max(peak_psd, 1e-12) / max(noise_floor, 1e-12)
        snr_db = 10.0 * np.log10(snr_ratio)

        peaks.append(
            Peak(
                freq_hz=float(round(f_refined, 4)),
                rel_power=float(round(rel_power, 6)),
                snr=float(round(snr_db, 2)),
            )
        )

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

    def compute_welch_psd(self, signal_arr: npt.NDArray[np.float64]) -> SpectrumOutput:
        """Compute PSD and return SpectrumOutput containing top peaks."""
        peaks = compute_welch_peaks(
            signal_arr,
            fps=self.fps,
            min_freq_hz=self.min_freq_hz,
            top_k=self.top_k,
        )
        sig_detrended = signal.detrend(signal_arr, type="linear")
        seg_len = min(len(sig_detrended), 1024)
        fft_len = max(seg_len, 4096)
        freqs, psd = signal.welch(
            sig_detrended,
            fs=self.fps,
            window="hann",
            nperseg=seg_len,
            nfft=fft_len,
        )
        fund_hz = peaks[0].freq_hz if peaks else None
        snr_val = peaks[0].snr if peaks else 0.0
        reliable = bool(peaks and is_peak_reliable(peaks[0]))
        conf = "high" if reliable else "low confidence"
        return SpectrumOutput(
            frequencies_hz=freqs,
            psd=psd,
            dominant_peaks=peaks,
            fundamental_frequency_hz=fund_hz,
            snr_db=snr_val,
            is_reliable=reliable,
            confidence=conf,
        )

    def analyze(self, signal_arr: npt.NDArray[np.float64]) -> List[Peak]:
        """Analyze displacement array and return top 3 peaks."""
        return compute_welch_peaks(
            signal_arr,
            fps=self.fps,
            min_freq_hz=self.min_freq_hz,
            top_k=self.top_k,
        )
