"""
Spectral analysis and dominant modal peak detection module for Project SETU.

Computes the Power Spectral Density (PSD) using Welch's method (linear detrending,
Hann windowing, and segment averaging) from a 1D subpixel structural displacement signal.
Extracts top dominant frequency peaks as Peak objects with sub-bin quadratic interpolation,
relative spectral power, and peak-prominence signal-to-noise ratio (SNR), ignoring low-frequency
drift (< 0.5 Hz).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import scipy.signal as signal

from setu.types import Peak


@dataclass
class SpectrumOutput:
    """Spectral analysis telemetry container preserved for backward compatibility."""

    frequencies_hz: npt.NDArray[np.float64]
    psd: npt.NDArray[np.float64]
    dominant_peaks: List[Peak]
    fundamental_frequency_hz: Optional[float]
    snr_db: float = 0.0


def compute_welch_peaks(
    displacement: Union[npt.NDArray[np.float64], List[float]],
    fps: float,
    min_freq_hz: float = 0.5,
    top_k: int = 3,
    nperseg: Optional[int] = None,
    nfft: Optional[int] = None,
) -> List[Peak]:
    """Given a 1D displacement signal and fps, compute Welch PSD and extract top peaks.

    Steps:
    1. Linear detrending to eliminate DC drift and linear camera motion.
    2. Hann windowing and Welch averaging to suppress spectral leakage.
    3. Truncate frequency band below min_freq_hz (default 0.5 Hz).
    4. Detect peaks via prominence, refine frequencies with parabolic peak interpolation.
    5. Calculate relative power and peak-prominence SNR for the top peaks.

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
        raw_prom = float(prominences[idx])
        peak_psd = float(p_band[k])

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

        rel_power = peak_psd / total_band_power
        snr = raw_prom / noise_floor

        peaks.append(
            Peak(
                freq_hz=float(round(f_refined, 4)),
                rel_power=float(round(rel_power, 6)),
                snr=float(round(snr, 2)),
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
        return SpectrumOutput(
            frequencies_hz=freqs,
            psd=psd,
            dominant_peaks=peaks,
            fundamental_frequency_hz=fund_hz,
            snr_db=snr_val,
        )

    def analyze(self, signal_arr: npt.NDArray[np.float64]) -> List[Peak]:
        """Analyze displacement array and return top 3 peaks."""
        return compute_welch_peaks(
            signal_arr,
            fps=self.fps,
            min_freq_hz=self.min_freq_hz,
            top_k=self.top_k,
        )
