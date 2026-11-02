"""
Spectral analysis and dominant frequency estimation module for Project SETU.

Step 4 in SETU Pipeline:
Computes the Power Spectral Density (PSD) using Welch's method and 1D FFT
to isolate dominant natural vibration frequencies and modal damping ratios.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np
import numpy.typing as npt


@dataclass
class SpectralPeak:
    """Detected dominant vibration mode from the frequency spectrum.

    Attributes:
        frequency_hz: Resonant modal frequency in Hertz.
        psd_power: Spectral power density (amplitude^2 / Hz).
        prominence: Relative peak prominence over broadband noise floor.
        estimated_damping: Damping ratio estimate via half-power (-3dB) bandwidth method.
    """

    frequency_hz: float
    psd_power: float
    prominence: float
    estimated_damping: Optional[float] = None


@dataclass
class SpectrumOutput:
    """Spectral analysis results and detected modal peaks.

    Attributes:
        frequencies_hz: Frequency array (Hz).
        psd: Power spectral density values corresponding to frequencies.
        dominant_peaks: Detected natural resonant modes sorted by power.
        fundamental_frequency_hz: First identified natural structural mode.
        snr_db: Signal-to-noise ratio estimate in dB.
    """

    frequencies_hz: npt.NDArray[np.float64]
    psd: npt.NDArray[np.float64]
    dominant_peaks: List[SpectralPeak]
    fundamental_frequency_hz: Optional[float]
    snr_db: float = 0.0


class WelchSpectrumAnalyzer:
    """Performs Welch's average periodogram PSD estimation and modal peak picking."""

    def __init__(self, fps: float = 30.0, nperseg: Optional[int] = None) -> None:
        """Initialize the spectral analyzer.

        Args:
            fps: Signal sampling frequency in Hz.
            nperseg: Length of each segment for Welch PSD averaging.
        """
        self.fps = fps
        self.nperseg = nperseg

    def compute_welch_psd(
        self,
        signal: npt.NDArray[np.float64],
    ) -> SpectrumOutput:
        """Compute Welch PSD and extract dominant structural frequencies.

        Args:
            signal: 1D structural displacement time series.

        Returns:
            SpectrumOutput containing frequency axis, PSD curve, and identified peaks.
        """
        # STUB: To be implemented in pipeline phase
        freqs = np.linspace(0, self.fps / 2.0, 128, dtype=np.float64)
        psd = np.zeros_like(freqs)

        return SpectrumOutput(
            frequencies_hz=freqs,
            psd=psd,
            dominant_peaks=[],
            fundamental_frequency_hz=None,
            snr_db=0.0,
        )
