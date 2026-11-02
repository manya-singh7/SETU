"""Tests for spectral analysis with synthetic signals."""

import numpy as np
from setu.spectrum import WelchSpectrumAnalyzer


def test_synthetic_frequency_structure():
    """Verify analyzer instantiation and interface on synthetic sinusoidal motion."""
    analyzer = WelchSpectrumAnalyzer(fps=30.0)

    # Generate synthetic 30-second vibration at known frequency f0 = 1.8 Hz
    duration = 30.0
    fps = 30.0
    t = np.linspace(0, duration, int(duration * fps), endpoint=False)
    f0 = 1.8
    # Known signal with amplitude 0.5 px
    synthetic_signal = 0.5 * np.sin(2 * np.pi * f0 * t)

    res = analyzer.compute_welch_psd(synthetic_signal)
    assert len(res.frequencies_hz) > 0
    assert len(res.psd) == len(res.frequencies_hz)
