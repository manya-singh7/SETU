from pathlib import Path

import numpy as np
from scipy.io import wavfile

from setu.sonify import OUTPUT_SAMPLE_RATE, make_sonification


FIXTURES = Path(__file__).parent / "fixtures"


def test_sonifies_measured_trace_to_a_valid_wav(tmp_path):
    measured = np.loadtxt(FIXTURES / "sonify_displacement.csv", delimiter=",", skiprows=1)
    destination = tmp_path / "vibration.wav"

    speedup, pitch = make_sonification(measured, 100.0, destination)

    rate, audio = wavfile.read(destination)
    assert rate == OUTPUT_SAMPLE_RATE == 44_100
    assert audio.ndim == 1 and audio.size <= 10 * rate
    assert np.max(np.abs(audio)) <= 1.0
    assert 200.0 <= pitch <= 400.0
    assert np.isclose(pitch, 300.0, atol=5.0)
    assert speedup > 1.0
    # Verify that the written signal contains the measured pitch.
    f, power = __import__("scipy").signal.welch(audio, fs=rate, nperseg=min(len(audio), 8192))
    assert abs(float(f[np.argmax(power)]) - pitch) < 8.0


def test_explicit_speedup_is_reported(tmp_path):
    values = np.sin(2 * np.pi * 4.0 * np.arange(400) / 100.0)
    speedup, pitch = make_sonification(values, 100, tmp_path / "explicit.wav", speedup=80)
    assert speedup == 80
    assert np.isclose(pitch, 320)
