"""Turn a measured displacement trace into a short, frequency-shifted WAV."""

from __future__ import annotations

from fractions import Fraction
from math import gcd
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from scipy import signal
from scipy.io import wavfile


OUTPUT_SAMPLE_RATE = 44_100
MAX_DURATION_SECONDS = 10.0


def _dominant_frequency(values: np.ndarray, fps: float) -> float:
    detrended = signal.detrend(values, type="linear")
    frequencies, power = signal.welch(detrended, fs=fps, nperseg=min(len(values), 4096))
    band = (frequencies >= 0.2) & (frequencies <= min(30.0, fps / 2.0))
    if not np.any(band) or float(np.max(power[band])) <= 0:
        raise ValueError("displacement has no measurable vibration peak")
    indices = np.flatnonzero(band)
    peak_index = int(indices[np.argmax(power[band])])
    return float(frequencies[peak_index])


def make_sonification(
    displacement: object,
    fps: float,
    out_wav: Union[str, Path],
    speedup: Optional[float] = None,
) -> Tuple[float, float]:
    """Write measured displacement as audio and return (speed-up, resulting Hz).

    With no explicit speed-up, the observed dominant vibration is shifted toward
    300 Hz. The output is a resampled copy of the measurement; no tone is added.
    """
    values = np.asarray(displacement, dtype=np.float64).reshape(-1)
    if values.size < 16 or not np.all(np.isfinite(values)):
        raise ValueError("displacement must contain at least 16 finite samples")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be a positive finite number")

    peak_hz = _dominant_frequency(values, float(fps))
    factor = 300.0 / peak_hz if speedup is None else float(speedup)
    if not np.isfinite(factor) or factor <= 0:
        raise ValueError("speedup must be a positive finite number")
    resulting_pitch = peak_hz * factor

    # Playing the same measured samples at fps*factor shifts every frequency by
    # factor. Polyphase resampling then converts that playback to 44.1 kHz.
    effective_rate = float(fps) * factor
    ratio = Fraction(OUTPUT_SAMPLE_RATE / effective_rate).limit_denominator(100_000)
    divisor = gcd(ratio.numerator, ratio.denominator)
    audio = signal.resample_poly(values, ratio.numerator // divisor, ratio.denominator // divisor)
    audio = audio[: int(MAX_DURATION_SECONDS * OUTPUT_SAMPLE_RATE)]
    if audio.size == 0:
        raise ValueError("displacement produced an empty audio signal")

    fade_count = min(int(0.02 * OUTPUT_SAMPLE_RATE), audio.size // 2)
    if fade_count:
        fade = np.linspace(0.0, 1.0, fade_count, endpoint=True)
        audio[:fade_count] *= fade
        audio[-fade_count:] *= fade[::-1]
    peak = float(np.max(np.abs(audio)))
    if peak > 0:
        audio = audio / peak * 0.95

    destination = Path(out_wav)
    destination.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(destination, OUTPUT_SAMPLE_RATE, np.asarray(audio, dtype=np.float32))
    return float(factor), float(resulting_pitch)
