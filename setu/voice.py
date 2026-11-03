"""Audio normalization and Muse Voice Transcribe integration."""
import wave
from pathlib import Path
from typing import Tuple

import av

from setu.meta_client import MetaClient, get_meta_client


def normalize_audio(source: str, destination: str) -> Tuple[float, str]:
    """Decode browser audio with PyAV into mono 24 kHz signed 16-bit PCM WAV."""
    container = av.open(source)
    try:
        stream = next(s for s in container.streams if s.type == "audio")
        duration = float(stream.duration * stream.time_base) if stream.duration else float(container.duration or 0) / av.time_base
        resampler = av.AudioResampler(format="s16", layout="mono", rate=24000)
        with wave.open(destination, "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(24000)
            for frame in container.decode(stream):
                for converted in resampler.resample(frame):
                    out.writeframes(converted.to_ndarray().tobytes())
            for converted in resampler.resample(None):
                out.writeframes(converted.to_ndarray().tobytes())
    finally:
        container.close()
    with wave.open(destination, "rb") as wav:
        duration = wav.getnframes() / wav.getframerate()
    return duration, destination


def transcribe_audio(source: str, language: str, client: MetaClient = None, max_duration_s: float = None) -> str:
    """Normalize and transcribe the uploaded browser recording."""
    normalized = str(Path(source).with_name(Path(source).stem + "_normalized.wav"))
    try:
        duration, wav_path = normalize_audio(source, normalized)
    except Exception:
        Path(normalized).unlink(missing_ok=True)
        raise
    try:
        if max_duration_s is not None and duration > max_duration_s:
            raise ValueError("Audio exceeds 2 minutes.")
        return (client or get_meta_client()).transcribe_audio(wav_path, language, duration)
    finally:
        Path(normalized).unlink(missing_ok=True)
