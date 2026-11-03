"""Offline coverage for Muse Voice normalization and evidence-locked tags."""
import wave
from pathlib import Path

from setu.field_notes import extract_field_note
from setu.meta_client import MetaClient
from setu.voice import transcribe_audio


def _wav(path):
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes(b"\x00\x00" * 2400)


def test_mock_voice_transcription_and_audio_cleanup():
    source = Path(__file__).with_name("voice_test_input.wav")
    try:
        _wav(source)
        client = MetaClient(mode="mock")
        transcript = transcribe_audio(str(source), "en", client=client)
        assert transcript == "The concrete has a crack and rust near the water."
        assert not source.with_name("voice_test_input_normalized.wav").exists()
    finally:
        source.unlink(missing_ok=True)


def test_mock_field_tags_apply_exact_evidence_lock():
    result = extract_field_note("The concrete has a crack and rust near the water.", MetaClient(mode="mock"))
    assert {tag["tag"] for tag in result["tags"]} == {"crack", "rust"}
    assert all(tag["evidence"] in "The concrete has a crack and rust near the water." for tag in result["tags"])
    assert not any(tag["evidence"] == "fabricated quote" for tag in result["tags"])
