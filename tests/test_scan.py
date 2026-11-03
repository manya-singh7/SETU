"""
Integration tests for SETU end-to-end vibration screening (setu.scan.analyze_video).

Verifies Requirement 5:
1. In mock mode with no network, the synthetic 3.7 Hz video produces a 'usable' verdict
   with a 95% confidence interval containing 3.70 Hz, and measurement.psd populated.
2. An unreadable or non-existent file produces a 'failed' verdict with no frequency/psd telemetry.
3. Pipeline artifacts (result.json, displacement.png, spectrum.png) are written to out_dir.
4. All result objects adhere strictly to the shared result format JSON contract.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import pytest

from setu.meta_client import MetaClient
from setu.osm import OverpassBridgeClient
from setu.result import ScanResult
from setu.scan import analyze_video
from tests.synth import generate_synthetic_video


@pytest.fixture
def mock_clients(monkeypatch):
    """Configure mock mode with zero network requests."""
    monkeypatch.setenv("SETU_META_MODE", "mock")
    monkeypatch.setenv("META_API_MODE", "mock")
    monkeypatch.setenv("USE_LOCAL_FALLBACKS", "true")
    client = MetaClient(mode="mock")
    osm_cli = OverpassBridgeClient(mode="mock")
    return client, osm_cli


def test_analyze_video_synthetic_3hz7_usable(tmp_path: Path, mock_clients):
    """Verify that a synthetic 3.7 Hz video yields 'usable' with CI containing 3.7."""
    meta_cli, osm_cli = mock_clients
    video_dest = str(tmp_path / "synth_3hz7.mp4")
    out_dir = tmp_path / "scan_output"

    # Generate 6.0s synthetic video oscillating at 3.7 Hz
    synth = generate_synthetic_video(
        output_path=video_dest,
        fps=60.0,
        duration_sec=6.0,
        width=640,
        height=480,
        freq_hz=3.70,
        amplitude_px=0.20,
        camera_jitter=False,
        seed=42,
    )

    result = analyze_video(
        path=synth.video_path,
        lat=13.0827,
        lon=80.2707,
        target_prompt="concrete pillar",
        background_prompt="building",
        magnify=False,
        is_demo=False,
        out_dir=out_dir,
        meta_client=meta_cli,
        osm_client=osm_cli,
    )

    assert isinstance(result, ScanResult)
    assert result.schema_version == "1.0"
    assert result.is_demo is False
    assert result.status == "complete"

    # 1. Verdict must be "usable" with empty reasons list
    assert result.verdict.level == "usable", (
        f"Expected verdict 'usable', got {result.verdict.level} ({result.verdict.reasons})"
    )
    assert result.verdict.reasons == []

    # 2. Dominant frequency must be 3.7 Hz (±0.1 Hz)
    dom_f = result.measurement.dominant_frequency_hz
    assert dom_f is not None
    assert abs(dom_f - 3.70) <= 0.10, f"Dominant frequency {dom_f} deviates from 3.70 Hz"

    # 3. 95% Confidence interval must contain 3.70 Hz
    ci = result.measurement.frequency_ci_hz
    assert ci is not None, "Expected populated frequency_ci_hz for usable result"
    assert len(ci) == 2
    assert ci[0] <= 3.70 <= ci[1], f"Confidence interval {ci} does not contain nominal 3.70 Hz"

    # 4. Measurement peaks list populated
    assert len(result.measurement.peaks) >= 1
    p0 = result.measurement.peaks[0]
    assert abs(p0.frequency_hz - 3.70) <= 0.10
    assert p0.relative_power > 0.50

    # 5. PSD populated with <= 300 points between 0 and 30 Hz
    assert result.measurement.psd is not None
    assert 0 < len(result.measurement.psd.freq_hz) <= 300
    assert len(result.measurement.psd.freq_hz) == len(result.measurement.psd.power)
    assert max(result.measurement.psd.freq_hz) <= 30.0
    assert min(result.measurement.psd.freq_hz) >= 0.0

    # 6. Output artifacts exist in out_dir
    assert result.outputs.displacement_plot == "displacement.png"
    assert result.outputs.spectrum_plot == "spectrum.png"
    assert os.path.isfile(out_dir / "result.json")
    assert os.path.isfile(out_dir / "displacement.png")
    assert os.path.isfile(out_dir / "spectrum.png")

    # 7. JSON serialization roundtrip
    result_json = result.to_json()
    parsed = json.loads(result_json)
    assert parsed["schema_version"] == "1.0"
    assert parsed["is_demo"] is False
    assert parsed["verdict"]["level"] == "usable"
    assert parsed["measurement"]["dominant_frequency_hz"] == dom_f
    assert "psd" in parsed["measurement"]
    assert len(parsed["measurement"]["psd"]["freq_hz"]) > 0


def test_analyze_video_unreadable_file_returns_failed(tmp_path: Path, mock_clients):
    """Verify that an unreadable or non-existent file returns 'failed' with no frequency."""
    meta_cli, osm_cli = mock_clients
    out_dir = tmp_path / "failed_output"

    result = analyze_video(
        path="completely_non_existent_file.mp4",
        lat=13.0827,
        lon=80.2707,
        target_prompt="concrete pillar",
        background_prompt="building",
        out_dir=out_dir,
        meta_client=meta_cli,
        osm_client=osm_cli,
    )

    assert isinstance(result, ScanResult)
    assert result.verdict.level == "failed"
    assert len(result.verdict.reasons) > 0

    # Invariant: failed result must have null frequencies, empty peaks, and null psd
    assert result.measurement.dominant_frequency_hz is None
    assert result.measurement.frequency_ci_hz is None
    assert result.measurement.peaks == []
    assert result.measurement.psd is None

    # Validation must pass without raising ValueError
    result.validate()
    json_str = result.to_json()
    parsed = json.loads(json_str)
    assert parsed["verdict"]["level"] == "failed"
    assert parsed["measurement"]["dominant_frequency_hz"] is None
    assert parsed["measurement"]["frequency_ci_hz"] is None
    assert parsed["measurement"]["peaks"] == []
    assert parsed["measurement"]["psd"] is None


def test_analyze_video_corrupt_file_returns_failed(tmp_path: Path, mock_clients):
    """Verify corrupted file returns failed result gracefully with no exception raised."""
    meta_cli, osm_cli = mock_clients
    corrupt_file = tmp_path / "bad.mp4"
    corrupt_file.write_bytes(b"INVALID_HEADER_GARBAGE")

    result = analyze_video(
        path=str(corrupt_file),
        lat=13.0827,
        lon=80.2707,
        meta_client=meta_cli,
        osm_client=osm_cli,
    )

    assert result.verdict.level == "failed"
    assert result.measurement.dominant_frequency_hz is None
    assert result.measurement.frequency_ci_hz is None
    assert result.measurement.peaks == []
    assert result.measurement.psd is None


def test_analyze_video_with_magnification_produces_artifact(tmp_path: Path, mock_clients):
    """Verify analyze_video with magnify=True produces magnified.mp4 and records stage_timings."""
    meta_cli, osm_cli = mock_clients
    video_dest = str(tmp_path / "synth_short_mag.mp4")
    out_dir = tmp_path / "mag_output"

    synth = generate_synthetic_video(
        output_path=video_dest,
        fps=60.0,
        duration_sec=5.0,
        freq_hz=3.70,
        amplitude_px=0.20,
        camera_jitter=False,
        seed=42,
    )

    timings = {}
    result = analyze_video(
        path=synth.video_path,
        lat=13.0827,
        lon=80.2707,
        target_prompt="concrete pillar",
        background_prompt="building",
        magnify=True,
        out_dir=out_dir,
        meta_client=meta_cli,
        osm_client=osm_cli,
        stage_timings=timings,
    )

    assert result.verdict.level == "usable"
    assert result.outputs.magnified_video == "magnified.mp4"
    assert os.path.isfile(out_dir / "magnified.mp4")
    assert "video checks" in timings
    assert "magnification" in timings
    assert timings["magnification"] > 0.0
