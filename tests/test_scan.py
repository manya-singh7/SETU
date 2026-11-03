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
import sys
from types import ModuleType, SimpleNamespace
import numpy as np
import pytest

from setu.meta_client import MetaClient
from setu.osm import OverpassBridgeClient
from setu.result import ScanResult
from setu.scan import analyze_video
from setu.types import MotionResult
from tests.synth import generate_synthetic_video


@pytest.fixture
def mock_clients(monkeypatch):
    """Configure mock mode with zero network requests."""
    monkeypatch.setenv("SETU_META_MODE", "mock")
    monkeypatch.setenv("META_API_MODE", "mock")
    monkeypatch.setenv("USE_LOCAL_FALLBACKS", "true")
    client = MetaClient(mode="mock")
    osm_cli = OverpassBridgeClient(mode="mock")
    import setu.weather as weather_module
    monkeypatch.setattr(weather_module, "get_weather", lambda *args: {
        "temperature_c": None, "humidity_pct": None, "wind_speed_kmh": None,
    })
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
    assert result.schema_version == "1.4"
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
    assert parsed["schema_version"] == "1.4"
    assert parsed["is_demo"] is False
    assert parsed["verdict"]["level"] == "usable"
    assert parsed["measurement"]["dominant_frequency_hz"] == dom_f
    assert "psd" in parsed["measurement"]
    assert len(parsed["measurement"]["psd"]["freq_hz"]) > 0


def test_analyze_video_uses_mock_sam_video_fixture_for_traffic(tmp_path: Path, mock_clients, monkeypatch, caplog):
    """Full scan forwards its mock MetaClient into traffic and records fixture tracks."""
    import logging

    meta_cli, osm_cli = mock_clients
    fixture_text = "\n".join(
        f"<{frame}f>0<|box;x1=10;y1=10;x2=30;y2=30;w=640;h=480|><|mask;x=0;y=0;data=1,1,!|>"
        for frame in range(8)
    )
    original_load_fixture = meta_cli._load_fixture
    monkeypatch.setattr(meta_cli, "_load_fixture", lambda name: {
        "status": "completed", "output_text": fixture_text, "_mock_data": True,
    } if name == "sam_3_1_video_response.json" else original_load_fixture(name))
    synth = generate_synthetic_video(
        output_path=str(tmp_path / "traffic_scan.mp4"), fps=30.0, duration_sec=5.0,
        width=640, height=480, freq_hz=3.7, amplitude_px=0.2, camera_jitter=False, seed=14,
    )
    caplog.set_level(logging.WARNING)
    result = analyze_video(
        path=synth.video_path, lat=13.0827, lon=80.2707,
        target_prompt="concrete pillar", background_prompt="building",
        out_dir=tmp_path / "traffic_scan_output", meta_client=meta_cli, osm_client=osm_cli,
    )
    assert result.traffic.events
    assert {event.type for event in result.traffic.events} == {"truck", "bus", "car"}
    assert "Traffic tracking failed" not in caplog.text


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


def _install_optional_feature_scan_stubs(monkeypatch, tmp_path, *, fail=False):
    import setu.scan as scan_module

    video_path = tmp_path / "feature_scan.mp4"
    generate_synthetic_video(
        output_path=str(video_path), fps=60.0, duration_sec=6.0,
        width=640, height=480, freq_hz=3.7, amplitude_px=0.2, seed=11,
    )
    masks = []
    candidates = []
    for index, x in enumerate((180, 270, 360)):
        mask = np.zeros((480, 640), dtype=bool)
        mask[100:350, x:x + 50] = True
        masks.append(mask)
        candidates.append({"label": f"post {index + 1}", "bbox": (x, 100, 50, 250), "mask": mask})
    background = np.zeros((480, 640), dtype=bool)
    background[370:450, 35:155] = True
    output = SimpleNamespace(
        target_bbox=(175, 95, 60, 260), background_bbox=(35, 370, 120, 80),
        target_mask=masks[0],
        target_source="sam", background_source="sam", method="mock", warnings=[],
        metadata={"all_target_instances": candidates},
    )
    monkeypatch.setattr(
        scan_module, "get_segmenter",
        lambda **kwargs: SimpleNamespace(segment=lambda *args, **kw: output),
    )
    monkeypatch.setattr(scan_module, "compute_persistence", lambda **kwargs: {"fraction_valid": 1.0, "passed": True})
    t = np.arange(360, dtype=np.float64) / 60.0
    trace = 0.2 * np.sin(2.0 * np.pi * 3.7 * t)
    monkeypatch.setattr(
        scan_module, "track_motion",
        lambda **kwargs: MotionResult(
            displacement=trace, fps=60.0,
            quality={"has_background": True, "clipping_detected": False, "background_residual_px": 0.0},
            verdict="usable", background_displacement=np.zeros_like(trace),
        ),
    )

    import setu.vibration_map as vibration_map_module
    import setu.sonify as sonify_module
    import setu.weather as weather_module
    import setu.device_motion as device_module

    def maybe_raise(name):
        if fail:
            raise RuntimeError(f"{name} mock failure")

    def build_map(video_path, instances, background_mask, out_png, **kwargs):
        maybe_raise("map")
        return [
            {"label": label, "bbox": [x, 100, 50, 250], "frequency_hz": 3.7,
             "amplitude_px": 0.1 + i * 0.05, "verdict": "usable"}
            for i, (label, mask) in enumerate(instances)
            for x in [int(np.where(mask)[1].min())]
        ]

    captured = {}

    def sonify(displacement, fps, out_wav):
        maybe_raise("sonification")
        captured["displacement"] = np.asarray(displacement).copy()
        captured["fps"] = fps
        Path(out_wav).write_bytes(b"mock wav")
        return 10.0, 37.0

    monkeypatch.setattr(vibration_map_module, "build_vibration_map", build_map)
    monkeypatch.setattr(sonify_module, "make_sonification", sonify)
    monkeypatch.setattr(weather_module, "get_weather", lambda *args: (_ for _ in ()).throw(RuntimeError("weather mock failure")) if fail else {
        "temperature_c": 18.0, "humidity_pct": 60.0, "wind_speed_kmh": 7.2,
    })
    monkeypatch.setattr(device_module, "analyze_device_motion", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("device mock failure")) if fail else {
        "resting_on_structure": True, "dominant_frequency_hz": 3.7,
        "confidence_interval_hz": [3.6, 3.8], "sampling_rate_hz": 100.0,
        "comparison": {"agreement": True}, "message": None,
    })
    traffic = ModuleType("setu.traffic")
    traffic.track_traffic = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("traffic mock failure")) if fail else {
        "events": [{"t_start": 1.0, "t_end": 2.0, "type": "truck"}],
        "response_plot": "traffic-response.png",
    }
    monkeypatch.setitem(sys.modules, "setu.traffic", traffic)
    out_dir = tmp_path / "scan"
    out_dir.mkdir()
    (out_dir / "device_motion.json").write_text("[]", encoding="utf-8")
    return str(video_path), out_dir, captured, trace


def test_schema_14_optional_features_and_displacement_export(tmp_path, mock_clients, monkeypatch):
    video_path, out_dir, captured, trace = _install_optional_feature_scan_stubs(monkeypatch, tmp_path)
    meta_cli, osm_cli = mock_clients
    result = analyze_video(
        path=video_path, lat=13.0, lon=80.0, out_dir=out_dir,
        meta_client=meta_cli, osm_client=osm_cli,
    )
    assert result.status == "complete" and result.verdict.level == "usable"
    assert result.schema_version == "1.4"
    assert result.segmentation.target_bbox == [180.0, 100.0, 50.0, 250.0]
    assert len(result.measurement.instances) == 3
    assert len(result.measurement.mode_shape) == 3
    assert [point.position_norm for point in result.measurement.mode_shape] == pytest.approx([0.0, 0.5, 1.0])
    assert result.outputs.vibration_map == "vibration_map.png"
    assert result.outputs.sound == "sound.wav"
    assert result.sonification.pitch_hz == 37.0
    assert result.weather.wind_speed_ms == pytest.approx(2.0)
    assert result.device_check.status == "agrees"
    assert result.traffic.events[0].type == "truck"
    assert result.traffic.response_plot == "traffic-response.png"
    saved_trace = np.load(out_dir / "displacement.npy")
    assert np.array_equal(saved_trace, trace)
    assert np.array_equal(captured["displacement"], saved_trace)
    assert captured["fps"] == result.video.fps == 60.0


def test_optional_feature_errors_are_recorded_without_failing_scan(tmp_path, mock_clients, monkeypatch):
    video_path, out_dir, _, _ = _install_optional_feature_scan_stubs(monkeypatch, tmp_path, fail=True)
    meta_cli, osm_cli = mock_clients
    result = analyze_video(
        path=video_path, lat=13.0, lon=80.0, out_dir=out_dir,
        meta_client=meta_cli, osm_client=osm_cli,
    )
    assert result.status == "complete"
    assert result.verdict.level == "usable"
    warnings = " ".join(result.segmentation.warnings)
    for reason in ("vibration map", "sonification", "weather", "device check", "traffic"):
        assert reason in warnings
    assert result.outputs.vibration_map is None
    assert result.outputs.sound is None
    assert result.weather.temperature_c is None
    assert result.device_check.note == "device mock failure"
