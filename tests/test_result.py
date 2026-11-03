"""
Unit tests for SETU Shared Result Format and ScanResult dataclasses.

Verifies:
1. Exact compliance with tests/fixtures/scan_result_example.json including:
   - Top-level 'is_demo': false
   - measurement.psd: {'freq_hz': [...], 'power': [...]}
2. Serialization and deserialization via to_json() / from_json().
3. Invariant validation: when verdict.level is 'failed', measurement.dominant_frequency_hz,
   frequency_ci_hz, and psd must be null, and peaks must be empty.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import pytest

from setu.result import (
    AssessmentInfo,
    CostsUSD,
    InterpretationInfo,
    Location,
    MeasurementInfo,
    MeasurementPeak,
    OutputsInfo,
    PSDData,
    QualityGateInfo,
    ScanResult,
    SegmentationInfo,
    SetupInfo,
    Verdict,
    VideoInfo,
)


def test_scan_result_example_fixture_roundtrip():
    """Verify ScanResult parses and reproduces scan_result_example.json exactly."""
    fixture_path = os.path.join(
        os.path.dirname(__file__), "fixtures", "scan_result_example.json"
    )
    assert os.path.exists(fixture_path), f"Fixture missing: {fixture_path}"

    with open(fixture_path, "r", encoding="utf-8") as f:
        raw_json = f.read()

    expected_dict = json.loads(raw_json)
    res = ScanResult.from_json(raw_json)

    # Top-level keys match
    actual_dict = res.to_dict()
    assert set(actual_dict.keys()) == set(expected_dict.keys())

    # Check key scalar values
    assert res.schema_version == "1.0"
    assert res.scan_id == "scan_20261103_a1b2c3"
    assert res.is_demo is False
    assert res.status == "complete"
    assert res.verdict.level == "usable"
    assert res.verdict.reasons == []

    # Check location
    assert res.location.lat == 13.0827
    assert res.location.lon == 80.2707

    # Check video
    assert res.video.filename == "clip.mp4"
    assert res.video.fps == 60.0
    assert res.video.duration_s == 30.0
    assert res.video.width == 1920
    assert res.video.height == 1080
    assert res.video.frame_count == 1800

    # Check segmentation
    assert res.segmentation.target_prompt == "concrete pillar"
    assert res.segmentation.background_prompt == "building"
    assert res.segmentation.target_source == "sam"
    assert res.segmentation.background_source == "sam"
    assert res.segmentation.target_persistence == 0.98
    assert res.segmentation.background_persistence == 0.99
    assert res.segmentation.warnings == []

    # Check measurement
    assert res.measurement.units == "px"
    assert res.measurement.scale is None
    assert res.measurement.dominant_frequency_hz == 3.70
    assert res.measurement.frequency_ci_hz == [3.68, 3.72]
    assert len(res.measurement.peaks) == 1
    p0 = res.measurement.peaks[0]
    assert p0.frequency_hz == 3.70
    assert p0.relative_power == 0.91
    assert p0.snr_db == 38.5
    assert res.measurement.background_residual_px == 0.016

    # Check PSD
    assert res.measurement.psd is not None
    assert len(res.measurement.psd.freq_hz) > 0
    assert len(res.measurement.psd.freq_hz) <= 300
    assert len(res.measurement.psd.freq_hz) == len(res.measurement.psd.power)
    assert max(res.measurement.psd.freq_hz) <= 30.0

    # Check interpretation
    assert res.interpretation.bridge is None
    assert res.interpretation.bridge_warning == "No mapped bridge within 150 m"
    assert res.interpretation.quality_gate.is_structural_element is True
    assert res.interpretation.quality_gate.has_static_background is True
    assert res.interpretation.quality_gate.appears_stable is True
    assert res.interpretation.assessment.severity == "none"
    assert res.interpretation.assessment.assessor == "local_rules"
    assert res.interpretation.assessment.confidence == "low"

    # Check outputs
    assert res.outputs.magnified_video == "magnified.mp4"
    assert res.outputs.spectrum_plot == "spectrum.png"
    assert res.outputs.displacement_plot == "displacement.png"

    # Check costs & disclaimer
    assert res.costs_usd.sam == 0.0025
    assert res.costs_usd.muse_spark == 0.0013
    assert res.costs_usd.total == 0.0038
    assert "SETU does not assess safety" in res.disclaimer

    # Round trip JSON matches
    roundtrip = ScanResult.from_json(res.to_json())
    assert roundtrip.to_dict() == actual_dict


def test_failed_result_validation_enforces_no_frequency_and_no_psd():
    """Verify that a 'failed' verdict strictly rejects frequency and PSD telemetry."""
    # Valid failed result (all frequency fields null/empty)
    failed_valid = ScanResult(
        verdict=Verdict(level="failed", reasons=["Sensor clipping detected"]),
        measurement=MeasurementInfo(
            dominant_frequency_hz=None,
            frequency_ci_hz=None,
            peaks=[],
            psd=None,
        ),
    )
    # validate() must pass
    failed_valid.validate()

    # Invalid Case 1: dominant_frequency_hz is not null
    with pytest.raises(ValueError, match="measurement.dominant_frequency_hz"):
        ScanResult(
            verdict=Verdict(level="failed", reasons=["Tracking lost"]),
            measurement=MeasurementInfo(
                dominant_frequency_hz=3.70,
                frequency_ci_hz=None,
                peaks=[],
                psd=None,
            ),
        )

    # Invalid Case 2: frequency_ci_hz is not null
    with pytest.raises(ValueError, match="measurement.frequency_ci_hz"):
        ScanResult(
            verdict=Verdict(level="failed", reasons=["Tracking lost"]),
            measurement=MeasurementInfo(
                dominant_frequency_hz=None,
                frequency_ci_hz=[3.68, 3.72],
                peaks=[],
                psd=None,
            ),
        )

    # Invalid Case 3: peaks is not empty
    with pytest.raises(ValueError, match="measurement.peaks"):
        ScanResult(
            verdict=Verdict(level="failed", reasons=["Tracking lost"]),
            measurement=MeasurementInfo(
                dominant_frequency_hz=None,
                frequency_ci_hz=None,
                peaks=[MeasurementPeak(frequency_hz=3.70, relative_power=0.9, snr_db=30.0)],
                psd=None,
            ),
        )

    # Invalid Case 4: psd is not null
    with pytest.raises(ValueError, match="measurement.psd"):
        ScanResult(
            verdict=Verdict(level="failed", reasons=["Tracking lost"]),
            measurement=MeasurementInfo(
                dominant_frequency_hz=None,
                frequency_ci_hz=None,
                peaks=[],
                psd=PSDData(freq_hz=[3.7], power=[0.5]),
            ),
        )


def test_weak_result_allows_frequency_and_psd():
    """Verify that a 'weak' verdict allows frequency, CI, and PSD."""
    weak_res = ScanResult(
        verdict=Verdict(level="weak", reasons=["No stationary background in frame"]),
        measurement=MeasurementInfo(
            dominant_frequency_hz=3.70,
            frequency_ci_hz=[3.65, 3.75],
            peaks=[MeasurementPeak(frequency_hz=3.70, relative_power=0.85, snr_db=25.0)],
            psd=PSDData(freq_hz=[3.70], power=[0.9]),
        ),
    )
    weak_res.validate()
    d = weak_res.to_dict()
    assert d["verdict"]["level"] == "weak"
    assert d["measurement"]["dominant_frequency_hz"] == 3.70
    assert d["measurement"]["frequency_ci_hz"] == [3.65, 3.75]
    assert d["measurement"]["psd"]["freq_hz"] == [3.70]


def test_schema_11_additions_and_schema_10_compatibility():
    current = ScanResult()
    data = current.to_dict()
    assert data["schema_version"] == "1.4"
    assert data["field_notes"] == []
    assert data["measurement"]["instances"] == []
    assert data["outputs"]["vibration_map"] is None
    assert data["outputs"]["sound"] is None
    assert data["measurement"]["mode_shape"] == []
    assert data["sonification"] is None
    assert data["weather"] is None
    assert data["device_check"]["status"] == "not_recorded"
    assert data["traffic"] == {"events": [], "response_plot": None}
    assert data["magnification_status"] == "skipped"
    assert data["cable"] == {
        "is_cable": False,
        "tension_change_pct": None,
        "tension_change_range_pct": None,
        "compared_to_scan_id": None,
        "compared_to_date": None,
    }
    assert data["setup"] == {
        "source": "default", "structure_type": None,
        "recording_issues": [], "advice": None,
    }
    legacy = ScanResult.from_dict({"schema_version": "1.0", "measurement": {}, "outputs": {}})
    legacy_data = legacy.to_dict()
    assert legacy.schema_version == "1.0"
    assert "field_notes" not in legacy_data
    assert "instances" not in legacy_data["measurement"]
    assert "cable" not in legacy_data
    legacy_11 = ScanResult.from_dict({"schema_version": "1.1", "measurement": {}, "outputs": {}})
    assert legacy_11.to_dict()["schema_version"] == "1.1"
    assert "cable" not in legacy_11.to_dict()
    legacy_12 = ScanResult.from_dict({"schema_version": "1.2", "measurement": {}, "outputs": {}})
    assert "setup" not in legacy_12.to_dict()
    legacy_13 = ScanResult.from_dict({"schema_version": "1.3", "measurement": {}, "outputs": {}})
    assert "sound" not in legacy_13.to_dict()["outputs"]
    assert "sonification" not in legacy_13.to_dict()
    assert "magnification_status" not in legacy_13.to_dict()
    assert "target_bbox" not in legacy_13.to_dict()["segmentation"]


def test_schema_14_sound_environment_device_traffic_round_trip():
    raw = {
        "schema_version": "1.4",
        "outputs": {"sound": "bridge_sound.wav"},
        "segmentation": {"target_bbox": [10, 20, 30, 40]},
        "sonification": {"speedup": 75.0, "pitch_hz": 300.0},
        "weather": {"temperature_c": 26.5, "humidity_pct": 65.0,
                    "wind_speed_ms": 2.0, "source": "open-meteo"},
        "device_check": {"status": "agrees", "device_frequency_hz": 3.7,
                         "device_ci_hz": [3.6, 3.8], "sample_rate_hz": 50.0,
                         "note": "Frequency intervals overlap."},
        "traffic": {"events": [{"t_start": 2.0, "t_end": 5.0, "type": "truck"}],
                    "response_plot": "traffic-response.png"},
        "measurement": {"mode_shape": [{"label": "midspan", "position_norm": 0.5,
                                        "amplitude_px": 0.12}]},
    }
    current = ScanResult.from_dict(raw)
    restored = ScanResult.from_json(current.to_json())
    assert restored.to_dict() == current.to_dict()
    assert restored.outputs.sound == "bridge_sound.wav"
    assert restored.segmentation.target_bbox == [10.0, 20.0, 30.0, 40.0]
    assert restored.measurement.mode_shape[0].position_norm == 0.5


def test_schema_13_setup_metadata_round_trip():
    current = ScanResult(
        schema_version="1.3",
        setup=SetupInfo(
            source="muse_spark", structure_type="bridge pier",
            recording_issues=["camera movement"], advice="Brace the phone.",
        ),
    )
    restored = ScanResult.from_json(current.to_json())
    assert restored.to_dict() == current.to_dict()
    assert restored.setup.source == "muse_spark"
    assert restored.setup.structure_type == "bridge pier"


def test_schema_12_cable_round_trip():
    current = ScanResult.from_dict({
        "schema_version": "1.2",
        "cable": {
            "is_cable": True,
            "tension_change_pct": -9.75,
            "tension_change_range_pct": [-11.1, -8.4],
            "compared_to_scan_id": "scan_old",
            "compared_to_date": "2026-11-03T10:00:00+00:00",
        },
    })
    restored = ScanResult.from_json(current.to_json())
    assert restored.to_dict() == current.to_dict()
    assert restored.cable.is_cable is True
    assert restored.cable.tension_change_pct == -9.75
