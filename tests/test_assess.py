"""Unit tests for anomaly assessment, forced JSON schema, and safety philosophy."""

import json
from unittest.mock import MagicMock
import numpy as np
import pytest

from setu.assess import (
    AnomalyAssessmentResult,
    LocalRuleBasedAssessor,
    MuseSparkAssessor,
    _validate_schema,
    get_assessor,
)
from setu.meta_client import MetaClient
from setu.osm import OSMBridgeInfo
from setu.spectrum import SpectrumOutput


def test_safety_philosophy_never_declares_safe():
    """Verify that SETU's assessment NEVER outputs a 'safe' declaration."""
    assessor = LocalRuleBasedAssessor()

    spectrum_normal = SpectrumOutput(
        frequencies_hz=np.linspace(0, 15, 100),
        psd=np.zeros(100),
        dominant_peaks=[],
        fundamental_frequency_hz=1.25,
    )
    bridge_sample = OSMBridgeInfo(
        osm_id="way/12345",
        name="Sample Bridge A - Chennai",
        bridge_type="beam",
        material="concrete",
        length_meters=80.0,
        latitude=13.06,
        longitude=80.28,
        distance_meters=20.0,
    )

    result = assessor.assess(bridge=bridge_sample, spectrum=spectrum_normal)

    summary_lower = result.summary.lower()
    rec_lower = result.recommended_action.lower()

    # Must never declare the structure "safe" or "sound"
    assert "certified safe" not in summary_lower
    assert "safe to use" not in summary_lower
    assert "no inspection needed" not in rec_lower
    assert result.severity in ("none", "low", "medium", "high")


def test_forced_json_schema_compliance():
    """Verify that both assessors produce dictionaries matching the forced JSON schema."""
    assessor = LocalRuleBasedAssessor()

    bridge = OSMBridgeInfo(
        osm_id="way/999",
        name="Sample Span",
        bridge_type="truss",
        material="steel",
        length_meters=100.0,
    )

    result = assessor.assess(
        bridge=bridge,
        peaks=[{"freq_hz": 1.0, "rel_power": 1.0, "snr": 20.0}],
        quality_report={"background_residual_px": 0.04, "snr": 22.0, "warnings": []},
        previous_scans=[{"freq_hz": 1.02}, {"freq_hz": 1.01}],
    )

    data = result.to_dict()
    assert _validate_schema(data) is True
    assert data["severity"] in ("none", "low", "medium", "high")
    assert isinstance(data["reasons"], list)
    assert data["confidence"] in ("low", "medium", "high")
    assert isinstance(data["data_quality_issues"], list)
    assert isinstance(data["recommended_action"], str)
    assert data["assessor"] in ("muse_spark", "local_rules")


def test_history_weighted_over_type_ranges():
    """Verify that bridge's own previous scan history determines expected frequency bounds."""
    assessor = LocalRuleBasedAssessor()

    # 50m bridge generic formula would expect ~2.0 Hz (100 / 50m)
    # But previous scans show it historically operates at ~3.5 Hz
    bridge = OSMBridgeInfo(
        osm_id="way/777",
        name="Sample Historical Span",
        length_meters=50.0,
    )

    result = assessor.assess(
        bridge=bridge,
        peaks=[{"freq_hz": 3.48, "rel_power": 1.0, "snr": 20.0}],
        previous_scans=[{"freq_hz": 3.52}, {"freq_hz": 3.50}, {"freq_hz": 3.49}],
    )

    # Should recognize 3.48 Hz as normal baseline based on history, not an anomaly
    assert result.severity == "none"
    assert result.expected_range_hz is not None
    assert 3.2 < result.expected_range_hz[0] < 3.4
    assert 3.6 < result.expected_range_hz[1] < 3.8


def test_poor_data_quality_reports_issue_instead_of_defect():
    """Verify that poor video quality flags data quality issue instead of structural defect."""
    assessor = LocalRuleBasedAssessor()

    bridge = OSMBridgeInfo(osm_id="way/888", name="Windy Span", length_meters=80.0)

    # High background residual motion (camera shake) and low SNR
    quality = {
        "background_residual_px": 0.18,
        "snr": 9.2,
        "warnings": ["Severe camera tripod buffeting", "Intermittent pedestrian occlusion"],
    }

    result = assessor.assess(
        bridge=bridge,
        peaks=[{"freq_hz": 0.65, "rel_power": 1.0, "snr": 9.2}],
        quality_report=quality,
    )

    assert len(result.data_quality_issues) > 0
    assert result.confidence == "low"
    assert result.severity != "high"


def test_muse_spark_assessor_mock_mode():
    """Verify MuseSparkAssessor runs offline in mock mode and complies with forced schema."""
    client = MetaClient(mode="mock")
    assessor = MuseSparkAssessor(client=client)

    bridge = OSMBridgeInfo(osm_id="way/123", name="Sample Bridge A - Chennai", length_meters=80.0)
    spectrum = SpectrumOutput(
        frequencies_hz=np.linspace(0, 15, 100),
        psd=np.zeros(100),
        dominant_peaks=[],
        fundamental_frequency_hz=1.62,
    )

    result = assessor.assess(bridge=bridge, spectrum=spectrum)

    assert result.assessor == "muse_spark"
    assert result.severity in ("none", "low", "medium", "high")
    assert _validate_schema(result.to_dict()) is True


def test_muse_spark_fallback_to_local_rules():
    """Verify that if Meta API throws an exception, MuseSparkAssessor falls back cleanly."""
    mock_client = MagicMock()
    mock_client.chat_completion.side_effect = RuntimeError("Meta API service timeout")

    assessor = MuseSparkAssessor(client=mock_client)

    bridge = OSMBridgeInfo(osm_id="way/123", name="Fallback Bridge", length_meters=80.0)
    spectrum = SpectrumOutput(
        frequencies_hz=np.linspace(0, 15, 100),
        psd=np.zeros(100),
        dominant_peaks=[],
        fundamental_frequency_hz=1.25,
    )

    result = assessor.assess(bridge=bridge, spectrum=spectrum)

    # Clean fallback without raising uncaught exception
    assert result.assessor == "local_rules"
    assert result.severity in ("none", "low", "medium", "high")
