"""Unit tests for SETU interpretation pipeline, Rule 1, and Evidence Lock (Rule 2)."""

import json
from unittest.mock import MagicMock
import numpy as np
import pytest

from setu.assess import LocalRuleBasedAssessor, MuseSparkAssessor
from setu.interpret import NO_BRIDGE_WARNING, interpret, interpret_scan
from setu.meta_client import MetaClient
from setu.osm import OSMBridgeInfo, OverpassBridgeClient


@pytest.fixture
def dummy_frame():
    """Return a simple dummy frame array for vision testing."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


def test_interpret_normal_flow_mock_mode(dummy_frame):
    """Verify standard interpretation synthesizing OSM, Quality Gate, and Evidence-Locked assessment."""
    client = MetaClient(mode="mock")
    osm_client = OverpassBridgeClient(mode="mock")

    # Inputs matching muse_spark_assess.json (1.62 Hz, 3.25 Hz, 2.15 Hz historic baseline)
    spectrum = {
        "peaks": [
            {"freq_hz": 1.62, "rel_power": 0.82, "snr": 20.0},
            {"freq_hz": 3.25, "rel_power": 0.15, "snr": 15.0},
        ],
        "verdict": "Reliable",
        "reasons": ["Dominant structural resonant peak detected at 1.62 Hz"],
    }
    previous_scans = [{"freq_hz": 2.15, "timestamp": "2026-10-15"}]
    quality_report = {"background_residual_px": 0.04, "snr": 20.0, "warnings": []}
    segmentation_provenance = {"target_bbox": [256, 168, 128, 168], "method": "sam"}

    # GPS coordinates for Napier Bridge
    gps = (13.067200, 80.284600)

    result = interpret(
        frame0=dummy_frame,
        gps=gps,
        spectrum=spectrum,
        quality_report=quality_report,
        segmentation_provenance=segmentation_provenance,
        previous_scans=previous_scans,
        client=client,
        osm_client=osm_client,
    )

    # 1. Structure of return dictionary
    assert isinstance(result, dict)
    assert "osm" in result
    assert "quality_gate" in result
    assert "assessment" in result

    # 2. OSM Bridge Info
    osm_bridge = result["osm"]
    assert isinstance(osm_bridge, OSMBridgeInfo)
    assert osm_bridge.name == "Napier Bridge"
    assert osm_bridge.osm_id == "way/24891152"

    # 3. Quality Gate
    qg = result["quality_gate"]
    assert isinstance(qg, dict)
    assert qg["is_structural_element"] is True
    assert qg["appears_stable"] is True
    assert "issues" in qg

    # 4. Assessment
    assessment = result["assessment"]
    assert assessment.assessor == "muse_spark"
    assert assessment.severity in ("none", "low", "medium", "high")
    assert assessment.expected_range_hz == [1.90, 2.30]


def test_rule_1_verdict_weak_suppresses_muse_spark(dummy_frame):
    """Rule 1: If verdict is 'weak', do NOT ask Muse Spark. Return severity 'none' with reason

    'Insufficient data quality for assessment' plus the verdict's reasons.
    """
    mock_client = MagicMock()
    # If Muse Spark were called, chat_completion would be triggered
    mock_client.chat_completion.side_effect = AssertionError("Muse Spark must NOT be called for weak input!")

    assessor = MuseSparkAssessor(client=mock_client)

    spectrum = {
        "peaks": [{"freq_hz": 0.55, "rel_power": 0.3, "snr": 5.2}],
        "verdict": "weak",
        "reasons": ["Displacement SNR 5.2 dB below threshold", "Possible low-frequency drift"],
    }

    result = interpret(
        frame0=dummy_frame,
        gps=None,
        spectrum=spectrum,
        assessor=assessor,
        client=mock_client,
    )

    # Muse Spark was never called
    mock_client.chat_completion.assert_not_called()

    assessment = result["assessment"]
    assert assessment.severity == "none"
    assert assessment.assessor == "local_rules"
    assert "Insufficient data quality for assessment" in assessment.reasons[0]
    assert "Displacement SNR 5.2 dB below threshold" in assessment.reasons[1]
    assert "Possible low-frequency drift" in assessment.reasons[2]


def test_rule_1_verdict_failed_suppresses_muse_spark(dummy_frame):
    """Rule 1: If verdict is 'failed', do NOT ask Muse Spark for anomaly assessment."""
    mock_client = MagicMock()
    mock_client.chat_completion.side_effect = AssertionError("Muse Spark must NOT be called for failed input!")

    assessor = MuseSparkAssessor(client=mock_client)

    spectrum = {
        "peaks": [],
        "verdict": "failed",
        "reasons": ["Excessive camera shake exceeding compensation boundary", "Lost optical track"],
    }

    result = interpret(
        frame0=dummy_frame,
        gps=None,
        spectrum=spectrum,
        assessor=assessor,
        client=mock_client,
    )

    mock_client.chat_completion.assert_not_called()

    assessment = result["assessment"]
    assert assessment.severity == "none"
    assert "Insufficient data quality for assessment" in assessment.reasons[0]
    assert "Excessive camera shake exceeding compensation boundary" in assessment.reasons[1]


def test_rule_2_evidence_lock_rejection_and_fallback_to_local_rules(dummy_frame):
    """Rule 2: Check every number in Muse Spark's output.

    If any number cannot be traced to inputs or OSM data, retry once, then fall back
    to local rules assessor and record why.
    """
    client = MetaClient(mode="mock")

    # Inputs with frequency 5.50 Hz, but NO 1.62 Hz, 2.15 Hz, or 3.25 Hz
    spectrum = {
        "peaks": [{"freq_hz": 5.50, "rel_power": 0.90, "snr": 25.0}],
        "verdict": "Reliable",
        "reasons": ["Dominant peak at 5.50 Hz"],
    }

    # Muse Spark mock fixture returns 1.62 Hz, 2.15 Hz, 3.25 Hz, [1.90, 2.30]
    # None of these numbers exist in our input telemetry!
    result = interpret(
        frame0=dummy_frame,
        gps=None,
        spectrum=spectrum,
        client=client,
    )

    assessment = result["assessment"]
    # Should fall back to local rules because numbers in fixture could not be grounded in 5.50 Hz input
    assert assessment.assessor == "local_rules"
    assert any("Evidence Lock rejection" in r for r in assessment.reasons)
    assert any("untraced numbers" in str(issue).lower() or "evidence lock" in str(issue).lower() for issue in assessment.data_quality_issues)


def test_rule_2_evidence_lock_with_hallucinated_fixture(dummy_frame):
    """Rule 2: Explicitly tests that hallucinated numbers (e.g. 9.87 Hz, 99.9%) trigger Evidence Lock fallback."""
    client = MetaClient(mode="mock")
    assessor = MuseSparkAssessor(client=client)

    # Monkeypatch assessor to use hallucinated fixture
    original_assess = assessor.assess

    def mocked_assess(*args, **kwargs):
        kwargs["fixture_name"] = "muse_spark_assess_hallucinated.json"
        return original_assess(*args, **kwargs)

    assessor.assess = mocked_assess

    spectrum = {
        "peaks": [{"freq_hz": 2.0, "rel_power": 0.8, "snr": 20.0}],
        "verdict": "Reliable",
        "reasons": ["Clean mode detected at 2.0 Hz"],
    }

    result = interpret(
        frame0=dummy_frame,
        gps=None,
        spectrum=spectrum,
        assessor=assessor,
        client=client,
    )

    assessment = result["assessment"]
    assert assessment.assessor == "local_rules"
    assert any("Evidence Lock rejection" in r for r in assessment.reasons)


def test_no_mapped_bridge_within_150m_warning(dummy_frame):
    """Verify that when no bridge is within 150m, exact warning string is returned."""
    client = MetaClient(mode="mock")
    osm_client = OverpassBridgeClient(mode="mock")

    # GPS far away from any fixture bridges (e.g. middle of the ocean)
    gps_far = (0.000000, 0.000000)

    spectrum = {
        "peaks": [{"freq_hz": 1.62, "rel_power": 0.85, "snr": 20.0}],
        "verdict": "Reliable",
        "reasons": ["Peak at 1.62 Hz"],
    }

    result = interpret(
        frame0=dummy_frame,
        gps=gps_far,
        spectrum=spectrum,
        client=client,
        osm_client=osm_client,
    )

    assert result["osm"] == NO_BRIDGE_WARNING
    assert result["osm"] == "No mapped bridge within 150 m of this location"


def test_no_gps_returns_warning(dummy_frame):
    """Verify that None GPS returns exact warning string."""
    client = MetaClient(mode="mock")
    spectrum = {"peaks": [], "verdict": "weak", "reasons": []}

    result = interpret(
        frame0=dummy_frame,
        gps=None,
        spectrum=spectrum,
        client=client,
    )

    assert result["osm"] == "No mapped bridge within 150 m of this location"
