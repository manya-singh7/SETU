"""Tests for anomaly assessment and safety philosophy verification."""

from setu.assess import LocalRuleBasedAssessor
from setu.models import TriageSeverity
from setu.osm import OSMBridgeInfo
from setu.spectrum import SpectrumOutput
import numpy as np


def test_safety_philosophy_never_declares_safe():
    """Verify that SETU's assessment NEVER outputs a 'safe' declaration."""
    assessor = LocalRuleBasedAssessor()

    # Case 1: Routine signal matching nominal expectations
    spectrum_normal = SpectrumOutput(
        frequencies_hz=np.linspace(0, 15, 100),
        psd=np.zeros(100),
        dominant_peaks=[],
        fundamental_frequency_hz=1.25,
    )
    bridge_sample = OSMBridgeInfo(
        osm_id="way/12345",
        name="Local River Span",
        bridge_type="beam",
        material="concrete",
        length_meters=80.0,
        latitude=28.61,
        longitude=77.20,
        distance_meters=20.0,
    )

    result = assessor.assess(bridge_sample, spectrum_normal)

    # Must never declare the structure "safe" or "sound"
    summary_lower = result.summary.lower()
    rec_lower = result.recommended_action.lower()

    assert "certified safe" not in summary_lower
    assert "safe to use" not in summary_lower
    assert "no inspection needed" not in rec_lower
    assert result.severity in [TriageSeverity.REQUIRES_INSPECTION, TriageSeverity.UNUSUAL, TriageSeverity.INCONCLUSIVE]


def test_unusual_frequency_flags_triage():
    """Verify that an abnormal frequency shift flags UNUSUAL severity."""
    assessor = LocalRuleBasedAssessor()

    # 80m beam bridge usually has ~1.25 Hz; 0.2 Hz is abnormally low
    spectrum_low = SpectrumOutput(
        frequencies_hz=np.linspace(0, 15, 100),
        psd=np.zeros(100),
        dominant_peaks=[],
        fundamental_frequency_hz=0.25,
    )
    bridge_sample = OSMBridgeInfo(
        osm_id="way/12345",
        name="Local River Span",
        bridge_type="beam",
        material="concrete",
        length_meters=80.0,
        latitude=28.61,
        longitude=77.20,
        distance_meters=20.0,
    )

    result = assessor.assess(bridge_sample, spectrum_low)
    assert result.severity == TriageSeverity.UNUSUAL
    assert result.anomaly_score > 0.7
