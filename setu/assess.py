"""
Anomaly assessment module for Project SETU.

Step 6 (Part B) in SETU Pipeline:
Passes OpenStreetMap structural metadata and dominant vibration frequencies to
Muse Spark 1.3 to formulate a cautious, non-authoritative anomaly assessment.

SAFETY PRINCIPLE:
SETU NEVER declares a bridge safe. Video-based vibration analysis is an early-warning
triage heuristic to flag unusual structural dynamics for certified civil engineer review.

Both live Muse Spark 1.3 and local rule-based heuristics are placed behind BaseAssessor interface.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from setu.config import settings
from setu.models import TriageSeverity
from setu.osm import OSMBridgeInfo
from setu.spectrum import SpectrumOutput


@dataclass
class AnomalyAssessmentResult:
    """Outcome of structural anomaly screening.

    Attributes:
        severity: Urgency flag (UNUSUAL, INCONCLUSIVE, REQUIRES_INSPECTION).
        anomaly_score: Normalized anomaly indicator [0.0 - 1.0].
        summary: Clear narrative summary of spectral findings vs bridge geometry.
        recommended_action: Directive for human civil engineers.
        model_name: Identifier of the assessing model or fallback rule engine.
        metadata: Detailed reasoning traces and prompt context.
    """

    severity: TriageSeverity
    anomaly_score: float
    summary: str
    recommended_action: str
    model_name: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseAssessor(ABC):
    """Abstract interface for bridge vibration anomaly assessment."""

    @abstractmethod
    def assess(
        self,
        bridge: Optional[OSMBridgeInfo],
        spectrum: SpectrumOutput,
    ) -> AnomalyAssessmentResult:
        """Formulate a cautious anomaly evaluation.

        Args:
            bridge: OpenStreetMap bridge metadata (or None if unmapped).
            spectrum: Extracted dominant frequencies and PSD.

        Returns:
            AnomalyAssessmentResult flagging whether engineer review is recommended.
        """
        pass


class MuseSparkAssessor(BaseAssessor):
    """Meta Muse Spark 1.3 client via the official OpenAI-compatible Meta Model API.

    Model: muse-spark-1.3
    Base URL: https://api.meta.ai/v1
    """

    SAFETY_SYSTEM_PROMPT = """You are an AI civil infrastructure screening assistant for Project SETU.
CRITICAL SAFETY PHILOSOPHY:
1. You NEVER declare a bridge 'safe', 'sound', 'healthy', or 'risk-free'.
2. Video-based frequency telemetry from mobile phones is an early screening heuristic only.
3. Your role is strictly to compare observed modal frequencies against expected ranges for the bridge type/material/span, identify possible anomalous shifts, and flag items that look unusual for professional civil engineer review.
4. Output must conclude with a clear recommendation for on-site visual/sensor inspection.
"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> None:
        self.api_key = api_key or settings.META_API_KEY
        self.base_url = base_url or settings.META_API_BASE_URL
        self.model_name = model_name or settings.MUSE_MODEL_NAME

    def assess(
        self,
        bridge: Optional[OSMBridgeInfo],
        spectrum: SpectrumOutput,
    ) -> AnomalyAssessmentResult:
        """Call Muse Spark 1.3 via Meta Model API."""
        # STUB: To be implemented in pipeline phase
        raise NotImplementedError("Muse Spark 1.3 live integration stub - use LocalRuleBasedAssessor for local testing.")


class LocalRuleBasedAssessor(BaseAssessor):
    """Offline deterministic rule-based assessor for local development and testing.

    Cross-references fundamental frequency against empirical bridge engineering formulas
    without requiring external LLM API keys.
    """

    def assess(
        self,
        bridge: Optional[OSMBridgeInfo],
        spectrum: SpectrumOutput,
    ) -> AnomalyAssessmentResult:
        """Evaluate frequencies using empirical structural rules."""
        fund_hz = spectrum.fundamental_frequency_hz
        if fund_hz is None:
            return AnomalyAssessmentResult(
                severity=TriageSeverity.INCONCLUSIVE,
                anomaly_score=0.5,
                summary="Insufficient spectral peak prominence detected in 30s video recording.",
                recommended_action="Re-record 30s video with closer framing on high-contrast pillar/joint. Schedule physical inspection.",
                model_name="local-rule-engine",
            )

        # Baseline empirical estimate if bridge length is available
        if bridge and bridge.length_meters and bridge.length_meters > 0:
            expected_hz = 100.0 / bridge.length_meters
            ratio = fund_hz / expected_hz
            if ratio < 0.6 or ratio > 1.8:
                return AnomalyAssessmentResult(
                    severity=TriageSeverity.UNUSUAL,
                    anomaly_score=0.78,
                    summary=(
                        f"Observed fundamental frequency of {fund_hz:.2f} Hz deviates significantly "
                        f"from expected nominal range (~{expected_hz:.2f} Hz) for {bridge.name or 'bridge'} "
                        f"(length: {bridge.length_meters}m, type: {bridge.bridge_type or 'unknown'})."
                    ),
                    recommended_action="High-priority engineer on-site inspection recommended. Check for stiffness degradation.",
                    model_name="local-rule-engine",
                )

        return AnomalyAssessmentResult(
            severity=TriageSeverity.REQUIRES_INSPECTION,
            anomaly_score=0.35,
            summary=f"Detected fundamental frequency at {fund_hz:.2f} Hz. Routine baseline recorded.",
            recommended_action="Add to civil engineer triage review queue. SETU never certifies structures safe.",
            model_name="local-rule-engine",
        )


def get_assessor() -> BaseAssessor:
    """Factory providing the configured anomaly assessor (Muse Spark 1.3 or local fallback)."""
    if settings.USE_LOCAL_FALLBACKS or not settings.META_API_KEY:
        return LocalRuleBasedAssessor()
    return MuseSparkAssessor()
