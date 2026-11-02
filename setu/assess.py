"""
Anomaly assessment module for Project SETU.

Step 6 (Part B) in SETU Pipeline:
Evaluates structural vibration telemetry and OpenStreetMap metadata to formulate
a cautious, non-authoritative anomaly triage assessment.

SAFETY PRINCIPLE:
SETU NEVER declares a bridge safe. Video-based vibration analysis is an early-warning
screening heuristic to triage unusual structural dynamics for certified civil engineer review.
Frequency shifts alone cannot confirm damage; temperature and traffic shift frequencies by several percent.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

from setu.config import settings
from setu.meta_client import MetaClient, get_meta_client
from setu.osm import OSMBridgeInfo
from setu.spectrum import SpectrumOutput

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert AI civil infrastructure screening assistant for Project SETU.

CRITICAL SAFETY AND REASONING PHILOSOPHY:
1. NEVER declare a bridge 'safe', 'sound', 'healthy', or 'risk-free'.
2. Video-based modal frequency telemetry from mobile phones is an early-warning screening heuristic only; frequency alone CANNOT confirm structural damage.
3. Ambient temperature fluctuations and live traffic naturally shift bridge natural frequency by a few percent (diurnal envelope); do not over-flag minor variances.
4. Weight this bridge's own historical vibration frequency baseline ABOVE generic structural type-based formulas.
5. If data quality is poor (excessive camera shake, high background residual, low SNR), explicitly report it in data_quality_issues instead of flagging a structural defect.
6. Conclude with a clear recommendation for on-site physical civil engineer inspection.

You MUST respond ONLY with valid JSON strictly matching this schema:
{
  "severity": "none" | "low" | "medium" | "high",
  "reasons": ["detailed string explaining the finding"],
  "expected_range_hz": [min_hz, max_hz] or null,
  "confidence": "low" | "medium" | "high",
  "data_quality_issues": ["description of any noise, camera shake, or low SNR issues"],
  "recommended_action": "actionable civil engineering directive",
  "assessor": "muse_spark"
}"""


@dataclass
class AnomalyAssessmentResult:
    """Outcome of structural anomaly screening matching forced JSON schema."""

    severity: str  # "none" | "low" | "medium" | "high"
    reasons: List[str]
    expected_range_hz: Optional[List[float]] = None
    confidence: str = "medium"  # "low" | "medium" | "high"
    data_quality_issues: List[str] = field(default_factory=list)
    recommended_action: str = ""
    assessor: str = "local_rules"  # "muse_spark" | "local_rules"

    # Backwards compatibility attributes:
    anomaly_score: float = 0.0
    summary: str = ""
    model_name: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.summary:
            self.summary = "; ".join(self.reasons) if self.reasons else "Assessment completed."
        if not self.model_name:
            self.model_name = self.assessor
        if self.anomaly_score == 0.0:
            score_map = {"high": 0.85, "medium": 0.65, "low": 0.35, "none": 0.10}
            self.anomaly_score = score_map.get(str(self.severity).lower(), 0.3)

    def to_dict(self) -> Dict[str, Any]:
        """Return forced JSON schema representation."""
        return {
            "severity": self.severity,
            "reasons": self.reasons,
            "expected_range_hz": self.expected_range_hz,
            "confidence": self.confidence,
            "data_quality_issues": self.data_quality_issues,
            "recommended_action": self.recommended_action,
            "assessor": self.assessor,
        }

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)

    def get(self, item: str, default: Any = None) -> Any:
        return getattr(self, item, default)


class BaseAssessor(ABC):
    """Abstract interface for bridge vibration anomaly assessment."""

    @abstractmethod
    def assess(
        self,
        bridge: Optional[OSMBridgeInfo] = None,
        spectrum: Optional[SpectrumOutput] = None,
        peaks: Optional[List[Dict[str, Any]]] = None,
        quality_report: Optional[Dict[str, Any]] = None,
        previous_scans: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> AnomalyAssessmentResult:
        """Formulate a cautious anomaly evaluation."""
        pass


class LocalRuleBasedAssessor(BaseAssessor):
    """Offline deterministic rule-based assessor."""

    def assess(
        self,
        bridge: Optional[OSMBridgeInfo] = None,
        spectrum: Optional[SpectrumOutput] = None,
        peaks: Optional[List[Dict[str, Any]]] = None,
        quality_report: Optional[Dict[str, Any]] = None,
        previous_scans: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> AnomalyAssessmentResult:
        """Evaluate frequencies using empirical structural rules and baseline comparison."""
        fund_hz: Optional[float] = None
        if spectrum and spectrum.fundamental_frequency_hz is not None:
            fund_hz = float(spectrum.fundamental_frequency_hz)
        elif peaks and len(peaks) > 0:
            p0 = peaks[0]
            fund_hz = float(p0.get("freq_hz") if isinstance(p0, dict) else getattr(p0, "freq_hz", 0.0))

        # Check data quality report
        data_quality_issues: List[str] = []
        if quality_report:
            warnings = quality_report.get("warnings") or []
            data_quality_issues.extend([str(w) for w in warnings])
            res_px = quality_report.get("background_residual_px")
            if res_px is not None and res_px > 0.10:
                data_quality_issues.append(f"Elevated background motion residual: {res_px:.2f} px (> 0.10 px limit)")
            snr = quality_report.get("snr")
            if snr is not None and snr < 12.0:
                data_quality_issues.append(f"Low signal-to-noise ratio: {snr:.1f} dB (< 12 dB limit)")

        # Case 1: Inconclusive peak
        if fund_hz is None or fund_hz <= 0.0:
            return AnomalyAssessmentResult(
                severity="low",
                reasons=["Insufficient spectral peak prominence detected in 30s video recording."],
                expected_range_hz=None,
                confidence="low",
                data_quality_issues=data_quality_issues or ["No distinct resonant peak identified."],
                recommended_action="Re-record 30s video with closer framing on high-contrast pillar/joint. Schedule physical civil engineer inspection.",
                assessor="local_rules",
            )

        # Baseline comparison: priority given to historical baseline over type-based formula
        expected_range_hz: Optional[List[float]] = None
        if previous_scans and len(previous_scans) > 0:
            past_freqs = [
                float(s.get("freq_hz"))
                for s in previous_scans
                if isinstance(s, dict) and s.get("freq_hz")
            ]
            if past_freqs:
                hist_avg = sum(past_freqs) / len(past_freqs)
                # Allow ~8% normal diurnal/temperature variance
                expected_range_hz = [round(hist_avg * 0.92, 2), round(hist_avg * 1.08, 2)]

        if not expected_range_hz and bridge and bridge.length_meters and bridge.length_meters > 0:
            nominal_hz = 100.0 / bridge.length_meters
            expected_range_hz = [round(nominal_hz * 0.75, 2), round(nominal_hz * 1.25, 2)]

        # If data quality is heavily degraded, report that instead of flagging structural defect
        if len(data_quality_issues) >= 2:
            return AnomalyAssessmentResult(
                severity="low",
                reasons=[
                    f"Fundamental peak extracted at {fund_hz:.2f} Hz, but data quality issues degrade measurement fidelity.",
                    "Vibration shift cannot be confirmed due to camera jitter or tracking noise.",
                ],
                expected_range_hz=expected_range_hz,
                confidence="low",
                data_quality_issues=data_quality_issues,
                recommended_action="Re-record video with stationary phone bracing. Certified physical inspection required for baseline validation.",
                assessor="local_rules",
            )

        if expected_range_hz:
            exp_min, exp_max = expected_range_hz[0], expected_range_hz[1]
            if fund_hz < exp_min * 0.80 or fund_hz > exp_max * 1.25:
                # >20% abnormal deviation
                return AnomalyAssessmentResult(
                    severity="high",
                    reasons=[
                        f"Fundamental frequency of {fund_hz:.2f} Hz deviates significantly from baseline range [{exp_min:.2f} - {exp_max:.2f} Hz].",
                        f"Target structure: {bridge.name if bridge else 'Bridge'} ({bridge.length_meters if bridge else '?'}m).",
                    ],
                    expected_range_hz=expected_range_hz,
                    confidence="high" if not data_quality_issues else "medium",
                    data_quality_issues=data_quality_issues,
                    recommended_action="High-priority engineer on-site inspection recommended. Check for stiffness degradation and bearing seats.",
                    assessor="local_rules",
                )
            elif fund_hz < exp_min or fund_hz > exp_max:
                # Moderate deviation (5-20%)
                return AnomalyAssessmentResult(
                    severity="medium",
                    reasons=[
                        f"Observed fundamental frequency of {fund_hz:.2f} Hz moderately deviates from baseline [{exp_min:.2f} - {exp_max:.2f} Hz].",
                        "May reflect seasonal thermal variance, live traffic loading, or localized structural softening.",
                    ],
                    expected_range_hz=expected_range_hz,
                    confidence="medium",
                    data_quality_issues=data_quality_issues,
                    recommended_action="Schedule engineer review to cross-reference against thermal and traffic conditions. SETU does not declare structures safe.",
                    assessor="local_rules",
                )

        # Normal baseline within expected range
        return AnomalyAssessmentResult(
            severity="none",
            reasons=[
                f"Detected fundamental frequency at {fund_hz:.2f} Hz.",
                "Observed frequency aligns within expected baseline operational envelope.",
            ],
            expected_range_hz=expected_range_hz,
            confidence="high" if not data_quality_issues else "medium",
            data_quality_issues=data_quality_issues,
            recommended_action="Routine baseline telemetry recorded. Add to civil engineer triage review queue. SETU never certifies structures safe.",
            assessor="local_rules",
        )


def _validate_schema(data: Any) -> bool:
    """Validate that parsed dictionary complies with the required assessment schema."""
    if not isinstance(data, dict):
        return False
    required_keys = {"severity", "reasons", "expected_range_hz", "confidence", "data_quality_issues", "recommended_action"}
    if not required_keys.issubset(data.keys()):
        return False
    if str(data.get("severity", "")).lower() not in ("none", "low", "medium", "high"):
        return False
    if not isinstance(data.get("reasons"), list):
        return False
    if str(data.get("confidence", "")).lower() not in ("low", "medium", "high"):
        return False
    exp = data.get("expected_range_hz")
    if exp is not None and (not isinstance(exp, list) or len(exp) != 2):
        return False
    return True


class MuseSparkAssessor(BaseAssessor):
    """Meta Muse Spark 1.3 client routed through MetaClient."""

    def __init__(
        self,
        client: Optional[MetaClient] = None,
        fallback_assessor: Optional[BaseAssessor] = None,
    ) -> None:
        self.client = client or get_meta_client()
        self.fallback = fallback_assessor or LocalRuleBasedAssessor()

    def _build_user_prompt(
        self,
        bridge: Optional[OSMBridgeInfo],
        spectrum: Optional[SpectrumOutput],
        peaks: Optional[List[Dict[str, Any]]],
        quality_report: Optional[Dict[str, Any]],
        previous_scans: Optional[List[Dict[str, Any]]],
    ) -> str:
        """Format telemetry and structural context into clear user prompt."""
        # Normalize peaks
        peak_list: List[Dict[str, Any]] = []
        if peaks:
            for p in peaks:
                if isinstance(p, dict):
                    peak_list.append(p)
                else:
                    peak_list.append({
                        "freq_hz": getattr(p, "freq_hz", 0.0),
                        "rel_power": getattr(p, "rel_power", 0.0),
                        "snr": getattr(p, "snr", 0.0),
                    })
        elif spectrum and spectrum.dominant_peaks:
            for dp in spectrum.dominant_peaks:
                peak_list.append({"freq_hz": float(dp), "rel_power": 1.0, "snr": 15.0})
        elif spectrum and spectrum.fundamental_frequency_hz:
            peak_list.append({"freq_hz": float(spectrum.fundamental_frequency_hz), "rel_power": 1.0, "snr": 15.0})

        bridge_info = {
            "name": bridge.name if bridge else "Unknown Bridge",
            "type": bridge.bridge_type if bridge else "Unknown",
            "material": bridge.material if bridge else "Unknown",
            "length_meters": bridge.length_meters if bridge else None,
            "tags": bridge.tags if bridge else {},
        }

        payload = {
            "bridge_metadata": bridge_info,
            "dominant_peaks": peak_list,
            "quality_report": quality_report or {},
            "previous_scans_history": previous_scans or [],
        }
        return f"Evaluate the following bridge vibration telemetry and return JSON assessment:\n{json.dumps(payload, indent=2)}"

    def assess(
        self,
        bridge: Optional[OSMBridgeInfo] = None,
        spectrum: Optional[SpectrumOutput] = None,
        peaks: Optional[List[Dict[str, Any]]] = None,
        quality_report: Optional[Dict[str, Any]] = None,
        previous_scans: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> AnomalyAssessmentResult:
        """Assess vibration telemetry using Muse Spark via MetaClient."""
        user_prompt = self._build_user_prompt(bridge, spectrum, peaks, quality_report, previous_scans)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        def try_parse_response(resp: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            # Handle mock fixture direct data
            if "_mock_data" in resp and _validate_schema(resp["_mock_data"]):
                return resp["_mock_data"]
            choices = resp.get("choices", [])
            if not choices:
                return None
            content = choices[0].get("message", {}).get("content", "")
            if not content:
                return None
            try:
                # Strip markdown code blocks if model wrapped json in ```json ... ```
                clean_content = content.strip()
                if clean_content.startswith("```"):
                    lines = clean_content.splitlines()
                    if lines[0].startswith("```"):
                        lines = lines[1:]
                    if lines and lines[-1].startswith("```"):
                        lines = lines[:-1]
                    clean_content = "\n".join(lines).strip()
                parsed = json.loads(clean_content)
                if _validate_schema(parsed):
                    return parsed
            except Exception as e:
                logger.warning(f"JSON parsing error from Muse Spark response: {e}")
            return None

        # Call 1: Primary attempt
        try:
            response = self.client.chat_completion(
                messages=messages,
                response_format={"type": "json_object"},
                fixture_name="muse_spark_assess.json",
                extra_inputs={"task": "assess", "bridge": bridge.name if bridge else ""},
            )
            parsed_data = try_parse_response(response)

            # If validation failed, retry ONCE
            if parsed_data is None:
                logger.info("Muse Spark output invalid schema; retrying once...")
                retry_messages = list(messages)
                choices = response.get("choices", [])
                raw_out = choices[0].get("message", {}).get("content", "") if choices else ""
                retry_messages.append({"role": "assistant", "content": raw_out})
                retry_messages.append({
                    "role": "user",
                    "content": "Your response did not match the required JSON schema. Return ONLY valid JSON with keys: severity, reasons, expected_range_hz, confidence, data_quality_issues, recommended_action, assessor.",
                })

                retry_response = self.client.chat_completion(
                    messages=retry_messages,
                    response_format={"type": "json_object"},
                    fixture_name="muse_spark_assess.json",
                )
                parsed_data = try_parse_response(retry_response)

            if parsed_data is not None:
                parsed_data["assessor"] = "muse_spark"
                exp = parsed_data.get("expected_range_hz")
                if exp:
                    exp = [float(exp[0]), float(exp[1])]

                return AnomalyAssessmentResult(
                    severity=str(parsed_data["severity"]).lower(),
                    reasons=[str(r) for r in parsed_data.get("reasons", [])],
                    expected_range_hz=exp,
                    confidence=str(parsed_data.get("confidence", "medium")).lower(),
                    data_quality_issues=[str(d) for d in parsed_data.get("data_quality_issues", [])],
                    recommended_action=str(parsed_data.get("recommended_action", "")),
                    assessor="muse_spark",
                )

        except Exception as exc:
            logger.warning(f"MuseSparkAssessor call failed: {exc}. Falling back to LocalRuleBasedAssessor.")

        # Fallback to LocalRuleBasedAssessor if live call or validation failed
        fallback_res = self.fallback.assess(
            bridge=bridge,
            spectrum=spectrum,
            peaks=peaks,
            quality_report=quality_report,
            previous_scans=previous_scans,
        )
        return fallback_res


def get_assessor() -> BaseAssessor:
    """Factory providing the configured anomaly assessor."""
    client = get_meta_client()
    # In mock or live mode, route via MuseSparkAssessor with LocalRuleBasedAssessor fallback
    return MuseSparkAssessor(client=client)
