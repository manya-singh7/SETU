"""
Pipeline interpretation module for Project SETU.

Synthesizes structural vibration telemetry, visual quality gating, and OpenStreetMap
infrastructure metadata into a unified interpretation record with safety-critical
evidence locking.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from setu.assess import (
    AnomalyAssessmentResult,
    BaseAssessor,
    LocalRuleBasedAssessor,
    MuseSparkAssessor,
    check_evidence_lock,
    collect_allowed_numbers,
    get_assessor,
)
from setu.config import settings
from setu.meta_client import MetaClient, get_meta_client
from setu.osm import OSMBridgeInfo, OverpassBridgeClient, get_nearest_bridge
from setu.quality_gate import QualityGate
from setu.spectrum import SpectrumOutput

logger = logging.getLogger(__name__)

NO_BRIDGE_WARNING = "No mapped bridge within 150 m of this location"


def _extract_spectrum_info(spectrum: Any) -> Tuple[List[Dict[str, Any]], str, List[str]]:
    """Extract normalized peaks list, verdict string, and verdict reasons/flags."""
    peaks: List[Dict[str, Any]] = []
    verdict: str = ""
    reasons: List[str] = []

    if isinstance(spectrum, dict):
        raw_peaks = spectrum.get("peaks") or spectrum.get("dominant_peaks") or []
        verdict = str(spectrum.get("verdict") or "")
        reasons_val = spectrum.get("reasons") or spectrum.get("flags") or []
        if isinstance(reasons_val, (list, tuple)):
            reasons = [str(r) for r in reasons_val]
        elif reasons_val:
            reasons = [str(reasons_val)]
    elif isinstance(spectrum, (list, tuple)):
        raw_peaks = list(spectrum)
        if raw_peaks:
            p0 = raw_peaks[0]
            verdict = str(getattr(p0, "verdict", "") or (p0.get("verdict") if isinstance(p0, dict) else ""))
            flags = (
                getattr(p0, "flags", None)
                or getattr(p0, "reasons", None)
                or (p0.get("flags") if isinstance(p0, dict) else None)
            )
            if flags:
                reasons = [str(f) for f in flags] if isinstance(flags, (list, tuple)) else [str(flags)]
    else:
        # SpectrumOutput or similar object
        raw_peaks = getattr(spectrum, "dominant_peaks", None) or getattr(spectrum, "peaks", None) or []
        verdict = str(getattr(spectrum, "verdict", "") or getattr(spectrum, "confidence", ""))
        flags = getattr(spectrum, "reasons", None) or getattr(spectrum, "flags", None)
        if flags:
            reasons = [str(f) for f in flags] if isinstance(flags, (list, tuple)) else [str(flags)]
        if not verdict and raw_peaks:
            p0 = raw_peaks[0]
            verdict = str(getattr(p0, "verdict", "") or "")
            if not reasons:
                flags0 = getattr(p0, "flags", None) or getattr(p0, "reasons", None)
                if flags0:
                    reasons = [str(f) for f in flags0] if isinstance(flags0, (list, tuple)) else [str(flags0)]

    for p in raw_peaks:
        if isinstance(p, dict):
            peaks.append(p)
        else:
            peaks.append({
                "freq_hz": getattr(p, "freq_hz", 0.0),
                "rel_power": getattr(p, "rel_power", 0.0),
                "snr": getattr(p, "snr", 0.0),
                "flags": getattr(p, "flags", []),
                "verdict": getattr(p, "verdict", ""),
            })

    return peaks, verdict.strip(), reasons


def interpret(
    frame0: Union[np.ndarray, str, Path, bytes],
    gps: Optional[Tuple[float, float]],
    spectrum: Any,
    quality_report: Optional[Dict[str, Any]] = None,
    segmentation_provenance: Optional[Dict[str, Any]] = None,
    previous_scans: Optional[List[Dict[str, Any]]] = None,
    client: Optional[MetaClient] = None,
    assessor: Optional[BaseAssessor] = None,
    quality_gate: Optional[QualityGate] = None,
    osm_client: Optional[OverpassBridgeClient] = None,
) -> Dict[str, Any]:
    """Execute end-to-end interpretation of scan telemetry.

    Synthesizes:
      - OSM bridge metadata (nearest within 150m or warning string)
      - Quality gate visual check of frame 0 via Muse Spark Vision
      - Anomaly assessment from setu/assess.py under Rule 1 (weak input suppression)
        and Rule 2 (Evidence Lock with one retry before fallback to local rules)

    Args:
        frame0: Frame 0 image as numpy array, file path, or bytes.
        gps: (latitude, longitude) coordinate tuple, or None.
        spectrum: Peaks list, dict with verdict and reasons, or SpectrumOutput.
        quality_report: Optional tracking quality telemetry dict.
        segmentation_provenance: Optional segmentation metadata dict.
        previous_scans: Optional list of historical scan records for this bridge.
        client: Optional MetaClient override.
        assessor: Optional BaseAssessor override.
        quality_gate: Optional QualityGate override.
        osm_client: Optional OverpassBridgeClient override.

    Returns:
        Dictionary with keys:
          - "osm": OSMBridgeInfo or "No mapped bridge within 150 m of this location"
          - "quality_gate": Dict with is_structural_element, has_static_background, appears_stable, issues
          - "assessment": AnomalyAssessmentResult
    """
    # 1. Determine active client & mock mode
    meta_cli = client or get_meta_client()
    is_mock = getattr(meta_cli, "mode", "mock") == "mock"

    # 2. OSM Bridge Lookup
    nearest_bridge: Optional[OSMBridgeInfo] = None
    if gps is not None and isinstance(gps, (tuple, list)) and len(gps) >= 2:
        try:
            lat, lon = float(gps[0]), float(gps[1])
            c_osm = osm_client or OverpassBridgeClient(mode="mock" if is_mock else None)
            nearest_bridge = c_osm.find_nearest_bridge(lat, lon, radius_meters=150.0)
        except Exception as exc:
            logger.warning(f"OSM lookup failed: {exc}")
            nearest_bridge = None

    osm_result: Union[OSMBridgeInfo, str]
    if nearest_bridge is not None:
        osm_result = nearest_bridge
    else:
        osm_result = NO_BRIDGE_WARNING

    # 3. Quality Gate on Frame 0
    qgate = quality_gate or QualityGate(client=meta_cli)
    quality_gate_result = qgate.check_frame(frame0)

    # 4. Extract spectrum peaks, verdict, and reasons
    peaks, verdict_str, verdict_reasons = _extract_spectrum_info(spectrum)
    v_norm = verdict_str.lower().strip()

    # Rule 1: If the verdict is "weak" or "failed", don't ask Muse Spark for an
    # anomaly assessment. Return severity "none" with reason
    # "Insufficient data quality for assessment" plus the verdict's reasons.
    # Weak input is never an anomaly.
    if v_norm in ("weak", "failed") or "weak" in v_norm or "failed" in v_norm:
        logger.info(f"Spectrum verdict is '{verdict_str}'; suppressing anomaly screening under Rule 1.")
        combined_reasons = ["Insufficient data quality for assessment"] + verdict_reasons
        assessment_result = AnomalyAssessmentResult(
            severity="none",
            reasons=combined_reasons,
            expected_range_hz=None,
            confidence="low",
            data_quality_issues=combined_reasons,
            recommended_action="Insufficient data quality for assessment. Re-record video with stationary phone bracing and visible stationary background.",
            assessor="local_rules",
        )
        return {
            "osm": osm_result,
            "quality_gate": quality_gate_result,
            "assessment": assessment_result,
        }

    # 5. Assessment with Evidence Lock (Rule 2)
    active_assessor = assessor or MuseSparkAssessor(client=meta_cli)
    local_assessor = LocalRuleBasedAssessor()

    allowed_numbers = collect_allowed_numbers(
        bridge=nearest_bridge,
        spectrum=spectrum,
        peaks=peaks,
        quality_report=quality_report,
        segmentation_provenance=segmentation_provenance,
        previous_scans=previous_scans,
    )

    try:
        assessment_result = active_assessor.assess(
            bridge=nearest_bridge,
            spectrum=spectrum if isinstance(spectrum, SpectrumOutput) else None,
            peaks=peaks,
            quality_report=quality_report,
            previous_scans=previous_scans,
        )
    except Exception as exc:
        logger.warning(f"Primary assessor call failed: {exc}. Falling back to LocalRuleBasedAssessor.")
        assessment_result = local_assessor.assess(
            bridge=nearest_bridge,
            spectrum=spectrum if isinstance(spectrum, SpectrumOutput) else None,
            peaks=peaks,
            quality_report=quality_report,
            previous_scans=previous_scans,
        )

    # Rule 2: Evidence Lock
    # Check every number in reasons and expected_range_hz.
    if assessment_result.assessor == "muse_spark":
        untraced = check_evidence_lock(assessment_result, allowed_numbers)
        if untraced:
            logger.info(f"Evidence Lock check failed with untraced numbers {untraced}; retrying once...")
            retry_result: Optional[AnomalyAssessmentResult] = None
            try:
                retry_instruction = (
                    f"Your previous response included numbers that could not be verified against the input telemetry or OSM metadata: {untraced}. "
                    f"Under the SETU Evidence Lock, EVERY number in reasons and expected_range_hz MUST match a value present in the inputs or OSM bridge data. "
                    f"Please re-analyze and return valid JSON using ONLY verified numbers from the inputs."
                )
                if isinstance(active_assessor, MuseSparkAssessor):
                    retry_result = active_assessor.assess(
                        bridge=nearest_bridge,
                        spectrum=spectrum if isinstance(spectrum, SpectrumOutput) else None,
                        peaks=peaks,
                        quality_report=quality_report,
                        previous_scans=previous_scans,
                        retry_prompt=retry_instruction,
                    )
                else:
                    retry_result = active_assessor.assess(
                        bridge=nearest_bridge,
                        spectrum=spectrum if isinstance(spectrum, SpectrumOutput) else None,
                        peaks=peaks,
                        quality_report=quality_report,
                        previous_scans=previous_scans,
                    )
            except Exception as retry_exc:
                logger.warning(f"Evidence Lock retry call failed: {retry_exc}")
                retry_result = None

            if retry_result is not None and retry_result.assessor == "muse_spark":
                retry_untraced = check_evidence_lock(retry_result, allowed_numbers)
                if not retry_untraced:
                    assessment_result = retry_result
                else:
                    logger.warning(
                        f"Evidence Lock check failed after retry with untraced numbers {retry_untraced}. Falling back to LocalRuleBasedAssessor."
                    )
                    fallback = local_assessor.assess(
                        bridge=nearest_bridge,
                        spectrum=spectrum if isinstance(spectrum, SpectrumOutput) else None,
                        peaks=peaks,
                        quality_report=quality_report,
                        previous_scans=previous_scans,
                    )
                    reason_msg = (
                        f"Evidence Lock rejection: Untraced numbers {retry_untraced} detected in Muse Spark output after retry; fell back to LocalRuleBasedAssessor."
                    )
                    fallback.reasons.insert(0, reason_msg)
                    fallback.data_quality_issues.append(reason_msg)
                    assessment_result = fallback
            else:
                logger.warning("Evidence Lock retry produced no valid result. Falling back to LocalRuleBasedAssessor.")
                fallback = local_assessor.assess(
                    bridge=nearest_bridge,
                    spectrum=spectrum if isinstance(spectrum, SpectrumOutput) else None,
                    peaks=peaks,
                    quality_report=quality_report,
                    previous_scans=previous_scans,
                )
                reason_msg = (
                    f"Evidence Lock rejection: Untraced numbers {untraced} detected in Muse Spark output and retry failed; fell back to LocalRuleBasedAssessor."
                )
                fallback.reasons.insert(0, reason_msg)
                fallback.data_quality_issues.append(reason_msg)
                assessment_result = fallback

    return {
        "osm": osm_result,
        "quality_gate": quality_gate_result,
        "assessment": assessment_result,
    }


interpret_scan = interpret
