"""
Pipeline execution engine for Project SETU.

Implements `analyze_video` uniting the full vibration analysis pipeline:
  1. video_checks.py: integrity, duration, fps, resolution.
  2. segment.py: structural target and static background segmentation.
  3. motion.py: subpixel tracking with background jitter compensation.
  4. spectrum.py: Welch power spectral density, resonant modal peaks, and verdict.
  5. interpret.py: OSM bridge lookup, quality gate, and evidence-locked assessment.
  6. (optional) evm.py: Eulerian video magnification centered at dominant frequency.

Guarantees:
  - Returns canonical ScanResult dataclass matching the shared result format.
  - Any failing stage gracefully returns a "failed" result with diagnostic reasons,
    never raising an unhandled exception.
  - Muse Spark assessment is queried ONLY when the verdict is "usable" (enforced
    by interpret.py).
  - Validation: when verdict is "failed", dominant_frequency_hz, frequency_ci_hz,
    and psd are null, and peaks is empty.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid

import cv2
import matplotlib
matplotlib.use("Agg")  # Headless backend
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal

from setu.assess import AnomalyAssessmentResult, BaseAssessor
from setu.evm import EVMDisplayConfig, magnify_video
from setu.interpret import NO_BRIDGE_WARNING, interpret
from setu.meta_client import MetaClient, get_meta_client
from setu.motion import track_motion
from setu.osm import OSMBridgeInfo, OverpassBridgeClient
from setu.quality_gate import QualityGate
from setu.result import (
    AssessmentInfo,
    CostsUSD,
    InterpretationInfo,
    Location,
    DeviceCheckInfo,
    MeasurementInfo,
    MeasurementInstance,
    MeasurementPeak,
    ModeShapePoint,
    OutputsInfo,
    PSDData,
    QualityGateInfo,
    ScanResult,
    SegmentationInfo,
    SonificationInfo,
    TrafficEvent,
    TrafficInfo,
    Verdict,
    VideoInfo,
    WeatherInfo,
)
from setu.segment import compute_persistence, get_segmenter, mask_to_bbox
from setu.spectrum import (
    compute_welch_peaks,
    determine_spectrum_verdict,
    extract_background_residual_motion,
)
from setu.types import SegmentResult
from setu.video_checks import check_video

logger = logging.getLogger(__name__)


def _feature_warning(warnings: List[str], feature: str, exc: object) -> None:
    message = f"Optional {feature} feature unavailable: {exc}"
    logger.warning(message)
    warnings.append(message)


def _boxes_overlap(
    b1: Tuple[int, int, int, int],
    b2: Tuple[int, int, int, int],
    threshold: float = 0.25,
) -> bool:
    """Check if two bounding boxes (x, y, w, h) significantly overlap."""
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    if inter_area <= 0:
        return False
    min_area = min(w1 * h1, w2 * h2)
    return (inter_area / min_area) > threshold if min_area > 0 else False


def _build_failed_result(
    scan_id: str,
    created_at: str,
    reasons: List[str],
    location: Location,
    video_info: VideoInfo,
    target_prompt: str = "concrete pillar",
    background_prompt: str = "building",
    target_source: str = "none",
    background_source: str = "none",
    target_persistence: float = 0.0,
    background_persistence: Optional[float] = None,
    is_demo: bool = False,
    out_dir: Optional[Union[str, Path]] = None,
) -> ScanResult:
    """Construct a schema-compliant failed ScanResult and optionally write to out_dir."""
    failed_reasons = list(reasons) if reasons else ["Pipeline processing failed."]
    result = ScanResult(
        schema_version="1.4",
        scan_id=scan_id,
        is_demo=is_demo,
        created_at=created_at,
        status="complete",
        verdict=Verdict(level="failed", reasons=failed_reasons),
        location=location,
        video=video_info,
        segmentation=SegmentationInfo(
            target_prompt=target_prompt,
            background_prompt=background_prompt,
            target_source=target_source,
            background_source=background_source,
            target_persistence=target_persistence,
            background_persistence=background_persistence,
            warnings=failed_reasons,
        ),
        measurement=MeasurementInfo(
            units="px",
            scale=None,
            dominant_frequency_hz=None,
            frequency_ci_hz=None,
            peaks=[],
            psd=None,
            background_residual_px=0.0,
        ),
        interpretation=InterpretationInfo(
            bridge=None,
            bridge_warning=NO_BRIDGE_WARNING,
            quality_gate=QualityGateInfo(
                is_structural_element=False,
                has_static_background=False,
                appears_stable=False,
                issues=failed_reasons,
            ),
            assessment=AssessmentInfo(
                severity="none",
                reasons=["Data quality insufficient for screening: " + r for r in failed_reasons],
                expected_range_hz=None,
                confidence="low",
                recommended_action="Re-record video with stationary phone bracing and visible static background.",
                assessor="local_rules",
            ),
        ),
        outputs=OutputsInfo(),
        costs_usd=CostsUSD(sam=0.0, muse_spark=0.0, total=0.0),
    )

    if out_dir:
        try:
            os.makedirs(str(out_dir), exist_ok=True)
            out_json_path = os.path.join(str(out_dir), "result.json")
            with open(out_json_path, "w", encoding="utf-8") as f:
                f.write(result.to_json(indent=2))
        except Exception as exc:
            logger.warning("Failed to write result.json to %s: %s", out_dir, exc)

    return result


def analyze_video(
    path: Union[str, Path],
    lat: float,
    lon: float,
    target_prompt: str = "concrete pillar",
    background_prompt: str = "building",
    magnify: bool = False,
    is_demo: bool = False,
    out_dir: Optional[Union[str, Path]] = None,
    meta_client: Optional[MetaClient] = None,
    osm_client: Optional[OverpassBridgeClient] = None,
    assessor: Optional[BaseAssessor] = None,
    quality_gate: Optional[QualityGate] = None,
    stage_timings: Optional[Dict[str, float]] = None,
) -> ScanResult:
    """Execute end-to-end vibration screening pipeline on a video file.

    Stages:
      1. video_checks.py: File integrity, frame decodability, duration [5, 60]s,
         fps [15, 240], and resolution >= 320x240.
      2. segment.py: Structural target and background anchor segmentation on Frame 0.
      3. motion.py: Subpixel displacement tracking with background micro-jitter compensation.
      4. spectrum.py: Welch PSD modal analysis, peak prominence, SNR, CI, PSD downsampling, and verdict.
      5. interpret.py: Infrastructure synthesis (OSM bridge metadata, visual quality gate,
         evidence-locked anomaly screening).
      6. (optional) evm.py: Phase-based motion magnification.

    Args:
        path: Path to input video file.
        lat: Recording location latitude in degrees.
        lon: Recording location longitude in degrees.
        target_prompt: Natural language description of vibrating structural element.
        background_prompt: Natural language description of static reference anchor.
        magnify: Whether to generate Eulerian video magnification artifact (default False).
        is_demo: Boolean indicating whether this is a demo mode execution (default False).
        out_dir: Optional output directory to write result.json and visualization plots.
        meta_client: Optional MetaClient override (e.g. for mock mode in tests).
        osm_client: Optional OverpassBridgeClient override.
        assessor: Optional BaseAssessor override.
        quality_gate: Optional QualityGate override.

    Returns:
        ScanResult object matching SETU shared result format schema.
    """
    now_dt = datetime.now().astimezone()
    scan_id = f"scan_{now_dt.strftime('%Y%m%d')}_{uuid.uuid4().hex[:6]}"
    created_at = now_dt.isoformat()
    location = Location(lat=float(lat), lon=float(lon))
    video_path_str = str(path)

    # -------------------------------------------------------------------------
    # STAGE 1: Video Integrity & Technical Requirements
    # -------------------------------------------------------------------------
    t_stage1_start = time.perf_counter()
    v_res = check_video(video_path_str)
    t_video_checks = time.perf_counter() - t_stage1_start
    if not v_res.passed:
        logger.warning("Stage 1 Video Checks failed: %s", v_res.reasons)
        v_info = v_res.video_info or VideoInfo(filename=os.path.basename(video_path_str))
        return _build_failed_result(
            scan_id=scan_id,
            created_at=created_at,
            reasons=v_res.reasons,
            location=location,
            video_info=v_info,
            target_prompt=target_prompt,
            background_prompt=background_prompt,
            is_demo=is_demo,
            out_dir=out_dir,
        )

    v_info = v_res.video_info or VideoInfo(filename=os.path.basename(video_path_str))
    frame0 = v_res.frame0
    fps = v_info.fps
    width = v_info.width
    height = v_info.height

    if frame0 is None:
        return _build_failed_result(
            scan_id=scan_id,
            created_at=created_at,
            reasons=["Failed to decode reference Frame 0 from video."],
            location=location,
            video_info=v_info,
            target_prompt=target_prompt,
            background_prompt=background_prompt,
            is_demo=is_demo,
            out_dir=out_dir,
        )

    # -------------------------------------------------------------------------
    # STAGE 2: Structural Segmentation & Track Persistence
    # -------------------------------------------------------------------------
    t_stage2_start = time.perf_counter()
    try:
        segmenter = get_segmenter(meta_client=meta_client)
        seg_output = segmenter.segment(
            frame0,
            target_prompt=target_prompt,
            background_prompt=background_prompt,
        )
    except Exception as seg_exc:
        logger.warning("Stage 2 Segmentation failed with exception: %s", seg_exc)
        return _build_failed_result(
            scan_id=scan_id,
            created_at=created_at,
            reasons=[f"Structural segmentation stage failed: {seg_exc}"],
            location=location,
            video_info=v_info,
            target_prompt=target_prompt,
            background_prompt=background_prompt,
            is_demo=is_demo,
            out_dir=out_dir,
        )

    frame0_target_mask = np.asarray(getattr(seg_output, "target_mask", np.zeros((height, width), dtype=bool)), dtype=bool)
    if frame0_target_mask.shape == (height, width) and np.any(frame0_target_mask):
        target_box = mask_to_bbox(frame0_target_mask)
    else:
        target_box = seg_output.target_bbox
    if not target_box or target_box[2] <= 0 or target_box[3] <= 0:
        return _build_failed_result(
            scan_id=scan_id,
            created_at=created_at,
            reasons=["Target segmentation failed: unable to isolate structural element in frame 0."],
            location=location,
            video_info=v_info,
            target_prompt=target_prompt,
            background_prompt=background_prompt,
            target_source=seg_output.target_source,
            background_source=seg_output.background_source,
            is_demo=is_demo,
            out_dir=out_dir,
        )

    # Disambiguate background bounding box if colliding with target (e.g. mock SAM repeating box)
    bg_box = seg_output.background_bbox
    if bg_box is not None and bg_box[2] > 0 and bg_box[3] > 0:
        if _boxes_overlap(target_box, bg_box):
            # Target and background cannot occupy the same structural patch
            logger.info("Background bbox overlaps target bbox; shifting background to stationary reference zone.")
            bg_box = (
                int(width * 0.05),
                int(height * 0.75),
                max(20, int(width * 0.20)),
                max(20, int(height * 0.15)),
            )

    # Evaluate track persistence across sampled frames
    target_pers_val = 1.0
    bg_pers_val = 1.0
    try:
        cap = cv2.VideoCapture(video_path_str)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        n_samples = min(60, max(10, total_frames))
        step = max(1, total_frames // n_samples)
        sampled_frames: List[np.ndarray] = []
        for i in range(total_frames):
            if i % step == 0 and len(sampled_frames) < n_samples:
                ret_s, f_s = cap.read()
                if ret_s and f_s is not None:
                    sampled_frames.append(f_s)
                else:
                    break
            else:
                if not cap.grab():
                    break
        cap.release()

        if sampled_frames:
            t_pers = compute_persistence(
                frames=sampled_frames,
                bbox=target_box,
                fps=fps,
                correlation_threshold=0.70,
                min_persistence_ratio=0.85,
                sample_stride=1,
            )
            target_pers_val = round(float(t_pers.get("fraction_valid", 1.0)), 2)
            if not t_pers.get("passed", True):
                pers_fail_msg = (
                    f"Target persistence failure: fraction of valid frames ({target_pers_val:.1%}) "
                    f"below threshold {t_pers.get('min_persistence_ratio', 0.85):.1%}. Target occluded or lost."
                )
                return _build_failed_result(
                    scan_id=scan_id,
                    created_at=created_at,
                    reasons=[pers_fail_msg],
                    location=location,
                    video_info=v_info,
                    target_prompt=target_prompt,
                    background_prompt=background_prompt,
                    target_source=seg_output.target_source,
                    background_source=seg_output.background_source,
                    target_persistence=target_pers_val,
                    background_persistence=bg_pers_val,
                    is_demo=is_demo,
                    out_dir=out_dir,
                )

            if bg_box is not None and bg_box[2] > 0 and bg_box[3] > 0:
                b_pers = compute_persistence(
                    frames=sampled_frames,
                    bbox=bg_box,
                    fps=fps,
                    correlation_threshold=0.70,
                    min_persistence_ratio=0.85,
                    sample_stride=1,
                )
                bg_pers_val = round(float(b_pers.get("fraction_valid", 1.0)), 2)
            else:
                bg_pers_val = None
    except Exception as pers_exc:
        logger.debug("Persistence sampling exception: %s", pers_exc)
        target_pers_val = 1.0
        bg_pers_val = 1.0 if bg_box else None
    t_segmentation = time.perf_counter() - t_stage2_start

    # -------------------------------------------------------------------------
    # STAGE 3: Subpixel Motion Tracking & Micro-Jitter Compensation
    # -------------------------------------------------------------------------
    t_stage3_start = time.perf_counter()
    target_mask = frame0_target_mask.copy() if frame0_target_mask.shape == (height, width) and np.any(frame0_target_mask) else np.zeros((height, width), dtype=bool)
    if not np.any(target_mask):
        tx, ty, tw, th = target_box
        target_mask[ty : ty + th, tx : tx + tw] = True

    has_background = bg_box is not None and bg_box[2] > 0 and bg_box[3] > 0
    bg_masks: List[np.ndarray] = []
    if has_background and bg_box is not None:
        bg_mask = np.zeros((height, width), dtype=bool)
        bx, by, bw, bh = bg_box
        bg_mask[by : by + bh, bx : bx + bw] = True
        bg_masks.append(bg_mask)

    seg_result = SegmentResult(
        target_masks=[target_mask],
        background_masks=bg_masks,
        fps=fps,
        method=seg_output.method,
        metadata=getattr(seg_output, "metadata", {}),
    )

    try:
        motion_result = track_motion(
            video_path=video_path_str,
            segment_result=seg_result,
            target_bbox=target_box,
            background_bbox=bg_box if has_background else None,
            method="phase_correlation",
        )
    except Exception as mot_exc:
        logger.warning("Stage 3 Motion tracking failed: %s", mot_exc)
        return _build_failed_result(
            scan_id=scan_id,
            created_at=created_at,
            reasons=[f"Subpixel motion tracking failed: {mot_exc}"],
            location=location,
            video_info=v_info,
            target_prompt=target_prompt,
            background_prompt=background_prompt,
            target_source=seg_output.target_source,
            background_source=seg_output.background_source,
            target_persistence=target_pers_val,
            background_persistence=bg_pers_val,
            is_demo=is_demo,
            out_dir=out_dir,
        )

    if motion_result.verdict == "failed":
        return _build_failed_result(
            scan_id=scan_id,
            created_at=created_at,
            reasons=motion_result.reasons or ["Subpixel motion tracking failed."],
            location=location,
            video_info=v_info,
            target_prompt=target_prompt,
            background_prompt=background_prompt,
            target_source=seg_output.target_source,
            background_source=seg_output.background_source,
            target_persistence=target_pers_val,
            background_persistence=bg_pers_val,
            is_demo=is_demo,
            out_dir=out_dir,
        )

    disp = motion_result.displacement
    qual = motion_result.quality
    has_bg_verified = qual.get("has_background", True) and has_background

    # Subpixel background residual is already computed during motion tracking (qual["background_residual_px"])
    # Omitting the redundant second full-video optical flow decode pass saves ~40s on 30s 1080p input.
    bg_disp = getattr(motion_result, "background_displacement", None)
    t_motion = time.perf_counter() - t_stage3_start

    # -------------------------------------------------------------------------
    # STAGE 4: Power Spectral Density & Dominant Resonant Frequency
    # -------------------------------------------------------------------------
    t_stage4_start = time.perf_counter()
    try:
        peaks = compute_welch_peaks(
            disp,
            fps=fps,
            min_freq_hz=0.5,
            top_k=3,
            background_displacement=bg_disp,
            has_background=has_bg_verified,
        )
    except Exception as spec_exc:
        logger.warning("Stage 4 Spectrum analysis failed: %s", spec_exc)
        return _build_failed_result(
            scan_id=scan_id,
            created_at=created_at,
            reasons=[f"Spectral peak extraction failed: {spec_exc}"],
            location=location,
            video_info=v_info,
            target_prompt=target_prompt,
            background_prompt=background_prompt,
            target_source=seg_output.target_source,
            background_source=seg_output.background_source,
            target_persistence=target_pers_val,
            background_persistence=bg_pers_val,
            is_demo=is_demo,
            out_dir=out_dir,
        )

    is_flat_sig = bool(np.all(np.abs(disp) < 1e-10))
    is_non_finite_sig = bool(not np.all(np.isfinite(disp)))
    clipping_flag = bool(qual.get("clipping_detected", False))

    verdict_level, verdict_reasons = determine_spectrum_verdict(
        peaks=peaks,
        has_background=has_bg_verified,
        is_flat=is_flat_sig,
        is_non_finite=is_non_finite_sig,
        tracking_failed=bool(motion_result.verdict == "failed"),
        clipping_detected=clipping_flag,
    )

    if motion_result.verdict == "weak":
        if verdict_level == "usable":
            verdict_level = "weak"
        for mr in motion_result.reasons:
            if mr not in verdict_reasons:
                verdict_reasons.append(mr)

    # Clean reasons for usable tier
    if verdict_level == "usable":
        verdict_reasons = []

    if verdict_level == "failed":
        return _build_failed_result(
            scan_id=scan_id,
            created_at=created_at,
            reasons=verdict_reasons or ["No reliable spectral modal peaks detected above 0.5 Hz."],
            location=location,
            video_info=v_info,
            target_prompt=target_prompt,
            background_prompt=background_prompt,
            target_source=seg_output.target_source,
            background_source=seg_output.background_source,
            target_persistence=target_pers_val,
            background_persistence=bg_pers_val,
            is_demo=is_demo,
            out_dir=out_dir,
        )

    # Compute Welch PSD for measurement.psd (downsampled to <= 300 points between 0 and 30 Hz)
    sig_detrend = signal.detrend(disp, type="linear")
    nperseg = min(len(sig_detrend), 1024)
    nfft = max(nperseg, 4096)
    freqs_raw, psd_raw = signal.welch(sig_detrend, fs=fps, window="hann", nperseg=nperseg, nfft=nfft)

    mask_30 = (freqs_raw >= 0.0) & (freqs_raw <= 30.0)
    f_30 = freqs_raw[mask_30]
    p_30 = psd_raw[mask_30]

    # Downsample to at most 300 points
    if len(f_30) > 300:
        step = int(np.ceil(len(f_30) / 300))
        f_30 = f_30[::step]
        p_30 = p_30[::step]

    psd_obj = PSDData(
        freq_hz=[round(float(f_val), 3) for f_val in f_30],
        power=[round(float(p_val), 6) for p_val in p_30],
    )

    # Populate measurement telemetry for usable or weak
    p0 = peaks[0] if peaks else None
    dom_freq = round(float(p0.freq_hz), 2) if p0 is not None else None
    freq_ci: Optional[List[float]] = None
    if p0 is not None:
        if getattr(p0, "freq_ci", None) is not None:
            freq_ci = [round(float(p0.freq_ci[0]), 2), round(float(p0.freq_ci[1]), 2)]
        elif hasattr(p0, "freq_ci_lower") and p0.freq_ci_lower is not None:
            freq_ci = [round(float(p0.freq_ci_lower), 2), round(float(p0.freq_ci_upper), 2)]

    measurement_peaks = [
        MeasurementPeak(
            frequency_hz=round(float(p.freq_hz), 2),
            relative_power=round(float(p.rel_power), 2),
            snr_db=round(float(p.snr), 1),
        )
        for p in peaks
    ]
    bg_residual_val = round(float(qual.get("background_residual_px", 0.0)), 4)

    measurement = MeasurementInfo(
        units="px",
        scale=None,
        dominant_frequency_hz=dom_freq,
        frequency_ci_hz=freq_ci,
        peaks=measurement_peaks,
        psd=psd_obj,
        background_residual_px=bg_residual_val,
    )
    t_spectrum = time.perf_counter() - t_stage4_start

    # -------------------------------------------------------------------------
    # STAGE 5: Synthesis & Interpretation (OSM, Quality Gate, Assessment)
    # -------------------------------------------------------------------------
    t_stage5_start = time.perf_counter()
    spectrum_input = {
        "peaks": peaks,
        "verdict": verdict_level,
        "reasons": verdict_reasons,
    }

    try:
        interp_res = interpret(
            frame0=frame0,
            gps=(lat, lon),
            spectrum=spectrum_input,
            quality_report=qual,
            segmentation_provenance=getattr(seg_output, "metadata", {}),
            client=meta_client,
            assessor=assessor,
            quality_gate=quality_gate,
            osm_client=osm_client,
        )
    except Exception as interp_exc:
        logger.warning("Stage 5 Interpretation exception: %s", interp_exc)
        interp_res = {
            "osm": NO_BRIDGE_WARNING,
            "quality_gate": {
                "is_structural_element": True,
                "has_static_background": has_bg_verified,
                "appears_stable": True,
                "issues": [],
            },
            "assessment": AnomalyAssessmentResult(
                severity="none",
                reasons=[f"Interpretation warning: {interp_exc}"],
                expected_range_hz=None,
                confidence="low",
                recommended_action="Review vibration telemetry manually.",
                assessor="local_rules",
            ),
        }

    # Format OSM bridge metadata
    raw_osm = interp_res.get("osm")
    bridge_dict: Optional[Dict[str, Any]] = None
    bridge_warning: Optional[str] = None
    if isinstance(raw_osm, str):
        bridge_warning = raw_osm
    elif isinstance(raw_osm, OSMBridgeInfo):
        bridge_dict = {
            "name": raw_osm.name,
            "bridge_type": raw_osm.bridge_type,
            "material": raw_osm.material,
            "distance_meters": raw_osm.distance_meters,
            "span_count": raw_osm.span_count,
            "osm_id": raw_osm.osm_id,
        }
    elif isinstance(raw_osm, dict):
        bridge_dict = raw_osm
    else:
        bridge_warning = NO_BRIDGE_WARNING

    # Format Quality Gate
    raw_qg = interp_res.get("quality_gate", {})
    quality_gate_info = QualityGateInfo(
        is_structural_element=bool(raw_qg.get("is_structural_element", True)),
        has_static_background=bool(raw_qg.get("has_static_background", True)),
        appears_stable=bool(raw_qg.get("appears_stable", True)),
        issues=list(raw_qg.get("issues", [])),
    )

    # Format Assessment
    raw_ass = interp_res.get("assessment")
    if isinstance(raw_ass, AnomalyAssessmentResult):
        exp_range = (
            [round(float(x), 2) for x in raw_ass.expected_range_hz]
            if raw_ass.expected_range_hz
            else None
        )
        assessment_info = AssessmentInfo(
            severity=str(raw_ass.severity),
            reasons=list(raw_ass.reasons),
            expected_range_hz=exp_range,
            confidence=str(raw_ass.confidence),
            recommended_action=str(raw_ass.recommended_action),
            assessor=str(raw_ass.assessor),
        )
    elif isinstance(raw_ass, dict):
        exp_range = raw_ass.get("expected_range_hz")
        assessment_info = AssessmentInfo(
            severity=str(raw_ass.get("severity", "none")),
            reasons=list(raw_ass.get("reasons", [])),
            expected_range_hz=[round(float(x), 2) for x in exp_range] if exp_range else None,
            confidence=str(raw_ass.get("confidence", "low")),
            recommended_action=str(raw_ass.get("recommended_action", "")),
            assessor=str(raw_ass.get("assessor", "local_rules")),
        )
    else:
        assessment_info = AssessmentInfo()

    interpretation = InterpretationInfo(
        bridge=bridge_dict,
        bridge_warning=bridge_warning,
        quality_gate=quality_gate_info,
        assessment=assessment_info,
    )
    t_interpretation = time.perf_counter() - t_stage5_start

    # -------------------------------------------------------------------------
    # STAGE 6: Optional Eulerian Video Magnification (EVM)
    # -------------------------------------------------------------------------
    t_stage6_start = time.perf_counter()
    magnified_file: Optional[str] = None
    if magnify and out_dir and dom_freq is not None and verdict_level != "failed":
        try:
            os.makedirs(str(out_dir), exist_ok=True)
            mag_dest = os.path.join(str(out_dir), "magnified.mp4")
            window_frames = max(1, int(round(fps * 10.0)))
            disp_arr = np.asarray(disp, dtype=np.float64)
            if len(disp_arr) > window_frames:
                sums = np.concatenate(([0.0], np.cumsum(disp_arr * disp_arr)))
                window_start = int(np.argmax(sums[window_frames:] - sums[:-window_frames]))
            else:
                window_start = 0
            window_start_sec = window_start / fps
            magnify_video(
                input_video_path=video_path_str,
                output_video_path=mag_dest,
                target_frequency_hz=dom_freq,
                target_bbox=target_box,
                amplification_factor=50.0,
                config=EVMDisplayConfig(max_crop_dimension=480),
                start_time_sec=window_start_sec,
            )
            magnified_file = "magnified.mp4"
        except Exception as evm_exc:
            logger.warning("Stage 6 EVM magnification skipped (%s)", evm_exc)
            magnified_file = None
    t_magnification = time.perf_counter() - t_stage6_start

    # -------------------------------------------------------------------------
    # Optional schema 1.4 features. Imports stay local because these modules
    # can be absent while their implementation or platform dependencies load.
    # Each feature owns its errors and cannot change the scan verdict.
    # -------------------------------------------------------------------------
    feature_warnings: List[str] = []
    vibration_map_file: Optional[str] = None
    sound_file: Optional[str] = None
    sonification_info: Optional[SonificationInfo] = None
    weather_info: Optional[WeatherInfo] = None
    device_check_info = DeviceCheckInfo()
    traffic_info = TrafficInfo()

    if out_dir:
        try:
            os.makedirs(str(out_dir), exist_ok=True)
            np.save(os.path.join(str(out_dir), "displacement.npy"), np.asarray(disp, dtype=np.float64))
        except Exception as exc:
            _feature_warning(feature_warnings, "displacement trace export", exc)

    metadata = getattr(seg_output, "metadata", {}) or {}
    target_instances = metadata.get("all_target_instances", []) if isinstance(metadata, dict) else []
    if seg_output.target_source == "sam" and isinstance(target_instances, list) and len(target_instances) >= 2:
        try:
            from setu import vibration_map as vibration_map_module

            if not out_dir:
                raise ValueError("scan output directory is required for the map artifact")
            if not has_background:
                raise ValueError("stationary background mask is unavailable")
            map_inputs = []
            for index, instance in enumerate(target_instances):
                if not isinstance(instance, dict) or instance.get("mask") is None:
                    continue
                label = str(instance.get("label") or instance.get("text") or f"instance {index + 1}")
                map_inputs.append((label, instance["mask"]))
            if len(map_inputs) >= 2:
                vibration_map_file = "vibration_map.png"
                map_path = os.path.join(str(out_dir), vibration_map_file)
                map_records = vibration_map_module.build_vibration_map(
                    video_path_str, map_inputs, bg_mask, map_path,
                    shared_frequency_hz=dom_freq,
                )
                measurement.instances = [
                    MeasurementInstance(
                        label=str(item["label"]),
                        bbox=[float(v) for v in item["bbox"]],
                        frequency_hz=(float(item["frequency_hz"]) if item.get("frequency_hz") is not None else None),
                        amplitude_px=(float(item["amplitude_px"]) if item.get("amplitude_px") is not None else None),
                        verdict=str(item.get("verdict", "failed")),
                    )
                    for item in map_records
                ]
                mode_shape = vibration_map_module.mode_shape_from_instances(
                    map_records, dominant_frequency_hz=dom_freq,
                )
                measurement.mode_shape = [ModeShapePoint(**point) for point in mode_shape]
            else:
                raise ValueError("SAM target instances did not include two usable masks")
        except Exception as exc:
            vibration_map_file = None
            _feature_warning(feature_warnings, "vibration map", exc)

    if verdict_level == "usable":
        try:
            if not out_dir:
                raise ValueError("scan output directory is required for the sound artifact")
            from setu.sonify import make_sonification

            sound_file = "sound.wav"
            speedup, pitch_hz = make_sonification(
                disp, fps, os.path.join(str(out_dir), sound_file),
            )
            sonification_info = SonificationInfo(speedup=float(speedup), pitch_hz=float(pitch_hz))
        except Exception as exc:
            sound_file = None
            sonification_info = None
            _feature_warning(feature_warnings, "sonification", exc)

    try:
        from setu.weather import get_weather

        raw_weather = get_weather(lat, lon, created_at) or {}
        wind_kmh = raw_weather.get("wind_speed_kmh")
        weather_info = WeatherInfo(
            temperature_c=raw_weather.get("temperature_c"),
            humidity_pct=raw_weather.get("humidity_pct"),
            wind_speed_ms=(float(wind_kmh) / 3.6 if wind_kmh is not None else None),
            source="open-meteo" if any(
                raw_weather.get(key) is not None
                for key in ("temperature_c", "humidity_pct", "wind_speed_kmh")
            ) else None,
        )
    except Exception as exc:
        weather_info = WeatherInfo()
        _feature_warning(feature_warnings, "weather", exc)

    if out_dir:
        device_path = Path(out_dir) / "device_motion.json"
        if device_path.is_file():
            try:
                from setu.device_motion import analyze_device_motion

                device_payload = json.loads(device_path.read_text(encoding="utf-8"))
                samples = device_payload.get("samples", device_payload) if isinstance(device_payload, dict) else device_payload
                device_result = analyze_device_motion(
                    samples,
                    camera_frequency_hz=dom_freq,
                    camera_ci_hz=freq_ci,
                )
                comparison = device_result.get("comparison") or {}
                if not device_result.get("resting_on_structure", True):
                    device_status = "not_resting"
                elif comparison.get("agreement") is True:
                    device_status = "agrees"
                elif comparison.get("agreement") is False:
                    device_status = "disagrees"
                else:
                    device_status = "not_recorded"
                device_check_info = DeviceCheckInfo(
                    status=device_status,
                    device_frequency_hz=device_result.get("dominant_frequency_hz"),
                    device_ci_hz=device_result.get("confidence_interval_hz"),
                    sample_rate_hz=device_result.get("sampling_rate_hz"),
                    note=device_result.get("message"),
                )
            except Exception as exc:
                device_check_info = DeviceCheckInfo(note=str(exc))
                _feature_warning(feature_warnings, "device check", exc)

    try:
        from setu.traffic import track_traffic

        traffic_result = track_traffic(video_path_str, meta_client=meta_client)
        if isinstance(traffic_result, dict):
            raw_events = traffic_result.get("events", [])
            response_plot = traffic_result.get("response_plot")
        else:
            raw_events = getattr(traffic_result, "events", [])
            response_plot = getattr(traffic_result, "response_plot", None)
        if isinstance(traffic_result, (dict, TrafficInfo)) or hasattr(traffic_result, "events"):
            traffic_info = TrafficInfo(
                events=[
                    item if isinstance(item, TrafficEvent) else TrafficEvent(
                        t_start=float(item["t_start"]),
                        t_end=float(item["t_end"]),
                        type=item["type"],
                    )
                    for item in raw_events
                ],
                response_plot=response_plot,
            )
    except ImportError as exc:
        _feature_warning(feature_warnings, "traffic", exc)
    except Exception as exc:
        traffic_info = TrafficInfo()
        _feature_warning(feature_warnings, "traffic", exc)

    # -------------------------------------------------------------------------
    # STAGE 7: Plot Generation (if out_dir provided)
    # -------------------------------------------------------------------------
    disp_plot_name: Optional[str] = None
    spec_plot_name: Optional[str] = None

    if out_dir:
        try:
            os.makedirs(str(out_dir), exist_ok=True)
            # A. Displacement Plot
            disp_dest = os.path.join(str(out_dir), "displacement.png")
            fig_d, ax_d = plt.subplots(figsize=(10, 4), dpi=150)
            t_vec = np.arange(len(disp)) / fps
            ax_d.plot(t_vec, disp, color="#0284c7", linewidth=1.2, label="Subpixel Displacement (px)")
            ax_d.set_title(f"SETU Subpixel Structural Displacement ({v_info.filename})", fontsize=11, fontweight="bold")
            ax_d.set_xlabel("Time (s)", fontsize=10)
            ax_d.set_ylabel("Displacement (px)", fontsize=10)
            ax_d.grid(True, linestyle="--", alpha=0.5)
            ax_d.legend(loc="upper right")
            plt.tight_layout()
            fig_d.savefig(disp_dest)
            plt.close(fig_d)
            disp_plot_name = "displacement.png"

            # B. Spectrum Plot
            spec_dest = os.path.join(str(out_dir), "spectrum.png")
            f_plot_mask = (freqs_raw >= 0.5) & (freqs_raw <= 20.0)
            fig_s, ax_s = plt.subplots(figsize=(10, 4), dpi=150)
            ax_s.plot(freqs_raw[f_plot_mask], psd_raw[f_plot_mask], color="#6366f1", linewidth=1.5, label="Welch PSD")
            plot_colors = ["#dc2626", "#ea580c", "#16a34a"]
            for idx_p, p_item in enumerate(peaks):
                col = plot_colors[idx_p % len(plot_colors)]
                ax_s.axvline(p_item.freq_hz, color=col, linestyle=":", alpha=0.7)
                ax_s.plot([p_item.freq_hz], [np.interp(p_item.freq_hz, freqs_raw[f_plot_mask], psd_raw[f_plot_mask])], "o", color=col)
                ax_s.annotate(
                    f"f{idx_p+1}={p_item.freq_hz:.2f} Hz\n({p_item.rel_power*100:.1f}%)",
                    xy=(p_item.freq_hz, np.interp(p_item.freq_hz, freqs_raw[f_plot_mask], psd_raw[f_plot_mask])),
                    xytext=(p_item.freq_hz + 0.3, np.interp(p_item.freq_hz, freqs_raw[f_plot_mask], psd_raw[f_plot_mask]) * 0.9 + 1e-8),
                    fontsize=8,
                    fontweight="bold",
                    color=col,
                    arrowprops=dict(arrowstyle="->", color=col, lw=1),
                )
            ax_s.set_title(f"SETU Vibration Power Spectral Density ({v_info.filename})", fontsize=11, fontweight="bold")
            ax_s.set_xlabel("Frequency (Hz)", fontsize=10)
            ax_s.set_ylabel("Power Spectral Density (px²/Hz)", fontsize=10)
            ax_s.grid(True, linestyle="--", alpha=0.5)
            ax_s.legend(loc="upper right")
            plt.tight_layout()
            fig_s.savefig(spec_dest)
            plt.close(fig_s)
            spec_plot_name = "spectrum.png"
        except Exception as plot_err:
            logger.warning("Stage 7 Plot generation note: %s", plot_err)

    # -------------------------------------------------------------------------
    # STAGE 8: Inference Costs & Result Construction
    # -------------------------------------------------------------------------
    sam_usd = 0.0025 if seg_output.target_source == "sam" else 0.0
    muse_usd = 0.0013 if assessment_info.assessor == "muse_spark" else 0.0
    total_usd = round(sam_usd + muse_usd, 4)

    costs = CostsUSD(sam=sam_usd, muse_spark=muse_usd, total=total_usd)
    outputs = OutputsInfo(
        magnified_video=magnified_file,
        spectrum_plot=spec_plot_name,
        displacement_plot=disp_plot_name,
        vibration_map=vibration_map_file,
        sound=sound_file,
    )
    segmentation = SegmentationInfo(
        target_prompt=target_prompt,
        background_prompt=background_prompt,
        target_source=seg_output.target_source,
        background_source=seg_output.background_source,
        target_persistence=target_pers_val,
        background_persistence=bg_pers_val,
        warnings=list(seg_output.warnings) + feature_warnings,
        target_bbox=[float(value) for value in target_box],
    )

    final_result = ScanResult(
        schema_version="1.4",
        scan_id=scan_id,
        is_demo=is_demo,
        created_at=created_at,
        status="complete",
        verdict=Verdict(level=verdict_level, reasons=verdict_reasons),
        location=location,
        video=v_info,
        segmentation=segmentation,
        measurement=measurement,
        interpretation=interpretation,
        outputs=outputs,
        costs_usd=costs,
        sonification=sonification_info,
        weather=weather_info,
        device_check=device_check_info,
        traffic=traffic_info,
    )

    if out_dir:
        try:
            os.makedirs(str(out_dir), exist_ok=True)
            out_json_path = os.path.join(str(out_dir), "result.json")
            with open(out_json_path, "w", encoding="utf-8") as f:
                f.write(final_result.to_json(indent=2))
        except Exception as f_err:
            logger.warning("Failed writing final result.json to %s: %s", out_dir, f_err)

    measured_timings = {
        "video checks": t_video_checks,
        "segmentation": t_segmentation,
        "motion": t_motion,
        "spectrum": t_spectrum,
        "interpretation": t_interpretation,
        "magnification": t_magnification,
    }
    if stage_timings is not None:
        stage_timings.update(measured_timings)
    final_result._stage_timings = measured_timings

    return final_result
