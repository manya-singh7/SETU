#!/usr/bin/env python3
"""
Noise Floor Calibration Script for Project SETU.

Evaluates subpixel displacement tracking on a real (or benchmark) video recording
of a completely still scene to measure the empirical physical tracking noise floor.
Compares the measured displacement RMS against the synthetic noise floor constant in
setu/motion.py (TRACKING_NOISE_FLOOR_RMS_PX = 0.008 px) and prints suggested thresholds
for production tuning without modifying setu/motion.py.

Usage:
    python scripts/calibrate_noise_floor.py --video path/to/still_scene.mp4
    python scripts/calibrate_noise_floor.py --video path/to/still_scene.mp4 --method both
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure SETU_META_MODE defaults to 'cached' per pipeline specifications
os.environ.setdefault("SETU_META_MODE", "cached")

# Add repository root to Python path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np

from setu.motion import (
    NOISE_FLOOR_FACTOR,
    TRACKING_NOISE_FLOOR_RMS_PX,
    track_motion,
)


def parse_bbox_arg(bbox_str: Optional[str]) -> Optional[Tuple[int, int, int, int]]:
    """Parse comma-separated 'x,y,w,h' string into an integer tuple."""
    if not bbox_str:
        return None
    parts = [int(p.strip()) for p in bbox_str.split(",")]
    if len(parts) != 4:
        raise ValueError(f"Invalid bounding box '{bbox_str}'. Expected 'x,y,w,h'.")
    return (parts[0], parts[1], parts[2], parts[3])


def calibrate_video_noise_floor(
    video_path: Union[str, Path],
    method: str = "phase_correlation",
    axis: str = "auto",
    target_bbox: Optional[Tuple[int, int, int, int]] = None,
    background_bbox: Optional[Tuple[int, int, int, int]] = None,
    factor: float = NOISE_FLOOR_FACTOR,
) -> Dict[str, Any]:
    """Run motion tracking on a still scene video and compute noise metrics.

    Args:
        video_path: Filepath to the video of a completely still scene.
        method: Motion extraction technique ('phase_correlation' or 'lucas_kanade').
        axis: Projection axis ('auto', 'x', 'y').
        target_bbox: Optional target ROI (x, y, w, h).
        background_bbox: Optional stationary background ROI (x, y, w, h).
        factor: Safety factor for detection threshold (default: NOISE_FLOOR_FACTOR = 3.0).

    Returns:
        Dictionary containing measured RMS, std, max displacement, comparison with synthetic
        constant, and suggested detection thresholds.
    """
    v_path = Path(video_path)
    if not v_path.exists():
        raise FileNotFoundError(f"Video file not found: {v_path}")

    cap = cv2.VideoCapture(str(v_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {v_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = frame_count / fps if fps > 0 else 0.0
    cap.release()

    # Auto-segment frame 0 if target or background bounding boxes are omitted
    if target_bbox is None or background_bbox is None:
        try:
            from setu.segment import get_segmenter
            cap = cv2.VideoCapture(str(v_path))
            ret, frame0 = cap.read()
            cap.release()
            if ret and frame0 is not None:
                seg = get_segmenter().segment(frame0)
                if target_bbox is None and seg.target_bbox and seg.target_bbox[2] > 0 and seg.target_bbox[3] > 0:
                    target_bbox = seg.target_bbox
                if background_bbox is None and seg.background_bbox and seg.background_bbox[2] > 0 and seg.background_bbox[3] > 0:
                    background_bbox = seg.background_bbox
        except Exception:
            pass

    # Run motion tracking with current motion.py baseline
    motion_res = track_motion(
        video_path=str(v_path),
        target_bbox=target_bbox,
        background_bbox=background_bbox,
        method=method,  # type: ignore[arg-type]
        axis=axis,  # type: ignore[arg-type]
    )

    disp = motion_res.displacement
    is_finite = len(disp) > 0 and bool(np.all(np.isfinite(disp)))

    if not is_finite:
        raw_rms = float("nan")
        detrended_rms = float("nan")
        disp_std = float("nan")
        ptp_val = float("nan")
        max_abs = float("nan")
    else:
        raw_rms = float(np.sqrt(np.mean(disp**2)))
        detrended_disp = disp - np.mean(disp)
        detrended_rms = float(np.sqrt(np.mean(detrended_disp**2)))
        disp_std = float(np.std(disp))
        ptp_val = float(np.ptp(disp))
        max_abs = float(np.max(np.abs(disp)))

    synthetic_noise_floor = float(TRACKING_NOISE_FLOOR_RMS_PX)
    ratio_to_synthetic = raw_rms / synthetic_noise_floor if is_finite and synthetic_noise_floor > 0 else float("nan")
    suggested_threshold = factor * raw_rms if is_finite else float("nan")
    conservative_threshold = 5.0 * raw_rms if is_finite else float("nan")

    return {
        "video_path": str(v_path),
        "filename": v_path.name,
        "method": method,
        "axis": axis,
        "dominant_axis": motion_res.quality.get("dominant_axis", "x"),
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_sec": round(duration_sec, 2),
        "displacement_raw_rms": round(raw_rms, 6),
        "displacement_detrended_rms": round(detrended_rms, 6),
        "displacement_std": round(disp_std, 6),
        "displacement_ptp": round(ptp_val, 6),
        "displacement_max_abs": round(max_abs, 6),
        "synthetic_noise_floor_rms": synthetic_noise_floor,
        "synthetic_default_threshold": round(synthetic_noise_floor * factor, 6),
        "ratio_to_synthetic": round(ratio_to_synthetic, 2) if np.isfinite(ratio_to_synthetic) else None,
        "suggested_factor": factor,
        "suggested_threshold": round(suggested_threshold, 6),
        "suggested_conservative_threshold": round(conservative_threshold, 6),
        "motion_verdict": motion_res.verdict,
        "motion_reasons": motion_res.reasons,
        "background_residual_px": motion_res.quality.get("background_residual_px", 0.0),
        "camera_motion_rms_px": motion_res.quality.get("camera_motion_rms_px", 0.0),
    }


def print_calibration_report(results: List[Dict[str, Any]]) -> None:
    """Print an engineer-friendly calibration table and recommendation summary."""
    print("\n" + "=" * 90)
    print("SETU TRACKING NOISE FLOOR CALIBRATION REPORT")
    print("=" * 90)

    for res in results:
        print(f"\n--- Video: {res['filename']} ({res['width']}x{res['height']} @ {res['fps']:.1f} fps, {res['duration_sec']}s) ---")
        print(f"Tracking Method:            {res['method']} (dominant axis: {res['dominant_axis']})")
        print(f"Displacement Raw RMS:       {res['displacement_raw_rms']:.6f} px")
        print(f"Displacement Detrended RMS: {res['displacement_detrended_rms']:.6f} px")
        print(f"Displacement Std Dev:       {res['displacement_std']:.6f} px")
        print(f"Displacement Peak-to-Peak:  {res['displacement_ptp']:.6f} px")
        print(f"Displacement Max Absolute:  {res['displacement_max_abs']:.6f} px")
        print(f"Background Residual RMS:    {res['background_residual_px']:.6f} px")
        print(f"Camera Micro-Jitter RMS:    {res['camera_motion_rms_px']:.6f} px")
        print("-" * 90)
        print("COMPARISON WITH SYNTHETIC BASELINE:")
        print(f"  Synthetic constant (TRACKING_NOISE_FLOOR_RMS_PX): {res['synthetic_noise_floor_rms']:.6f} px")
        print(f"  Synthetic default threshold (3.0x factor):         {res['synthetic_default_threshold']:.6f} px")
        if res['ratio_to_synthetic'] is not None:
            print(f"  Measured / Synthetic ratio:                       {res['ratio_to_synthetic']:.2f}x")
        print("-" * 90)
        print("SUGGESTED REAL-WORLD THRESHOLDS (Rose Criterion / 3-sigma LOD):")
        print(f"  Recommended threshold ({res['suggested_factor']:.1f}x measured RMS):      {res['suggested_threshold']:.6f} px")
        print(f"  Conservative threshold (5.0x measured RMS):     {res['suggested_conservative_threshold']:.6f} px")
        print(f"  Current motion.py verdict on this video:        {res['motion_verdict'].upper()}")
        if res['motion_reasons']:
            print(f"  Diagnostic reasons:                             {', '.join(res['motion_reasons'])}")

    print("\n" + "=" * 90)
    print("Note: setu/motion.py was NOT modified. Use the above values to calibrate constants")
    print("on physical field recordings when establishing production thresholds.")
    print("=" * 90 + "\n")


def main() -> int:
    """CLI entrypoint for calibrate_noise_floor.py."""
    parser = argparse.ArgumentParser(
        description="Calibrate motion tracking noise floor on a real video of a completely still scene."
    )
    parser.add_argument(
        "--video",
        "-v",
        required=True,
        type=str,
        help="Path to input MP4/video file of a completely still target/scene.",
    )
    parser.add_argument(
        "--method",
        "-m",
        choices=["phase_correlation", "lucas_kanade", "both"],
        default="phase_correlation",
        help="Motion extraction technique to evaluate (default: phase_correlation).",
    )
    parser.add_argument(
        "--axis",
        "-a",
        choices=["auto", "x", "y"],
        default="auto",
        help="Motion projection axis (default: auto).",
    )
    parser.add_argument(
        "--target-bbox",
        type=str,
        default=None,
        help="Target bounding box as 'x,y,w,h'. If omitted, default center crop is used.",
    )
    parser.add_argument(
        "--background-bbox",
        type=str,
        default=None,
        help="Background bounding box as 'x,y,w,h'. If omitted, stationary reference is estimated.",
    )
    parser.add_argument(
        "--factor",
        type=float,
        default=NOISE_FLOOR_FACTOR,
        help=f"Detection safety factor above noise floor (default: {NOISE_FLOOR_FACTOR}).",
    )

    args = parser.parse_args()

    t_box = parse_bbox_arg(args.target_bbox)
    b_box = parse_bbox_arg(args.background_bbox)

    methods = ["phase_correlation", "lucas_kanade"] if args.method == "both" else [args.method]

    results = []
    for meth in methods:
        res = calibrate_video_noise_floor(
            video_path=args.video,
            method=meth,
            axis=args.axis,
            target_bbox=t_box,
            background_bbox=b_box,
            factor=args.factor,
        )
        results.append(res)

    print_calibration_report(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
