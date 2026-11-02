"""
Project SETU End-to-End Command Line Interface.

Executes the screening pipeline from mobile phone video to spectral modal analysis:
1. Segments frame 0 via Meta SAM 3.1 or fallback (setu/segment.py).
2. Extracts subpixel structural displacement with background jitter cancellation (setu/motion.py).
3. Computes the Welch Power Spectral Density and dominant peaks (setu/spectrum.py).
4. Prints telemetry, segmentation method, and quality diagnostics.
5. Saves displacement and spectrum plots as PNGs next to the input video.

Usage:
    python -m setu.run VIDEO --target "prompt" --background "prompt" [--bbox-target x,y,w,h --bbox-background x,y,w,h]
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless CLI execution
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal

from setu.motion import track_motion
from setu.segment import get_segmenter
from setu.spectrum import RELIABILITY_THRESHOLD_SNR_DB, compute_welch_peaks, is_peak_reliable
from setu.types import MotionResult, SegmentResult


def _parse_bbox(bbox_str: Optional[str]) -> Optional[Tuple[int, int, int, int]]:
    """Parse comma-separated 'x,y,w,h' string into a 4-tuple of integers."""
    if not bbox_str:
        return None
    try:
        parts = [int(p.strip()) for p in bbox_str.split(",")]
        if len(parts) != 4:
            raise ValueError
        return (parts[0], parts[1], parts[2], parts[3])
    except Exception:
        raise argparse.ArgumentTypeError(
            f"Invalid bounding box format '{bbox_str}'. Expected 'x,y,w,h' (e.g. 270,90,100,280)"
        )


def main() -> None:
    """CLI entry point for Project SETU."""
    parser = argparse.ArgumentParser(
        prog="python -m setu.run",
        description="SETU Civil Infrastructure Vibration Telemetry CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("video", type=str, help="Path to input video file (e.g. 30s mobile recording)")
    parser.add_argument(
        "--target",
        "-t",
        type=str,
        default="corner of concrete pillar",
        help="Text prompt for vibrating structural target (default: 'corner of concrete pillar')",
    )
    parser.add_argument(
        "--background",
        "-b",
        type=str,
        default="stationary ground reference",
        help="Text prompt for stationary background anchor (default: 'stationary ground reference')",
    )
    parser.add_argument(
        "--bbox-target",
        type=_parse_bbox,
        default=None,
        help="Optional manual bounding box override for target 'x,y,w,h'",
    )
    parser.add_argument(
        "--bbox-background",
        type=_parse_bbox,
        default=None,
        help="Optional manual bounding box override for background 'x,y,w,h'",
    )
    parser.add_argument(
        "--method",
        "-m",
        choices=["phase_correlation", "lucas_kanade"],
        default="phase_correlation",
        help="Motion tracking technique (default: phase_correlation)",
    )

    args = parser.parse_args()

    video_path = os.path.abspath(args.video)
    if not os.path.isfile(video_path):
        print(f"Error: Video file not found at '{video_path}'", file=sys.stderr)
        sys.exit(1)

    print("\n" + "=" * 78)
    print("SETU: STRUCTURAL EVALUATION & TELEMETRY UTILITY - PIPELINE EXECUTION")
    print("=" * 78)
    print(f"Video Source: {video_path}")

    # 1. Read video metadata and frame 0
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video stream at '{video_path}'", file=sys.stderr)
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0.0 or np.isnan(fps):
        fps = 60.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_s = total_frames / fps if fps > 0 else 0.0

    ret, frame0 = cap.read()
    cap.release()

    if not ret or frame0 is None:
        print(f"Error: Failed to read frame 0 from video '{video_path}'", file=sys.stderr)
        sys.exit(1)

    print(f"Resolution:   {width}x{height} | {total_frames} frames @ {fps:.1f} fps ({duration_s:.1f}s)")

    # 2. Segment frame 0 via setu/segment.py
    print(f"\n[Step 1] Segmenting Frame 0...")
    segmenter = get_segmenter()
    seg_output = segmenter.segment(
        frame0,
        target_prompt=args.target,
        background_prompt=args.background,
    )

    seg_method = "manual" if (args.bbox_target or args.bbox_background) else seg_output.method
    target_box = args.bbox_target or seg_output.target_bbox
    bg_box = args.bbox_background or seg_output.background_bbox

    target_mask = np.zeros((height, width), dtype=bool)
    bg_mask = np.zeros((height, width), dtype=bool)
    tx, ty, tw, th = target_box
    bx, by, bw, bh = bg_box
    target_mask[ty : ty + th, tx : tx + tw] = True
    bg_mask[by : by + bh, bx : bx + bw] = True

    seg_result = SegmentResult(
        target_masks=[target_mask],
        background_masks=[bg_mask],
        fps=fps,
        method=seg_method,
    )

    print(f"  Segmentation Method: {seg_result.method}")
    print(f"  Target Prompt:       '{args.target}' -> BBox: {target_box}")
    print(f"  Background Prompt:   '{args.background}' -> BBox: {bg_box}")

    # 3. Extract subpixel motion via setu/motion.py
    print(f"\n[Step 2] Tracking Subpixel Motion & Compensating Micro-Jitter...")
    print(f"  Tracking Method:     {args.method}")
    motion_result: MotionResult = track_motion(
        video_path=video_path,
        segment_result=seg_result,
        target_bbox=target_box,
        background_bbox=bg_box,
        method=args.method,
    )

    disp = motion_result.displacement
    qual = motion_result.quality

    # 4. Compute Welch Power Spectral Density
    print(f"\n[Step 3] Computing Spectral Modal Frequency Peaks...")
    peaks = compute_welch_peaks(disp, fps=fps, min_freq_hz=0.5, top_k=3)

    is_reliable = bool(peaks and is_peak_reliable(peaks[0]))
    confidence_status = "HIGH CONFIDENCE" if is_reliable else "LOW CONFIDENCE"

    qual_warnings = list(qual.get("warnings", []))
    if not peaks:
        qual_warnings.append("Low confidence: No prominent modal frequency peaks detected above 0.5 Hz.")
    elif not is_reliable:
        top_snr = peaks[0].snr
        qual_warnings.append(
            f"Low confidence: Top peak SNR ({top_snr:.1f} dB) below {RELIABILITY_THRESHOLD_SNR_DB:.1f} dB reliability threshold."
        )
    qual["warnings"] = qual_warnings
    qual["confidence"] = confidence_status.lower()

    # Step 4: Quality Diagnostics Report
    print(f"\n[Step 4] Quality Diagnostics Report:")
    print(f"  Overall Confidence:       {confidence_status}")
    print(f"  Background Residual RMS:  {qual.get('background_residual_px', 0.0):.4f} px")
    print(f"  Displacement Signal SNR:  {qual.get('snr', 0.0):.1f}")
    print(f"  Camera Motion (RMS):      {qual.get('camera_motion_rms_px', 0.0):.4f} px")
    print(f"  Dominant Motion Axis:     {qual.get('dominant_axis', 'x').upper()}")

    warnings = qual.get("warnings", [])
    if warnings:
        print("  Warnings:")
        for w in warnings:
            print(f"    ! {w}")
    else:
        print("  Warnings:                 None (Clean Tracking)")

    # Spectral Peak Telemetry
    print("\n" + "-" * 75)
    print(f"{'Rank':<6} | {'Frequency':<14} | {'Relative Power':<18} | {'Peak SNR':<12} | {'Confidence'}")
    print("-" * 75)
    for idx, p in enumerate(peaks, 1):
        p_conf = "High" if is_peak_reliable(p) else "Low Confidence"
        print(f"{idx:<6} | {p.freq_hz:>6.3f} Hz       | {p.rel_power * 100:>8.2f} %         | {p.snr:>6.1f} dB    | {p_conf}")
    print("-" * 75)

    if peaks:
        conf_suffix = "" if is_reliable else " [LOW CONFIDENCE - Modal peak below 10.0 dB threshold]"
        print(f"\nDominant Fundamental Structural Mode (f1): {peaks[0].freq_hz:.3f} Hz{conf_suffix}")
    else:
        print("\nNo prominent spectral peaks identified above 0.5 Hz [LOW CONFIDENCE].")

    # 5. Save Plots Next to Video
    video_dir = os.path.dirname(video_path)
    video_stem = os.path.splitext(os.path.basename(video_path))[0]

    disp_plot_path = os.path.join(video_dir, f"{video_stem}_displacement.png")
    spec_plot_path = os.path.join(video_dir, f"{video_stem}_spectrum.png")

    time_vec = np.arange(len(disp)) / fps

    # A. Displacement Plot
    fig_d, ax_d = plt.subplots(figsize=(10, 4), dpi=150)
    ax_d.plot(time_vec, disp, color="#0284c7", linewidth=1.2, label=f"Subpixel Displacement ({args.method})")
    ax_d.set_title(f"SETU Subpixel Structural Displacement: {video_stem}", fontsize=12, fontweight="bold")
    ax_d.set_xlabel("Time (seconds)", fontsize=10)
    ax_d.set_ylabel("Displacement (pixels)", fontsize=10)
    ax_d.grid(True, linestyle="--", alpha=0.5)
    ax_d.legend(loc="upper right")
    plt.tight_layout()
    fig_d.savefig(disp_plot_path)
    plt.close(fig_d)

    # B. Spectrum Plot
    sig_detrended = signal.detrend(disp, type="linear")
    nperseg = min(len(sig_detrended), 1024)
    nfft = max(nperseg, 4096)
    freqs, psd = signal.welch(sig_detrended, fs=fps, window="hann", nperseg=nperseg, nfft=nfft)

    mask = (freqs >= 0.5) & (freqs <= 20.0)
    freqs_plot = freqs[mask]
    psd_plot = psd[mask]

    fig_s, ax_s = plt.subplots(figsize=(10, 4), dpi=150)
    ax_s.plot(freqs_plot, psd_plot, color="#6366f1", linewidth=1.5, label="Welch PSD")

    # Mark top peaks
    colors = ["#dc2626", "#ea580c", "#16a34a"]
    for i, p in enumerate(peaks):
        c = colors[i % len(colors)]
        ax_s.axvline(p.freq_hz, color=c, linestyle=":", alpha=0.7)
        ax_s.plot([p.freq_hz], [np.interp(p.freq_hz, freqs_plot, psd_plot)], "o", color=c)
        ax_s.annotate(
            f"f{i+1}={p.freq_hz:.2f} Hz\n({p.rel_power*100:.1f}%)",
            xy=(p.freq_hz, np.interp(p.freq_hz, freqs_plot, psd_plot)),
            xytext=(p.freq_hz + 0.3, np.interp(p.freq_hz, freqs_plot, psd_plot) * 0.9 + 1e-8),
            fontsize=8,
            fontweight="bold",
            color=c,
            arrowprops=dict(arrowstyle="->", color=c, lw=1),
        )

    ax_s.set_title(f"SETU Vibration Power Spectral Density (Welch PSD): {video_stem}", fontsize=12, fontweight="bold")
    ax_s.set_xlabel("Frequency (Hz)", fontsize=10)
    ax_s.set_ylabel("Power Spectral Density (px²/Hz)", fontsize=10)
    ax_s.grid(True, linestyle="--", alpha=0.5)
    ax_s.legend(loc="upper right")
    plt.tight_layout()
    fig_s.savefig(spec_plot_path)
    plt.close(fig_s)

    print(f"\n[Step 5] Telemetry Plots Saved:")
    print(f"  Displacement Plot:  {disp_plot_path}")
    print(f"  Spectrum Plot:      {spec_plot_path}")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
