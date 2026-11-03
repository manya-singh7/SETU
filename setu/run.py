"""
Project SETU Command Line Interface.

A thin CLI wrapper around `setu.scan.analyze_video`.
Executes the screening pipeline on a mobile or test video, writes
`result.json` and diagnostic plots into the output folder, and prints
a structured summary to stdout.

Usage:
    python -m setu.run VIDEO [--target PROMPT] [--background PROMPT] [--out-dir DIR] [--lat LAT] [--lon LON] [--magnify]
"""

from __future__ import annotations

import argparse
import os
import sys

from setu.scan import analyze_video


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
        default="concrete pillar",
        help="Text prompt for vibrating structural target (default: 'concrete pillar')",
    )
    parser.add_argument(
        "--background",
        "-b",
        type=str,
        default="building",
        help="Text prompt for stationary background anchor (default: 'building')",
    )
    parser.add_argument(
        "--lat",
        type=float,
        default=13.0827,
        help="GPS Latitude of recording location (default: 13.0827)",
    )
    parser.add_argument(
        "--lon",
        type=float,
        default=80.2707,
        help="GPS Longitude of recording location (default: 80.2707)",
    )
    parser.add_argument(
        "--out-dir",
        "-o",
        type=str,
        default=None,
        help="Output directory for result.json and diagnostic plots (defaults to <video_stem>_out)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        default=False,
        help="Flag scan result as demo mode",
    )
    parser.add_argument(
        "--magnify",
        action="store_true",
        default=False,
        help="Run Eulerian Video Magnification (EVM) centered at dominant frequency",
    )

    args = parser.parse_args()

    video_path = os.path.abspath(args.video)
    if not os.path.isfile(video_path):
        print(f"Error: Video file not found at '{video_path}'", file=sys.stderr)
        sys.exit(1)

    # Default out_dir is <video_dir>/<stem>_out
    if args.out_dir:
        out_dir = os.path.abspath(args.out_dir)
    else:
        video_dir = os.path.dirname(video_path)
        video_stem = os.path.splitext(os.path.basename(video_path))[0]
        out_dir = os.path.join(video_dir, f"{video_stem}_out")

    os.makedirs(out_dir, exist_ok=True)

    print("\n" + "=" * 78)
    print("SETU: CIVIL INFRASTRUCTURE VIBRATION SCREENING CLI")
    print("=" * 78)
    print(f"Video Source: {video_path}")
    print(f"Output Dir:   {out_dir}")
    print(f"Prompts:      Target: '{args.target}' | Background: '{args.background}'")
    print(f"Location:     Lat {args.lat:.4f}, Lon {args.lon:.4f}")
    print("-" * 78)

    # Execute pipeline via analyze_video
    result = analyze_video(
        path=video_path,
        lat=args.lat,
        lon=args.lon,
        target_prompt=args.target,
        background_prompt=args.background,
        magnify=args.magnify,
        is_demo=args.demo,
        out_dir=out_dir,
    )

    # Print summary
    print(f"\nScan ID:          {result.scan_id}")
    print(f"Verdict Level:    {result.verdict.level.upper()}")
    if result.verdict.reasons:
        print(f"Verdict Reasons:  {', '.join(result.verdict.reasons)}")

    if result.verdict.level == "failed":
        print("\nScan Status:      FAILED")
        print("Telemetry:        No reliable frequency detected.")
    else:
        dom_hz = result.measurement.dominant_frequency_hz
        ci = result.measurement.frequency_ci_hz
        ci_str = f"[{ci[0]:.2f}, {ci[1]:.2f}] Hz" if ci else "N/A"
        print(f"Dominant Mode:    {dom_hz:.2f} Hz (95% CI: {ci_str})")
        print(f"Peaks Detected:   {len(result.measurement.peaks)}")
        for idx, p in enumerate(result.measurement.peaks, 1):
            print(f"  #{idx}: {p.frequency_hz:.2f} Hz | Rel Power: {p.relative_power*100:.1f}% | SNR: {p.snr_db:.1f} dB")

    # Outputs
    print("\nGenerated Artifacts:")
    result_json_path = os.path.join(out_dir, "result.json")
    print(f"  Result JSON:        {result_json_path}")
    if result.outputs.displacement_plot:
        print(f"  Displacement Plot:  {os.path.join(out_dir, result.outputs.displacement_plot)}")
    if result.outputs.spectrum_plot:
        print(f"  Spectrum Plot:      {os.path.join(out_dir, result.outputs.spectrum_plot)}")
    if result.outputs.magnified_video:
        print(f"  Magnified Video:    {os.path.join(out_dir, result.outputs.magnified_video)}")

    print(f"\nInference Cost:   ${result.costs_usd.total:.4f} USD")
    print(f"Disclaimer:       {result.disclaimer}")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
