#!/usr/bin/env python3
"""
Batch Video Footage Analysis Script for Project SETU.

Scans a folder of civil infrastructure video recordings (default: uploads/), runs
the full SETU vibration screening pipeline (setu.scan.analyze_video) on each video,
using prompts from an optional prompts.json located in the folder (target, background,
lat, lon), saves result.json and visualization plots to uploads/results/<video name>/,
and prints a comprehensive summary table.

If a Phyphox accelerometer CSV with the same base name exists (e.g. bridge.csv for bridge.mp4),
cross-sensor validation (validation.compare) is automatically executed and its findings
(accelerometer peak frequency and agreement status) are added to the summary table.

Usage:
    python scripts/analyze_footage.py
    python scripts/analyze_footage.py --folder path/to/videos
    python scripts/analyze_footage.py uploads/ --magnify
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure SETU_META_MODE defaults to 'cached' per pipeline specifications
os.environ.setdefault("SETU_META_MODE", "cached")

# Add repository root to Python path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from setu.result import ScanResult
from setu.scan import analyze_video
from validation.compare import compare_phyphox_to_camera

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}


def load_prompts_config(folder_path: Path) -> Dict[str, Any]:
    """Load prompts configuration from prompts.json in the target folder if available."""
    prompts_file = folder_path / "prompts.json"
    if not prompts_file.exists():
        return {}

    try:
        with open(prompts_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data
            return {}
    except Exception as exc:
        print(f"[WARNING] Failed to parse {prompts_file}: {exc}. Using default prompts.")
        return {}


def resolve_video_prompts(
    video_file: Path, prompts_config: Dict[str, Any]
) -> Tuple[str, str, float, float]:
    """Resolve target prompt, background prompt, latitude, and longitude for a video.

    Supports:
    1. Per-video configuration mapping by filename or stem:
       {"video1.mp4": {"target": "pillar", "background": "bedrock", "lat": 12.97, "lon": 77.59}}
    2. Global defaults at the top level or under a 'defaults' key:
       {"target": "concrete pillar", "background": "building", "lat": 0.0, "lon": 0.0}
    """
    # Base defaults
    target = "concrete pillar"
    background = "building"
    lat = 0.0
    lon = 0.0

    # 1. Check top-level or 'defaults' section for global overrides
    defaults_section = prompts_config.get("defaults") or prompts_config.get("default") or {}
    if isinstance(defaults_section, dict):
        target = defaults_section.get("target") or defaults_section.get("target_prompt") or target
        background = defaults_section.get("background") or defaults_section.get("background_prompt") or background
        lat = float(defaults_section.get("lat") or defaults_section.get("latitude") or lat)
        lon = float(defaults_section.get("lon") or defaults_section.get("longitude") or lon)

    target = prompts_config.get("target") or prompts_config.get("target_prompt") or target
    background = prompts_config.get("background") or prompts_config.get("background_prompt") or background
    lat = float(prompts_config.get("lat") or prompts_config.get("latitude") or lat)
    lon = float(prompts_config.get("lon") or prompts_config.get("longitude") or lon)

    # 2. Check for video-specific override
    file_key = None
    if video_file.name in prompts_config:
        file_key = video_file.name
    elif video_file.stem in prompts_config:
        file_key = video_file.stem

    if file_key and isinstance(prompts_config[file_key], dict):
        cfg = prompts_config[file_key]
        target = cfg.get("target") or cfg.get("target_prompt") or target
        background = cfg.get("background") or cfg.get("background_prompt") or background
        lat = float(cfg.get("lat") or cfg.get("latitude") or lat)
        lon = float(cfg.get("lon") or cfg.get("longitude") or lon)

    return target, background, lat, lon


def find_matching_phyphox_csv(video_file: Path) -> Optional[Path]:
    """Look for an accelerometer CSV with the same base name next to the video."""
    folder = video_file.parent
    stem = video_file.stem

    candidates = [
        video_file.with_suffix(".csv"),
        folder / f"{stem}_phyphox.csv",
        folder / f"{stem}_accel.csv",
        folder / f"{stem}_accelerometer.csv",
    ]

    for cand in candidates:
        if cand.exists() and cand.is_file():
            return cand
    return None


def format_frequency_ci(scan_result: ScanResult) -> str:
    """Format dominant frequency and confidence interval for table display."""
    f0 = scan_result.measurement.dominant_frequency_hz
    ci = scan_result.measurement.frequency_ci_hz

    if f0 is None or scan_result.verdict.level == "failed":
        return "N/A"

    if ci is not None and len(ci) == 2:
        ci_half = abs(ci[1] - ci[0]) / 2.0
        return f"{f0:.2f} +/- {ci_half:.2f} Hz"
    return f"{f0:.2f} Hz"


def print_summary_table(rows: List[Dict[str, str]], has_accel: bool) -> None:
    """Print an ASCII summary table of the batch analysis results."""
    print("\n" + "=" * (135 if has_accel else 110))
    print("SETU BATCH FOOTAGE VIBRATION ANALYSIS SUMMARY")
    print("=" * (135 if has_accel else 110))

    if not rows:
        print("No videos analyzed.")
        print("=" * (135 if has_accel else 110) + "\n")
        return

    if has_accel:
        header = (
            f"{'File':<22} | {'Verdict':<8} | {'Frequency +/- CI':<19} | "
            f"{'Target/BG Source':<18} | {'Run Time':<8} | {'Accel Freq':<12} | {'Agreement':<14} | {'Reasons'}"
        )
    else:
        header = (
            f"{'File':<24} | {'Verdict':<8} | {'Frequency +/- CI':<20} | "
            f"{'Target/BG Source':<20} | {'Run Time':<10} | {'Reasons'}"
        )

    print(header)
    print("-" * (135 if has_accel else 110))

    for r in rows:
        if has_accel:
            row_str = (
                f"{r['File']:<22} | {r['Verdict']:<8} | {r['Frequency']:<19} | "
                f"{r['Sources']:<18} | {r['Run Time']:<8} | {r.get('Accel Freq', 'N/A'):<12} | "
                f"{r.get('Agreement', 'N/A'):<14} | {r['Reasons']}"
            )
        else:
            row_str = (
                f"{r['File']:<24} | {r['Verdict']:<8} | {r['Frequency']:<20} | "
                f"{r['Sources']:<20} | {r['Run Time']:<10} | {r['Reasons']}"
            )
        print(row_str)

    print("=" * (135 if has_accel else 110))
    usable_count = sum(1 for r in rows if r["Verdict"].lower() == "usable")
    weak_count = sum(1 for r in rows if r["Verdict"].lower() == "weak")
    failed_count = sum(1 for r in rows if r["Verdict"].lower() == "failed")
    print(f"Summary: {len(rows)} videos processed | Usable: {usable_count} | Weak: {weak_count} | Failed: {failed_count}")
    print("=" * (135 if has_accel else 110) + "\n")


def analyze_footage_folder(
    folder_path: Union[str, Path] = "uploads",
    results_dir: Optional[Union[str, Path]] = None,
    magnify: bool = False,
    is_demo: bool = False,
) -> List[Dict[str, Any]]:
    """Scan and process all video recordings in the specified folder.

    Args:
        folder_path: Input directory containing video footage and optional prompts.json.
        results_dir: Output base directory (defaults to folder_path / 'results').
        magnify: Whether to generate Eulerian motion magnification videos.
        is_demo: Whether running in demo mode.

    Returns:
        List of dictionaries with processing results for each video.
    """
    f_path = Path(folder_path)
    if not f_path.exists():
        print(f"[ERROR] Input folder '{f_path}' does not exist.")
        return []

    res_base = Path(results_dir) if results_dir else (f_path / "results")
    res_base.mkdir(parents=True, exist_ok=True)

    prompts_config = load_prompts_config(f_path)

    # Collect video files (excluding any files inside results/)
    video_files: List[Path] = []
    for item in sorted(f_path.iterdir()):
        if item.is_file() and item.suffix.lower() in VIDEO_EXTENSIONS:
            video_files.append(item)

    if not video_files:
        print(f"[INFO] No video files found in '{f_path}' matching extensions: {sorted(VIDEO_EXTENSIONS)}")
        return []

    print(f"\n[INFO] Found {len(video_files)} video(s) in '{f_path}'. Output destination: '{res_base}'")

    has_any_accel = False
    table_rows: List[Dict[str, str]] = []
    batch_records: List[Dict[str, Any]] = []

    for idx, v_file in enumerate(video_files, 1):
        print(f"\n[{idx}/{len(video_files)}] Processing '{v_file.name}'...")
        target_prompt, bg_prompt, lat, lon = resolve_video_prompts(v_file, prompts_config)
        print(f"       Prompts: target='{target_prompt}', background='{bg_prompt}' | Location: ({lat:.4f}, {lon:.4f})")

        video_out_dir = res_base / v_file.stem
        video_out_dir.mkdir(parents=True, exist_ok=True)

        t_start = time.perf_counter()
        scan_res = analyze_video(
            path=str(v_file),
            lat=lat,
            lon=lon,
            target_prompt=target_prompt,
            background_prompt=bg_prompt,
            magnify=magnify,
            is_demo=is_demo,
            out_dir=video_out_dir,
        )
        elapsed_sec = time.perf_counter() - t_start

        # Cross-sensor validation if matching phyphox CSV exists
        accel_csv = find_matching_phyphox_csv(v_file)
        accel_freq_str = "N/A"
        agreement_str = "N/A"

        if accel_csv is not None:
            has_any_accel = True
            print(f"       Found matching accelerometer CSV: '{accel_csv.name}'")
            if scan_res.measurement.dominant_frequency_hz is not None and scan_res.verdict.level != "failed":
                try:
                    comp_plot = video_out_dir / "comparison.png"
                    comp_res = compare_phyphox_to_camera(
                        phyphox_csv=accel_csv,
                        camera_source={
                            "frequency_hz": scan_res.measurement.dominant_frequency_hz,
                            "frequency_ci": scan_res.measurement.frequency_ci_hz,
                        },
                        output_png=comp_plot,
                    )
                    accel_freq_str = f"{comp_res.accelerometer_peak_hz:.2f} Hz"
                    agree_label = "AGREE" if comp_res.is_agreement else "DISAGREE"
                    agreement_str = f"{comp_res.verdict} ({agree_label})"
                    print(f"       Accelerometer Peak: {accel_freq_str} | Agreement: {agreement_str} (Diff: {comp_res.relative_diff_pct:.1f}%)")
                except Exception as comp_err:
                    print(f"       [WARNING] Accelerometer comparison failed: {comp_err}")
                    accel_freq_str = "Error"
                    agreement_str = "Error"
            else:
                accel_freq_str = "N/A"
                agreement_str = "N/A (Scan Failed)"

        # Prepare summary table row
        reasons_list = scan_res.verdict.reasons
        reasons_display = "; ".join(reasons_list) if reasons_list else "None"
        # Truncate overly long reason strings for terminal readability
        if len(reasons_display) > 45:
            reasons_display = reasons_display[:42] + "..."

        t_src = scan_res.segmentation.target_source
        bg_src = scan_res.segmentation.background_source
        sources_str = f"{t_src}/{bg_src}"

        row_data = {
            "File": v_file.name,
            "Verdict": scan_res.verdict.level.upper(),
            "Frequency": format_frequency_ci(scan_res),
            "Sources": sources_str,
            "Run Time": f"{elapsed_sec:.1f}s",
            "Reasons": reasons_display,
            "Accel Freq": accel_freq_str,
            "Agreement": agreement_str,
        }
        table_rows.append(row_data)

        batch_records.append(
            {
                "video": str(v_file),
                "output_dir": str(video_out_dir),
                "scan_result": scan_res,
                "elapsed_sec": elapsed_sec,
                "accel_csv": str(accel_csv) if accel_csv else None,
                "accel_freq_str": accel_freq_str,
                "agreement_str": agreement_str,
            }
        )

    print_summary_table(table_rows, has_accel=has_any_accel)
    return batch_records


def main() -> int:
    """CLI entrypoint for analyze_footage.py."""
    parser = argparse.ArgumentParser(
        description="Batch vibration analysis for Project SETU video footage."
    )
    parser.add_argument(
        "folder",
        nargs="?",
        default="uploads",
        type=str,
        help="Directory containing video footage to analyze (default: uploads/).",
    )
    parser.add_argument(
        "--results-dir",
        "-r",
        default=None,
        type=str,
        help="Directory to save result.json and visualization plots (default: <folder>/results/).",
    )
    parser.add_argument(
        "--magnify",
        "-m",
        action="store_true",
        help="Generate Eulerian motion magnification video artifact for passing scans.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run in demo mode.",
    )

    args = parser.parse_args()

    analyze_footage_folder(
        folder_path=args.folder,
        results_dir=args.results_dir,
        magnify=args.magnify,
        is_demo=args.demo,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
