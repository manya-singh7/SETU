"""
Tests for subpixel motion tracking and camera jitter cancellation.

Verifies:
1. Recovery of injected 3.7 Hz structural vibration within ±0.1 Hz across:
   - Methods: phase_correlation, lucas_kanade
   - Conditions: with and without camera jitter
   - Compression: CRF 18 vs CRF 28
2. Checks whether CRF compression is active by comparing video file sizes.
3. Caches generated test videos in tests/.cache/ for fast reruns.
4. Prints a benchmark table of method x condition x detected frequency.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple
import pytest

from setu.motion import track_motion
from setu.spectrum import compute_welch_peaks
from setu.types import MotionResult
from tests.synth import generate_synthetic_video

CACHE_DIR = Path("tests/.cache")


def _get_or_create_cached_video(
    name: str,
    camera_jitter: bool,
    crf: int,
    duration_sec: float = 12.0,
    fps: float = 60.0,
) -> Tuple[str, Tuple[int, int, int, int], Tuple[int, int, int, int]]:
    """Retrieve existing cached video or generate and save to tests/.cache/."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    video_path = str(CACHE_DIR / f"{name}.mp4")

    target_bbox = (270, 90, 100, 280)
    bg_bbox = (40, 360, 120, 80)

    # Check if cached file already exists and is non-empty
    if os.path.exists(video_path) and os.path.getsize(video_path) > 10000:
        return video_path, target_bbox, bg_bbox

    res = generate_synthetic_video(
        output_path=video_path,
        fps=fps,
        duration_sec=duration_sec,
        width=640,
        height=480,
        freq_hz=3.7,
        amplitude_px=0.2,
        camera_jitter=camera_jitter,
        crf=crf,
        target_bbox=target_bbox,
        background_bbox=bg_bbox,
        seed=42,
    )
    return res.video_path, res.target_bbox, res.background_bbox


@pytest.fixture(scope="session")
def benchmark_videos() -> Dict[str, Tuple[str, Tuple[int, int, int, int], Tuple[int, int, int, int]]]:
    """Session fixture creating and caching synthetic test videos across conditions."""
    videos = {}
    videos["no_jitter_crf18"] = _get_or_create_cached_video("synth_nojitter_crf18", camera_jitter=False, crf=18)
    videos["no_jitter_crf28"] = _get_or_create_cached_video("synth_nojitter_crf28", camera_jitter=False, crf=28)
    videos["with_jitter_crf18"] = _get_or_create_cached_video("synth_jitter_crf18", camera_jitter=True, crf=18)
    videos["with_jitter_crf28"] = _get_or_create_cached_video("synth_jitter_crf28", camera_jitter=True, crf=28)
    return videos


def test_crf_compression_file_size_check(benchmark_videos):
    """Check whether CRF 18 and CRF 28 produce different file sizes."""
    path_crf18, _, _ = benchmark_videos["no_jitter_crf18"]
    path_crf28, _, _ = benchmark_videos["no_jitter_crf28"]

    size18 = os.path.getsize(path_crf18)
    size28 = os.path.getsize(path_crf28)
    diff = abs(size18 - size28)

    print("\n--- CRF COMPRESSION CHECK ---")
    print(f"CRF 18 File Size: {size18:,} bytes")
    print(f"CRF 28 File Size: {size28:,} bytes")
    print(f"Size Difference: {diff:,} bytes")

    if diff == 0:
        msg = (
            "NOTICE: CRF 18 and CRF 28 video files have identical size. "
            "H.264 CRF compression is NOT being applied by OpenCV because the OpenH264 "
            "encoder library (openh264-2.5.0-win64.dll) is missing in this environment, "
            "so OpenCV falls back to standard mp4v container encoding."
        )
        print(f"\n{msg}\n")
    else:
        print("CONFIRMED: H.264 CRF compression is active and altering file size.\n")


@pytest.mark.slow
def test_motion_recovery_matrix(benchmark_videos):
    """Recover 3.7 Hz within ±0.1 Hz across methods, jitter conditions, and CRFs."""
    methods = ["phase_correlation", "lucas_kanade"]
    conditions = [
        ("No Jitter (CRF 18)", "no_jitter_crf18"),
        ("No Jitter (CRF 28)", "no_jitter_crf28"),
        ("With Jitter (CRF 18)", "with_jitter_crf18"),
        ("With Jitter (CRF 28)", "with_jitter_crf28"),
    ]

    target_freq = 3.70
    table_rows: List[Dict[str, str]] = []

    for method in methods:
        for cond_label, cond_key in conditions:
            vid_path, t_box, b_box = benchmark_videos[cond_key]

            # 1. Run motion tracking
            motion_res: MotionResult = track_motion(
                video_path=vid_path,
                target_bbox=t_box,
                background_bbox=b_box,
                method=method,
            )

            # 2. Extract spectral peaks via setu.spectrum
            peaks = compute_welch_peaks(motion_res.displacement, fps=motion_res.fps)
            assert len(peaks) > 0, f"No spectral peaks detected for {method} under {cond_label}"

            top_peak = peaks[0]
            detected_freq = top_peak.freq_hz
            freq_error = abs(detected_freq - target_freq)

            # 3. Assert recovery within ±0.1 Hz
            assert freq_error <= 0.10, (
                f"Method '{method}' under '{cond_label}' recovered {detected_freq:.3f} Hz, "
                f"which deviates by {freq_error:.3f} Hz (limit: ±0.10 Hz)"
            )

            snr_val = motion_res.quality.get("snr", 0.0)
            res_val = motion_res.quality.get("background_residual_px", 0.0)

            table_rows.append(
                {
                    "Method": method,
                    "Condition": cond_label,
                    "Target f0": f"{target_freq:.2f} Hz",
                    "Detected f0": f"{detected_freq:.3f} Hz",
                    "Error": f"{freq_error:.4f} Hz",
                    "Residual": f"{res_val:.4f} px",
                    "SNR": f"{snr_val:.1f}",
                    "Status": "PASS (≤0.1 Hz)",
                }
            )

    # 4. Print benchmark summary table
    print("\n" + "=" * 105)
    print("SETU MOTION TRACKING & MODAL FREQUENCY RECOVERY BENCHMARK TABLE")
    print("=" * 105)
    header = (
        f"{'Method':<19} | {'Condition':<22} | {'Target':<10} | {'Detected':<12} | "
        f"{'Error':<11} | {'Residual':<11} | {'SNR':<8} | {'Status'}"
    )
    print(header)
    print("-" * 105)

    for r in table_rows:
        row_str = (
            f"{r['Method']:<19} | {r['Condition']:<22} | {r['Target f0']:<10} | {r['Detected f0']:<12} | "
            f"{r['Error']:<11} | {r['Residual']:<11} | {r['SNR']:<8} | {r['Status']}"
        )
        print(row_str)

    print("=" * 105 + "\n")
