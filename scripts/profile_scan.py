"""
Profiling script for Project SETU vibration screening pipeline.

Task Requirements:
1. Create a 30 s, 1920x1080, 60 fps H.264 test video of a target vibrating at 3.7 Hz
   (using a simple generator within this script; tests/synth.py is NOT modified).
2. Run analyze_video on it in SETU_META_MODE=mock, with and without magnify=True.
3. Print the elapsed time for each stage:
   - video checks
   - segmentation
   - motion
   - spectrum
   - interpretation
   - magnification
   and peak memory.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
import tracemalloc
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Force mock mode with zero external network requests
os.environ["SETU_META_MODE"] = "mock"
os.environ["META_API_MODE"] = "mock"
os.environ["USE_LOCAL_FALLBACKS"] = "true"

from setu.meta_client import MetaClient
from setu.osm import OverpassBridgeClient
from setu.result import ScanResult
from setu.scan import analyze_video


def generate_benchmark_video(
    output_path: str,
    fps: float = 60.0,
    duration_sec: float = 30.0,
    width: int = 1920,
    height: int = 1080,
    freq_hz: float = 3.7,
    amplitude_px: float = 0.2,
    target_bbox: Tuple[int, int, int, int] = (800, 200, 320, 680),
    background_bbox: Tuple[int, int, int, int] = (100, 750, 350, 250),
    seed: int = 42,
) -> str:
    """Generate a 30 s 1920x1080 @ 60 fps H.264 video with a target vibrating at 3.7 Hz."""
    print(f"[Generator] Synthesizing benchmark video: {duration_sec}s, {width}x{height} @ {fps} fps ({freq_hz} Hz)...")
    t0 = time.time()
    rng = np.random.default_rng(seed)
    n_frames = int(round(fps * duration_sec))
    time_series = np.arange(n_frames, dtype=np.float64) / fps
    displacements = amplitude_px * np.sin(2.0 * np.pi * freq_hz * time_series)

    # 1. Textured static background
    bg = rng.integers(50, 90, size=(height // 4, width // 4, 3), dtype=np.uint8)
    bg = cv2.resize(bg, (width, height), interpolation=cv2.INTER_LINEAR)

    # Stationary background anchor with distinct 2D checkerboard texture
    bx, by, bw, bh = background_bbox
    bg_anchor = rng.integers(70, 150, size=(bh, bw, 3), dtype=np.uint8)
    for y_s in range(0, bh - 16, 16):
        for x_s in range(0, bw - 16, 16):
            if ((y_s // 16) + (x_s // 16)) % 2 == 0:
                bg_anchor[y_s : y_s + 16, x_s : x_s + 16] = 230
            else:
                bg_anchor[y_s : y_s + 16, x_s : x_s + 16] = 30
    bg[by : by + bh, bx : bx + bw] = bg_anchor

    # 2. Textured target pillar with subpixel margins
    margin = 32
    tx, ty, tw, th = target_bbox
    patch_w = tw + 2 * margin
    patch_h = th + 2 * margin
    pillar = rng.integers(140, 200, size=(patch_h // 2, patch_w // 2, 3), dtype=np.uint8)
    pillar = cv2.resize(pillar, (patch_w, patch_h), interpolation=cv2.INTER_LINEAR)
    for x_s in range(margin + 20, patch_w - margin, 40):
        pillar[:, x_s : x_s + 4] = 80
    pillar[:, margin : margin + 4] = 240  # highlight edge
    pillar[:, patch_w - margin - 4 : patch_w - margin] = 40  # shadow edge

    # 3. Fast H.264 encoding via PyAV (with cv2 fallback)
    has_av = False
    try:
        import av

        has_av = True
    except ImportError:
        pass

    if has_av:
        container = av.open(output_path, mode="w")
        stream = container.add_stream("libx264", rate=int(round(fps)))
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        stream.options = {"preset": "ultrafast", "crf": "23"}

        for i in range(n_frames):
            dx = displacements[i]
            m = np.float32([[1.0, 0.0, dx + margin], [0.0, 1.0, float(margin)]])
            warped = cv2.warpAffine(
                pillar, m, (patch_w, patch_h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT
            )
            frame = bg.copy()
            frame[ty : ty + th, tx : tx + tw] = warped[margin : margin + th, margin : margin + tw]

            av_frame = av.VideoFrame.from_ndarray(frame, format="bgr24")
            for packet in stream.encode(av_frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
        container.close()
    else:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, float(fps), (width, height))
        for i in range(n_frames):
            dx = displacements[i]
            m = np.float32([[1.0, 0.0, dx + margin], [0.0, 1.0, float(margin)]])
            warped = cv2.warpAffine(
                pillar, m, (patch_w, patch_h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT
            )
            frame = bg.copy()
            frame[ty : ty + th, tx : tx + tw] = warped[margin : margin + th, margin : margin + tw]
            writer.write(frame)
        writer.release()

    gen_time = time.time() - t0
    fsize_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[Generator] Benchmark video generated in {gen_time:.2f}s ({fsize_mb:.2f} MB, {n_frames} frames).")
    return output_path


def run_profile_pass(
    video_path: str,
    magnify: bool,
    out_dir: str,
) -> Tuple[Dict[str, float], float, float, ScanResult]:
    """Execute analyze_video with per-stage timing and peak memory tracking."""
    meta_client = MetaClient(mode="mock")
    osm_client = OverpassBridgeClient(mode="mock")

    tracemalloc.start()
    t_start = time.perf_counter()

    timings: Dict[str, float] = {}
    result = analyze_video(
        path=video_path,
        lat=13.0827,
        lon=80.2707,
        target_prompt="concrete pillar",
        background_prompt="building",
        magnify=magnify,
        out_dir=out_dir,
        meta_client=meta_client,
        osm_client=osm_client,
        stage_timings=timings,
    )

    total_time = time.perf_counter() - t_start
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak_bytes / (1024 * 1024)

    if not timings and hasattr(result, "_stage_timings"):
        timings = getattr(result, "_stage_timings")

    return timings, total_time, peak_mb, result


def print_stage_report(title: str, timings: Dict[str, float], total_time: float, peak_mb: float, result: ScanResult) -> None:
    """Print clean formatted report of profiling metrics."""
    print("=" * 68)
    print(f" {title.upper()}")
    print("=" * 68)
    print(f" Verdict : {result.verdict.level} (reasons: {result.verdict.reasons})")
    dom_f = result.measurement.dominant_frequency_hz
    ci = result.measurement.frequency_ci_hz
    print(f" Frequency : {dom_f} Hz, CI: {ci}")
    print("-" * 68)
    print(f" {'Pipeline Stage':<25} | {'Duration (s)':<15} | {'Percentage':<10}")
    print("-" * 68)

    stage_order = [
        "video checks",
        "segmentation",
        "motion",
        "spectrum",
        "interpretation",
        "magnification",
    ]
    tracked_sum = 0.0
    for stage in stage_order:
        sec = timings.get(stage, 0.0)
        tracked_sum += sec
        pct = (sec / total_time * 100.0) if total_time > 0 else 0.0
        print(f" {stage:<25} | {sec:>12.3f} s | {pct:>9.1f}%")

    other_time = max(0.0, total_time - tracked_sum)
    other_pct = (other_time / total_time * 100.0) if total_time > 0 else 0.0
    print(f" {'other/overhead':<25} | {other_time:>12.3f} s | {other_pct:>9.1f}%")
    print("-" * 68)
    print(f" {'Total Pipeline Time':<25} | {total_time:>12.3f} s | 100.0%")
    print(f" {'Peak Memory Usage':<25} | {peak_mb:>12.2f} MB |")
    print("=" * 68)
    print()


def main() -> None:
    temp_dir = tempfile.mkdtemp(prefix="setu_profile_")
    try:
        video_path = os.path.join(temp_dir, "benchmark_1080p_30s.mp4")
        generate_benchmark_video(video_path)

        # Pass 1: WITHOUT magnification
        out_dir_nomag = os.path.join(temp_dir, "out_no_mag")
        print("\n>>> Running analyze_video WITHOUT magnification (magnify=False)...")
        timings_nomag, total_nomag, peak_nomag, res_nomag = run_profile_pass(
            video_path=video_path,
            magnify=False,
            out_dir=out_dir_nomag,
        )
        print_stage_report("Scan Profile (magnify=False)", timings_nomag, total_nomag, peak_nomag, res_nomag)

        # Pass 2: WITH magnification
        out_dir_mag = os.path.join(temp_dir, "out_with_mag")
        print("\n>>> Running analyze_video WITH magnification (magnify=True)...")
        timings_mag, total_mag, peak_mag, res_mag = run_profile_pass(
            video_path=video_path,
            magnify=True,
            out_dir=out_dir_mag,
        )
        print_stage_report("Scan Profile (magnify=True)", timings_mag, total_mag, peak_mag, res_mag)

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
