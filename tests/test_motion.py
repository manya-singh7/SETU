"""
Comprehensive benchmark and verification tests for subpixel motion tracking.

Verifies:
1. Cache keys uniquely hashed by every generation parameter (frequency, amplitude,
   jitter, noise, moving_object, CRF, seed, duration, fps).
2. Standard vibration conditions:
   - No jitter (CRF 18, CRF 28)
   - With 9 Hz camera micro-jitter (CRF 18, CRF 28)
3. Harder synthetic stress cases:
   - Low subpixel amplitude (0.05 px)
   - Close-frequency camera jitter (3.0 Hz component vs 3.7 Hz target)
   - 2-second background moving object occlusion
4. Evaluates methods: phase_correlation vs lucas_kanade.
5. Strict tolerance assertion: |f_detected - 3.70| <= 0.10 Hz.
6. Prints a complete benchmark table reporting PASS / FAIL status for each condition.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple
import pytest

from setu.motion import track_motion
from setu.spectrum import (
    compute_welch_peaks,
    extract_background_residual_motion,
    is_peak_reliable,
)
from setu.types import MotionResult, SegmentResult
from tests.synth import generate_synthetic_video

CACHE_DIR = Path("tests/.cache")


def _generate_cache_key(prefix: str, params: Dict[str, Any]) -> str:
    """Generate a deterministic filename incorporating all generation parameters."""
    param_str = (
        f"fps={params.get('fps', 60.0):.1f}_"
        f"dur={params.get('duration_sec', 10.0):.1f}_"
        f"f={params.get('freq_hz', 3.7):.2f}_"
        f"a={params.get('amplitude_px', 0.2):.3f}_"
        f"jit={int(params.get('camera_jitter', False))}_"
        f"jfreq={params.get('jitter_tremor_freq', 9.0):.2f}_"
        f"jamp={params.get('jitter_tremor_amp', 0.08):.3f}_"
        f"noise={params.get('gaussian_noise_std', 0.0):.2f}_"
        f"mov={int(params.get('moving_object', False))}_"
        f"crf={params.get('crf', 23)}_"
        f"seed={params.get('seed', 42)}"
    )
    param_hash = hashlib.sha256(param_str.encode()).hexdigest()[:12]
    return f"{prefix}_{param_hash}.mp4"


def _get_or_create_benchmark_video(
    prefix: str,
    params: Dict[str, Any],
) -> Tuple[str, Tuple[int, int, int, int], Tuple[int, int, int, int]]:
    """Retrieve existing cached video or generate and save to tests/.cache/."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    filename = _generate_cache_key(prefix, params)
    video_path = str(CACHE_DIR / filename)

    target_bbox = (270, 90, 100, 280)
    bg_bbox = (40, 360, 120, 80)

    if os.path.exists(video_path) and os.path.getsize(video_path) > 10000:
        return video_path, target_bbox, bg_bbox

    res = generate_synthetic_video(
        output_path=video_path,
        fps=params.get("fps", 60.0),
        duration_sec=params.get("duration_sec", 10.0),
        width=640,
        height=480,
        freq_hz=params.get("freq_hz", 3.7),
        amplitude_px=params.get("amplitude_px", 0.2),
        camera_jitter=params.get("camera_jitter", False),
        jitter_tremor_freq=params.get("jitter_tremor_freq", 9.0),
        jitter_tremor_amp=params.get("jitter_tremor_amp", 0.08),
        gaussian_noise_std=params.get("gaussian_noise_std", 0.0),
        crf=params.get("crf", 23),
        moving_object=params.get("moving_object", False),
        moving_object_start_sec=params.get("moving_object_start_sec", 3.0),
        moving_object_duration_sec=params.get("moving_object_duration_sec", 2.0),
        target_bbox=target_bbox,
        background_bbox=bg_bbox,
        seed=params.get("seed", 42),
    )
    return res.video_path, res.target_bbox, res.background_bbox


@pytest.fixture(scope="session")
def benchmark_suite() -> Dict[str, Tuple[str, Tuple[int, int, int, int], Tuple[int, int, int, int]]]:
    """Session fixture creating cached test videos with unique parameter-hashed filenames."""
    suite = {}

    # 1. Standard Conditions
    suite["no_jitter_crf18"] = _get_or_create_benchmark_video(
        "std_nojitter_crf18",
        {"camera_jitter": False, "crf": 18, "amplitude_px": 0.2, "freq_hz": 3.7},
    )
    suite["no_jitter_crf28"] = _get_or_create_benchmark_video(
        "std_nojitter_crf28",
        {"camera_jitter": False, "crf": 28, "amplitude_px": 0.2, "freq_hz": 3.7},
    )
    suite["with_jitter_crf18"] = _get_or_create_benchmark_video(
        "std_jitter_crf18",
        {"camera_jitter": True, "jitter_tremor_freq": 9.0, "crf": 18, "amplitude_px": 0.2, "freq_hz": 3.7},
    )
    suite["with_jitter_crf28"] = _get_or_create_benchmark_video(
        "std_jitter_crf28",
        {"camera_jitter": True, "jitter_tremor_freq": 9.0, "crf": 28, "amplitude_px": 0.2, "freq_hz": 3.7},
    )

    # 2. Harder Synthetic Cases
    suite["hard_low_amplitude"] = _get_or_create_benchmark_video(
        "hard_low_amp",
        {"camera_jitter": True, "jitter_tremor_freq": 9.0, "amplitude_px": 0.05, "freq_hz": 3.7, "crf": 18},
    )
    suite["hard_close_jitter"] = _get_or_create_benchmark_video(
        "hard_close_jitter",
        {"camera_jitter": True, "jitter_tremor_freq": 3.0, "jitter_tremor_amp": 0.12, "amplitude_px": 0.2, "freq_hz": 3.7, "crf": 18},
    )
    suite["hard_moving_object"] = _get_or_create_benchmark_video(
        "hard_moving_obj",
        {"camera_jitter": True, "jitter_tremor_freq": 9.0, "moving_object": True, "amplitude_px": 0.2, "freq_hz": 3.7, "crf": 18},
    )

    return suite


def test_crf_compression_file_size_check(benchmark_suite):
    """Check whether CRF 18 and CRF 28 produce substantially different file sizes."""
    path_crf18, _, _ = benchmark_suite["no_jitter_crf18"]
    path_crf28, _, _ = benchmark_suite["no_jitter_crf28"]

    size18 = os.path.getsize(path_crf18)
    size28 = os.path.getsize(path_crf28)
    diff = abs(size18 - size28)
    ratio = (size18 - size28) / max(size18, 1) * 100.0

    print("\n--- CRF COMPRESSION CHECK ---")
    print(f"CRF 18 Video File: {os.path.basename(path_crf18)} ({size18:,} bytes)")
    print(f"CRF 28 Video File: {os.path.basename(path_crf28)} ({size28:,} bytes)")
    print(f"Size Difference:   {diff:,} bytes ({ratio:.1f}% size reduction with CRF 28)")

    assert size18 > size28, (
        f"CRF 18 ({size18} bytes) must be larger than CRF 28 ({size28} bytes)."
    )
    assert diff > 0.20 * size18, (
        f"CRF 18 and CRF 28 do not differ substantially! Diff: {diff:,} bytes ({ratio:.1f}%)."
    )
    print(f"CONFIRMED: Real libx264 CRF compression is active ({diff:,} bytes saved with CRF 28).\n")


@pytest.mark.slow
def test_motion_recovery_matrix(benchmark_suite):
    """Run full benchmark matrix across standard and harder cases without loosening tolerance."""
    methods = ["phase_correlation", "lucas_kanade"]
    conditions = [
        ("No Jitter (CRF 18)", "no_jitter_crf18"),
        ("No Jitter (CRF 28)", "no_jitter_crf28"),
        ("With Jitter (CRF 18)", "with_jitter_crf18"),
        ("With Jitter (CRF 28)", "with_jitter_crf28"),
        ("Hard: Low Amp (0.05 px)", "hard_low_amplitude"),
        ("Hard: Close Jitter (3.0 Hz)", "hard_close_jitter"),
        ("Hard: Moving Object (2s)", "hard_moving_object"),
    ]

    target_freq = 3.70
    tolerance = 0.10  # Strict tolerance: |f - 3.70| <= 0.10 Hz

    # Pre-extract background residual displacement per unique cached video path (computed once per video)
    bg_cache: Dict[str, Any] = {}
    for _, cond_key in conditions:
        v_path, _, b_box = benchmark_suite[cond_key]
        if v_path not in bg_cache:
            bg_cache[v_path] = extract_background_residual_motion(v_path, b_box)

    table_rows: List[Dict[str, str]] = []
    passing_count = 0
    failing_count = 0

    for method in methods:
        for cond_label, cond_key in conditions:
            vid_path, t_box, b_box = benchmark_suite[cond_key]
            bg_disp = bg_cache[vid_path]

            # 1. Run motion tracking
            motion_res: MotionResult = track_motion(
                video_path=vid_path,
                target_bbox=t_box,
                background_bbox=b_box,
                method=method,
            )

            # 2. Extract spectral peaks with background-referenced and prominence checks
            peaks = compute_welch_peaks(
                motion_res.displacement,
                fps=motion_res.fps,
                background_displacement=bg_disp,
            )

            if peaks:
                top_peak = peaks[0]
                detected_freq = top_peak.freq_hz
                freq_error = abs(detected_freq - target_freq)
                passed = freq_error <= tolerance
                is_reliable = getattr(top_peak, "is_reliable", is_peak_reliable(top_peak))
                bg_ratio = getattr(top_peak, "bg_ratio", float("inf"))
                bg_ratio_db = getattr(top_peak, "bg_ratio_db", float("inf"))
                verdict_str = "Reliable" if is_reliable else "Low Confidence"
                bg_ratio_str = (
                    f"{bg_ratio:8.1f} ({bg_ratio_db:4.1f} dB)"
                    if bg_ratio < 1e7
                    else f">1e7 ({bg_ratio_db:4.1f} dB)"
                )
            else:
                detected_freq = 0.0
                freq_error = 999.0
                passed = False
                is_reliable = False
                verdict_str = "Low Confidence"
                bg_ratio_str = "N/A"

            if passed:
                passing_count += 1
                status_str = f"PASS (<={tolerance} Hz)"
            else:
                failing_count += 1
                status_str = f"FAIL (>{tolerance} Hz)"

            res_val = motion_res.quality.get("background_residual_px", 0.0)
            rel_p_str = f"{top_peak.rel_power * 100:.1f}%" if peaks else "N/A"
            snr_db_str = f"{top_peak.snr:.1f} dB" if peaks else "N/A"

            table_rows.append(
                {
                    "Method": method,
                    "Condition": cond_label,
                    "Target f0": f"{target_freq:.2f} Hz",
                    "Detected f0": f"{detected_freq:.3f} Hz" if peaks else "None",
                    "Error": f"{freq_error:.4f} Hz" if peaks else "N/A",
                    "Residual": f"{res_val:.4f} px",
                    "Rel Power": rel_p_str,
                    "Peak SNR": snr_db_str,
                    "BG Ratio": bg_ratio_str,
                    "Verdict": verdict_str,
                    "Status": status_str,
                }
            )

    # 3. Print Benchmark Table
    print("\n" + "=" * 165)
    print("SETU MOTION TRACKING & MODAL FREQUENCY RECOVERY BENCHMARK TABLE")
    print("=" * 165)
    header = (
        f"{'Method':<19} | {'Condition':<26} | {'Target':<10} | {'Detected':<12} | "
        f"{'Error':<11} | {'Residual':<11} | {'Rel Power':<10} | {'Peak SNR':<10} | "
        f"{'BG Ratio':<24} | {'Verdict':<15} | {'Status'}"
    )
    print(header)
    print("-" * 165)

    for r in table_rows:
        row_str = (
            f"{r['Method']:<19} | {r['Condition']:<26} | {r['Target f0']:<10} | {r['Detected f0']:<12} | "
            f"{r['Error']:<11} | {r['Residual']:<11} | {r['Rel Power']:<10} | {r['Peak SNR']:<10} | "
            f"{r['BG Ratio']:<24} | {r['Verdict']:<15} | {r['Status']}"
        )
        print(row_str)

    print("=" * 165)
    print(f"Summary: {passing_count} PASSED, {failing_count} FAILED across {len(table_rows)} evaluated runs.")
    print("=" * 165 + "\n")

    # Assertions:
    # 1. The two 0.05 px failing cases must be flagged as "Low Confidence"
    # 2. All 12 passing cases must be flagged as "Reliable" and strictly PASS without loosening tolerance
    for r in table_rows:
        if "Hard: Low Amp" in r["Condition"]:
            assert r["Verdict"] == "Low Confidence", (
                f"Failing case {r['Method']} under {r['Condition']} must be flagged Low Confidence, got {r['Verdict']}"
            )
            assert "FAIL" in r["Status"], "Tolerance must not be loosened on low amplitude 0.05 px stress case"
        else:
            assert r["Verdict"] == "Reliable", (
                f"Passing case {r['Method']} under {r['Condition']} must be Reliable, got {r['Verdict']}"
            )
            assert "PASS" in r["Status"], (
                f"Standard/passing case {r['Method']} under {r['Condition']} failed: {r['Error']}"
            )


def test_no_stationary_background_handling(benchmark_suite):
    """Verify that when SAM finds no stationary background (background_source 'none'):
    - setu/motion.py runs without crashing.
    - Camera stabilisation is skipped with a clear warning.
    - Background-ratio check is skipped.
    - Result is marked 'Unverified: no stationary background in frame'.
    - Result is NEVER labelled 'Reliable'.
    """
    import cv2
    import numpy as np

    v_path, t_box, _ = benchmark_suite["no_jitter_crf18"]

    cap = cv2.VideoCapture(v_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    target_mask = np.zeros((height, width), dtype=bool)
    tx, ty, tw, th = t_box
    target_mask[ty : ty + th, tx : tx + tw] = True

    seg_result = SegmentResult(
        target_masks=[target_mask],
        background_masks=[],  # Empty background masks when SAM finds no stationary background
        fps=60.0,
        method="fallback",
    )

    # 1. Run motion tracking without background mask or background bbox
    motion_res: MotionResult = track_motion(
        video_path=v_path,
        segment_result=seg_result,
        target_bbox=t_box,
        background_bbox=None,
        method="phase_correlation",
    )

    # 2. Verify stabilisation was skipped and clear warning added
    assert motion_res.quality.get("has_background") is False, "Expected has_background=False in quality"
    warnings = motion_res.quality.get("warnings", [])
    assert any("no stationary background" in w.lower() for w in warnings), (
        f"Expected clear warning about missing background in {warnings}"
    )
    assert motion_res.quality.get("confidence") == "Unverified: no stationary background in frame"
    assert motion_res.quality.get("verdict") == "Unverified: no stationary background in frame"

    # 3. Verify spectral analysis skips background-ratio check and marks peaks unverified
    peaks = compute_welch_peaks(
        motion_res.displacement,
        fps=motion_res.fps,
        has_background=False,
    )

    assert len(peaks) >= 1
    for p in peaks:
        assert p.is_reliable is False, "Must NEVER be labelled Reliable without a background"
        assert is_peak_reliable(p) is False
        assert p.verdict == "Unverified: no stationary background in frame"
        assert p.bg_ratio is None, "Background ratio check must be skipped"
