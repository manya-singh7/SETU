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
import numpy as np

from setu.motion import _select_phase_patches, track_motion
from setu.spectrum import (
    compute_welch_peaks,
    extract_background_residual_motion,
    is_peak_reliable,
)
from setu.types import MotionResult, SegmentResult
from tests.synth import generate_synthetic_video

CACHE_DIR = Path("tests/.cache")


def test_phase_patch_selection_is_textured_masked_and_non_overlapping():
    image = np.zeros((384, 384), dtype=np.uint8)
    image[::2, :] = 255
    mask = np.ones(image.shape, dtype=bool)
    mask[:128, :128] = False

    patches = _select_phase_patches(image, mask)

    assert 1 <= len(patches) <= 5
    assert all(np.all(mask[y : y + h, x : x + w]) for x, y, w, h in patches)
    for i, (x1, y1, w1, h1) in enumerate(patches):
        for x2, y2, w2, h2 in patches[i + 1 :]:
            assert x1 + w1 <= x2 or x2 + w2 <= x1 or y1 + h1 <= y2 or y2 + h2 <= y1


def test_roi_affine_warp_matches_full_frame_crop():
    """The ROI affine transform must match cropping a full-frame affine warp."""
    import cv2

    rng = np.random.default_rng(81)
    frame = rng.integers(0, 256, (173, 219), dtype=np.uint8)
    x, y, w, h = 37, 29, 61, 72
    matrix = np.array([[0.9998, -0.017, 2.31], [0.017, 0.9998, -1.72]], dtype=np.float64)
    shifted = matrix.copy()
    shifted[:, 2] -= np.array([x, y])
    grid_x, grid_y = np.meshgrid(np.arange(x, x + w), np.arange(y, y + h))
    points = np.stack((grid_x, grid_y), axis=-1).reshape(1, -1, 2).astype(np.float64)
    full_coordinates = cv2.transform(points, matrix)
    roi_coordinates = cv2.transform(points, shifted)
    roi_coordinates += np.array([[[x, y]]], dtype=np.float64)
    np.testing.assert_allclose(roi_coordinates, full_coordinates, atol=1e-6, rtol=0)


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
                verdict_str = top_peak.verdict  # "usable" or "weak"
                passed = freq_error <= tolerance and verdict_str == "usable"
                bg_ratio = getattr(top_peak, "bg_ratio", float("inf"))
                bg_ratio_db = getattr(top_peak, "bg_ratio_db", float("inf"))
                bg_ratio_str = (
                    f"{bg_ratio:8.1f} ({bg_ratio_db:4.1f} dB)"
                    if bg_ratio is not None and bg_ratio < 1e7
                    else f">1e7 ({bg_ratio_db:4.1f} dB)" if bg_ratio is not None else "N/A"
                )
            else:
                detected_freq = 0.0
                freq_error = 999.0
                passed = False
                verdict_str = "failed"
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
    # 1. The two 0.05 px failing cases must be flagged as "weak"
    # 2. All 12 passing cases must be flagged as "usable" and strictly PASS without loosening tolerance
    for r in table_rows:
        if "Hard: Low Amp" in r["Condition"]:
            assert r["Verdict"] == "weak", (
                f"Failing case {r['Method']} under {r['Condition']} must be flagged 'weak', got {r['Verdict']}"
            )
            assert "FAIL" in r["Status"], "Tolerance must not be loosened on low amplitude 0.05 px stress case"
        else:
            assert r["Verdict"] == "usable", (
                f"Passing case {r['Method']} under {r['Condition']} must be 'usable', got {r['Verdict']}"
            )
            assert "PASS" in r["Status"], (
                f"Standard/passing case {r['Method']} under {r['Condition']} failed: {r['Error']}"
            )


def test_no_stationary_background_handling(benchmark_suite):
    """Verify that when SAM finds no stationary background (background_source 'none'):
    - setu/motion.py runs without crashing.
    - Camera stabilisation is skipped with a clear warning.
    - Background-ratio check is skipped.
    - Result is marked 'weak' with unverified reason.
    - Result is NEVER labelled 'usable'.
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
    assert motion_res.quality.get("verdict") == "weak"
    assert motion_res.verdict == "weak"
    assert any("no stationary background" in r.lower() for r in motion_res.quality.get("reasons", []))

    # 3. Verify spectral analysis skips background-ratio check and marks peaks weak
    peaks = compute_welch_peaks(
        motion_res.displacement,
        fps=motion_res.fps,
        has_background=False,
    )

    assert len(peaks) >= 1
    for p in peaks:
        assert p.is_reliable is False, "Must NEVER be labelled Reliable/usable without a background"
        assert is_peak_reliable(p) is False
        assert p.verdict == "weak"
        assert p.bg_ratio is None, "Background ratio check must be skipped"


def test_clipping_check_flags_weak_verdict(tmp_path):
    """Verify that sensor saturation/clipping in target mask:
    - Detects clipped frames when >15% of pixels are at 0 or 255.
    - If >10% of frames have clipping, flags clipping_detected=True.
    - Adds 'weak' verdict with reason explaining saturation reduces sub-pixel accuracy.
    """
    import cv2
    import numpy as np

    fps = 30.0
    n_frames = 30
    width, height = 320, 240
    target_bbox = (100, 50, 60, 60)
    bg_bbox = (20, 150, 60, 40)

    temp_video = str(tmp_path / "clipping_test.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(temp_video, fourcc, fps, (width, height))

    tx, ty, tw, th = target_bbox
    bx, by, bw, bh = bg_bbox

    for _ in range(n_frames):
        frame = np.ones((height, width, 3), dtype=np.uint8) * 128
        # Add textured background
        frame[by : by + bh, bx : bx + bw] = np.random.randint(50, 200, (bh, bw, 3), dtype=np.uint8)
        # Saturate 50% of the target area to 255 (blown out / overexposed)
        frame[ty : ty + th, tx : tx + tw] = 120
        frame[ty : ty + th // 2, tx : tx + tw] = 255  # 50% of target saturated to 255
        writer.write(frame)
    writer.release()

    res = track_motion(
        video_path=temp_video,
        target_bbox=target_bbox,
        background_bbox=bg_bbox,
        method="phase_correlation",
    )

    assert res.quality.get("clipping_detected") is True, "Expected clipping_detected=True"
    assert res.quality.get("verdict") == "weak", f"Expected verdict 'weak', got {res.quality.get('verdict')}"
    reasons = res.quality.get("reasons", [])
    assert any("saturation" in r.lower() or "clipping" in r.lower() for r in reasons)
    assert any("reduces sub-pixel accuracy" in r.lower() for r in reasons)


def test_motion_result_units_and_scale(benchmark_suite):
    """Verify that MotionResult contains explicit units='px' and scale=None."""
    v_path, t_box, b_box = benchmark_suite["no_jitter_crf18"]
    res = track_motion(v_path, target_bbox=t_box, background_bbox=b_box)

    assert hasattr(res, "units")
    assert res.units == "px"
    assert hasattr(res, "scale")
    assert res.scale is None
    assert res.quality.get("units") == "px"
    assert res.quality.get("scale") is None
    assert res.verdict in ("usable", "weak", "failed")
    assert isinstance(res.reasons, list)


def test_still_synthetic_video_gives_failed_verdict(tmp_path):
    """Verify that a synthetic video of a completely still target:
    - Results in target displacement RMS within the tracking noise floor.
    - Sets verdict to 'failed'.
    - Sets failure reason to 'No measurable motion above tracking noise'.
    """
    video_path = str(tmp_path / "still_target.mp4")
    res = generate_synthetic_video(
        output_path=video_path,
        fps=60.0,
        duration_sec=3.0,
        amplitude_px=0.0,
        camera_jitter=False,
        crf=23,
        seed=42,
    )

    # 1. Phase correlation
    res_pc = track_motion(
        video_path=video_path,
        target_bbox=res.target_bbox,
        background_bbox=res.background_bbox,
        method="phase_correlation",
    )
    assert res_pc.verdict == "failed", f"Expected verdict 'failed', got {res_pc.verdict}"
    assert "No measurable motion above tracking noise" in res_pc.reasons
    assert res_pc.quality["verdict"] == "failed"
    assert "No measurable motion above tracking noise" in res_pc.quality["reasons"]

    # 2. Lucas-Kanade
    res_lk = track_motion(
        video_path=video_path,
        target_bbox=res.target_bbox,
        background_bbox=res.background_bbox,
        method="lucas_kanade",
    )
    assert res_lk.verdict == "failed", f"Expected verdict 'failed', got {res_lk.verdict}"
    assert "No measurable motion above tracking noise" in res_lk.reasons
    assert res_lk.quality["verdict"] == "failed"


def test_motion_tracking_nan_inf_gives_failed_verdict(tmp_path, monkeypatch):
    """Verify that if displacement contains NaN or Inf, verdict is 'failed' with reason
    'Motion tracking produced invalid values'.
    """
    video_path = str(tmp_path / "nan_motion.mp4")
    res = generate_synthetic_video(
        output_path=video_path,
        fps=30.0,
        duration_sec=1.0,
        amplitude_px=0.2,
        camera_jitter=False,
        seed=42,
    )
    # Monkeypatch cv2.phaseCorrelate to simulate invalid tracking values
    monkeypatch.setattr(
        "cv2.phaseCorrelate",
        lambda *args, **kwargs: ((float("nan"), float("nan")), 0.0),
    )
    motion_res = track_motion(
        video_path=video_path,
        target_bbox=res.target_bbox,
        background_bbox=res.background_bbox,
        method="phase_correlation",
    )
    assert motion_res.verdict == "failed", f"Expected verdict 'failed', got {motion_res.verdict}"
    assert "Motion tracking produced invalid values" in motion_res.reasons
    assert motion_res.quality["verdict"] == "failed"
    assert "Motion tracking produced invalid values" in motion_res.quality["reasons"]

