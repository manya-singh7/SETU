"""Unit tests for validation.compare module.

Tests cover:
- Phyphox CSV export parsing with flexible column headers, delimiters, and axis configurations.
- Accelerometer Welch PSD computation and 0.5-30 Hz structural modal peak detection.
- Sub-bin log-parabolic frequency refinement.
- Dominant axis identification (isolates y-axis 12 Hz peak from synthetic fixture).
- Camera frequency extraction from numeric inputs, AnalysisResult contracts, and ScanResult mocks.
- Cross-sensor comparison with justified tolerance criteria (PASS on match, FAIL on mismatch).
- Composite normalized spectrum plot generation and file validation.
- CLI argument parsing, mutually exclusive inputs, and formatted output reporting.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import tempfile
from unittest.mock import patch
import sys

import numpy as np
import pytest

from validation.compare import (
    DEFAULT_MAX_FREQ_HZ,
    DEFAULT_MIN_FREQ_HZ,
    DEFAULT_TOLERANCE_ABS_HZ,
    DEFAULT_TOLERANCE_REL_PCT,
    AccelerometerSpectrum,
    ComparisonResult,
    compare_frequency,
    comparison_validation_dict,
    compare_phyphox_to_camera,
    compute_axis_psd,
    estimate_accelerometer_uncertainty,
    extract_camera_frequency,
    find_dominant_accelerometer_peak,
    load_phyphox_csv,
    main,
    plot_comparison,
)


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "phyphox_sample.csv"


# ---------------------------------------------------------------------------
# Fixture Validation & CSV Parsing Tests
# ---------------------------------------------------------------------------

def test_phyphox_sample_fixture_exists():
    """Verify that tests/fixtures/phyphox_sample.csv exists and contains data."""
    assert FIXTURE_PATH.exists(), f"Fixture file not found: {FIXTURE_PATH}"
    assert FIXTURE_PATH.stat().st_size > 1000, "Fixture file appears empty or truncated"


def test_load_phyphox_csv_standard():
    """Verify standard phyphox CSV export loading and axis detection."""
    times, axes, fs = load_phyphox_csv(FIXTURE_PATH)

    assert len(times) >= 1000
    assert np.isclose(fs, 100.0, atol=1.0), f"Expected ~100 Hz sampling rate, got {fs:.2f} Hz"
    assert "x" in axes
    assert "y" in axes
    assert "z" in axes
    assert len(axes["y"]) == len(times)
    assert np.all(np.isfinite(axes["y"]))


def test_load_phyphox_csv_flexible_headers(tmp_path: Path):
    """Test flexible column header variations accepted by load_phyphox_csv."""
    t = np.linspace(0.0, 5.0, 500)
    ax = 0.1 * np.sin(2 * np.pi * 3.0 * t)
    ay = 1.5 * np.cos(2 * np.pi * 12.0 * t)

    # Variation 1: Short names ("time", "ax", "ay")
    csv1 = tmp_path / "var1.csv"
    with open(csv1, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "ax", "ay"])
        for ti, xi, yi in zip(t, ax, ay):
            writer.writerow([f"{ti:.3f}", f"{xi:.4f}", f"{yi:.4f}"])

    _, axes1, fs1 = load_phyphox_csv(csv1)
    assert "x" in axes1 and "y" in axes1
    assert "z" not in axes1
    assert np.isclose(fs1, 100.0, atol=2.0)

    # Variation 2: Descriptive names ("Time (s)", "Acceleration x (m/s^2)", "Acceleration y (m/s^2)")
    csv2 = tmp_path / "var2.csv"
    with open(csv2, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Time (s)", "Acceleration x (m/s^2)", "Acceleration y (m/s^2)"])
        for ti, xi, yi in zip(t, ax, ay):
            writer.writerow([f"{ti:.3f}", f"{xi:.4f}", f"{yi:.4f}"])

    _, axes2, fs2 = load_phyphox_csv(csv2)
    assert "x" in axes2 and "y" in axes2

    # Variation 3: Semicolon delimited with European comma decimal separator
    csv3 = tmp_path / "var3.csv"
    with open(csv3, "w", encoding="utf-8") as f:
        f.write("Time (s);Linear Acceleration x;Linear Acceleration y\n")
        for ti, xi, yi in zip(t[:50], ax[:50], ay[:50]):
            t_str = f"{ti:.3f}".replace(".", ",")
            x_str = f"{xi:.4f}".replace(".", ",")
            y_str = f"{yi:.4f}".replace(".", ",")
            f.write(f"{t_str};{x_str};{y_str}\n")

    _, axes3, fs3 = load_phyphox_csv(csv3)
    assert "x" in axes3 and "y" in axes3


def test_load_phyphox_csv_error_handling(tmp_path: Path):
    """Test error handling on missing files or invalid CSV structures."""
    # Non-existent file
    with pytest.raises(FileNotFoundError):
        load_phyphox_csv(tmp_path / "non_existent.csv")

    # Empty file
    empty_csv = tmp_path / "empty.csv"
    empty_csv.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="insufficient data rows"):
        load_phyphox_csv(empty_csv)

    # Non-numeric rows
    invalid_csv = tmp_path / "invalid.csv"
    invalid_csv.write_text("Time,X,Y\nfoo,bar,baz\n", encoding="utf-8")
    with pytest.raises(ValueError, match="fewer than 10 valid"):
        load_phyphox_csv(invalid_csv)


# ---------------------------------------------------------------------------
# Welch PSD & Peak Extraction Tests
# ---------------------------------------------------------------------------

def test_compute_axis_psd_synthetic_pure_sine():
    """Verify Welch PSD computation and sub-bin accuracy on a pure sinusoid."""
    fs = 100.0
    t = np.arange(0.0, 10.0, 1.0 / fs)
    target_freq = 7.35  # non-integer frequency to test sub-bin interpolation
    sig = 2.0 * np.sin(2.0 * np.pi * target_freq * t)

    freqs, psd, peak_f, peak_p = compute_axis_psd(sig, fs=fs, min_freq_hz=0.5, max_freq_hz=30.0)

    assert np.isclose(peak_f, target_freq, atol=0.05), f"Expected {target_freq} Hz, got {peak_f:.4f} Hz"
    assert peak_p > 0.5


def test_find_dominant_accelerometer_peak_sample_fixture():
    """Verify that find_dominant_accelerometer_peak picks y-axis with 12 Hz peak from fixture."""
    _, axes, fs = load_phyphox_csv(FIXTURE_PATH)
    dom_axis, dom_spectrum, all_spectra = find_dominant_accelerometer_peak(axes, fs=fs)

    # In phyphox_sample.csv, y has dominant 12 Hz oscillation
    assert dom_axis == "y"
    assert dom_spectrum.axis == "y"
    assert np.isclose(dom_spectrum.peak_frequency_hz, 12.0, atol=0.1), (
        f"Expected peak at 12.0 Hz, got {dom_spectrum.peak_frequency_hz:.3f} Hz"
    )
    # y peak power must be substantially greater than x and z
    assert dom_spectrum.peak_power > all_spectra["x"].peak_power * 5.0


# ---------------------------------------------------------------------------
# Camera Frequency Extraction Tests
# ---------------------------------------------------------------------------

def test_extract_camera_frequency_various_sources(tmp_path: Path):
    """Test extracting camera frequency from float, numeric string, and simple JSON."""
    # 1. Direct float
    f1, s1 = extract_camera_frequency(12.0)
    assert f1 == 12.0
    assert s1 is None

    # 2. String number
    f2, s2 = extract_camera_frequency("12.05")
    assert f2 == 12.05
    assert s2 is None

    # 3. Simple JSON with top-level frequency_hz only
    json_path = tmp_path / "simple_res.json"
    json_path.write_text(json.dumps({"frequency_hz": 12.0}), encoding="utf-8")
    f3, ci3 = extract_camera_frequency(json_path)
    assert f3 == 12.0
    assert ci3 is None

    # 4. Simple JSON with top-level frequency_hz and frequency_ci
    json_ci_path = tmp_path / "ci_res.json"
    json_ci_path.write_text(
        json.dumps({"frequency_hz": 12.0, "frequency_ci": [11.95, 12.05]}),
        encoding="utf-8",
    )
    f4, ci4 = extract_camera_frequency(json_ci_path)
    assert f4 == 12.0
    assert ci4 == (11.95, 12.05)

    # 5. Direct dict with frequency_hz and frequency_ci
    f5, ci5 = extract_camera_frequency({"frequency_hz": 11.98, "frequency_ci": [11.9, 12.1]})
    assert f5 == 11.98
    assert ci5 == (11.9, 12.1)

    # 6. Reject old/nested result formats (AnalysisResult / ScanResult without top-level frequency_hz)
    old_analysis = {
        "spectrum": {
            "fundamental_frequency_hz": 12.0,
            "peaks": [{"frequency_hz": 12.0}],
        }
    }
    with pytest.raises(ValueError, match="missing required top-level 'frequency_hz'"):
        extract_camera_frequency(old_analysis)

    # 7. Invalid empty dict raises ValueError
    with pytest.raises(ValueError, match="missing required top-level 'frequency_hz'"):
        extract_camera_frequency({})


# ---------------------------------------------------------------------------
# Comparison & Agreement Verdict Tests
# ---------------------------------------------------------------------------

def test_agreement_matching_camera_frequency():
    """Verify that a matching camera frequency yields agreement PASS."""
    # Accelerometer peak is ~12.0 Hz from fixture, camera is 12.0 Hz
    res = compare_phyphox_to_camera(
        phyphox_csv=FIXTURE_PATH,
        camera_source=12.0,
        tolerance_rel_pct=5.0,
        tolerance_abs_hz=0.30,
    )

    assert res.is_agreement is True
    assert res.verdict == "PASS"
    assert res.absolute_diff_hz < 0.1
    assert res.relative_diff_pct < 1.0
    assert res.dominant_axis == "y"
    assert "Modal frequency agreement" in res.tolerance_justification


def test_agreement_mismatching_camera_frequency():
    """Verify that a mismatching camera frequency (e.g. 15.0 Hz vs 12.0 Hz) yields FAIL."""
    res = compare_phyphox_to_camera(
        phyphox_csv=FIXTURE_PATH,
        camera_source=15.0,
        tolerance_rel_pct=5.0,
        tolerance_abs_hz=0.30,
    )

    assert res.is_agreement is False
    assert res.verdict == "FAIL"
    assert np.isclose(res.absolute_diff_hz, 3.0, atol=0.1)
    assert res.relative_diff_pct > 20.0  # 3.0 / 12.0 = 25%


def test_compare_with_result_json_matching(tmp_path: Path):
    """Verify comparison using a simple result JSON with frequency_hz and frequency_ci."""
    result_json = tmp_path / "setu_output.json"
    result_data = {
        "frequency_hz": 12.0,
        "frequency_ci": [11.95, 12.05],
    }
    result_json.write_text(json.dumps(result_data), encoding="utf-8")

    res = compare_phyphox_to_camera(
        phyphox_csv=FIXTURE_PATH,
        camera_source=result_json,
    )

    assert res.is_agreement is True
    assert res.verdict == "PASS"
    assert res.camera_peak_hz == 12.0
    assert res.camera_ci == (11.95, 12.05)
    assert np.isclose(res.camera_ci_half_width, 0.05)
    assert res.combined_uncertainty >= 0.05
    assert res.uncertainty_note is None


def test_estimate_accelerometer_uncertainty_overlapping_segments():
    """Verify accelerometer CI estimation via overlapping segments (95% Student's t)."""
    _, axes, fs = load_phyphox_csv(FIXTURE_PATH)
    dom_axis, dom_spectrum, _ = find_dominant_accelerometer_peak(axes, fs=fs)

    uncert = estimate_accelerometer_uncertainty(
        signal_arr=axes[dom_axis],
        fs=fs,
        center_freq_hz=dom_spectrum.peak_frequency_hz,
        n_segments=5,
        overlap=0.5,
    )

    assert uncert["n_segments"] == 5
    assert uncert["ci_half_width"] > 0.0
    assert uncert["ci_lower"] <= dom_spectrum.peak_frequency_hz <= uncert["ci_upper"]
    assert "Student's t" in uncert["method"]


def test_camera_without_ci_uses_accelerometer_alone():
    """Verify that when camera has no CI (--freq given), accelerometer uncertainty alone is used."""
    res = compare_phyphox_to_camera(
        phyphox_csv=FIXTURE_PATH,
        camera_source=12.0,
    )

    assert res.camera_ci is None
    assert res.camera_ci_half_width is None
    assert res.combined_uncertainty == res.accel_ci_half_width
    assert res.uncertainty_note is not None
    assert "accelerometer uncertainty alone" in res.uncertainty_note


# ---------------------------------------------------------------------------
# Plot Generation Tests
# ---------------------------------------------------------------------------

def test_plot_generation(tmp_path: Path):
    """Verify plot generation produces a valid, non-empty PNG file with labeled peaks."""
    out_plot = tmp_path / "comparison_plot.png"

    res = compare_phyphox_to_camera(
        phyphox_csv=FIXTURE_PATH,
        camera_source=12.0,
        output_png=out_plot,
    )

    assert res.plot_path == str(out_plot)
    assert out_plot.exists()
    assert out_plot.stat().st_size > 5000  # PNG image must be non-trivial

    # Verify PNG header
    with open(out_plot, "rb") as f:
        header = f.read(8)
    assert header == b"\x89PNG\r\n\x1a\n", "Generated plot is not a valid PNG file"


# ---------------------------------------------------------------------------
# Command Line Interface (CLI) Execution Tests
# ---------------------------------------------------------------------------

def test_cli_execution_matching(tmp_path: Path, capsys: pytest.CaptureFixture):
    """Verify CLI execution with matching frequency flag."""
    out_png = tmp_path / "cli_plot.png"
    test_args = [
        "validation.compare",
        "--phyphox",
        str(FIXTURE_PATH),
        "--freq",
        "12.0",
        "--out",
        str(out_png),
    ]

    with patch.object(sys, "argv", test_args):
        exit_code = main()

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "PASS (Passes validation criteria)" in captured.out
    assert "12.000 Hz" in captured.out
    assert out_png.exists()


def test_cli_execution_mismatching(tmp_path: Path, capsys: pytest.CaptureFixture):
    """Verify CLI execution with mismatching frequency flag."""
    out_png = tmp_path / "cli_mismatch_plot.png"
    test_args = [
        "validation.compare",
        "--phyphox",
        str(FIXTURE_PATH),
        "--freq",
        "15.0",
        "--out",
        str(out_png),
    ]

    with patch.object(sys, "argv", test_args):
        exit_code = main()

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "FAIL (Exceeds tolerance margin)" in captured.out
    assert "25.00%" in captured.out
    assert out_png.exists()


def test_cli_writes_proof_json_with_normalized_bounded_spectra(tmp_path: Path):
    """Proof JSON contains real peaks, confidence intervals, and normalized spectra."""
    out_png = tmp_path / "proof_plot.png"
    out_json = tmp_path / "proof.json"
    camera_freqs = np.linspace(0.0, 35.0, 701)
    camera_power = np.exp(-((camera_freqs - 12.0) / 0.4) ** 2)
    result_json = tmp_path / "camera_result.json"
    result_json.write_text(json.dumps({
        "frequency_hz": 12.0,
        "frequency_ci": [11.9, 12.1],
        "measurement": {"psd": {"freq_hz": camera_freqs.tolist(), "power": camera_power.tolist()}},
    }), encoding="utf-8")

    test_args = ["validation.compare", "--phyphox", str(FIXTURE_PATH), "--result", str(result_json), "--out", str(out_png), "--json", str(out_json)]
    with patch.object(sys, "argv", test_args):
        assert main() == 0

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["camera_frequency_hz"] == 12.0
    assert payload["camera_frequency_ci_hz"] == [11.9, 12.1]
    assert payload["accelerometer_frequency_ci_hz"] is not None
    assert payload["difference_hz"] >= 0
    assert payload["difference_pct"] >= 0
    assert payload["agreement"] is True
    assert payload["agreement_verdict"] == "PASS"
    for key in ("camera_spectrum", "accelerometer_spectrum"):
        spectrum = payload[key]
        assert spectrum["available"] is True
        assert 1 <= len(spectrum["points"]) <= 300
        assert max(point["frequency_hz"] for point in spectrum["points"]) <= 30.0
        assert max(point["normalized_power"] for point in spectrum["points"]) == pytest.approx(1.0)
        assert all(0 <= point["normalized_power"] <= 1 for point in spectrum["points"])


def test_validation_json_does_not_invent_camera_spectrum_for_frequency_only_input():
    res = compare_phyphox_to_camera(phyphox_csv=FIXTURE_PATH, camera_source=12.0)
    payload = comparison_validation_dict(res)
    assert payload["camera_frequency_ci_hz"] is None
    assert payload["camera_spectrum"] == {"available": False, "points": []}
    assert payload["accelerometer_spectrum"]["available"] is True


def test_cli_execution_with_result_json(tmp_path: Path, capsys: pytest.CaptureFixture):
    """Verify CLI execution with simple JSON --result flag."""
    out_png = tmp_path / "cli_result_plot.png"
    res_json = tmp_path / "result.json"
    res_json.write_text(json.dumps({"frequency_hz": 12.0, "frequency_ci": [11.95, 12.05]}), encoding="utf-8")

    test_args = [
        "validation.compare",
        "--phyphox",
        str(FIXTURE_PATH),
        "--result",
        str(res_json),
        "--out",
        str(out_png),
    ]

    with patch.object(sys, "argv", test_args):
        exit_code = main()

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "PASS (Passes validation criteria)" in captured.out
    assert "12.000 Hz" in captured.out
    assert "Percentage Difference:" in captured.out
    assert out_png.exists()
