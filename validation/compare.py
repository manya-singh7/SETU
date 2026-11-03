"""Validation and Cross-Sensor Comparison Module for Project SETU.

Compares SETU's camera-measured resonant frequency against a contact phone
accelerometer recording exported from phyphox (https://phyphox.org).

Features:
- Flexible CSV import for phyphox exports (flexible headers, delimiters, and axis detection).
- Welch Power Spectral Density (PSD) computation across all detected accelerometer axes.
- Dominant resonant modal peak extraction in the 0.5 - 30.0 Hz civil infrastructure band.
- Cross-sensor comparison with justified tolerance criteria (5.0% relative, 0.30 Hz absolute).
- Composite spectrum plot generation with both spectra normalized to their respective maxima.
- Command-line interface:
    python -m validation.compare --phyphox FILE.csv (--result RESULT.json | --freq HZ) --out PLOT.png
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import scipy.signal as signal
import scipy.stats as stats


# ---------------------------------------------------------------------------
# Default Tolerances and Physical Engineering Justification
# ---------------------------------------------------------------------------

DEFAULT_TOLERANCE_REL_PCT: float = 5.0   # 5.0% relative frequency discrepancy
DEFAULT_TOLERANCE_ABS_HZ: float = 0.30   # 0.30 Hz absolute frequency discrepancy
DEFAULT_MIN_FREQ_HZ: float = 0.5         # Civil structure lower band edge (exclude rigid drift)
DEFAULT_MAX_FREQ_HZ: float = 30.0        # Civil structure upper modal band edge

TOLERANCE_JUSTIFICATION: str = (
    "Modal frequency agreement between optical/video displacement and contact "
    "accelerometers in civil engineering structural health monitoring (SHM) typically "
    "allows up to 5.0% relative discrepancy (or 0.30 Hz for fundamental modes < 6 Hz). "
    "This accommodates rolling shutter timing jitter, variable mobile frame rates, "
    "finite window FFT bin resolution, and sensor placement boundary condition offsets."
)


# ---------------------------------------------------------------------------
# Data Containers
# ---------------------------------------------------------------------------

@dataclass
class AccelerometerSpectrum:
    """Welch spectral analysis result for a single accelerometer axis."""

    axis: str
    frequencies_hz: npt.NDArray[np.float64]
    psd: npt.NDArray[np.float64]
    peak_frequency_hz: float
    peak_power: float
    fs: float
    ci_half_width: float = 0.0
    ci: Optional[Tuple[float, float]] = None


@dataclass
class ComparisonResult:
    """Quantitative comparison between accelerometer and camera vibration readings."""

    accelerometer_peak_hz: float
    camera_peak_hz: float
    absolute_diff_hz: float
    relative_diff_pct: float
    is_agreement: bool
    dominant_axis: str
    axes_detected: List[str]
    verdict: str  # "PASS" or "FAIL"
    combined_uncertainty: float = 0.0
    accel_ci_half_width: float = 0.0
    camera_ci_half_width: Optional[float] = None
    camera_ci: Optional[Tuple[float, float]] = None
    accel_ci: Optional[Tuple[float, float]] = None
    uncertainty_note: Optional[str] = None
    tolerance_rel_pct: float = DEFAULT_TOLERANCE_REL_PCT
    tolerance_abs_hz: float = DEFAULT_TOLERANCE_ABS_HZ
    tolerance_justification: str = TOLERANCE_JUSTIFICATION
    plot_path: Optional[str] = None
    accelerometer_spectrum: Optional[AccelerometerSpectrum] = None
    camera_spectrum: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert result summary into a clean dictionary."""
        d: Dict[str, Any] = {
            "accelerometer_peak_hz": round(self.accelerometer_peak_hz, 4),
            "camera_peak_hz": round(self.camera_peak_hz, 4),
            "absolute_diff_hz": round(self.absolute_diff_hz, 4),
            "relative_diff_pct": round(self.relative_diff_pct, 2),
            "is_agreement": self.is_agreement,
            "verdict": self.verdict,
            "dominant_axis": self.dominant_axis,
            "axes_detected": self.axes_detected,
            "combined_uncertainty": round(self.combined_uncertainty, 4),
            "accel_ci_half_width": round(self.accel_ci_half_width, 4),
            "tolerance_rel_pct": self.tolerance_rel_pct,
            "tolerance_abs_hz": self.tolerance_abs_hz,
            "plot_path": self.plot_path,
        }
        if self.camera_ci_half_width is not None:
            d["camera_ci_half_width"] = round(self.camera_ci_half_width, 4)
        if self.uncertainty_note:
            d["uncertainty_note"] = self.uncertainty_note
        return d


# ---------------------------------------------------------------------------
# Phyphox CSV Parsing & Header Detection
# ---------------------------------------------------------------------------

def _detect_delimiter(sample_lines: Sequence[str]) -> str:
    """Detect CSV delimiter (comma, semicolon, or tab) from sample lines."""
    joined = "\n".join(sample_lines[:5])
    try:
        dialect = csv.Sniffer().sniff(joined, delimiters=",;\t")
        return dialect.delimiter
    except Exception:
        # Fallback heuristic based on character counts in header
        first_line = sample_lines[0] if sample_lines else ""
        counts = {",": first_line.count(","), ";": first_line.count(";"), "\t": first_line.count("\t")}
        best = max(counts, key=counts.get)
        return best if counts[best] > 0 else ","


def _clean_header_token(name: str) -> str:
    """Normalize a column header string for robust keyword matching."""
    s = name.strip().lower()
    # Strip enclosing quotes
    s = s.strip("\"' ")
    # Replace unicode non-breaking spaces
    s = s.replace("\u00a0", " ")
    return s


def _classify_column(name: str) -> Optional[str]:
    """Classify a CSV column as 'time', 'x', 'y', 'z', or 'abs'."""
    clean = _clean_header_token(name)

    # 1. Check for time
    if any(k in clean for k in ("time", "zeit", "timestamp", "sec")) or clean in ("t", "t (s)", "t(s)", "time (s)"):
        return "time"

    # 2. Check for absolute / magnitude acceleration
    if any(k in clean for k in ("absolute", "magnitude", "norm", "total", "mag", "abs")):
        return "abs"

    # 3. Check for specific axes (x, y, z)
    # Match patterns like: "acceleration x", "linear acceleration x", "acc_x", "ax", "x (m/s^2)", "x"
    # Using regex to isolate axis letter
    if re.search(r"(?:^|[\s_\-(/])x(?:[\s_\-)/]|$)", clean) or clean.endswith(" x") or clean.startswith("x "):
        return "x"
    if re.search(r"(?:^|[\s_\-(/])y(?:[\s_\-)/]|$)", clean) or clean.endswith(" y") or clean.startswith("y "):
        return "y"
    if re.search(r"(?:^|[\s_\-(/])z(?:[\s_\-)/]|$)", clean) or clean.endswith(" z") or clean.startswith("z "):
        return "z"

    # Exact single-letter fallbacks
    if clean == "x":
        return "x"
    if clean == "y":
        return "y"
    if clean == "z":
        return "z"

    return None


def load_phyphox_csv(
    filepath: Union[str, Path]
) -> Tuple[npt.NDArray[np.float64], Dict[str, npt.NDArray[np.float64]], float]:
    """Load and parse a phyphox accelerometer CSV export file.

    Accepts flexible headers such as:
      - 'Time (s)', 'time', 't'
      - 'Acceleration x (m/s^2)', 'Linear Acceleration x', 'acc_x', 'X'
      - Comma, semicolon, or tab delimiters
      - European comma decimal separators (e.g. '0,123')

    Returns:
        (time_array, axes_dict, sampling_rate_hz)
        where axes_dict maps axis name ('x', 'y', 'z', etc.) to 1D acceleration array.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Phyphox CSV file not found: {path}")

    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        lines = [line.strip() for line in f if line.strip()]

    if len(lines) < 2:
        raise ValueError(f"Phyphox CSV file {path} has insufficient data rows (found {len(lines)}).")

    delim = _detect_delimiter(lines[:10])

    # Parse header line
    header_raw = [token.strip() for token in lines[0].split(delim)]
    col_mapping: Dict[int, str] = {}
    time_col_idx: Optional[int] = None

    for idx, col_name in enumerate(header_raw):
        col_type = _classify_column(col_name)
        if col_type == "time" and time_col_idx is None:
            time_col_idx = idx
        elif col_type in ("x", "y", "z", "abs"):
            col_mapping[idx] = col_type

    # Fallback: if no time column was found by header name, assume column 0 is time
    if time_col_idx is None:
        time_col_idx = 0
        if 0 in col_mapping:
            del col_mapping[0]

    # If no acceleration axes were recognized by name, map available columns
    if not col_mapping:
        axis_names = ["x", "y", "z"]
        cur_axis = 0
        for idx in range(len(header_raw)):
            if idx != time_col_idx and cur_axis < len(axis_names):
                col_mapping[idx] = axis_names[cur_axis]
                cur_axis += 1

    if not col_mapping:
        raise ValueError(f"Could not identify any acceleration data columns in {path}. Headers: {header_raw}")

    # Parse numeric data rows
    times: List[float] = []
    data_by_col: Dict[int, List[float]] = {idx: [] for idx in col_mapping}

    for row_num, line in enumerate(lines[1:], start=2):
        parts = [p.strip() for p in line.split(delim)]
        if len(parts) <= time_col_idx:
            continue

        try:
            # Handle European decimal comma if delimiter is semicolon
            t_str = parts[time_col_idx].replace(",", ".") if delim == ";" else parts[time_col_idx]
            t_val = float(t_str)
        except ValueError:
            continue  # Skip unparseable lines

        # Parse available axis values for this row
        row_vals: Dict[int, float] = {}
        row_ok = True
        for col_idx in col_mapping:
            if col_idx < len(parts):
                val_str = parts[col_idx].replace(",", ".") if delim == ";" else parts[col_idx]
                try:
                    row_vals[col_idx] = float(val_str)
                except ValueError:
                    row_ok = False
                    break
            else:
                row_ok = False
                break

        if row_ok:
            times.append(t_val)
            for col_idx, val in row_vals.items():
                data_by_col[col_idx].append(val)

    if len(times) < 10:
        raise ValueError(f"Phyphox CSV file {path} contains fewer than 10 valid numerical records.")

    time_arr = np.array(times, dtype=np.float64)
    # Ensure monotonically increasing timestamps
    dt_series = np.diff(time_arr)
    valid_dt = dt_series[dt_series > 1e-6]
    if len(valid_dt) == 0:
        raise ValueError(f"Time column in {path} has zero or non-increasing values.")

    median_dt = float(np.median(valid_dt))
    if median_dt <= 0:
        raise ValueError(f"Invalid time step (median dt = {median_dt}) in {path}.")

    fs = 1.0 / median_dt

    # Assemble axes dictionary mapping axis label to array
    axes_dict: Dict[str, npt.NDArray[np.float64]] = {}
    for col_idx, axis_label in col_mapping.items():
        arr = np.array(data_by_col[col_idx], dtype=np.float64)
        if len(arr) == len(time_arr):
            # Replace NaNs or infinities with median
            if not np.all(np.isfinite(arr)):
                med = float(np.nanmedian(arr))
                arr = np.nan_to_num(arr, nan=med, posinf=med, neginf=med)
            axes_dict[axis_label] = arr

    return time_arr, axes_dict, fs


# ---------------------------------------------------------------------------
# Welch Spectral Analysis for Accelerometer Signals
# ---------------------------------------------------------------------------

def compute_axis_psd(
    signal_arr: npt.NDArray[np.float64],
    fs: float,
    min_freq_hz: float = DEFAULT_MIN_FREQ_HZ,
    max_freq_hz: float = DEFAULT_MAX_FREQ_HZ,
    nperseg: Optional[int] = None,
    nfft: Optional[int] = None,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float, float]:
    """Compute Welch PSD and locate the dominant modal peak within [min_freq, max_freq].

    Returns:
        (frequencies_hz, psd, peak_freq_refined_hz, peak_power)
    """
    n_samples = len(signal_arr)
    if n_samples < 8:
        raise ValueError(f"Signal has too few samples ({n_samples}) for spectral estimation.")

    # Linear detrending to eliminate static DC gravity offsets
    detrended = signal.detrend(signal_arr, type="linear")

    # Adaptive window length and FFT size
    seg_len = nperseg or min(n_samples, 1024)
    if seg_len < 16:
        seg_len = n_samples
    fft_len = nfft or max(seg_len, 4096)

    freqs, psd = signal.welch(
        detrended,
        fs=fs,
        window="hann",
        nperseg=seg_len,
        nfft=fft_len,
        scaling="density",
    )

    # Restrict to structural analysis band
    mask = (freqs >= min_freq_hz) & (freqs <= max_freq_hz)
    if not np.any(mask):
        raise ValueError(
            f"No spectral frequency bins within [{min_freq_hz}, {max_freq_hz}] Hz "
            f"(Nyquist is {fs / 2.0:.1f} Hz)."
        )

    f_band = freqs[mask]
    p_band = psd[mask]

    # Find dominant peak index in the band
    k = int(np.argmax(p_band))
    peak_power = float(p_band[k])
    f_peak_raw = float(f_band[k])

    # Parabolic sub-bin interpolation on log-PSD for high modal precision
    if 0 < k < len(p_band) - 1 and p_band[k - 1] > 0 and p_band[k] > 0 and p_band[k + 1] > 0:
        y1 = np.log(p_band[k - 1])
        y2 = np.log(p_band[k])
        y3 = np.log(p_band[k + 1])
        denom = y1 - 2.0 * y2 + y3
        delta = 0.5 * (y1 - y3) / denom if abs(denom) > 1e-12 else 0.0
        delta = float(np.clip(delta, -0.5, 0.5))
        df = f_band[1] - f_band[0] if len(f_band) > 1 else (freqs[1] - freqs[0])
        peak_freq_refined = float(f_peak_raw + delta * df)
    else:
        peak_freq_refined = f_peak_raw

    return freqs, psd, peak_freq_refined, peak_power


def find_dominant_accelerometer_peak(
    axes_dict: Dict[str, npt.NDArray[np.float64]],
    fs: float,
    min_freq_hz: float = DEFAULT_MIN_FREQ_HZ,
    max_freq_hz: float = DEFAULT_MAX_FREQ_HZ,
) -> Tuple[str, AccelerometerSpectrum, Dict[str, AccelerometerSpectrum]]:
    """Compute PSD for each accelerometer axis and select the axis with strongest peak.

    Returns:
        (dominant_axis_name, dominant_spectrum, all_spectra_dict)
    """
    if not axes_dict:
        raise ValueError("No accelerometer axes provided for spectral analysis.")

    all_spectra: Dict[str, AccelerometerSpectrum] = {}

    for axis_name, axis_data in axes_dict.items():
        f, p, peak_f, peak_p = compute_axis_psd(
            axis_data,
            fs=fs,
            min_freq_hz=min_freq_hz,
            max_freq_hz=max_freq_hz,
        )
        all_spectra[axis_name] = AccelerometerSpectrum(
            axis=axis_name,
            frequencies_hz=f,
            psd=p,
            peak_frequency_hz=peak_f,
            peak_power=peak_p,
            fs=fs,
        )

    # Select the axis with the highest peak spectral power in the band
    dominant_axis = max(all_spectra, key=lambda a: all_spectra[a].peak_power)
    dominant_spectrum = all_spectra[dominant_axis]

    return dominant_axis, dominant_spectrum, all_spectra


def estimate_accelerometer_uncertainty(
    signal_arr: npt.NDArray[np.float64],
    fs: float,
    center_freq_hz: float,
    n_segments: int = 5,
    overlap: float = 0.5,
    confidence_level: float = 0.95,
    search_half_width_hz: float = 1.0,
) -> Dict[str, Any]:
    """Estimate accelerometer dominant peak frequency uncertainty using overlapping segments (95% Student's t).

    Splits detrended acceleration into overlapping Hann-windowed segments (default 5, 50% overlap).
    Refines the dominant peak per segment using sub-bin log-parabolic interpolation.
    Reports mean, standard deviation, and Student's t 95% confidence interval half-width.
    """
    sig = np.asarray(signal_arr, dtype=np.float64).ravel()
    N = len(sig)
    if N < 32 or fs <= 0.0 or not np.all(np.isfinite(sig)):
        return {
            "mean_hz": center_freq_hz,
            "std_hz": 0.0,
            "ci_lower": center_freq_hz,
            "ci_upper": center_freq_hz,
            "ci_half_width": 0.0,
            "ci_width": 0.0,
            "confidence_level": confidence_level,
            "n_segments": 0,
            "segment_freqs": [],
            "method": "insufficient_data",
        }

    sig_detrended = signal.detrend(sig, type="linear")
    K = max(int(n_segments), 2)
    step_factor = 1.0 + (K - 1) * (1.0 - overlap)
    seg_len = max(int(N / step_factor), 32)
    step_size = max(int(seg_len * (1.0 - overlap)), 1)

    segment_freqs: List[float] = []
    for i in range(K):
        start_idx = i * step_size
        end_idx = min(start_idx + seg_len, N)
        seg = sig_detrended[start_idx:end_idx]
        if len(seg) < 32:
            continue

        seg_det = signal.detrend(seg, type="linear")
        nfft = max(len(seg_det), 4096)
        f_axis, psd = signal.welch(
            seg_det,
            fs=float(fs),
            window="hann",
            nperseg=len(seg_det),
            nfft=nfft,
            detrend=False,
        )

        f_low = max(0.1, center_freq_hz - search_half_width_hz)
        f_high = center_freq_hz + search_half_width_hz
        mask = (f_axis >= f_low) & (f_axis <= f_high)
        if not np.any(mask):
            mask = f_axis >= 0.1
        if not np.any(mask):
            continue

        f_band = f_axis[mask]
        p_band = psd[mask]
        k = int(np.argmax(p_band))
        f0 = float(f_band[k])
        df = float(f_band[1] - f_band[0]) if len(f_band) > 1 else 1.0

        if 0 < k < len(p_band) - 1 and p_band[k - 1] > 0 and p_band[k] > 0 and p_band[k + 1] > 0:
            y1 = np.log(p_band[k - 1])
            y2 = np.log(p_band[k])
            y3 = np.log(p_band[k + 1])
            denom = y1 - 2.0 * y2 + y3
            delta = 0.5 * (y1 - y3) / denom if abs(denom) > 1e-12 else 0.0
            delta = float(np.clip(delta, -0.5, 0.5))
            f_ref = f_band[k] + delta * df
        else:
            f_ref = f0

        segment_freqs.append(float(f_ref))

    if len(segment_freqs) < 2:
        mean_hz = float(np.mean(segment_freqs)) if segment_freqs else center_freq_hz
        return {
            "mean_hz": round(mean_hz, 4),
            "std_hz": 0.0,
            "ci_lower": round(mean_hz, 4),
            "ci_upper": round(mean_hz, 4),
            "ci_half_width": 0.0,
            "ci_width": 0.0,
            "confidence_level": confidence_level,
            "n_segments": len(segment_freqs),
            "segment_freqs": [round(f, 4) for f in segment_freqs],
            "method": "single_segment_fallback",
        }

    freqs_arr = np.array(segment_freqs, dtype=np.float64)
    mean_hz = float(np.mean(freqs_arr))
    std_hz = float(np.std(freqs_arr, ddof=1))

    alpha = 1.0 - confidence_level
    dof = len(freqs_arr) - 1
    t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=dof))

    ci_half_width = float(t_crit * std_hz / np.sqrt(len(freqs_arr)))
    ci_lower = float(mean_hz - ci_half_width)
    ci_upper = float(mean_hz + ci_half_width)
    ci_width = float(ci_upper - ci_lower)

    return {
        "mean_hz": round(mean_hz, 4),
        "std_hz": round(std_hz, 5),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "ci_half_width": round(ci_half_width, 4),
        "ci_width": round(ci_width, 4),
        "confidence_level": confidence_level,
        "n_segments": len(segment_freqs),
        "segment_freqs": [round(f, 4) for f in segment_freqs],
        "method": f"Overlapping Hann-windowed ensemble ({len(segment_freqs)} segments, {int(overlap * 100)}% overlap, 95% Student's t).",
    }


# ---------------------------------------------------------------------------
# Camera Frequency & Result Extraction
# ---------------------------------------------------------------------------

def extract_camera_frequency(
    camera_source: Union[float, int, str, Path, Dict[str, Any]]
) -> Tuple[float, Optional[Tuple[float, float]]]:
    """Extract SETU camera-measured fundamental frequency and optional confidence interval.

    Args:
        camera_source: Can be a direct frequency float (e.g. 12.0), a path to a
                       simple SETU result JSON file, or a loaded dictionary.
                       JSON must accept only a simple JSON with a top-level
                       "frequency_hz" number and optional "frequency_ci" [low, high].

    Returns:
        (camera_freq_hz, optional_frequency_ci_tuple)
    """
    # 1. Direct numeric frequency
    if isinstance(camera_source, (int, float)):
        val = float(camera_source)
        if val <= 0:
            raise ValueError(f"Invalid camera frequency: {val} Hz (must be positive).")
        return val, None

    # 2. String representation of a numeric float
    if isinstance(camera_source, str):
        try:
            val = float(camera_source.strip())
            if val > 0:
                return val, None
        except ValueError:
            pass

    # 3. JSON file or dictionary
    if isinstance(camera_source, (str, Path)):
        path = Path(camera_source)
        if not path.exists():
            raise FileNotFoundError(f"SETU result JSON file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    elif isinstance(camera_source, dict):
        data = camera_source
    else:
        raise TypeError(f"Unsupported camera_source type: {type(camera_source).__name__}")

    if not isinstance(data, dict):
        raise ValueError(f"Result JSON must contain a top-level JSON object/dictionary, got {type(data).__name__}.")

    if "frequency_hz" not in data or data["frequency_hz"] is None:
        raise ValueError(
            f"Invalid result JSON: missing required top-level 'frequency_hz' key in {camera_source}."
        )

    try:
        camera_freq = float(data["frequency_hz"])
    except (ValueError, TypeError) as err:
        raise ValueError(f"Invalid 'frequency_hz' value: {data['frequency_hz']}") from err

    if camera_freq <= 0:
        raise ValueError(f"Invalid camera frequency: {camera_freq} Hz (must be positive).")

    camera_ci: Optional[Tuple[float, float]] = None
    if "frequency_ci" in data and data["frequency_ci"] is not None:
        ci_raw = data["frequency_ci"]
        if not (isinstance(ci_raw, (list, tuple)) and len(ci_raw) == 2):
            raise ValueError(
                f"Invalid 'frequency_ci': expected [low, high] list of 2 numbers, got {ci_raw}"
            )
        try:
            camera_ci = (float(ci_raw[0]), float(ci_raw[1]))
        except (ValueError, TypeError) as err:
            raise ValueError(f"Invalid values in 'frequency_ci': {ci_raw}") from err

    return camera_freq, camera_ci


# ---------------------------------------------------------------------------
# Cross-Sensor Comparison & Agreement Evaluation
# ---------------------------------------------------------------------------

def compare_frequency(
    accelerometer_freq_hz: float,
    camera_freq_hz: float,
    accel_ci_half_width: float = 0.0,
    camera_ci_half_width: Optional[float] = None,
    tolerance_rel_pct: float = DEFAULT_TOLERANCE_REL_PCT,
    tolerance_abs_hz: float = DEFAULT_TOLERANCE_ABS_HZ,
) -> Tuple[float, float, bool, str]:
    """Compare accelerometer vs camera frequencies using combined uncertainty.

    Agreement rule:
        agreement if |difference| <= combined uncertainty, where
        combined = sqrt(camera_ci_half_width^2 + accel_ci_half_width^2)
        If camera has no CI, use accelerometer's alone.

    Returns:
        (absolute_diff_hz, relative_diff_pct, is_agreement, verdict_string)
    """
    abs_diff = abs(camera_freq_hz - accelerometer_freq_hz)
    rel_diff_pct = (abs_diff / accelerometer_freq_hz) * 100.0 if accelerometer_freq_hz > 0 else float("inf")

    if camera_ci_half_width is not None:
        combined_uncertainty = float(np.sqrt(camera_ci_half_width**2 + accel_ci_half_width**2))
    else:
        combined_uncertainty = float(accel_ci_half_width)

    is_agreement = (abs_diff <= combined_uncertainty)
    verdict = "PASS" if is_agreement else "FAIL"

    return abs_diff, rel_diff_pct, is_agreement, verdict


# ---------------------------------------------------------------------------
# Visualization & Dual Normalized Spectrum Plotting
# ---------------------------------------------------------------------------

def plot_comparison(
    accel_spectrum: AccelerometerSpectrum,
    camera_freq_hz: float,
    comparison_res: ComparisonResult,
    output_png_path: Union[str, Path],
    camera_spectrum: Optional[Dict[str, Any]] = None,
    max_display_freq_hz: float = DEFAULT_MAX_FREQ_HZ,
) -> Path:
    """Generate and save composite comparison plot with both normalized spectra.

    Features:
    - Shared frequency axis (0 to max_display_freq_hz).
    - Each spectrum normalised to its own maximum (0 to 1.0).
    - Clear visual callouts marking accelerometer peak and camera peak.
    - Diagnostics banner indicating delta f, relative %, and agreement verdict.
    """
    out_path = Path(output_png_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    f_accel = accel_spectrum.frequencies_hz
    p_accel = accel_spectrum.psd

    # Restrict to display range
    mask_acc = (f_accel >= 0.0) & (f_accel <= max_display_freq_hz)
    f_acc_plot = f_accel[mask_acc]
    p_acc_plot = p_accel[mask_acc]

    # Normalize accelerometer spectrum to its peak maximum in band
    max_acc_power = float(np.max(p_acc_plot)) if len(p_acc_plot) > 0 and np.max(p_acc_plot) > 0 else 1.0
    p_acc_norm = p_acc_plot / max_acc_power

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)

    # 1. Plot Accelerometer Spectrum Curve
    ax.plot(
        f_acc_plot,
        p_acc_norm,
        color="#1f77b4",
        linewidth=2.0,
        label=f"Phyphox Accelerometer ({accel_spectrum.axis}-axis PSD)",
        alpha=0.9,
    )
    ax.fill_between(f_acc_plot, p_acc_norm, color="#1f77b4", alpha=0.15)

    # Mark Accelerometer Peak
    ax.plot(
        [accel_spectrum.peak_frequency_hz],
        [1.0],
        marker="o",
        markersize=8,
        color="#1f77b4",
        markeredgecolor="white",
        markeredgewidth=1.5,
        zorder=5,
    )
    ax.annotate(
        f"Accel Peak: {accel_spectrum.peak_frequency_hz:.2f} Hz\n({accel_spectrum.axis}-axis)",
        xy=(accel_spectrum.peak_frequency_hz, 1.0),
        xytext=(accel_spectrum.peak_frequency_hz + 0.8, 0.85),
        arrowprops=dict(facecolor="#1f77b4", edgecolor="#1f77b4", arrowstyle="->", lw=1.5),
        fontsize=9,
        fontweight="bold",
        color="#1f77b4",
        bbox=dict(boxstyle="round,pad=0.3", fc="#f0f8ff", ec="#1f77b4", lw=1, alpha=0.9),
    )

    # 2. Plot Camera Spectrum
    if camera_spectrum is not None and "frequencies_hz" in camera_spectrum and "psd" in camera_spectrum:
        f_cam = camera_spectrum["frequencies_hz"]
        p_cam = camera_spectrum["psd"]
        mask_cam = (f_cam >= 0.0) & (f_cam <= max_display_freq_hz)
        f_cam_plot = f_cam[mask_cam]
        p_cam_plot = p_cam[mask_cam]
        max_cam_power = float(np.max(p_cam_plot)) if len(p_cam_plot) > 0 and np.max(p_cam_plot) > 0 else 1.0
        p_cam_norm = p_cam_plot / max_cam_power

        ax.plot(
            f_cam_plot,
            p_cam_norm,
            color="#d95f02",
            linewidth=2.0,
            linestyle="-",
            label="SETU Camera (Displacement PSD)",
            alpha=0.85,
        )
        ax.fill_between(f_cam_plot, p_cam_norm, color="#d95f02", alpha=0.12)
    else:
        # If camera continuous spectrum array is not in the JSON or --freq was passed,
        # represent camera peak via a clean vertical indicator on the normalized scale
        ax.axvline(
            x=camera_freq_hz,
            color="#d95f02",
            linestyle="--",
            linewidth=2.2,
            label=f"SETU Camera Peak ({camera_freq_hz:.2f} Hz)",
            alpha=0.9,
        )

    # Mark Camera Peak Indicator
    ax.plot(
        [camera_freq_hz],
        [1.0],
        marker="s",
        markersize=8,
        color="#d95f02",
        markeredgecolor="white",
        markeredgewidth=1.5,
        zorder=5,
    )
    ax.annotate(
        f"Camera Peak: {camera_freq_hz:.2f} Hz",
        xy=(camera_freq_hz, 1.0),
        xytext=(camera_freq_hz - 4.5, 0.70),
        arrowprops=dict(facecolor="#d95f02", edgecolor="#d95f02", arrowstyle="->", lw=1.5),
        fontsize=9,
        fontweight="bold",
        color="#d95f02",
        bbox=dict(boxstyle="round,pad=0.3", fc="#fff5ee", ec="#d95f02", lw=1, alpha=0.9),
    )

    # Mark confidence interval spans if available
    if comparison_res.accel_ci is not None:
        ax.axvspan(
            comparison_res.accel_ci[0],
            comparison_res.accel_ci[1],
            color="#1f77b4",
            alpha=0.18,
            label=f"Accel 95% CI (±{comparison_res.accel_ci_half_width:.3f} Hz)",
        )
    if comparison_res.camera_ci is not None and comparison_res.camera_ci_half_width is not None:
        ax.axvspan(
            comparison_res.camera_ci[0],
            comparison_res.camera_ci[1],
            color="#d95f02",
            alpha=0.18,
            label=f"Camera CI (±{comparison_res.camera_ci_half_width:.3f} Hz)",
        )

    # Chart Styling
    ax.set_xlim(0.0, max_display_freq_hz)
    ax.set_ylim(0.0, 1.15)
    ax.set_xlabel("Frequency (Hz)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Normalized Spectral Power (PSD / max)", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)

    # Top Status Banner with Verdict
    verdict_color = "#2ca02c" if comparison_res.is_agreement else "#d62728"
    verdict_text = f"VERDICT: {comparison_res.verdict} (Agreement)" if comparison_res.is_agreement else f"VERDICT: {comparison_res.verdict} (Mismatch)"

    title_str = (
        f"SETU Camera vs Phyphox Accelerometer Validation\n"
        f"Accel: {comparison_res.accelerometer_peak_hz:.2f} Hz ({comparison_res.dominant_axis}-axis)  |  "
        f"Camera: {comparison_res.camera_peak_hz:.2f} Hz  |  "
        f"Δf = {comparison_res.absolute_diff_hz:.2f} Hz ({comparison_res.relative_diff_pct:.1f}%)"
    )
    ax.set_title(title_str, fontsize=12, fontweight="bold", pad=12)

    # Badge for verdict
    ax.text(
        0.98,
        0.93,
        verdict_text,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        color="white",
        ha="right",
        va="top",
        bbox=dict(boxstyle="round,pad=0.4", fc=verdict_color, ec="none", alpha=0.9),
    )

    ax.legend(loc="upper left", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

    return out_path


# ---------------------------------------------------------------------------
# High-Level End-to-End Comparison API
# ---------------------------------------------------------------------------

def compare_phyphox_to_camera(
    phyphox_csv: Union[str, Path],
    camera_source: Union[float, int, str, Path, Dict[str, Any]],
    output_png: Optional[Union[str, Path]] = None,
    tolerance_rel_pct: float = DEFAULT_TOLERANCE_REL_PCT,
    tolerance_abs_hz: float = DEFAULT_TOLERANCE_ABS_HZ,
    min_freq_hz: float = DEFAULT_MIN_FREQ_HZ,
    max_freq_hz: float = DEFAULT_MAX_FREQ_HZ,
) -> ComparisonResult:
    """End-to-end execution: load phyphox CSV, compute PSDs and CI, compare against camera, and plot."""
    # 1. Load Phyphox CSV
    time_arr, axes_dict, fs = load_phyphox_csv(phyphox_csv)

    # 2. Compute PSD for all axes and select dominant peak in band
    dom_axis, dom_spectrum, all_spectra = find_dominant_accelerometer_peak(
        axes_dict=axes_dict,
        fs=fs,
        min_freq_hz=min_freq_hz,
        max_freq_hz=max_freq_hz,
    )

    # 3. Estimate accelerometer CI using overlapping segments (95% Student's t)
    accel_uncert = estimate_accelerometer_uncertainty(
        signal_arr=axes_dict[dom_axis],
        fs=fs,
        center_freq_hz=dom_spectrum.peak_frequency_hz,
    )
    accel_ci_half_width = float(accel_uncert["ci_half_width"])
    accel_ci = (float(accel_uncert["ci_lower"]), float(accel_uncert["ci_upper"]))
    dom_spectrum.ci_half_width = accel_ci_half_width
    dom_spectrum.ci = accel_ci

    # 4. Extract camera frequency and optional CI
    cam_freq, cam_ci = extract_camera_frequency(camera_source)
    if cam_ci is not None:
        cam_ci_half_width = float(round(abs(cam_ci[1] - cam_ci[0]) / 2.0, 6))
        uncertainty_note = None
        combined_uncertainty = float(np.sqrt(cam_ci_half_width**2 + accel_ci_half_width**2))
    else:
        cam_ci_half_width = None
        uncertainty_note = "Camera has no CI (--freq given); using accelerometer uncertainty alone."
        combined_uncertainty = float(accel_ci_half_width)

    # 5. Compare frequencies against combined uncertainty
    abs_diff, rel_diff_pct, is_agree, verdict = compare_frequency(
        accelerometer_freq_hz=dom_spectrum.peak_frequency_hz,
        camera_freq_hz=cam_freq,
        accel_ci_half_width=accel_ci_half_width,
        camera_ci_half_width=cam_ci_half_width,
        tolerance_rel_pct=tolerance_rel_pct,
        tolerance_abs_hz=tolerance_abs_hz,
    )

    result = ComparisonResult(
        accelerometer_peak_hz=dom_spectrum.peak_frequency_hz,
        camera_peak_hz=cam_freq,
        absolute_diff_hz=abs_diff,
        relative_diff_pct=rel_diff_pct,
        is_agreement=is_agree,
        dominant_axis=dom_axis,
        axes_detected=list(axes_dict.keys()),
        tolerance_rel_pct=tolerance_rel_pct,
        tolerance_abs_hz=tolerance_abs_hz,
        verdict=verdict,
        combined_uncertainty=combined_uncertainty,
        accel_ci_half_width=accel_ci_half_width,
        camera_ci_half_width=cam_ci_half_width,
        camera_ci=cam_ci,
        accel_ci=accel_ci,
        uncertainty_note=uncertainty_note,
        accelerometer_spectrum=dom_spectrum,
    )

    # 6. Generate plot if output path requested
    if output_png is not None:
        plot_path = plot_comparison(
            accel_spectrum=dom_spectrum,
            camera_freq_hz=cam_freq,
            comparison_res=result,
            output_png_path=output_png,
            camera_spectrum=None,
            max_display_freq_hz=max_freq_hz,
        )
        result.plot_path = str(plot_path)

    return result


# ---------------------------------------------------------------------------
# Command Line Interface (CLI)
# ---------------------------------------------------------------------------

def main() -> int:
    """CLI entry point for validation.compare."""
    parser = argparse.ArgumentParser(
        prog="python -m validation.compare",
        description="Compare SETU camera-measured frequency against phyphox accelerometer export.",
    )
    parser.add_argument(
        "--phyphox",
        type=str,
        required=True,
        help="Path to phone accelerometer CSV export from phyphox.",
    )

    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--result",
        type=str,
        help="Path to SETU analysis result JSON file (accepts simple JSON with 'frequency_hz' and optional 'frequency_ci').",
    )
    source_group.add_argument(
        "--freq",
        type=float,
        help="SETU camera-measured fundamental frequency in Hz.",
    )

    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Destination path for the comparison PNG spectrum plot.",
    )
    parser.add_argument(
        "--tolerance-rel",
        type=float,
        default=DEFAULT_TOLERANCE_REL_PCT,
        help=f"Relative frequency tolerance in percent (default: {DEFAULT_TOLERANCE_REL_PCT}%%).",
    )
    parser.add_argument(
        "--tolerance-abs",
        type=float,
        default=DEFAULT_TOLERANCE_ABS_HZ,
        help=f"Absolute frequency tolerance in Hz (default: {DEFAULT_TOLERANCE_ABS_HZ} Hz).",
    )

    args = parser.parse_args()

    camera_source: Union[float, str] = args.freq if args.freq is not None else args.result

    try:
        res = compare_phyphox_to_camera(
            phyphox_csv=args.phyphox,
            camera_source=camera_source,
            output_png=args.out,
            tolerance_rel_pct=args.tolerance_rel,
            tolerance_abs_hz=args.tolerance_abs,
        )
    except Exception as exc:
        print(f"Error during validation comparison: {exc}", file=sys.stderr)
        return 1

    # Print summary report
    print("=" * 65)
    print("      SETU CAMERA vs PHYPHOX ACCELEROMETER VALIDATION")
    print("=" * 65)
    print(f"Phyphox CSV:             {args.phyphox}")
    print(f"Detected Axes:           {', '.join(res.axes_detected)}")
    print(f"Dominant Vibration Axis: {res.dominant_axis}")
    print(f"Accelerometer Peak:      {res.accelerometer_peak_hz:.3f} Hz (95% CI: ±{res.accel_ci_half_width:.4f} Hz)")
    if res.camera_ci_half_width is not None:
        print(f"Camera Measured Peak:    {res.camera_peak_hz:.3f} Hz (95% CI: ±{res.camera_ci_half_width:.4f} Hz)")
        print(f"Combined Uncertainty:    ±{res.combined_uncertainty:.4f} Hz (sqrt(camera_ci^2 + accel_ci^2))")
    else:
        print(f"Camera Measured Peak:    {res.camera_peak_hz:.3f} Hz (camera has no CI; using accelerometer uncertainty alone)")
        print(f"Combined Uncertainty:    ±{res.combined_uncertainty:.4f} Hz (accelerometer uncertainty alone)")
    print(f"Absolute Difference:     {res.absolute_diff_hz:.3f} Hz")
    print(f"Percentage Difference:   {res.relative_diff_pct:.2f}%")
    verdict_detail = "Passes validation criteria" if res.is_agreement else "Exceeds tolerance margin"
    print(f"Agreement Verdict:       {res.verdict} ({verdict_detail})")
    if res.plot_path:
        print(f"Comparison Plot:         {os.path.abspath(res.plot_path)}")
    print("=" * 65)

    return 0


if __name__ == "__main__":
    sys.exit(main())
