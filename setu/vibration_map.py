"""Measure and visualize the dominant vibration frequency of segmented instances."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from setu.motion import track_motion
from setu.segment import mask_to_bbox
from setu.spectrum import compute_welch_peaks, determine_spectrum_verdict
from setu.types import SegmentResult
from setu.video_io import read_video_frame0

MAX_INSTANCES = 16
ROI_MARGIN_PX = 12


def _read_first_frame(video_path: Union[str, os.PathLike[str]]) -> Tuple[np.ndarray, float]:
    import av
    frame = read_video_frame0(video_path)
    with av.open(str(video_path)) as container:
        fps = float(container.streams.video[0].average_rate or 60.0)
    if not math.isfinite(fps) or fps <= 0:
        fps = 60.0
    return frame, fps


def _valid_mask(mask: Any, shape: Tuple[int, int], name: str) -> np.ndarray:
    arr = np.asarray(mask)
    if arr.ndim != 2 or arr.shape != shape:
        raise ValueError(f"{name} must be a 2D mask matching frame dimensions {shape}.")
    return arr.astype(bool, copy=False)


def _padded_bbox(bbox: Tuple[int, int, int, int], width: int, height: int) -> Tuple[int, int, int, int]:
    x, y, w, h = bbox
    x1, y1 = max(0, x - ROI_MARGIN_PX), max(0, y - ROI_MARGIN_PX)
    x2, y2 = min(width, x + w + ROI_MARGIN_PX), min(height, y + h + ROI_MARGIN_PX)
    return x1, y1, x2 - x1, y2 - y1


def _record_instance(
    label: str,
    bbox: Tuple[int, int, int, int],
    displacement: Optional[np.ndarray],
    peaks: list,
    verdict: str,
    fps: float,
    amplitude_frequency_hz: Optional[float] = None,
) -> Dict[str, Any]:
    peak = peaks[0] if peaks else None
    amp = None
    if displacement is not None and len(displacement) and np.all(np.isfinite(displacement)):
        if peak is not None or amplitude_frequency_hz is not None:
            # Fit the shared spectral component, rather than using total trace
            # range (which includes noise and other modes).
            times = np.arange(len(displacement), dtype=np.float64)
            if fps > 0:
                fit_frequency = amplitude_frequency_hz if amplitude_frequency_hz is not None else float(peak.freq_hz)
                phase = 2.0 * np.pi * fit_frequency * times / fps
                design = np.column_stack((np.sin(phase), np.cos(phase), np.ones_like(phase)))
                coeff, *_ = np.linalg.lstsq(design, displacement, rcond=None)
                amp = float(np.hypot(coeff[0], coeff[1]))
        if amp is None:
            amp = float((np.max(displacement) - np.min(displacement)) / 2.0)
    return {
        "label": label,
        "bbox": [int(v) for v in bbox],
        "frequency_hz": float(peak.freq_hz) if peak is not None else None,
        "amplitude_px": amp,
        "verdict": verdict,
    }


def mode_shape_from_instances(
    instances: Sequence[Mapping[str, Any]],
    dominant_frequency_hz: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Return normalized mode-shape points for at least three collinear records."""
    usable = [item for item in instances
              if item.get("verdict") == "usable"
              and item.get("amplitude_px") is not None
              and item.get("frequency_hz") is not None]
    if dominant_frequency_hz is not None:
        usable = [item for item in usable
                  if abs(float(item["frequency_hz"]) - dominant_frequency_hz)
                  <= max(0.15, dominant_frequency_hz * 0.03)]
    if len(usable) < 3:
        return []
    centers = np.asarray([
        (float(item["bbox"][0]) + float(item["bbox"][2]) / 2.0,
         float(item["bbox"][1]) + float(item["bbox"][3]) / 2.0)
        for item in usable
    ], dtype=np.float64)
    centered = centers - centers.mean(axis=0)
    eigenvalues, eigenvectors = np.linalg.eigh(centered.T @ centered / len(usable))
    if eigenvalues[-1] <= 1e-12 or eigenvalues[0] / eigenvalues[-1] > 0.10:
        return []
    axis = eigenvectors[:, -1]
    if axis[0] < 0.0 or (abs(axis[0]) <= 1e-12 and axis[1] < 0.0):
        axis = -axis
    projection = centered @ axis
    lo, hi = float(np.min(projection)), float(np.max(projection))
    if hi - lo <= 1e-9:
        return []
    result = [
        {"label": str(item.get("label", "instance")),
         "position_norm": float((position - lo) / (hi - lo)),
         "amplitude_px": float(item["amplitude_px"])}
        for item, position in zip(usable, projection)
    ]
    return sorted(result, key=lambda item: item["position_norm"])


def _hex_bgr(value: str) -> Tuple[int, int, int]:
    return tuple(int(value[i : i + 2], 16) for i in (5, 3, 1))


LOW_FREQUENCY_COLOR = _hex_bgr("#0B6E4F")
HIGH_FREQUENCY_COLOR = _hex_bgr("#F2C230")
UNUSABLE_COLOR = (150, 150, 150)


def _frequency_color(freq: float, low: float, high: float) -> Tuple[int, int, int]:
    ratio = 0.5 if high <= low else float(np.clip((freq - low) / (high - low), 0.0, 1.0))
    return tuple(int(round(a + (b - a) * ratio)) for a, b in zip(LOW_FREQUENCY_COLOR, HIGH_FREQUENCY_COLOR))


def _draw_map(
    frame: np.ndarray,
    instances: Sequence[Tuple[str, np.ndarray, Dict[str, Any]]],
    out_png: Union[str, os.PathLike[str]],
) -> None:
    image = frame.copy()
    usable_freqs = [float(record["frequency_hz"]) for _, _, record in instances
                    if record["verdict"] == "usable" and record["frequency_hz"] is not None]
    low = min(usable_freqs, default=0.0)
    high = max(usable_freqs, default=0.0)

    for index, (label, mask, record) in enumerate(instances):
        color = UNUSABLE_COLOR
        if record["verdict"] == "usable" and record["frequency_hz"] is not None:
            color = _frequency_color(record["frequency_hz"], low, high)
            overlay = image.copy()
            overlay[mask] = color
            image[mask] = cv2.addWeighted(image, 0.48, overlay, 0.52, 0)[mask]
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cv2.drawContours(image, contours, -1, color, 2, cv2.LINE_AA)
        x, y, w, h = record["bbox"]
        if record["verdict"] != "usable" or record["frequency_hz"] is None:
            text = f"{label}: not enough data"
        else:
            text = f"{label} {record['frequency_hz']:.2f} Hz"
        text_y = max(18, y - 6 - (index % 2) * 20)
        text_origin = (max(0, x), text_y)
        cv2.putText(image, text, text_origin, cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (20, 20, 20), 3, cv2.LINE_AA)
        cv2.putText(image, text, text_origin, cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, color, 1, cv2.LINE_AA)

    # Compact continuous legend, with frequencies taken from usable instances only.
    bar_x, bar_y, bar_w, bar_h = 24, max(10, image.shape[0] - 60), min(240, max(100, image.shape[1] // 3)), 14
    for col in range(bar_w):
        t = col / max(bar_w - 1, 1)
        c = tuple(int(round(a + (b - a) * t)) for a, b in zip(LOW_FREQUENCY_COLOR, HIGH_FREQUENCY_COLOR))
        cv2.line(image, (bar_x + col, bar_y), (bar_x + col, bar_y + bar_h), c, 1)
    cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (25, 25, 25), 1)
    if usable_freqs:
        lo_text = f"{low:.2f} Hz"
        hi_text = f"{high:.2f} Hz"
    else:
        lo_text = hi_text = "n/a"
    cv2.putText(image, lo_text, (bar_x, bar_y + bar_h + 18), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (20, 20, 20), 1, cv2.LINE_AA)
    hi_w = cv2.getTextSize(hi_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0][0]
    cv2.putText(image, hi_text, (bar_x + bar_w - hi_w, bar_y + bar_h + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.putText(image, "Dominant frequency", (bar_x, bar_y + bar_h + 37),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)

    output = Path(out_png)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), image):
        raise OSError(f"Could not write vibration map PNG to {output}.")


def build_vibration_map(
    video_path: Union[str, os.PathLike[str]],
    instances: Sequence[Tuple[str, np.ndarray]],
    background_mask: np.ndarray,
    out_png: Union[str, os.PathLike[str]],
    shared_frequency_hz: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Measure up to 16 segmented instances and draw their usable frequencies.

    Every instance is analyzed with the same background-stabilized tracking and
    spectrum verdict rules used by the main pipeline. Tracking is restricted to
    that instance's bounding rectangle plus a small margin, while the supplied
    full-frame background mask provides camera stabilization features.

    Args:
        video_path: Video to analyze; its first frame is the visualization base.
        instances: ``(label, mask)`` pairs, where each mask is a full-frame
            binary mask for one concept instance.
        background_mask: Full-frame binary mask of stationary background.
        out_png: Destination PNG path.

    Returns:
        A list of schema 1.1 measurement-instance dictionaries, one per input
        instance up to the 16-instance cap.
    """
    frame, fps = _read_first_frame(video_path)
    height, width = frame.shape[:2]
    bg_mask = _valid_mask(background_mask, (height, width), "background_mask")
    if not np.any(bg_mask):
        raise ValueError("background_mask must contain stationary reference pixels.")
    if instances is None:
        raise ValueError("instances must be a sequence of (label, mask) pairs")

    rendered: List[Tuple[str, np.ndarray, Dict[str, Any]]] = []
    for item in list(instances)[:MAX_INSTANCES]:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise ValueError("Each instance must be a (label, mask) pair.")
        raw_label, raw_mask = item
        label = str(raw_label).strip() or "instance"
        mask = _valid_mask(raw_mask, (height, width), f"mask for {label!r}")
        if not np.any(mask):
            raise ValueError(f"Mask for {label!r} is empty.")
        bbox = tuple(int(v) for v in mask_to_bbox(mask))
        padded_bbox = _padded_bbox(bbox, width, height)
        target_segment = SegmentResult(
            target_masks=[mask], background_masks=[bg_mask], fps=fps,
        )
        displacement: Optional[np.ndarray] = None
        peaks = []
        try:
            motion = track_motion(
                video_path=str(video_path), segment_result=target_segment,
                target_bbox=padded_bbox, method="phase_correlation", axis="auto",
            )
            displacement = np.asarray(motion.displacement, dtype=np.float64)
            if motion.verdict != "failed":
                peaks = compute_welch_peaks(
                    displacement, fps=float(motion.fps), top_k=3,
                    background_displacement=getattr(motion, "background_displacement", None),
                    has_background=True,
                )
            verdict, _ = determine_spectrum_verdict(
                peaks,
                has_background=True,
                clipping_detected=bool(motion.quality.get("clipping_detected", False)),
                tracking_failed=motion.verdict == "failed",
                tracking_error="; ".join(motion.reasons) if motion.reasons else None,
            )
            if motion.verdict == "weak" and verdict == "usable":
                verdict = "weak"
        except Exception:
            # An individual untrackable instance does not prevent reporting the others.
            verdict = "failed"
            peaks = []
            displacement = None
        record = _record_instance(
            label, bbox, displacement, peaks, verdict, fps,
            amplitude_frequency_hz=shared_frequency_hz,
        )
        rendered.append((label, mask, record))

    _draw_map(frame, rendered, out_png)
    return [record for _, _, record in rendered]


# Earlier utility entry points retained for callers that only need a marked-up frame.
def render_vibration_map(
    frame_or_video: Union[str, os.PathLike[str], np.ndarray],
    instances: Sequence[Mapping[str, Any]],
    output_path: Optional[Union[str, os.PathLike[str]]] = None,
    *,
    alpha: float = 0.38,
) -> np.ndarray:
    """Render amplitude-colored boxes as a lightweight visualization helper."""
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")
    if isinstance(frame_or_video, np.ndarray):
        frame = frame_or_video.copy()
    else:
        frame, _ = _read_first_frame(frame_or_video)
    height, width = frame.shape[:2]
    for instance in instances:
        bbox = instance.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            raise ValueError("Each vibration instance must have bbox [x, y, width, height].")
        x, y, w, h = [int(v) for v in bbox]
        x1, y1, x2, y2 = max(0, x), max(0, y), min(width, x + w), min(height, y + h)
        if x1 >= x2 or y1 >= y2:
            raise ValueError("Bounding box does not intersect the image.")
        color = (160, 160, 160)
        if instance.get("amplitude_px") is not None:
            level = int(np.clip(abs(float(instance["amplitude_px"])) * 255, 0, 255))
            color = tuple(int(c) for c in cv2.applyColorMap(np.array([[level]], np.uint8), cv2.COLORMAP_TURBO)[0, 0])
        roi = frame[y1:y2, x1:x2]
        fill = np.empty_like(roi)
        fill[:] = color
        frame[y1:y2, x1:x2] = cv2.addWeighted(roi, 1 - alpha, fill, alpha, 0)
        label = str(instance.get("label", "target"))
        frequency = instance.get("frequency_hz")
        detail = "n/a" if frequency is None else f"{float(frequency):.2f} Hz"
        verdict = str(instance.get("verdict", "weak"))
        cv2.rectangle(frame, (x1, y1), (x2 - 1, y2 - 1), color, 2, cv2.LINE_AA)
        cv2.putText(frame, f"{label}: {detail} | quality {verdict}", (x1, max(16, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output), frame):
            raise OSError(f"Could not write vibration map to {str(output)!r}.")
    return frame


def create_vibration_map(
    frame_or_video: Union[str, os.PathLike[str], np.ndarray],
    instances: Sequence[Mapping[str, Any]],
    output_path: Optional[Union[str, os.PathLike[str]]] = None,
    *, alpha: float = 0.38,
) -> np.ndarray:
    """Compatibility alias for :func:`render_vibration_map`."""
    return render_vibration_map(frame_or_video, instances, output_path, alpha=alpha)
