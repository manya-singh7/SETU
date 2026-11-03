"""
Structural segmentation module for Project SETU.

Step 2 in SETU Pipeline:
Segments the structural target from a text prompt (e.g. "corner of concrete pillar")
and a stationary background reference (e.g. "bedrock / distant building") using Meta SAM 3.1.

Integrates with Meta's OpenAI-compatible Model API (sam-3.1 at https://api.meta.ai/v1)
routed through setu/meta_client.py for unified mock/cached/live dispatch, usage logging,
and budget cap enforcement.

Supports Promptable Concept Segmentation (PCS), manual bounding box ROI segmentation,
multi-instance candidate ranking, and fixed-mask frame replication for downstream
subpixel motion cancellation.

Safety Principle:
SETU NEVER declares a bridge safe. Vision-based vibration screening provides an early
screening heuristic only, routing flagged readings to human civil engineers for on-site inspection.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import base64
from dataclasses import dataclass, field
import logging
import math
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import numpy.typing as npt
import requests

from setu.config import settings
from setu.meta_client import (
    SAM_COST_PER_IMAGE,
    BudgetExceededError,
    MetaAPIError,
    MetaClient,
    get_meta_client,
)
from setu.types import SegmentResult

logger = logging.getLogger(__name__)

# Check if official meta_sam_parser package is available
try:
    import meta_sam_parser
    HAS_META_SAM_PARSER = True
except ImportError:
    HAS_META_SAM_PARSER = False


# ---------------------------------------------------------------------------
# Single-Frame Output & Prompt Types
# ---------------------------------------------------------------------------

@dataclass
class SegmentationOutput:
    """Output masks and bounding boxes for a single video frame.

    Attributes:
        target_mask: Binary mask of shape (H, W) for structural region of interest.
        background_mask: Binary mask of shape (H, W) for stationary reference anchor (or None if not found).
        target_bbox: Bounding box tuple (x, y, w, h) around structural target.
        background_bbox: Bounding box tuple (x, y, w, h) around stationary background (or None if not found).
        target_prompt: Text prompt used (e.g. "corner of concrete pillar").
        background_prompt: Reference prompt used (e.g. "stationary ground reference").
        confidence: Optional confidence score reported by SAM, or None.
        method: Method used to generate segmentation ("sam", "manual", or "fallback").
        target_source: Provenance of target mask ("sam", "manual", "fallback", or "none").
        background_source: Provenance of background mask ("sam", "manual", "fallback", or "none").
        warnings: List of diagnostic warnings.
        metadata: Model, instance candidates, and execution details.
    """

    target_mask: npt.NDArray[np.bool_]
    background_mask: Optional[npt.NDArray[np.bool_]] = None
    target_bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
    background_bbox: Optional[Tuple[int, int, int, int]] = None
    target_prompt: str = "corner of concrete pillar"
    background_prompt: str = "stationary ground reference"
    confidence: Optional[float] = None
    method: Literal["sam", "manual", "fallback"] = "fallback"
    target_source: Literal["sam", "manual", "fallback", "none"] = "fallback"
    background_source: Literal["sam", "manual", "fallback", "none"] = "none"
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PointPrompt:
    """A 2D coordinate point prompt for interactive segmentation."""

    x: float
    y: float
    label: int = 1


@dataclass
class BoxPrompt:
    """Bounding box prompt (x, y, w, h) in pixel coordinates."""

    x: int
    y: int
    w: int
    h: int


# ---------------------------------------------------------------------------
# Utility & Conversion Functions
# ---------------------------------------------------------------------------

def mask_to_bbox(mask: npt.NDArray[np.bool_]) -> Tuple[int, int, int, int]:
    """Calculate bounding box (x, y, w, h) from a 2D boolean mask.

    Returns:
        (x, y, w, h) bounding box, or (0, 0, 0, 0) if mask is completely empty.
    """
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not np.any(rows) or not np.any(cols):
        return (0, 0, 0, 0)

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return (int(cmin), int(rmin), int(cmax - cmin + 1), int(rmax - rmin + 1))


def bbox_to_mask(bbox: Tuple[int, int, int, int], shape: Tuple[int, int]) -> npt.NDArray[np.bool_]:
    """Convert a bounding box (x, y, w, h) into a 2D boolean mask of specified (H, W) shape."""
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    x, y, bw, bh = bbox
    x1 = max(0, min(x, w))
    y1 = max(0, min(y, h))
    x2 = max(0, min(x + bw, w))
    y2 = max(0, min(y + bh, h))

    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = True
    return mask


def shift_mask(mask: npt.NDArray[np.bool_], dx: int, dy: int) -> npt.NDArray[np.bool_]:
    """Translate a 2D boolean mask by integer pixel offsets (dx, dy)."""
    h, w = mask.shape
    new_mask = np.zeros_like(mask)

    src_y1 = max(0, -dy)
    src_y2 = min(h, h - dy)
    src_x1 = max(0, -dx)
    src_x2 = min(w, w - dx)

    dst_y1 = max(0, dy)
    dst_y2 = min(h, h + dy)
    dst_x1 = max(0, dx)
    dst_x2 = min(w, w + dx)

    if src_y2 > src_y1 and src_x2 > src_x1 and dst_y2 > dst_y1 and dst_x2 > dst_x1:
        new_mask[dst_y1:dst_y2, dst_x1:dst_x2] = mask[src_y1:src_y2, src_x1:src_x2]
    return new_mask


def compute_mask_centroid(mask: npt.NDArray[np.bool_]) -> Optional[Tuple[float, float]]:
    """Compute the (cx, cy) pixel centroid coordinates of a binary mask."""
    y_indices, x_indices = np.where(mask)
    if len(x_indices) == 0:
        return None
    return (float(np.mean(x_indices)), float(np.mean(y_indices)))


def compute_mask_area(mask: Optional[npt.NDArray[np.bool_]]) -> int:
    """Return total active pixel count of a binary mask, or 0 if None."""
    if mask is None:
        return 0
    return int(np.sum(mask))


def compute_mask_iou(mask1: npt.NDArray[np.bool_], mask2: npt.NDArray[np.bool_]) -> float:
    """Compute Intersection over Union (IoU) between two binary masks."""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)


def extract_bboxes_from_result(
    result: SegmentResult, frame_idx: int = 0
) -> Tuple[Tuple[int, int, int, int], Optional[Tuple[int, int, int, int]]]:
    """Extract (target_bbox, background_bbox) from a SegmentResult for a given frame index."""
    if not result.target_masks:
        raise ValueError("SegmentResult contains empty target mask list.")
    idx = max(0, min(frame_idx, len(result.target_masks) - 1))
    t_box = mask_to_bbox(result.target_masks[idx])
    b_box = mask_to_bbox(result.background_masks[idx]) if result.background_masks else None
    return t_box, b_box


def validate_segment_result(result: SegmentResult) -> Dict[str, Any]:
    """Validate a SegmentResult for structural integrity and pipeline compliance."""
    n_target = len(result.target_masks)
    n_bg = len(result.background_masks)
    errors: List[str] = []
    warnings: List[str] = []

    if n_target == 0:
        errors.append("Target mask list is empty.")
    if n_bg == 0:
        warnings.append("No stationary background found; include ground, buildings or trees in frame")
    elif n_target != n_bg:
        errors.append(f"Mismatched mask counts: target has {n_target}, background has {n_bg}.")

    if result.fps <= 0:
        errors.append(f"Invalid frame rate fps={result.fps}; must be positive.")

    avg_t_area = 0.0
    avg_b_area = 0.0
    max_iou = 0.0

    if n_target > 0:
        t_areas = [compute_mask_area(m) for m in result.target_masks]
        avg_t_area = float(np.mean(t_areas))
        if min(t_areas) == 0:
            warnings.append("One or more target masks contain zero active pixels.")

    if n_bg > 0:
        b_areas = [compute_mask_area(m) for m in result.background_masks]
        avg_b_area = float(np.mean(b_areas))
        if min(b_areas) == 0:
            warnings.append("One or more background masks contain zero active pixels.")

    if n_target > 0 and n_bg > 0 and n_target == n_bg:
        ious = [compute_mask_iou(tm, bm) for tm, bm in zip(result.target_masks, result.background_masks)]
        max_iou = float(max(ious)) if ious else 0.0
        if max_iou > 0.15:
            warnings.append(f"High overlap between target and stationary background (max IoU: {max_iou:.2f}).")

    # Check target persistence failure
    target_pers = getattr(result, "metadata", {}).get("target_persistence")
    if target_pers is not None and not target_pers.get("passed", True):
        failure_msg = getattr(
            result,
            "metadata",
            {},
        ).get(
            "failure_reason",
            f"Target persistence failure: fraction of valid frames ({target_pers.get('fraction_valid', 0.0):.1%}) below threshold.",
        )
        errors.append(failure_msg)

    return {
        "is_valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "frame_count": n_target,
        "fps": result.fps,
        "method": result.method,
        "avg_target_area_px": avg_t_area,
        "avg_background_area_px": avg_b_area,
        "max_target_background_iou": max_iou,
    }


def load_video_frames(
    video_source: Union[str, Path, Sequence[npt.NDArray[np.uint8]], npt.NDArray[np.uint8]],
    default_fps: float = 30.0,
    max_frames: Optional[int] = None,
) -> Tuple[List[npt.NDArray[np.uint8]], float]:
    """Load video frames and sampling frame rate from a file path or array structure."""
    if isinstance(video_source, (str, Path)):
        path_str = str(video_source)
        if not os.path.exists(path_str):
            raise FileNotFoundError(f"Video or image file not found: {path_str}")

        # Check if direct still image
        ext = os.path.splitext(path_str)[1].lower()
        if ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
            img = cv2.imread(path_str)
            if img is None:
                raise ValueError(f"Failed to decode image file: {path_str}")
            return [img], float(default_fps)

        cap = cv2.VideoCapture(path_str)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video file: {path_str}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or np.isnan(fps):
            fps = default_fps

        frames: List[npt.NDArray[np.uint8]] = []
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
                if max_frames is not None and len(frames) >= max_frames:
                    break
        finally:
            cap.release()

        if not frames:
            raise ValueError(f"No frames could be decoded from video file: {path_str}")
        return frames, float(fps)

    if isinstance(video_source, np.ndarray):
        if video_source.ndim == 4:
            frames = [video_source[i].astype(np.uint8) for i in range(len(video_source))]
        elif video_source.ndim == 3:
            frames = [video_source[i].astype(np.uint8) for i in range(len(video_source))]
        elif video_source.ndim == 2:
            frames = [video_source.astype(np.uint8)]
        else:
            raise ValueError(f"Expected 2D, 3D, or 4D numpy array, got {video_source.ndim}D array.")

        if max_frames is not None:
            frames = frames[:max_frames]
        return frames, float(default_fps)

    if isinstance(video_source, (list, tuple)):
        frames = [np.asarray(f, dtype=np.uint8) for f in video_source]
        if max_frames is not None:
            frames = frames[:max_frames]
        if not frames:
            raise ValueError("Empty frame sequence provided.")
        return frames, float(default_fps)

    raise TypeError(f"Unsupported video source type: {type(video_source)}")


def _to_gray(frame: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    """Helper to convert a frame to grayscale if it is multichannel."""
    if frame.ndim == 3 and frame.shape[2] in (3, 4):
        return cv2.cvtColor(frame[:, :, :3], cv2.COLOR_BGR2GRAY)
    return frame


def track_roi_displacement(
    ref_gray: npt.NDArray[np.uint8],
    curr_gray: npt.NDArray[np.uint8],
    bbox: Tuple[int, int, int, int],
    search_margin: int = 12,
) -> Tuple[int, int]:
    """Template matching displacement tracking (used for optional QA drift monitoring)."""
    x, y, w, h = bbox
    H, W = ref_gray.shape[:2]

    x1 = max(0, min(x, W - 1))
    y1 = max(0, min(y, H - 1))
    x2 = max(x1 + 1, min(x + w, W))
    y2 = max(y1 + 1, min(y + h, H))

    template = ref_gray[y1:y2, x1:x2]
    th, tw = template.shape[:2]
    if th < 3 or tw < 3:
        return (0, 0)

    sw_x1 = max(0, x1 - search_margin)
    sw_y1 = max(0, y1 - search_margin)
    sw_x2 = min(W, x2 + search_margin)
    sw_y2 = min(H, y2 + search_margin)

    search_window = curr_gray[sw_y1:sw_y2, sw_x1:sw_x2]
    if search_window.shape[0] < th or search_window.shape[1] < tw:
        return (0, 0)

    res = cv2.matchTemplate(search_window, template, cv2.TM_SQDIFF)
    _, _, min_loc, _ = cv2.minMaxLoc(res)

    best_x = sw_x1 + min_loc[0]
    best_y = sw_y1 + min_loc[1]

    dx = int(best_x - x1)
    dy = int(best_y - y1)
    return dx, dy


def compute_patch_ncc(patch_ref: np.ndarray, patch_curr: np.ndarray) -> float:
    """Compute normalized cross-correlation between two grayscale image patches.

    Returns a Pearson correlation coefficient in [-1.0, 1.0].
    Guards against division by zero on uniform/solid-color patches.
    """
    if patch_ref.shape != patch_curr.shape or patch_ref.size == 0:
        return 0.0

    ref_f = patch_ref.astype(np.float32)
    curr_f = patch_curr.astype(np.float32)

    std_ref = float(np.std(ref_f))
    std_curr = float(np.std(curr_f))

    # Guard against division by zero on flat/uniform patches
    if std_ref < 1e-3 or std_curr < 1e-3:
        mean_diff = abs(float(np.mean(ref_f)) - float(np.mean(curr_f)))
        if std_ref < 1e-3 and std_curr < 1e-3:
            # Both flat: match if mean intensity matches within 10 levels
            return 1.0 if mean_diff <= 10.0 else 0.0
        # One is flat and other has texture (e.g. solid block occluding texture)
        return 0.0

    res = cv2.matchTemplate(curr_f, ref_f, cv2.TM_CCOEFF_NORMED)
    val = float(res[0, 0])
    return max(-1.0, min(1.0, val))


def compute_persistence(
    frames: Sequence[npt.NDArray[np.uint8]],
    bbox: Optional[Tuple[int, int, int, int]],
    fps: float = 30.0,
    correlation_threshold: float = 0.70,
    min_persistence_ratio: float = 0.85,
    sample_stride: Optional[int] = None,
) -> Dict[str, Any]:
    """Verify that a frame-0 mask/bounding box remains valid across a video sequence.

    Computes normalized cross-correlation (NCC) of each frame's patch against Frame 0.

    Threshold justification:
      - Normal civil structural vibration involves subpixel to micro-pixel motion (< 1-2 px),
        under which NCC for real textured structures remains high (>= 0.85 to 0.98).
      - Target occlusion (e.g. a vehicle or block) or camera pan changes patch structure,
        causing correlation to drop sharply (< 0.50).
      - A correlation threshold of 0.70 reliably discriminates valid structural motion
        from severe occlusions and camera drift.
      - A min_persistence_ratio of 0.85 requires 85% of checked frames to stay above 0.70,
        ensuring that videos with substantial occlusion (e.g. 30% of frames blocked) fail,
        while allowing minor momentary micro-transients on static recordings.

    Args:
        frames: Sequence of video frame arrays.
        bbox: Bounding box tuple (x, y, w, h) in pixel coordinates, or None.
        fps: Sampling frame rate in Hz.
        correlation_threshold: Minimum correlation score to consider a frame valid (default: 0.70).
        min_persistence_ratio: Minimum fraction of checked frames that must meet threshold (default: 0.85).
        sample_stride: Frame check interval for speed. Defaults to 1 for <= 60 frames, or max(1, len(frames) // 60).

    Returns:
        Dictionary reporting:
            - fraction_valid: float (fraction of checked frames above correlation_threshold)
            - lowest_correlation: float | None (minimum correlation observed)
            - lowest_frame_idx: int | None (frame index where lowest correlation occurred)
            - lowest_timestamp_sec: float | None (time in seconds where lowest correlation occurred)
            - threshold: float (correlation threshold used)
            - min_persistence_ratio: float (fraction threshold required)
            - passed: bool (whether fraction_valid >= min_persistence_ratio)
            - total_checked: int
            - valid_frames: int
    """
    if bbox is None or len(frames) == 0:
        return {
            "fraction_valid": 0.0,
            "lowest_correlation": None,
            "lowest_frame_idx": None,
            "lowest_timestamp_sec": None,
            "threshold": correlation_threshold,
            "min_persistence_ratio": min_persistence_ratio,
            "passed": False,
            "total_checked": 0,
            "valid_frames": 0,
            "reason": "No bounding box specified",
        }

    x, y, w, h = bbox
    H, W = frames[0].shape[:2]
    x1 = max(0, min(x, W - 1))
    y1 = max(0, min(y, H - 1))
    x2 = max(x1 + 1, min(x + w, W))
    y2 = max(y1 + 1, min(y + h, H))

    ref_gray = _to_gray(frames[0])[y1:y2, x1:x2]
    if ref_gray.size == 0:
        return {
            "fraction_valid": 0.0,
            "lowest_correlation": None,
            "lowest_frame_idx": None,
            "lowest_timestamp_sec": None,
            "threshold": correlation_threshold,
            "min_persistence_ratio": min_persistence_ratio,
            "passed": False,
            "total_checked": 0,
            "valid_frames": 0,
            "reason": "Empty patch bounding box",
        }

    n_frames = len(frames)
    stride = sample_stride if sample_stride is not None else (1 if n_frames <= 60 else max(1, n_frames // 60))

    valid_frames = 0
    total_checked = 0
    lowest_corr: Optional[float] = None
    lowest_idx: Optional[int] = None

    for i in range(0, n_frames, stride):
        curr_gray = _to_gray(frames[i])[y1:y2, x1:x2]
        score = compute_patch_ncc(ref_gray, curr_gray)
        total_checked += 1
        if score >= correlation_threshold:
            valid_frames += 1
        if lowest_corr is None or score < lowest_corr:
            lowest_corr = score
            lowest_idx = i

    fraction_valid = valid_frames / max(1, total_checked)
    passed = fraction_valid >= min_persistence_ratio
    lowest_time = (lowest_idx / fps) if (lowest_idx is not None and fps > 0) else None

    return {
        "fraction_valid": round(fraction_valid, 4),
        "lowest_correlation": round(lowest_corr, 4) if lowest_corr is not None else None,
        "lowest_frame_idx": lowest_idx,
        "lowest_timestamp_sec": round(lowest_time, 3) if lowest_time is not None else None,
        "threshold": correlation_threshold,
        "min_persistence_ratio": min_persistence_ratio,
        "passed": passed,
        "total_checked": total_checked,
        "valid_frames": valid_frames,
    }


# ---------------------------------------------------------------------------
# SAM 3.1 Grammar Parsing & Multi-Instance Ranking
# ---------------------------------------------------------------------------

def parse_sam_grammar(resp: Dict[str, Any], frame_shape: Tuple[int, int]) -> List[Dict[str, Any]]:
    """Parse Meta SAM 3.1 output text grammar or structured records into candidate instances.

    Each instance dictionary contains:
        instance_id: int
        bbox: (x, y, w, h)
        mask: (H, W) boolean numpy array
        area: int
        centroid: (cx, cy)
        confidence: Optional[float]
    """
    h_frame, w_frame = frame_shape
    instances: List[Dict[str, Any]] = []

    # Handle direct test/mock payload structures
    if "target_bbox" in resp:
        t_box = tuple(resp["target_bbox"])
        mask = bbox_to_mask(t_box, frame_shape)
        conf = resp.get("confidence")
        return [
            {
                "instance_id": 0,
                "bbox": t_box,
                "mask": mask,
                "area": compute_mask_area(mask),
                "centroid": compute_mask_centroid(mask),
                "confidence": float(conf) if conf is not None else None,
            }
        ]

    # Handle structured instance list if provided
    if "instances" in resp and isinstance(resp["instances"], list):
        for idx, inst in enumerate(resp["instances"]):
            box = tuple(inst.get("bbox", (0, 0, 0, 0)))
            mask = inst.get("mask")
            if mask is None:
                mask = bbox_to_mask(box, frame_shape)
            conf = inst.get("confidence", inst.get("score"))
            instances.append(
                {
                    "instance_id": idx,
                    "bbox": box,
                    "mask": mask,
                    "area": compute_mask_area(mask),
                    "centroid": compute_mask_centroid(mask),
                    "confidence": float(conf) if conf is not None else None,
                }
            )
        if instances:
            return instances

    # Extract output_text from Meta Model API responses
    text = ""
    for item in resp.get("output", []):
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                text += content.get("text", "")

    if not text.strip():
        return instances

    # Method 1: Official meta_sam_parser image_segmentation_format
    if HAS_META_SAM_PARSER:
        try:
            fmt = meta_sam_parser.image_segmentation_format()
            parser = fmt.create_parser()
            parser.push(text)
            finish = parser.finish(meta_sam_parser.CompletedOutcome())
            res = finish.result

            box_records = {r.object_id: r for r in res.records if r.kind == "box"}
            mask_records = {r.object_id: r for r in res.records if r.kind == "mask"}

            if mask_records or box_records:
                # Find source frame dimensions from box token w, h
                box_w_m = re.search(r"<\|box;[^>]*w=([0-9]+)[^>]*\|>", text)
                box_h_m = re.search(r"<\|box;[^>]*h=([0-9]+)[^>]*\|>", text)
                cw = float(box_w_m.group(1)) if box_w_m else float(w_frame)
                ch = float(box_h_m.group(1)) if box_h_m else float(h_frame)
                scale_x = w_frame / cw if cw > 0 else 1.0
                scale_y = h_frame / ch if ch > 0 else 1.0

                all_ids = list(dict.fromkeys(list(mask_records.keys()) + list(box_records.keys())))
                for idx, obj_id in enumerate(all_ids):
                    m_rec = mask_records.get(obj_id)
                    b_rec = box_records.get(obj_id)

                    conf = None
                    if m_rec is not None and m_rec.confidence is not None:
                        conf = float(m_rec.confidence)
                    elif b_rec is not None and b_rec.confidence is not None:
                        conf = float(b_rec.confidence)
                    elif "confidence" in resp and resp["confidence"] is not None:
                        conf = float(resp["confidence"])

                    if m_rec is not None and m_rec.bounds is not None:
                        b_left = m_rec.bounds.left
                        b_top = m_rec.bounds.top
                        b_right = m_rec.bounds.right
                        b_bottom = m_rec.bounds.bottom
                    elif b_rec is not None:
                        b_left = b_rec.left
                        b_top = b_rec.top
                        b_right = b_rec.right
                        b_bottom = b_rec.bottom
                    else:
                        continue

                    bx1 = max(0, min(int(round(b_left * scale_x)), w_frame - 1))
                    by1 = max(0, min(int(round(b_top * scale_y)), h_frame - 1))
                    bx2 = max(bx1 + 1, min(int(round(b_right * scale_x)), w_frame))
                    by2 = max(by1 + 1, min(int(round(b_bottom * scale_y)), h_frame))
                    bw = bx2 - bx1
                    bh = by2 - by1

                    mask = None
                    if m_rec is not None:
                        try:
                            raster = meta_sam_parser.decode_mask_to_raster(m_rec.mask)
                            patch = np.frombuffer(raster, dtype=bool).reshape(
                                m_rec.mask.height, m_rec.mask.width
                            )
                            resized_patch = (
                                cv2.resize(
                                    patch.astype(np.uint8),
                                    (bw, bh),
                                    interpolation=cv2.INTER_NEAREST,
                                )
                                > 0
                            )
                            full_mask = np.zeros((h_frame, w_frame), dtype=bool)
                            full_mask[by1:by2, bx1:bx2] = resized_patch
                            mask = full_mask
                        except Exception as dec_err:
                            logger.debug("meta_sam_parser raster decoding error: %s", dec_err)

                    if mask is None:
                        mask = bbox_to_mask((bx1, by1, bw, bh), frame_shape)

                    instances.append(
                        {
                            "instance_id": int(obj_id) if obj_id.isdigit() else idx,
                            "bbox": (bx1, by1, bw, bh),
                            "mask": mask,
                            "area": compute_mask_area(mask),
                            "centroid": compute_mask_centroid(mask),
                            "confidence": conf,
                        }
                    )

                if instances:
                    return instances
        except Exception as parser_err:
            logger.debug("meta_sam_parser parse error: %s", parser_err)

    # Method 2: Fallback token regex parser (e.g. for mock tokens or if parser missing)
    box_matches = list(re.finditer(r"<\|box;([^|>]+)\|>", text))
    mask_matches = list(re.finditer(r"<\|mask;([^|>]+)\|>", text))

    for idx, match in enumerate(box_matches):
        box_str = match.group(1)
        params: Dict[str, str] = {}
        for part in box_str.split(";"):
            if "=" in part:
                k, v = part.split("=", 1)
                params[k.strip().lower()] = v.strip()

        try:
            x1 = float(params.get("x1", 0))
            y1 = float(params.get("y1", 0))
            x2 = float(params.get("x2", 0))
            y2 = float(params.get("y2", 0))
            cw = float(params.get("w", w_frame))
            ch = float(params.get("h", h_frame))

            scale_x = w_frame / cw if cw > 0 else 1.0
            scale_y = h_frame / ch if ch > 0 else 1.0

            bx = max(0, min(int(round(x1 * scale_x)), w_frame - 1))
            by = max(0, min(int(round(y1 * scale_y)), h_frame - 1))
            bw = max(1, min(int(round((x2 - x1) * scale_x)), w_frame - bx))
            bh = max(1, min(int(round((y2 - y1) * scale_y)), h_frame - by))

            score_val = (
                params.get("score")
                or params.get("c")
                or params.get("confidence")
                or resp.get("confidence")
            )
            confidence = float(score_val) if score_val is not None else None

            mask = bbox_to_mask((bx, by, bw, bh), frame_shape)

            instances.append(
                {
                    "instance_id": idx,
                    "bbox": (bx, by, bw, bh),
                    "mask": mask,
                    "area": compute_mask_area(mask),
                    "centroid": compute_mask_centroid(mask),
                    "confidence": confidence,
                }
            )
        except (ValueError, KeyError) as parse_err:
            logger.debug("Failed to parse box token '%s': %s", box_str, parse_err)

    return instances


def select_best_instance(
    instances: List[Dict[str, Any]], frame_shape: Tuple[int, int]
) -> Dict[str, Any]:
    """Select the largest and most central instance from multiple prompt matches."""
    if not instances:
        raise ValueError("Cannot select from empty instance list.")
    if len(instances) == 1:
        return instances[0]

    h_frame, w_frame = frame_shape
    center_x = w_frame / 2.0
    center_y = h_frame / 2.0
    max_dist = max(1.0, math.hypot(center_x, center_y))
    max_area = max(1, max(inst["area"] for inst in instances))

    def score_instance(inst: Dict[str, Any]) -> float:
        centroid = inst.get("centroid") or (center_x, center_y)
        cx, cy = centroid
        dist = math.hypot(cx - center_x, cy - center_y)
        centrality = 1.0 - (dist / max_dist)
        area_ratio = inst["area"] / max_area
        return 0.6 * area_ratio + 0.4 * centrality

    return max(instances, key=score_instance)


# ---------------------------------------------------------------------------
# Base Segmenter Interface
# ---------------------------------------------------------------------------

class BaseSegmenter(ABC):
    """Abstract interface for structural target and background reference segmentation."""

    @abstractmethod
    def segment(
        self,
        frame: npt.NDArray[np.uint8],
        target_prompt: str = "corner of concrete pillar",
        background_prompt: str = "stationary ground reference",
        **kwargs: Any,
    ) -> SegmentationOutput:
        """Segment target and stationary reference from a single video frame.

        Args:
            frame: Video frame array (H, W, 3) or (H, W) in uint8 format.
            target_prompt: Open-vocabulary text description of structural target.
            background_prompt: Text description of static background reference anchor.
            **kwargs: Additional implementation-specific options.

        Returns:
            SegmentationOutput containing binary masks, bounding boxes, and optional confidence.
        """
        pass

    def segment_video(
        self,
        video_source: Union[str, Path, Sequence[npt.NDArray[np.uint8]], npt.NDArray[np.uint8]],
        target_prompt: str = "corner of concrete pillar",
        background_prompt: str = "stationary ground reference",
        fps: float = 30.0,
        qa_keyframes: bool = False,
        qa_keyframe_interval: int = 30,
        correlation_threshold: float = 0.70,
        min_persistence_ratio: float = 0.85,
        sample_stride: Optional[int] = None,
        fallback_background_prompts: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> SegmentResult:
        """Segment target and background across an entire video sequence.

        Segments ONLY the first frame (frame 0) and returns the same fixed masks
        for every frame. Motion stabilization and displacement tracking happen
        downstream (in setu/motion.py), so masks must remain fixed per frame.

        Evaluates track persistence over time using normalized cross-correlation
        against Frame 0. If target persistence falls below threshold, flags
        segmentation as failed with a diagnostic failure reason.
        """
        frames, detected_fps = load_video_frames(video_source, default_fps=fps)
        if not frames:
            raise ValueError("No frames provided to segment_video.")

        # Segment ONLY Frame 0
        first_frame = frames[0]
        initial_out = self.segment(
            first_frame,
            target_prompt=target_prompt,
            background_prompt=background_prompt,
            fallback_background_prompts=fallback_background_prompts,
            **kwargs,
        )

        # Return the same fixed masks for every frame
        n_frames = len(frames)
        target_masks: List[npt.NDArray[np.bool_]] = [
            initial_out.target_mask.copy() for _ in range(n_frames)
        ]
        if initial_out.background_mask is not None:
            background_masks: List[npt.NDArray[np.bool_]] = [
                initial_out.background_mask.copy() for _ in range(n_frames)
            ]
        else:
            background_masks = []

        # Track persistence across video frames
        target_pers = compute_persistence(
            frames=frames,
            bbox=initial_out.target_bbox,
            fps=detected_fps,
            correlation_threshold=correlation_threshold,
            min_persistence_ratio=min_persistence_ratio,
            sample_stride=sample_stride,
        )

        bg_pers = None
        if initial_out.background_bbox is not None:
            bg_pers = compute_persistence(
                frames=frames,
                bbox=initial_out.background_bbox,
                fps=detected_fps,
                correlation_threshold=correlation_threshold,
                min_persistence_ratio=min_persistence_ratio,
                sample_stride=sample_stride,
            )

        metadata = dict(initial_out.metadata)
        metadata["target_persistence"] = target_pers
        metadata["background_persistence"] = bg_pers

        if not target_pers["passed"]:
            failure_msg = (
                f"Target persistence failure: fraction of valid frames ({target_pers['fraction_valid']:.1%}) "
                f"below threshold {target_pers['min_persistence_ratio']:.1%} "
                f"(lowest correlation {target_pers['lowest_correlation']} at frame {target_pers['lowest_frame_idx']}, "
                f"t={target_pers['lowest_timestamp_sec']}s). Target likely occluded or camera panned away."
            )
            metadata["status"] = "failed"
            metadata["failed"] = True
            metadata["is_valid"] = False
            metadata["failure_reason"] = failure_msg
            logger.warning(failure_msg)

            if getattr(self, "raise_on_failure", False):
                raise RuntimeError(failure_msg)
        else:
            metadata["status"] = "completed"
            metadata["failed"] = False
            metadata["is_valid"] = True

        # Optional QA drift check on keyframes
        if qa_keyframes and n_frames > 1:
            ref_gray = _to_gray(first_frame)
            drift_reports = []
            for kf_idx in range(qa_keyframe_interval, n_frames, qa_keyframe_interval):
                curr_gray = _to_gray(frames[kf_idx])
                tdx, tdy = track_roi_displacement(ref_gray, curr_gray, initial_out.target_bbox)
                drift_reports.append({"frame": kf_idx, "target_drift_px": (tdx, tdy)})
            logger.info("QA keyframe drift check (%d keyframes): %s", len(drift_reports), drift_reports)

        result = SegmentResult(
            target_masks=target_masks,
            background_masks=background_masks,
            fps=detected_fps,
            method=initial_out.method,
            metadata=metadata,
        )
        return result


# ---------------------------------------------------------------------------
# Meta SAM 3.1 Implementation
# ---------------------------------------------------------------------------

class MetaSAM31Segmenter(BaseSegmenter):
    """Meta SAM 3.1 implementation using Promptable Concept Segmentation (PCS).

    Calls Meta Model API (sam-3.1) routed via setu/meta_client.py for unified
    mock/cached/live modes, disk caching, usage logging, and budget cap checks.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        api_base_url: Optional[str] = None,
        timeout: int = 25,
        raise_on_failure: bool = False,
        meta_client: Optional[MetaClient] = None,
        fallback_background_prompts: Optional[Sequence[str]] = ("building", "ground", "trees"),
    ) -> None:
        self.api_key = api_key or settings.META_API_KEY
        self.model_name = model_name or getattr(settings, "SAM_MODEL", "sam-3.1")
        self.api_base_url = api_base_url or settings.META_API_BASE_URL
        self.timeout = timeout
        self.raise_on_failure = raise_on_failure
        self.meta_client = meta_client
        self.fallback_background_prompts = (
            list(fallback_background_prompts) if fallback_background_prompts is not None else ["building", "ground", "trees"]
        )

    def _call_api_endpoint(
        self,
        frame: npt.NDArray[np.uint8],
        target_prompt: str,
        background_prompt: str,
        fallback_background_prompts: Optional[Sequence[str]] = None,
    ) -> Optional[SegmentationOutput]:
        """Call Meta SAM 3.1 Model API endpoint via MetaClient."""
        client = self.meta_client or get_meta_client()
        if client.mode == "live" and not client.api_key:
            return None

        # Encode frame 0 as JPEG base64
        ret, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if not ret:
            return None
        image_b64 = base64.b64encode(buf).decode("utf-8")

        # Truncate prompt to max 32 tokens
        t_prompt = " ".join(target_prompt.strip().split()[:32])
        b_prompt = " ".join(background_prompt.strip().split()[:32])

        warnings: List[str] = []
        try:
            target_resp = client.segment_image(image_b64, prompt=t_prompt, model=self.model_name)
            target_instances = parse_sam_grammar(target_resp, frame.shape[:2])

            if not target_instances:
                logger.warning("SAM 3.1 returned no target instances for prompt '%s'.", t_prompt)
                return None

            # Return largest/most central instance as target, keeping all in metadata
            target_inst = select_best_instance(target_instances, frame.shape[:2])
            target_box = target_inst["bbox"]
            target_mask = target_inst["mask"]
            target_conf = target_inst.get("confidence")
            target_source: Literal["sam", "manual", "fallback", "none"] = "sam"

            # Stationary background reference with candidate prompt fallback
            candidate_prompts = [b_prompt]
            fb_candidates = (
                fallback_background_prompts
                if fallback_background_prompts is not None
                else self.fallback_background_prompts
            )
            for fb in fb_candidates:
                fb_clean = " ".join(fb.strip().split()[:32])
                if (
                    fb_clean.lower() != candidate_prompts[0].lower()
                    and fb_clean not in candidate_prompts
                ):
                    candidate_prompts.append(fb_clean)
                    if len(candidate_prompts) >= 3:  # Initial prompt + up to 2 fallbacks
                        break

            bg_instances: List[Dict[str, Any]] = []
            matched_bg_prompt: Optional[str] = None
            background_prompts_attempted: List[Dict[str, Any]] = []

            for p_idx, p_cand in enumerate(candidate_prompts):
                try:
                    bg_resp = client.segment_image(image_b64, prompt=p_cand, model=self.model_name)
                    cand_instances = parse_sam_grammar(bg_resp, frame.shape[:2])
                    matched = len(cand_instances) > 0
                    background_prompts_attempted.append({
                        "prompt": p_cand,
                        "attempt": p_idx + 1,
                        "matched": matched,
                        "instance_count": len(cand_instances),
                        "cost": SAM_COST_PER_IMAGE,
                    })
                    logger.info(
                        "Background SAM call #%d for prompt '%s': matched=%s (%d instances), cost=$%.4f",
                        p_idx + 1,
                        p_cand,
                        matched,
                        len(cand_instances),
                        SAM_COST_PER_IMAGE,
                    )
                    if matched:
                        bg_instances = cand_instances
                        matched_bg_prompt = p_cand
                        break
                except Exception as bg_err:
                    logger.debug("Background SAM call for '%s' failed (%s).", p_cand, bg_err)
                    background_prompts_attempted.append({
                        "prompt": p_cand,
                        "attempt": p_idx + 1,
                        "matched": False,
                        "error": str(bg_err),
                        "cost": 0.0,
                    })

            if bg_instances:
                bg_inst = select_best_instance(bg_instances, frame.shape[:2])
                bg_box = bg_inst["bbox"]
                bg_mask = bg_inst["mask"]
                bg_source: Literal["sam", "manual", "fallback", "none"] = "sam"
            else:
                # Never substitute a hardcoded box
                bg_box = None
                bg_mask = None
                bg_source = "none"
                no_bg_msg = "No stationary background found; include ground, buildings or trees in frame"
                warnings.append(no_bg_msg)
                logger.warning(no_bg_msg)

            # Overall method must not say "sam" if any mask came from elsewhere
            if target_source == "fallback" or bg_source == "fallback":
                method: Literal["sam", "manual", "fallback"] = "fallback"
            elif target_source == "manual" or bg_source == "manual":
                method = "manual"
            elif target_source == "sam":
                method = "sam"
            else:
                method = "fallback"

            bg_prompt_used = matched_bg_prompt if matched_bg_prompt is not None else background_prompt
            return SegmentationOutput(
                target_mask=target_mask,
                background_mask=bg_mask,
                target_bbox=target_box,
                background_bbox=bg_box,
                target_prompt=target_prompt,
                background_prompt=bg_prompt_used,
                confidence=target_conf,
                method=method,
                target_source=target_source,
                background_source=bg_source,
                warnings=warnings,
                metadata={
                    "source": "meta_sam_3.1_api",
                    "model": self.model_name,
                    "mode": client.mode,
                    "target_source": target_source,
                    "background_source": bg_source,
                    "warnings": warnings,
                    "all_target_instances": target_instances,
                    "all_background_instances": bg_instances,
                    "target_instance_count": len(target_instances),
                    "background_instance_count": len(bg_instances),
                    "background_prompts_attempted": background_prompts_attempted,
                    "matched_background_prompt": matched_bg_prompt,
                    "background_sam_calls": len(background_prompts_attempted),
                    "background_sam_cost": sum(a.get("cost", 0.0) for a in background_prompts_attempted),
                    "parser": "meta_sam_parser" if HAS_META_SAM_PARSER else "builtin_grammar_decoder",
                },
            )
        except BudgetExceededError:
            raise
        except Exception as exc:
            logger.warning("Meta SAM API call via MetaClient failed (%s).", exc)
            return None

    def _heuristic_semantic_segmentation(
        self,
        frame: npt.NDArray[np.uint8],
        target_prompt: str,
        background_prompt: str,
    ) -> SegmentationOutput:
        """Adaptive prompt-guided heuristic segmentation for offline/fallback use.

        Returns method='fallback' and confidence=None (no made-up confidence).
        """
        h, w = frame.shape[:2]
        tp_lower = target_prompt.lower()
        bp_lower = background_prompt.lower()

        if any(kw in tp_lower for kw in ("cable", "stay", "tendon", "wire")):
            tw, th = int(w * 0.15), int(h * 0.35)
            tx, ty = int(w * 0.45), int(h * 0.20)
        elif any(kw in tp_lower for kw in ("deck", "girder", "beam", "slab", "span")):
            tw, th = int(w * 0.40), int(h * 0.15)
            tx, ty = int(w * 0.30), int(h * 0.45)
        elif any(kw in tp_lower for kw in ("pier", "pillar", "column", "abutment", "bearing")):
            tw, th = int(w * 0.20), int(h * 0.35)
            tx, ty = int(w * 0.40), int(h * 0.35)
        else:
            tw, th = int(w * 0.25), int(h * 0.25)
            tx, ty = int(w * 0.375), int(h * 0.375)

        if any(kw in bp_lower for kw in ("sky", "cloud", "horizon")):
            bw, bh = int(w * 0.20), int(h * 0.15)
            bx, by = int(w * 0.75), int(h * 0.05)
        elif any(kw in bp_lower for kw in ("building", "tower", "structure")):
            bw, bh = int(w * 0.15), int(h * 0.20)
            bx, by = int(w * 0.80), int(h * 0.25)
        else:
            bw, bh = int(w * 0.20), int(h * 0.15)
            bx, by = int(w * 0.05), int(h * 0.80)

        target_mask = bbox_to_mask((tx, ty, tw, th), (h, w))
        background_mask = bbox_to_mask((bx, by, bw, bh), (h, w))

        gray = _to_gray(frame)
        target_roi = gray[ty : ty + th, tx : tx + tw]
        if target_roi.size > 0 and np.std(target_roi) > 5.0:
            _, binary_roi = cv2.threshold(target_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            if 0.05 < np.mean(binary_roi > 0) < 0.95:
                refined_target = np.zeros((h, w), dtype=bool)
                refined_target[ty : ty + th, tx : tx + tw] = binary_roi > 0
                target_mask = refined_target

        return SegmentationOutput(
            target_mask=target_mask,
            background_mask=background_mask,
            target_bbox=(tx, ty, tw, th),
            background_bbox=(bx, by, bw, bh),
            target_prompt=target_prompt,
            background_prompt=background_prompt,
            confidence=None,  # No made-up confidence
            method="fallback",
            target_source="fallback",
            background_source="fallback",
            warnings=["Using offline heuristic fallback; real SAM 3.1 unreachable"],
            metadata={
                "mode": "sam3.1_concept_heuristic",
                "target_concept": target_prompt,
                "background_concept": background_prompt,
                "target_source": "fallback",
                "background_source": "fallback",
                "fallback_reason": "Real SAM 3.1 unreachable; offline heuristic used",
            },
        )

    def segment(
        self,
        frame: npt.NDArray[np.uint8],
        target_prompt: str = "corner of concrete pillar",
        background_prompt: str = "stationary ground reference",
        fallback_background_prompts: Optional[Sequence[str]] = None,
    ) -> SegmentationOutput:
        """Segment target and background using Meta SAM 3.1 API or offline concept heuristic."""
        client = self.meta_client or get_meta_client()
        if client.mode == "mock" or client.api_key:
            api_res = self._call_api_endpoint(
                frame,
                target_prompt,
                background_prompt,
                fallback_background_prompts=fallback_background_prompts,
            )
            if api_res is not None:
                return api_res

        if self.raise_on_failure:
            raise RuntimeError(
                f"Real Meta SAM 3.1 model/API not reachable and raise_on_failure=True."
            )

        logger.warning(
            "Real Meta SAM 3.1 API/model not reachable; using offline heuristic fallback. Output method flagged as 'fallback'."
        )
        return self._heuristic_semantic_segmentation(frame, target_prompt, background_prompt)


# ---------------------------------------------------------------------------
# Manual Segmenter Implementation
# ---------------------------------------------------------------------------

class ManualSegmenter(BaseSegmenter):
    """User-guided manual region-of-interest segmentation."""

    def __init__(
        self,
        target_bbox: Optional[Tuple[int, int, int, int]] = None,
        background_bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> None:
        self.target_bbox = target_bbox
        self.background_bbox = background_bbox

    def segment(
        self,
        frame: npt.NDArray[np.uint8],
        target_prompt: str = "manual target ROI",
        background_prompt: str = "manual background ROI",
        **kwargs: Any,
    ) -> SegmentationOutput:
        """Generate masks directly from specified manual bounding boxes."""
        h, w = frame.shape[:2]

        t_box = self.target_bbox or (int(w * 0.35), int(h * 0.35), int(w * 0.30), int(h * 0.30))
        b_box = self.background_bbox or (int(w * 0.05), int(h * 0.75), int(w * 0.20), int(h * 0.20))

        t_mask = bbox_to_mask(t_box, (h, w))
        b_mask = bbox_to_mask(b_box, (h, w))

        return SegmentationOutput(
            target_mask=t_mask,
            background_mask=b_mask,
            target_bbox=t_box,
            background_bbox=b_box,
            target_prompt=target_prompt,
            background_prompt=background_prompt,
            confidence=None,  # No made-up confidence
            method="manual",
            target_source="manual",
            background_source="manual",
            warnings=[],
            metadata={"mode": "manual_roi", "target_source": "manual", "background_source": "manual"},
        )


# ---------------------------------------------------------------------------
# Fallback Segmenter Implementation
# ---------------------------------------------------------------------------

class FallbackSegmenter(BaseSegmenter):
    """Local, offline-friendly deterministic fallback segmenter.

    Generates reproducible geometric ROIs. Always sets method="fallback" and confidence=None.
    """

    def segment(
        self,
        frame: npt.NDArray[np.uint8],
        target_prompt: str = "corner of concrete pillar",
        background_prompt: str = "stationary ground reference",
        **kwargs: Any,
    ) -> SegmentationOutput:
        """Generate deterministic target and background regions from frame dimensions."""
        h, w = frame.shape[:2]

        tw, th = int(w * 0.25), int(h * 0.25)
        tx, ty = int(w * 0.375), int(h * 0.375)
        target_mask = bbox_to_mask((tx, ty, tw, th), (h, w))

        bw, bh = int(w * 0.20), int(h * 0.15)
        bx, by = int(w * 0.05), int(h * 0.80)
        background_mask = bbox_to_mask((bx, by, bw, bh), (h, w))

        return SegmentationOutput(
            target_mask=target_mask,
            background_mask=background_mask,
            target_bbox=(tx, ty, tw, th),
            background_bbox=(bx, by, bw, bh),
            target_prompt=target_prompt,
            background_prompt=background_prompt,
            confidence=None,  # No made-up confidence
            method="fallback",
            target_source="fallback",
            background_source="fallback",
            warnings=[],
            metadata={"mode": "fallback_offline_segmenter", "target_source": "fallback", "background_source": "fallback"},
        )


# ---------------------------------------------------------------------------
# Factory & High-Level Convenience API
# ---------------------------------------------------------------------------

def get_segmenter(
    method: Literal["sam", "manual", "fallback"] = "sam",
    target_bbox: Optional[Tuple[int, int, int, int]] = None,
    background_bbox: Optional[Tuple[int, int, int, int]] = None,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    raise_on_failure: bool = False,
    meta_client: Optional[MetaClient] = None,
    fallback_background_prompts: Optional[Sequence[str]] = None,
) -> BaseSegmenter:
    """Factory providing the configured segmenter implementation."""
    if method == "manual":
        return ManualSegmenter(target_bbox=target_bbox, background_bbox=background_bbox)

    if method == "fallback":
        return FallbackSegmenter()

    return MetaSAM31Segmenter(
        api_key=api_key,
        model_name=model_name,
        raise_on_failure=raise_on_failure,
        meta_client=meta_client,
        fallback_background_prompts=fallback_background_prompts,
    )


def segment_video(
    video_source: Union[str, Path, Sequence[npt.NDArray[np.uint8]], npt.NDArray[np.uint8]],
    target_prompt: str = "corner of concrete pillar",
    background_prompt: str = "stationary ground reference",
    fps: float = 30.0,
    method: Literal["sam", "manual", "fallback"] = "sam",
    target_bbox: Optional[Tuple[int, int, int, int]] = None,
    background_bbox: Optional[Tuple[int, int, int, int]] = None,
    raise_on_failure: bool = False,
    meta_client: Optional[MetaClient] = None,
    fallback_background_prompts: Optional[Sequence[str]] = None,
    correlation_threshold: float = 0.70,
    min_persistence_ratio: float = 0.85,
    sample_stride: Optional[int] = None,
) -> SegmentResult:
    """Convenience function to segment video with fixed masks per frame."""
    segmenter = get_segmenter(
        method=method,
        target_bbox=target_bbox,
        background_bbox=background_bbox,
        raise_on_failure=raise_on_failure,
        meta_client=meta_client,
        fallback_background_prompts=fallback_background_prompts,
    )
    return segmenter.segment_video(
        video_source=video_source,
        target_prompt=target_prompt,
        background_prompt=background_prompt,
        fps=fps,
        fallback_background_prompts=fallback_background_prompts,
        correlation_threshold=correlation_threshold,
        min_persistence_ratio=min_persistence_ratio,
        sample_stride=sample_stride,
    )


def segment_manual(
    video_source: Union[str, Path, Sequence[npt.NDArray[np.uint8]], npt.NDArray[np.uint8]],
    target_bbox: Tuple[int, int, int, int],
    background_bbox: Tuple[int, int, int, int],
    fps: float = 30.0,
) -> SegmentResult:
    """Convenience function for manual ROI segmentation returning SegmentResult with method='manual'."""
    return segment_video(
        video_source=video_source,
        target_prompt="manual target ROI",
        background_prompt="manual background ROI",
        fps=fps,
        method="manual",
        target_bbox=target_bbox,
        background_bbox=background_bbox,
    )
