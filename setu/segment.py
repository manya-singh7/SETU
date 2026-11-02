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
from setu.meta_client import BudgetExceededError, MetaAPIError, MetaClient, get_meta_client
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
        background_mask: Binary mask of shape (H, W) for stationary reference anchor.
        target_bbox: Bounding box tuple (x, y, w, h) around structural target.
        background_bbox: Bounding box tuple (x, y, w, h) around stationary background.
        target_prompt: Text prompt used (e.g. "corner of concrete pillar").
        background_prompt: Reference prompt used (e.g. "stationary ground reference").
        confidence: Optional confidence score reported by SAM, or None if fallback.
        method: Method used to generate segmentation ("sam", "manual", or "fallback").
        metadata: Model, instance candidates, and execution details.
    """

    target_mask: npt.NDArray[np.bool_]
    background_mask: npt.NDArray[np.bool_]
    target_bbox: Tuple[int, int, int, int]
    background_bbox: Tuple[int, int, int, int]
    target_prompt: str = "corner of concrete pillar"
    background_prompt: str = "stationary ground reference"
    confidence: Optional[float] = None
    method: Literal["sam", "manual", "fallback"] = "fallback"
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


def compute_mask_area(mask: npt.NDArray[np.bool_]) -> int:
    """Return total active pixel count of a binary mask."""
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
) -> Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]:
    """Extract (target_bbox, background_bbox) from a SegmentResult for a given frame index."""
    if not result.target_masks or not result.background_masks:
        raise ValueError("SegmentResult contains empty mask lists.")
    idx = max(0, min(frame_idx, len(result.target_masks) - 1))
    t_box = mask_to_bbox(result.target_masks[idx])
    b_box = mask_to_bbox(result.background_masks[idx])
    return t_box, b_box


def validate_segment_result(result: SegmentResult) -> Dict[str, Any]:
    """Validate a SegmentResult for structural integrity and pipeline compliance."""
    n_target = len(result.target_masks)
    n_bg = len(result.background_masks)
    errors: List[str] = []
    warnings: List[str] = []

    if n_target == 0 or n_bg == 0:
        errors.append("Target or background mask list is empty.")
    elif n_target != n_bg:
        errors.append(f"Mismatched mask counts: target has {n_target}, background has {n_bg}.")

    if result.fps <= 0:
        errors.append(f"Invalid frame rate fps={result.fps}; must be positive.")

    avg_t_area = 0.0
    avg_b_area = 0.0
    max_iou = 0.0

    if n_target > 0 and n_bg > 0 and n_target == n_bg:
        t_areas = [compute_mask_area(m) for m in result.target_masks]
        b_areas = [compute_mask_area(m) for m in result.background_masks]
        avg_t_area = float(np.mean(t_areas))
        avg_b_area = float(np.mean(b_areas))

        if min(t_areas) == 0:
            warnings.append("One or more target masks contain zero active pixels.")
        if min(b_areas) == 0:
            warnings.append("One or more background masks contain zero active pixels.")

        ious = [compute_mask_iou(tm, bm) for tm, bm in zip(result.target_masks, result.background_masks)]
        max_iou = float(max(ious)) if ious else 0.0
        if max_iou > 0.15:
            warnings.append(f"High overlap between target and stationary background (max IoU: {max_iou:.2f}).")

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

    if not text:
        return instances

    # Parse special token box syntax: <|box;x1=...;y1=...;x2=...;y2=...;w=...;h=...|>
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

            score_val = params.get("score") or params.get("confidence") or resp.get("confidence")
            confidence = float(score_val) if score_val is not None else None

            mask = None
            if HAS_META_SAM_PARSER and idx < len(mask_matches):
                try:
                    mask_raw = mask_matches[idx].group(1)
                    raster = meta_sam_parser.decode_mask_to_raster(mask_raw)  # type: ignore[attr-defined]
                    if isinstance(raster, np.ndarray) and raster.shape != frame_shape:
                        raster = cv2.resize(raster.astype(np.uint8), (w_frame, h_frame), interpolation=cv2.INTER_NEAREST) > 0
                    mask = raster
                except Exception as decode_exc:
                    logger.debug("meta_sam_parser raster decoding error: %s", decode_exc)
                    mask = None

            if mask is None:
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
    ) -> SegmentationOutput:
        """Segment target and stationary reference from a single video frame.

        Args:
            frame: Video frame array (H, W, 3) or (H, W) in uint8 format.
            target_prompt: Open-vocabulary text description of structural target.
            background_prompt: Text description of static background reference anchor.

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
    ) -> SegmentResult:
        """Segment target and background across an entire video sequence.

        Segments ONLY the first frame (frame 0) and returns the same fixed masks
        for every frame. Motion stabilization and displacement tracking happen
        downstream (in setu/motion.py), so masks must remain fixed per frame.
        """
        frames, detected_fps = load_video_frames(video_source, default_fps=fps)
        if not frames:
            raise ValueError("No frames provided to segment_video.")

        # Segment ONLY Frame 0
        first_frame = frames[0]
        initial_out = self.segment(first_frame, target_prompt, background_prompt)

        # Return the same fixed masks for every frame
        n_frames = len(frames)
        target_masks: List[npt.NDArray[np.bool_]] = [
            initial_out.target_mask.copy() for _ in range(n_frames)
        ]
        background_masks: List[npt.NDArray[np.bool_]] = [
            initial_out.background_mask.copy() for _ in range(n_frames)
        ]

        # Optional QA drift check on keyframes
        if qa_keyframes and n_frames > 1:
            ref_gray = _to_gray(first_frame)
            drift_reports = []
            for kf_idx in range(qa_keyframe_interval, n_frames, qa_keyframe_interval):
                curr_gray = _to_gray(frames[kf_idx])
                tdx, tdy = track_roi_displacement(ref_gray, curr_gray, initial_out.target_bbox)
                drift_reports.append({"frame": kf_idx, "target_drift_px": (tdx, tdy)})
            logger.info("QA keyframe drift check (%d keyframes): %s", len(drift_reports), drift_reports)

        return SegmentResult(
            target_masks=target_masks,
            background_masks=background_masks,
            fps=detected_fps,
            method=initial_out.method,
        )


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
    ) -> None:
        self.api_key = api_key or settings.META_API_KEY
        self.model_name = model_name or getattr(settings, "SAM_MODEL", "sam-3.1")
        self.api_base_url = api_base_url or settings.META_API_BASE_URL
        self.timeout = timeout
        self.raise_on_failure = raise_on_failure
        self.meta_client = meta_client

    def _call_api_endpoint(
        self,
        frame: npt.NDArray[np.uint8],
        target_prompt: str,
        background_prompt: str,
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

            # Stationary background reference
            bg_instances: List[Dict[str, Any]] = []
            try:
                bg_resp = client.segment_image(image_b64, prompt=b_prompt, model=self.model_name)
                bg_instances = parse_sam_grammar(bg_resp, frame.shape[:2])
            except Exception as bg_err:
                logger.debug("Background SAM call failed (%s); using default ground anchor.", bg_err)

            h, w = frame.shape[:2]
            if bg_instances:
                bg_inst = select_best_instance(bg_instances, frame.shape[:2])
                bg_box = bg_inst["bbox"]
                bg_mask = bg_inst["mask"]
            else:
                bw, bh = int(w * 0.20), int(h * 0.15)
                bx, by = int(w * 0.05), int(h * 0.80)
                bg_box = (bx, by, bw, bh)
                bg_mask = bbox_to_mask(bg_box, (h, w))

            return SegmentationOutput(
                target_mask=target_mask,
                background_mask=bg_mask,
                target_bbox=target_box,
                background_bbox=bg_box,
                target_prompt=target_prompt,
                background_prompt=background_prompt,
                confidence=target_conf,
                method="sam",
                metadata={
                    "source": "meta_sam_3.1_api",
                    "model": self.model_name,
                    "mode": client.mode,
                    "all_target_instances": target_instances,
                    "all_background_instances": bg_instances,
                    "target_instance_count": len(target_instances),
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
            metadata={
                "mode": "sam3.1_concept_heuristic",
                "target_concept": target_prompt,
                "background_concept": background_prompt,
                "fallback_reason": "Real SAM 3.1 unreachable; offline heuristic used",
            },
        )

    def segment(
        self,
        frame: npt.NDArray[np.uint8],
        target_prompt: str = "corner of concrete pillar",
        background_prompt: str = "stationary ground reference",
    ) -> SegmentationOutput:
        """Segment target and background using Meta SAM 3.1 API or offline concept heuristic."""
        client = self.meta_client or get_meta_client()
        if client.mode == "mock" or client.api_key:
            api_res = self._call_api_endpoint(frame, target_prompt, background_prompt)
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
            metadata={"mode": "manual_roi"},
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
            metadata={"mode": "fallback_offline_segmenter"},
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
) -> SegmentResult:
    """Convenience function to segment video with fixed masks per frame."""
    segmenter = get_segmenter(
        method=method,
        target_bbox=target_bbox,
        background_bbox=background_bbox,
        raise_on_failure=raise_on_failure,
        meta_client=meta_client,
    )
    return segmenter.segment_video(
        video_source=video_source,
        target_prompt=target_prompt,
        background_prompt=background_prompt,
        fps=fps,
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
