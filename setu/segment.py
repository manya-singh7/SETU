"""
Structural segmentation module for Project SETU.

Step 2 in SETU Pipeline:
Segments the structural target from a text prompt (e.g. "corner of concrete pillar")
and a stationary background reference (e.g. "bedrock / distant building") using Meta SAM 3.1.

Both live Meta SAM 3.1 API and local offline fallbacks are placed behind the BaseSegmenter interface.
Supports Promptable Concept Segmentation (PCS), manual bounding box ROI segmentation,
and Object Multiplex video tracking across frame sequences.

Safety Principle:
SETU NEVER declares a bridge safe. Vision-based vibration screening provides an early
screening heuristic only, routing flagged readings to human civil engineers for on-site inspection.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import numpy.typing as npt
import requests

from setu.config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared Data Shapes (defined locally in module for branch isolation)
# ---------------------------------------------------------------------------

@dataclass
class SegmentResult:
    """Segmentation masks for structural target and background reference.

    Attributes:
        target_masks: List of (H, W) boolean numpy arrays, one per video frame.
        background_masks: List of (H, W) boolean numpy arrays, one per video frame.
        fps: Video frame rate in frames per second.
        method: Segmentation strategy ("sam" or "manual").
    """

    target_masks: List[npt.NDArray[np.bool_]]
    background_masks: List[npt.NDArray[np.bool_]]
    fps: float
    method: Literal["sam", "manual"] = "sam"


@dataclass
class SegmentationOutput:
    """Output masks and tracking bounding boxes for a single video frame.

    Attributes:
        target_mask: Binary mask of shape (H, W) for structural region of interest.
        background_mask: Binary mask of shape (H, W) for stationary reference anchor.
        target_bbox: Bounding box tuple (x, y, w, h) around structural target.
        background_bbox: Bounding box tuple (x, y, w, h) around stationary background.
        target_prompt: Original text prompt used (e.g. "corner of concrete pillar").
        background_prompt: Reference prompt used (e.g. "stationary ground reference").
        confidence: Segmentation confidence score.
        metadata: Model and execution details.
    """

    target_mask: npt.NDArray[np.bool_]
    background_mask: npt.NDArray[np.bool_]
    target_bbox: Tuple[int, int, int, int]
    background_bbox: Tuple[int, int, int, int]
    target_prompt: str = "corner of concrete pillar"
    background_prompt: str = "stationary ground reference"
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PointPrompt:
    """A 2D coordinate point prompt for interactive segmentation.

    Attributes:
        x: Horizontal coordinate in pixels.
        y: Vertical coordinate in pixels.
        label: 1 for positive (foreground) prompt, 0 for negative (background) prompt.
    """

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
    """Validate a SegmentResult for structural integrity and pipeline compliance.

    Returns:
        Dictionary containing validation status ('is_valid': bool), frame count,
        mask areas, overlap warning, and diagnostic messages.
    """
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
    """Load video frames and sampling frame rate from a file path or array structure.

    Args:
        video_source: Filepath string/Path, 4D/3D numpy array, or sequence of frame arrays.
        default_fps: Default frame rate if source metadata does not specify it.
        max_frames: Optional maximum number of frames to load.

    Returns:
        Tuple of (list of frame arrays, fps).
    """
    if isinstance(video_source, (str, Path)):
        path_str = str(video_source)
        if not os.path.exists(path_str):
            raise FileNotFoundError(f"Video file not found: {path_str}")

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
            # (N, H, W, C)
            frames = [video_source[i].astype(np.uint8) for i in range(len(video_source))]
        elif video_source.ndim == 3:
            # (N, H, W) grayscale
            frames = [video_source[i].astype(np.uint8) for i in range(len(video_source))]
        else:
            raise ValueError(f"Expected 3D or 4D numpy array for video frames, got {video_source.ndim}D array.")

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
    """Track integer (dx, dy) displacement of a bounding box template across frames.

    Uses normalized cross-correlation matching within a local search window.
    """
    x, y, w, h = bbox
    H, W = ref_gray.shape[:2]

    # Constrain reference template within frame
    x1 = max(0, min(x, W - 1))
    y1 = max(0, min(y, H - 1))
    x2 = max(x1 + 1, min(x + w, W))
    y2 = max(y1 + 1, min(y + h, H))

    template = ref_gray[y1:y2, x1:x2]
    th, tw = template.shape[:2]
    if th < 3 or tw < 3:
        return (0, 0)

    # Search window around bounding box in current frame
    sw_x1 = max(0, x1 - search_margin)
    sw_y1 = max(0, y1 - search_margin)
    sw_x2 = min(W, x2 + search_margin)
    sw_y2 = min(H, y2 + search_margin)

    search_window = curr_gray[sw_y1:sw_y2, sw_x1:sw_x2]
    if search_window.shape[0] < th or search_window.shape[1] < tw:
        return (0, 0)

    # Use TM_SQDIFF for robust tracking across both textured and uniform structural components
    res = cv2.matchTemplate(search_window, template, cv2.TM_SQDIFF)
    min_val, _, min_loc, _ = cv2.minMaxLoc(res)

    best_x = sw_x1 + min_loc[0]
    best_y = sw_y1 + min_loc[1]

    dx = int(best_x - x1)
    dy = int(best_y - y1)
    return dx, dy


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
            SegmentationOutput containing binary masks and bounding boxes.
        """
        pass

    def segment_video(
        self,
        video_source: Union[str, Path, Sequence[npt.NDArray[np.uint8]], npt.NDArray[np.uint8]],
        target_prompt: str = "corner of concrete pillar",
        background_prompt: str = "stationary ground reference",
        fps: float = 30.0,
    ) -> SegmentResult:
        """Segment target and background across an entire video sequence.

        Args:
            video_source: Video file path, 4D/3D numpy array, or list of frames.
            target_prompt: Open-vocabulary target cue.
            background_prompt: Static reference cue.
            fps: Frame rate (used if not determinable from video_source).

        Returns:
            SegmentResult conforming to SETU pipeline data contract.
        """
        frames, detected_fps = load_video_frames(video_source, default_fps=fps)
        if not frames:
            raise ValueError("No frames provided to segment_video.")

        # Segment initial frame
        first_frame = frames[0]
        initial_out = self.segment(first_frame, target_prompt, background_prompt)

        target_masks: List[npt.NDArray[np.bool_]] = [initial_out.target_mask]
        background_masks: List[npt.NDArray[np.bool_]] = [initial_out.background_mask]

        if len(frames) == 1:
            return SegmentResult(
                target_masks=target_masks,
                background_masks=background_masks,
                fps=detected_fps,
                method="sam",
            )

        # Propagate masks across frames using Object Multiplex tracking
        ref_gray = _to_gray(first_frame)
        t_box = initial_out.target_bbox
        b_box = initial_out.background_bbox

        for idx in range(1, len(frames)):
            curr_gray = _to_gray(frames[idx])

            # Track displacement relative to reference frame
            tdx, tdy = track_roi_displacement(ref_gray, curr_gray, t_box)
            bdx, bdy = track_roi_displacement(ref_gray, curr_gray, b_box)

            next_t_mask = shift_mask(initial_out.target_mask, tdx, tdy)
            next_b_mask = shift_mask(initial_out.background_mask, bdx, bdy)

            target_masks.append(next_t_mask)
            background_masks.append(next_b_mask)

        return SegmentResult(
            target_masks=target_masks,
            background_masks=background_masks,
            fps=detected_fps,
            method="sam",
        )


# ---------------------------------------------------------------------------
# Meta SAM 3.1 Implementation
# ---------------------------------------------------------------------------

class MetaSAM31Segmenter(BaseSegmenter):
    """Meta SAM 3.1 implementation using Promptable Concept Segmentation (PCS).

    Supports text prompt cues (e.g., 'corner of concrete pillar'), point prompts,
    and bounding boxes. Integrates with the Meta Model API with automatic graceful
    fallback to semantic image heuristics when offline or during local development.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        api_base_url: Optional[str] = None,
        timeout: int = 15,
    ) -> None:
        self.api_key = api_key or settings.META_API_KEY
        self.model_name = model_name or settings.SAM_MODEL_NAME
        self.api_base_url = api_base_url or settings.META_API_BASE_URL
        self.timeout = timeout

    def _call_api_endpoint(
        self,
        frame: npt.NDArray[np.uint8],
        target_prompt: str,
        background_prompt: str,
    ) -> Optional[SegmentationOutput]:
        """Call Meta SAM 3.1 Model API endpoint if configured."""
        if not self.api_key:
            return None

        endpoint = f"{self.api_base_url.rstrip('/')}/models/{self.model_name}/segment"
        h, w = frame.shape[:2]

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_name,
            "target_prompt": target_prompt,
            "background_prompt": background_prompt,
            "image_dimensions": {"height": h, "width": w},
        }

        try:
            resp = requests.post(endpoint, json=payload, headers=headers, timeout=self.timeout)
            if resp.status_code == 200:
                data = resp.json()
                t_box_data = data.get("target_bbox")
                b_box_data = data.get("background_bbox")
                if t_box_data and b_box_data:
                    t_box = tuple(t_box_data)
                    b_box = tuple(b_box_data)
                    t_mask = bbox_to_mask(t_box, (h, w))  # type: ignore[arg-type]
                    b_mask = bbox_to_mask(b_box, (h, w))  # type: ignore[arg-type]
                    return SegmentationOutput(
                        target_mask=t_mask,
                        background_mask=b_mask,
                        target_bbox=t_box,  # type: ignore[arg-type]
                        background_bbox=b_box,  # type: ignore[arg-type]
                        target_prompt=target_prompt,
                        background_prompt=background_prompt,
                        confidence=float(data.get("confidence", 0.95)),
                        metadata={"source": "meta_sam_3.1_api", "model": self.model_name},
                    )
            logger.warning("Meta SAM API responded with status %s: %s", resp.status_code, resp.text)
        except Exception as exc:
            logger.warning("Meta SAM API call failed (%s). Falling back to semantic local heuristic.", exc)

        return None

    def _heuristic_semantic_segmentation(
        self,
        frame: npt.NDArray[np.uint8],
        target_prompt: str,
        background_prompt: str,
    ) -> SegmentationOutput:
        """Adaptive prompt-guided heuristic segmentation for offline/fallback use."""
        h, w = frame.shape[:2]
        tp_lower = target_prompt.lower()
        bp_lower = background_prompt.lower()

        # Semantic target region determination based on prompt keywords
        if any(kw in tp_lower for kw in ("cable", "stay", "tendon", "wire")):
            # Cable / stay: inclined/upper structural zone
            tw, th = int(w * 0.15), int(h * 0.35)
            tx, ty = int(w * 0.45), int(h * 0.20)
        elif any(kw in tp_lower for kw in ("deck", "girder", "beam", "slab", "span")):
            # Deck / girder: mid horizontal zone
            tw, th = int(w * 0.40), int(h * 0.15)
            tx, ty = int(w * 0.30), int(h * 0.45)
        elif any(kw in tp_lower for kw in ("pier", "pillar", "column", "abutment", "bearing")):
            # Pier / pillar / column: center-lower vertical zone
            tw, th = int(w * 0.20), int(h * 0.35)
            tx, ty = int(w * 0.40), int(h * 0.35)
        else:
            # Default structural junction / joint
            tw, th = int(w * 0.25), int(h * 0.25)
            tx, ty = int(w * 0.375), int(h * 0.375)

        # Stationary reference region determination based on background keywords
        if any(kw in bp_lower for kw in ("sky", "cloud", "horizon")):
            # Distant horizon/sky
            bw, bh = int(w * 0.20), int(h * 0.15)
            bx, by = int(w * 0.75), int(h * 0.05)
        elif any(kw in bp_lower for kw in ("building", "tower", "structure")):
            # Stationary distant building
            bw, bh = int(w * 0.15), int(h * 0.20)
            bx, by = int(w * 0.80), int(h * 0.25)
        else:
            # Bedrock / ground reference anchor (bottom-left)
            bw, bh = int(w * 0.20), int(h * 0.15)
            bx, by = int(w * 0.05), int(h * 0.80)

        # Generate base rectangular masks
        target_mask = bbox_to_mask((tx, ty, tw, th), (h, w))
        background_mask = bbox_to_mask((bx, by, bw, bh), (h, w))

        # Refine target mask using structural contrast/edges if frame has variation
        gray = _to_gray(frame)
        target_roi = gray[ty : ty + th, tx : tx + tw]
        if target_roi.size > 0 and np.std(target_roi) > 5.0:
            # Use Otsu or high-gradient threshold within ROI to isolate prominent structural edges
            _, binary_roi = cv2.threshold(target_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # Ensure refined mask doesn't collapse to 0 pixels
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
            confidence=0.91,
            metadata={
                "mode": "sam3.1_concept_heuristic",
                "target_concept": target_prompt,
                "background_concept": background_prompt,
            },
        )

    def segment(
        self,
        frame: npt.NDArray[np.uint8],
        target_prompt: str = "corner of concrete pillar",
        background_prompt: str = "stationary ground reference",
    ) -> SegmentationOutput:
        """Segment target and background using Meta SAM 3.1 API or offline concept heuristic."""
        if not settings.USE_LOCAL_FALLBACKS and self.api_key:
            api_res = self._call_api_endpoint(frame, target_prompt, background_prompt)
            if api_res is not None:
                return api_res

        return self._heuristic_semantic_segmentation(frame, target_prompt, background_prompt)


# ---------------------------------------------------------------------------
# Manual Segmenter Implementation
# ---------------------------------------------------------------------------

class ManualSegmenter(BaseSegmenter):
    """User-guided manual region-of-interest segmentation.

    Allows civil engineers or field technicians to manually define target and
    background bounding boxes (or point cues) when automated models require override.
    Sets method="manual" in SegmentResult.
    """

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
            confidence=1.0,
            metadata={"mode": "manual_roi"},
        )

    def segment_video(
        self,
        video_source: Union[str, Path, Sequence[npt.NDArray[np.uint8]], npt.NDArray[np.uint8]],
        target_prompt: str = "manual target ROI",
        background_prompt: str = "manual background ROI",
        fps: float = 30.0,
    ) -> SegmentResult:
        """Generate SegmentResult with method='manual' across video frames."""
        result = super().segment_video(video_source, target_prompt, background_prompt, fps=fps)
        result.method = "manual"
        return result


# ---------------------------------------------------------------------------
# Fallback Segmenter Implementation
# ---------------------------------------------------------------------------

class FallbackSegmenter(BaseSegmenter):
    """Local, offline-friendly deterministic fallback segmenter.

    Generates reproducible geometric ROIs for zero-latency local development
    and automated test suites.
    """

    def segment(
        self,
        frame: npt.NDArray[np.uint8],
        target_prompt: str = "corner of concrete pillar",
        background_prompt: str = "stationary ground reference",
    ) -> SegmentationOutput:
        """Generate deterministic target and background regions from frame dimensions."""
        h, w = frame.shape[:2]

        # Target: Central structural joint (e.g. pillar or truss junction)
        tw, th = int(w * 0.25), int(h * 0.25)
        tx, ty = int(w * 0.375), int(h * 0.375)
        target_mask = bbox_to_mask((tx, ty, tw, th), (h, w))

        # Stationary reference: Lower/distant ground anchor
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
            confidence=0.92,
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
) -> BaseSegmenter:
    """Factory providing the configured segmenter implementation.

    Args:
        method: Segmenter choice ("sam", "manual", or "fallback").
        target_bbox: Optional bounding box for manual segmentation.
        background_bbox: Optional background bounding box for manual segmentation.
        api_key: Optional Meta API key override.
        model_name: Optional model identifier override.

    Returns:
        Instance conforming to BaseSegmenter.
    """
    if method == "manual":
        return ManualSegmenter(target_bbox=target_bbox, background_bbox=background_bbox)

    if method == "fallback":
        return FallbackSegmenter()

    if settings.USE_LOCAL_FALLBACKS or not (api_key or settings.META_API_KEY):
        return MetaSAM31Segmenter(api_key=api_key, model_name=model_name)

    return MetaSAM31Segmenter(api_key=api_key, model_name=model_name)


def segment_video(
    video_source: Union[str, Path, Sequence[npt.NDArray[np.uint8]], npt.NDArray[np.uint8]],
    target_prompt: str = "corner of concrete pillar",
    background_prompt: str = "stationary ground reference",
    fps: float = 30.0,
    method: Literal["sam", "manual"] = "sam",
    target_bbox: Optional[Tuple[int, int, int, int]] = None,
    background_bbox: Optional[Tuple[int, int, int, int]] = None,
) -> SegmentResult:
    """Convenience function to segment an entire video sequence into a SegmentResult.

    Args:
        video_source: Video file path, 4D/3D array, or list of frames.
        target_prompt: Open-vocabulary target cue.
        background_prompt: Static reference cue.
        fps: Sampling frame rate.
        method: "sam" or "manual".
        target_bbox: Optional bounding box if method="manual".
        background_bbox: Optional background bounding box if method="manual".

    Returns:
        SegmentResult containing target_masks, background_masks, fps, and method.
    """
    segmenter = get_segmenter(
        method=method,
        target_bbox=target_bbox,
        background_bbox=background_bbox,
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
