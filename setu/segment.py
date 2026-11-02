"""
Structural segmentation module for Project SETU.

Step 2 in SETU Pipeline:
Segments the structural target from a text prompt (e.g. "corner of concrete pillar")
and a stationary background reference (e.g. "bedrock / distant building") using Meta SAM 3.1.

Both live Meta SAM 3.1 and local fallbacks are placed behind BaseSegmenter interface.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple
import numpy as np
import numpy.typing as npt

from setu.config import settings


@dataclass
class SegmentationOutput:
    """Output masks and tracking bounding boxes for target and background reference.

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
    target_prompt: str
    background_prompt: str
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseSegmenter(ABC):
    """Abstract interface for structural target and background reference segmentation."""

    @abstractmethod
    def segment(
        self,
        frame: npt.NDArray[np.uint8],
        target_prompt: str = "corner of concrete pillar",
        background_prompt: str = "stationary ground reference",
    ) -> SegmentationOutput:
        """Segment target and stationary reference from a video frame using text prompts.

        Args:
            frame: Video frame array (H, W, 3) in BGR or RGB format.
            target_prompt: Open-vocabulary text description of the vibrating structural element.
            background_prompt: Text description of static background to cancel micro-jitter.

        Returns:
            SegmentationOutput containing binary masks and bounding boxes.
        """
        pass


class MetaSAM31Segmenter(BaseSegmenter):
    """Meta SAM 3.1 implementation using Promptable Concept Segmentation (PCS).

    Leverages open-vocabulary text prompting and Object Multiplex video tracking
    as specified in Meta SAM 3.1 documentation.
    """

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None) -> None:
        self.api_key = api_key or settings.META_API_KEY
        self.model_name = model_name or settings.SAM_MODEL_NAME

    def segment(
        self,
        frame: npt.NDArray[np.uint8],
        target_prompt: str = "corner of concrete pillar",
        background_prompt: str = "stationary ground reference",
    ) -> SegmentationOutput:
        """Call Meta SAM 3.1 API or local inference checkpoint."""
        # STUB: To be implemented in pipeline phase
        raise NotImplementedError("Meta SAM 3.1 live integration stub - use FallbackSegmenter for local testing.")


class FallbackSegmenter(BaseSegmenter):
    """Local, offline-friendly fallback segmenter.

    Generates synthetic ROIs and masks to allow full pipeline testing without
    requiring external GPU hardware or Meta API credentials.
    """

    def segment(
        self,
        frame: npt.NDArray[np.uint8],
        target_prompt: str = "corner of concrete pillar",
        background_prompt: str = "stationary ground reference",
    ) -> SegmentationOutput:
        """Generate deterministic target and background regions from frame dimensions."""
        h, w = frame.shape[:2]
        target_mask = np.zeros((h, w), dtype=bool)
        background_mask = np.zeros((h, w), dtype=bool)

        # Target: Central structural joint (e.g. pillar or truss junction)
        tw, th = int(w * 0.25), int(h * 0.25)
        tx, ty = int(w * 0.375), int(h * 0.375)
        target_mask[ty : ty + th, tx : tx + tw] = True

        # Stationary reference: Lower/distant ground anchor
        bw, bh = int(w * 0.2), int(h * 0.15)
        bx, by = int(w * 0.05), int(h * 0.8)
        background_mask[by : by + bh, bx : bx + bw] = True

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


def get_segmenter() -> BaseSegmenter:
    """Factory providing the configured segmenter (SAM 3.1 or fallback)."""
    if settings.USE_LOCAL_FALLBACKS or not settings.META_API_KEY:
        return FallbackSegmenter()
    return MetaSAM31Segmenter()
