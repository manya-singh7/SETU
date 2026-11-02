"""
Core shared data types for Project SETU.

Defines standardized dataclasses for segmentation results, motion tracking,
and spectral modal peaks across the pipeline modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Union
import numpy as np
import numpy.typing as npt


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
class Peak:
    """Detected dominant vibration frequency peak.

    Attributes:
        freq_hz: Dominant resonant frequency in Hertz.
        rel_power: Relative spectral power (normalized peak power / total PSD power).
        snr: Peak-prominence signal-to-noise ratio in decibels or linear scale.
    """

    freq_hz: float
    rel_power: float
    snr: float


@dataclass
class MotionResult:
    """Subpixel displacement tracking result.

    Attributes:
        displacement: 1D numpy array of structural displacement in pixels across time.
        fps: Sampling frame rate in Hz.
        quality: Diagnostic telemetry dictionary containing:
            - background_residual_px: float (RMS residual motion in stationary anchor)
            - snr: float (displacement tracking SNR estimate)
            - warnings: List[str] (quality warnings e.g. low contrast, clipping)
    """

    displacement: npt.NDArray[np.float64]
    fps: float
    quality: Dict[str, Any] = field(default_factory=dict)
