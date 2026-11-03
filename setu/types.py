"""
Core shared data types for Project SETU.

Defines standardized dataclasses for segmentation results, motion tracking,
and spectral modal peaks across the pipeline modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
import numpy as np
import numpy.typing as npt


@dataclass
class SegmentResult:
    """Segmentation masks for structural target and background reference.

    Attributes:
        target_masks: List of (H, W) boolean numpy arrays, one per video frame.
        background_masks: List of (H, W) boolean numpy arrays, one per video frame.
        fps: Video frame rate in frames per second.
        method: Segmentation strategy ("sam", "manual", or "fallback").
        metadata: Diagnostic and provenance metadata dictionary (default empty).
    """

    target_masks: List[npt.NDArray[np.bool_]]
    background_masks: List[npt.NDArray[np.bool_]]
    fps: float
    method: Literal["sam", "manual", "fallback"] = "sam"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Peak:
    """Detected dominant vibration frequency peak.

    Attributes:
        freq_hz: Dominant resonant frequency in Hertz.
        rel_power: Relative spectral power (normalized peak power / total PSD power).
        snr: Peak-prominence signal-to-noise ratio in decibels (dB) or linear scale.
        verdict: Quality evaluation tier ("usable" or "weak").
        reasons: Diagnostic reasons explaining any quality demerits.
        units: Physical unit of displacement measurements (default "px").
        scale: Calibrated spatial scaling factor in mm/pixel (default None).
        freq_uncertainty: Half-width of 95% confidence interval in Hz.
        freq_ci: 95% confidence interval (lower_hz, upper_hz) from segment estimation.
        is_reliable: Binary reliability flag (True if verdict == "usable").
        bg_ratio: Target PSD to background residual PSD ratio at peak frequency.
        bg_ratio_db: Target-to-background ratio in decibels (dB).
    """

    freq_hz: float
    rel_power: float
    snr: float
    verdict: Literal["usable", "weak"] = "usable"
    reasons: List[str] = field(default_factory=list)
    units: str = "px"
    scale: Optional[float] = None
    freq_uncertainty: Optional[float] = None
    freq_ci: Optional[Tuple[float, float]] = None
    is_reliable: bool = True
    bg_ratio: Optional[float] = None
    bg_ratio_db: Optional[float] = None

    @property
    def flags(self) -> List[str]:
        """Backward-compatible alias for reasons."""
        return self.reasons

    @flags.setter
    def flags(self, val: List[str]) -> None:
        self.reasons = val

    @property
    def rejection_reasons(self) -> List[str]:
        """Backward-compatible alias for reasons."""
        return self.reasons

    @rejection_reasons.setter
    def rejection_reasons(self, val: List[str]) -> None:
        self.reasons = val


@dataclass
class MotionResult:
    """Subpixel displacement tracking result.

    Attributes:
        displacement: 1D numpy array of structural displacement in pixels across time.
        fps: Sampling frame rate in Hz.
        quality: Diagnostic telemetry dictionary.
        units: Physical unit of displacement measurements (default "px").
        scale: Calibrated spatial scaling factor in mm/pixel (default None).
        verdict: Quality evaluation tier ("usable", "weak", or "failed").
        reasons: Diagnostic reasons explaining any tracking or reliability demerits.
    """

    displacement: npt.NDArray[np.float64]
    fps: float
    quality: Dict[str, Any] = field(default_factory=dict)
    units: str = "px"
    scale: Optional[float] = None
    verdict: Literal["usable", "weak", "failed"] = "usable"
    reasons: List[str] = field(default_factory=list)
    background_displacement: Optional[npt.NDArray[np.float64]] = None
