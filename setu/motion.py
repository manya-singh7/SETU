"""
Subpixel motion tracking and camera motion cancellation module for Project SETU.

Step 3 in SETU Pipeline:
Computes subpixel displacement of the structural target, tracks motion in the
stationary background reference anchor, and subtracts the background motion
from the target to yield a pure structural displacement time series free from camera micro-jitter.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple
import numpy as np
import numpy.typing as npt


@dataclass
class DisplacementSeries:
    """Displacement telemetry extracted across video frames.

    Attributes:
        time_seconds: 1D time array in seconds.
        raw_target_displacement: Raw target motion (N,) [pixels].
        background_displacement: Background reference motion (N,) [pixels].
        corrected_displacement: Pure structural displacement = target - background (N,) [pixels].
        sampling_rate_fps: Video framerate.
        metrics: Noise and jitter RMS statistics.
    """

    time_seconds: npt.NDArray[np.float64]
    raw_target_displacement: npt.NDArray[np.float64]
    background_displacement: npt.NDArray[np.float64]
    corrected_displacement: npt.NDArray[np.float64]
    sampling_rate_fps: float
    metrics: Dict[str, float] = field(default_factory=dict)


class SubpixelMotionTracker:
    """Calculates subpixel structural displacement cancelled against background jitter."""

    def __init__(self, fps: float = 30.0, method: str = "phase_correlation") -> None:
        """Initialize the subpixel motion tracker.

        Args:
            fps: Frame rate of the input video (Hz).
            method: Subpixel interpolation algorithm ('phase_correlation', 'lucas_kanade', 'gradient').
        """
        self.fps = fps
        self.method = method

    def track_displacement_series(
        self,
        frames: npt.NDArray[np.uint8],
        target_bbox: Tuple[int, int, int, int],
        background_bbox: Tuple[int, int, int, int],
    ) -> DisplacementSeries:
        """Track motion of target and background ROIs across frames and subtract background motion.

        Args:
            frames: Video frames array of shape (N, H, W, 3) or (N, H, W).
            target_bbox: Target structural component bounding box (x, y, w, h).
            background_bbox: Stationary background anchor bounding box (x, y, w, h).

        Returns:
            DisplacementSeries with corrected subpixel structural motion.
        """
        # STUB: To be implemented in pipeline phase
        n_frames = len(frames)
        t = np.arange(n_frames, dtype=np.float64) / self.fps
        zero_series = np.zeros(n_frames, dtype=np.float64)

        return DisplacementSeries(
            time_seconds=t,
            raw_target_displacement=zero_series.copy(),
            background_displacement=zero_series.copy(),
            corrected_displacement=zero_series.copy(),
            sampling_rate_fps=self.fps,
            metrics={"jitter_rms_px": 0.0, "structural_rms_px": 0.0},
        )
