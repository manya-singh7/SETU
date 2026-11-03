"""Validation module for Project SETU.

Tools for cross-sensor validation, benchmark comparison, and ground-truth verification
against contact accelerometer measurements (e.g. phyphox exports).
"""

from typing import Any

__all__ = [
    "AccelerometerSpectrum",
    "ComparisonResult",
    "compare_frequency",
    "compare_phyphox_to_camera",
    "compute_axis_psd",
    "find_dominant_accelerometer_peak",
    "load_phyphox_csv",
    "plot_comparison",
]


def __getattr__(name: str) -> Any:
    """Lazy module attribute access to prevent circular import warnings during CLI runs."""
    import validation.compare as _compare

    if hasattr(_compare, name):
        return getattr(_compare, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
