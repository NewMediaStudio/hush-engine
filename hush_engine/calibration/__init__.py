"""
Calibration module for Hush Engine

Provides automated threshold calibration and weight optimization
based on feedback data and precision-recall analysis.
"""

from .weight_calibrator import (
    WeightCalibrator,
    ModelMetrics,
    calibrate_from_feedback,
    get_calibrator,
)

__all__ = [
    "WeightCalibrator",
    "ModelMetrics",
    "calibrate_from_feedback",
    "get_calibrator",
]
