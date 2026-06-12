"""FFT-based detection of calibration grid frames.

This requires no trained model: a calibration grating shows a strong, square,
orthogonal harmonic comb, whereas tissue does not. The same measurement
features (peak prominence, squareness, axis orthogonality) classify a frame as
calibration or tissue. A learned classifier can later refine the borderline
cases, but this rule already separates the two cleanly on the reference data.
"""

from __future__ import annotations

from dataclasses import dataclass

from .measurer import measure_grid
from .models import MeasurementResult

CONFIDENCE_THRESHOLD = 0.60
SQUARENESS_THRESHOLD = 0.90
ORTHOGONALITY_TOLERANCE_DEG = 12.0


@dataclass
class Detection:
    is_calibration: bool
    score: float
    measurement: MeasurementResult


def detect_calibration(image_path: str) -> Detection:
    result = measure_grid(image_path)
    return Detection(
        is_calibration=_is_calibration(result),
        score=_grid_score(result),
        measurement=result,
    )


def _is_calibration(result: MeasurementResult) -> bool:
    return (
        result.axis_count >= 2
        and result.fft_confidence >= CONFIDENCE_THRESHOLD
        and result.grid_uniformity >= SQUARENESS_THRESHOLD
        and abs(result.axis_separation_deg - 90.0) <= ORTHOGONALITY_TOLERANCE_DEG
    )


def _grid_score(result: MeasurementResult) -> float:
    if result.axis_count < 2:
        return 0.0
    orthogonality = max(0.0, 1.0 - abs(result.axis_separation_deg - 90.0) / 90.0)
    return result.fft_confidence * result.grid_uniformity * orthogonality
