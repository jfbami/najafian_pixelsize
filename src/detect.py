"""FFT-based detection of calibration grid frames.

This requires no trained model: a calibration grating shows a strong, square,
orthogonal harmonic comb, whereas tissue does not. The same spectral features
(peak prominence, squareness, axis orthogonality, energy concentration)
classify a frame as calibration or tissue.

Detection and measurement are deliberately separate gates. Detection runs on
whatever is cheap -- typically a Drive thumbnail -- and uses only features that
are scale-invariant, so it does *not* require the frame to be measurable. A
thumbnail's grid period is usually below the aliasing limit that
`measurer._MIN_RESOLVABLE_SPACING` enforces, so `MeasurementResult.valid` is
routinely False for a correctly detected calibration thumbnail. The nm/pixel
number must always come from the full-resolution TIFF.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .measurer import measure_grid
from .models import MeasurementResult

# Validated on the reference set: calibration frames scored 0.965-0.970 against
# a strongest-tissue score of 0.499. Re-validate with `revalidate_detector.py`
# after changing any of these or after switching image source.
CONFIDENCE_THRESHOLD = 0.60
SQUARENESS_THRESHOLD = 0.90
ORTHOGONALITY_TOLERANCE_DEG = 12.0

# Broadband frames (tissue, noise) spread their energy; a grating concentrates
# it. Set far below any plausible grating value so it can only reject, never
# cost a real detection.
CONCENTRATION_THRESHOLD = 0.05


@dataclass
class Thresholds:
    confidence: float = CONFIDENCE_THRESHOLD
    squareness: float = SQUARENESS_THRESHOLD
    orthogonality_deg: float = ORTHOGONALITY_TOLERANCE_DEG
    concentration: float = CONCENTRATION_THRESHOLD

    @classmethod
    def from_config(cls, config: Optional[dict]) -> "Thresholds":
        section = (config or {}).get("detection", {})
        return cls(
            confidence=float(section.get("confidence_threshold", CONFIDENCE_THRESHOLD)),
            squareness=float(section.get("squareness_threshold", SQUARENESS_THRESHOLD)),
            orthogonality_deg=float(
                section.get("orthogonality_tolerance_deg", ORTHOGONALITY_TOLERANCE_DEG)
            ),
            concentration=float(
                section.get("concentration_threshold", CONCENTRATION_THRESHOLD)
            ),
        )


@dataclass
class Detection:
    is_calibration: bool
    score: float
    measurement: MeasurementResult


def detect_calibration(
    image_path: str, thresholds: Optional[Thresholds] = None
) -> Detection:
    result = measure_grid(image_path)
    return classify(result, thresholds)


def classify(
    result: MeasurementResult, thresholds: Optional[Thresholds] = None
) -> Detection:
    limits = thresholds or Thresholds()
    return Detection(
        is_calibration=_is_calibration(result, limits),
        score=_grid_score(result),
        measurement=result,
    )


def _is_calibration(result: MeasurementResult, limits: Thresholds) -> bool:
    return (
        result.axis_count >= 2
        and result.fft_confidence >= limits.confidence
        and result.grid_uniformity >= limits.squareness
        and abs(result.axis_separation_deg - 90.0) <= limits.orthogonality_deg
        and result.spectral_concentration >= limits.concentration
    )


def _grid_score(result: MeasurementResult) -> float:
    if result.axis_count < 2:
        return 0.0
    orthogonality = max(0.0, 1.0 - abs(result.axis_separation_deg - 90.0) / 90.0)
    return result.fft_confidence * result.grid_uniformity * orthogonality
