"""Pair a calibration frame to tissue by magnification and acquisition date.

Lab SOP: use a calibration grid taken at the same magnification on the same
scope, on the same day or as close as possible. Same-magnification is required;
date proximity decides among candidates and whether to auto-use or flag.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional

from .models import CalibrationCandidate, TiffMetadata


@dataclass
class MatchDecision:
    candidate: Optional[CalibrationCandidate]
    date_delta_days: Optional[int]
    auto_usable: bool
    reason: str


def choose_calibration(
    candidates: list[CalibrationCandidate],
    tissue: TiffMetadata,
    auto_use_within_days: int,
    max_date_window_days: int,
) -> MatchDecision:
    same_magnification = [
        candidate
        for candidate in candidates
        if candidate.metadata.magnification == tissue.magnification
    ]
    if not same_magnification:
        return MatchDecision(None, None, False, "no calibration at matching magnification")

    best = min(
        same_magnification,
        key=lambda candidate: _date_distance(candidate.metadata, tissue),
    )
    delta = _date_distance(best.metadata, tissue)

    if delta is None:
        return MatchDecision(best, None, False, "calibration found but dates unknown")
    if delta > max_date_window_days:
        return MatchDecision(best, delta, False, f"nearest calibration is {delta} days away")
    if delta > auto_use_within_days:
        return MatchDecision(best, delta, False, f"calibration {delta} days away exceeds auto-use window")
    return MatchDecision(best, delta, True, f"matched within {delta} days at same magnification")


def _date_distance(
    calibration: TiffMetadata, tissue: TiffMetadata
) -> Optional[int]:
    calibration_date = _parse(calibration.acquisition_date)
    tissue_date = _parse(tissue.acquisition_date)
    if calibration_date is None or tissue_date is None:
        return None
    return abs((calibration_date - tissue_date).days)


def _parse(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None
