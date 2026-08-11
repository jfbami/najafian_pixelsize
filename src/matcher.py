"""Pair a calibration frame to tissue by magnification, resolution and date.

Lab SOP: use a calibration grid taken at the same magnification on the same
scope, on the same day or as close as possible. Same-magnification is required;
date proximity decides among candidates and whether to auto-use or flag.

Matching resolution is required for the same reason. nm/pixel describes one
image's pixel grid, not the grating, so a calibration measured on a 2512px
frame does not describe a 1024px frame even at identical magnification: the
scale would be wrong by the resolution ratio. This lab saves in at least three
formats (roughly 1024, 2512 and 4576 pixels wide), so the mismatch is reachable
in practice and produces a confident, badly wrong number.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional

from .models import CalibrationCandidate, TiffMetadata

# Frames from one sensor format vary slightly because the burned-in info bar
# is not always the same height, so require closeness rather than equality.
_RESOLUTION_TOLERANCE = 0.02


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
    if tissue.magnification is None:
        return MatchDecision(None, None, False, "tissue magnification unknown")

    same_magnification = [
        candidate
        for candidate in candidates
        if candidate.metadata.magnification == tissue.magnification
    ]
    if not same_magnification:
        return MatchDecision(None, None, False, "no calibration at matching magnification")

    same_resolution = [
        candidate
        for candidate in same_magnification
        if _resolution_matches(candidate.metadata, tissue)
    ]
    if not same_resolution:
        return MatchDecision(
            None, None, False, _resolution_mismatch_reason(same_magnification, tissue)
        )

    # Candidates with an unreadable date sort last rather than crashing the
    # comparison, and are only chosen when nothing dated is available.
    best = min(
        same_resolution,
        key=lambda candidate: _sort_key(_date_distance(candidate.metadata, tissue)),
    )
    delta = _date_distance(best.metadata, tissue)

    if delta is None:
        return MatchDecision(best, None, False, "calibration found but dates unknown")
    if delta > max_date_window_days:
        return MatchDecision(best, delta, False, f"nearest calibration is {delta} days away")
    if delta > auto_use_within_days:
        return MatchDecision(best, delta, False, f"calibration {delta} days away exceeds auto-use window")
    return MatchDecision(
        best,
        delta,
        True,
        f"matched within {delta} days at same magnification and "
        f"{resolution_note(tissue)}",
    )


def _resolution_matches(candidate: TiffMetadata, tissue: TiffMetadata) -> bool:
    """Whether a calibration's pixel grid is the same format as the tissue's.

    Dimensions that are unknown cannot be compared, so they are permitted here
    and reported as unverified by `resolution_note` rather than silently
    passing as a confirmed match.
    """
    for left, right in (
        (candidate.image_width, tissue.image_width),
        (candidate.image_height, tissue.image_height),
    ):
        if left is None or right is None:
            continue
        if abs(left - right) / max(left, right) > _RESOLUTION_TOLERANCE:
            return False
    return True


def resolution_note(metadata: TiffMetadata) -> str:
    """Human-readable frame resolution, for result notes and CSV output."""
    if metadata.image_width is None or metadata.image_height is None:
        return "resolution unknown"
    return f"{metadata.image_width}x{metadata.image_height}"


def _resolution_mismatch_reason(
    candidates: list[CalibrationCandidate], tissue: TiffMetadata
) -> str:
    offered = sorted({resolution_note(c.metadata) for c in candidates})
    return (
        f"calibration exists at this magnification but only at "
        f"{', '.join(offered)}, and the tissue frame is "
        f"{resolution_note(tissue)}; the pixel scale would be wrong"
    )


def _sort_key(delta: Optional[int]) -> tuple[int, int]:
    """Order by date distance, pushing unknown dates behind every known one."""
    return (1, 0) if delta is None else (0, delta)


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
