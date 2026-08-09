"""Physical constants for the calibration standard.

The grid is a TedPella Prod. No. 607 cross-grating replica: 2160 lines/mm.
The pitch is derived from that ruling rather than hard-coding the rounded
0.463 um the SOP quotes, so the anchor has a single source of truth. This
value is the anchor of every measurement and must not be changed without a
new physical standard.

Accuracy note: the replica's ruling has its own manufacturing tolerance
(typically a few tenths of a percent, and not certified traceable). That
tolerance is a *systematic* floor on accuracy that no amount of FFT
precision removes - the repeatability figures the pipeline reports describe
precision only. Treat `PITCH_RELATIVE_UNCERTAINTY` as the accuracy floor
until the grating is measured against a traceable standard.
"""

from __future__ import annotations

import math

GRID_STANDARD_NAME = "TedPella Prod. No. 607"
GRID_LINES_PER_MM = 2160
NM_PER_UM = 1000.0
UM_PER_MM = 1000.0

# 1 / 2160 mm = 0.4629630 um. The SOP's 0.463 um is this rounded to 3 places.
GRID_PITCH_UM = UM_PER_MM / GRID_LINES_PER_MM
GRID_PITCH_NM = GRID_PITCH_UM * NM_PER_UM

# Manufacturer tolerance on the ruling; the accuracy floor of every result.
PITCH_RELATIVE_UNCERTAINTY = 0.002

# A measurement outside this range is physically implausible for biopsy EM and
# is treated as a failure rather than a result. Wide on purpose: it is a
# blunder guard (integer-factor errors, garbage frames), not a tolerance band.
PLAUSIBLE_NM_PER_PIXEL = (0.05, 500.0)


def nm_per_pixel(pixels_per_space: float) -> float:
    """Convert measured grid spacing in pixels to nanometers per pixel.

    Returns NaN for non-finite or non-positive spacings instead of raising, so
    a bad frame propagates as a missing measurement rather than an exception
    part-way through a batch run.
    """
    if not math.isfinite(pixels_per_space) or pixels_per_space <= 0.0:
        return float("nan")
    return GRID_PITCH_NM / pixels_per_space


def pixels_per_space(nm_per_pixel_value: float) -> float:
    """Inverse of `nm_per_pixel`; used by cross-checks against known scales."""
    if not math.isfinite(nm_per_pixel_value) or nm_per_pixel_value <= 0.0:
        return float("nan")
    return GRID_PITCH_NM / nm_per_pixel_value


def is_plausible(nm_per_pixel_value: float) -> bool:
    low, high = PLAUSIBLE_NM_PER_PIXEL
    return math.isfinite(nm_per_pixel_value) and low <= nm_per_pixel_value <= high


def total_relative_uncertainty(measurement_relative_uncertainty: float) -> float:
    """Combine measurement precision with the standard's own tolerance."""
    if not math.isfinite(measurement_relative_uncertainty):
        return float("nan")
    return math.hypot(measurement_relative_uncertainty, PITCH_RELATIVE_UNCERTAINTY)
