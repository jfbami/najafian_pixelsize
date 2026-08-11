"""Measure the straightest, cleanest part of a calibration grating.

A replica grating is rarely flat across a whole frame. Folds, wrinkles, tears
and contamination are normal, and a fold changes the *projected* pitch: where
the replica tilts out of the image plane, the grid foreshortens and reads
short. A whole-frame transform averages the good and bad regions together, so
the answer is pulled toward the distortion with nothing to signal it. On a
synthetic frame whose clean region has a pitch of exactly 100.0 px, a fold
across the outer 40% pulls the whole-frame reading to 98.22 px, a -1.78% bias:
nearly nine times the grating's own 0.2% tolerance.

This module tiles the frame, measures each tile independently, and keeps only
the tiles that agree on grid orientation and show a sharp, square comb. The
selection criteria are all properties of grid *quality* rather than of the
pitch value, so tiles are not chosen for reading a particular number. Even so,
selecting a subset on any fit-derived statistic can bias the result slightly,
which is why `RegionMeasurement.tile_spread_percent` reports the observed
disagreement between the surviving tiles rather than only a propagated error.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from . import calibration
from .measurer import measure_array
from .models import MeasurementResult

# A tile needs enough periods for a well-conditioned comb fit. Below about ten
# the per-tile uncertainty grows faster than the benefit of finer sampling.
_PERIODS_PER_TILE = 8.0
_MIN_TILE_PIXELS = 192
_TILE_OVERLAP = 0.5

# Straightness: how far a tile's grid orientation may sit from the frame's
# consensus orientation before the tile is treated as distorted.
_ANGLE_TOLERANCE_DEG = 1.5

_MIN_CONCENTRATION = 0.25
# Strict on purpose, and much stricter than detection uses. An out-of-plane
# tilt foreshortens one grid axis and leaves the other alone, so a tilted tile
# reads non-square while staying perfectly sharp. On a 23 degree tilt the
# squareness falls only to 0.92, so a loose threshold lets the fold through.
_MIN_SQUARENESS = 0.97
_MAX_ORTHOGONALITY_ERROR_DEG = 5.0
_MIN_TILES_FOR_SELECTION = 4


@dataclass
class TileMeasurement:
    """One tile's measurement and where it sits in the frame."""

    row0: int
    col0: int
    row1: int
    col1: int
    result: MeasurementResult
    angle_deviation_deg: float = float("nan")
    selected: bool = False

    @property
    def bounds(self) -> tuple[int, int, int, int]:
        return (self.row0, self.col0, self.row1, self.col1)


@dataclass
class RegionMeasurement:
    """Result of measuring only the well-formed regions of a grating.

    `measurement` is the combined answer over the selected tiles and is the
    value to use. `whole_frame` is the same frame measured in one piece, kept
    so the two can be compared: a large gap between them means the frame is
    distorted and the whole-frame number should not be trusted.
    """

    measurement: MeasurementResult
    whole_frame: MeasurementResult
    tile_size: int
    tiles: list[TileMeasurement] = field(default_factory=list)
    tile_spread_percent: float = float("nan")
    warning: Optional[str] = None

    @property
    def selected_tiles(self) -> list[TileMeasurement]:
        return [tile for tile in self.tiles if tile.selected]

    @property
    def disagreement_percent(self) -> float:
        """How far the whole-frame reading sits from the selected regions."""
        whole, region = self.whole_frame, self.measurement
        if not (whole.valid and region.valid):
            return float("nan")
        return (whole.pixels_per_space - region.pixels_per_space) / region.pixels_per_space * 100.0


def measure_straightest_region(image: np.ndarray) -> RegionMeasurement:
    """Measure the grating using only its straightest, cleanest tiles."""
    whole_frame = measure_array(image)
    tile_size = _tile_size(image, whole_frame)

    tiles = _measure_tiles(image, tile_size)
    valid = [tile for tile in tiles if tile.result.valid]
    if len(valid) < _MIN_TILES_FOR_SELECTION:
        return RegionMeasurement(
            measurement=whole_frame,
            whole_frame=whole_frame,
            tile_size=tile_size,
            tiles=tiles,
            warning=(
                f"only {len(valid)} of {len(tiles)} tiles were measurable; "
                f"falling back to the whole frame"
            ),
        )

    _score_straightness(valid)
    selected = _select(valid)
    for tile in selected:
        tile.selected = True

    combined, spread = _combine(selected)
    return RegionMeasurement(
        measurement=combined,
        whole_frame=whole_frame,
        tile_size=tile_size,
        tiles=tiles,
        tile_spread_percent=spread,
        warning=_warning(selected, valid, tiles),
    )


def _tile_size(image: np.ndarray, whole_frame: MeasurementResult) -> int:
    """Tile edge in pixels, sized to hold enough grid periods to fit well."""
    span = min(image.shape)
    if whole_frame.valid and math.isfinite(whole_frame.pixels_per_space):
        wanted = int(math.ceil(_PERIODS_PER_TILE * whole_frame.pixels_per_space))
    else:
        wanted = span // 2
    return int(max(_MIN_TILE_PIXELS, min(wanted, span)))


def _measure_tiles(image: np.ndarray, tile_size: int) -> list[TileMeasurement]:
    stride = max(1, int(tile_size * (1.0 - _TILE_OVERLAP)))
    tiles: list[TileMeasurement] = []
    for row0 in _starts(image.shape[0], tile_size, stride):
        for col0 in _starts(image.shape[1], tile_size, stride):
            patch = np.ascontiguousarray(
                image[row0 : row0 + tile_size, col0 : col0 + tile_size]
            )
            tiles.append(
                TileMeasurement(
                    row0=row0,
                    col0=col0,
                    row1=row0 + tile_size,
                    col1=col0 + tile_size,
                    result=measure_array(patch),
                )
            )
    return tiles


def _starts(length: int, tile_size: int, stride: int) -> list[int]:
    if tile_size >= length:
        return [0]
    starts = list(range(0, length - tile_size + 1, stride))
    if starts[-1] != length - tile_size:
        starts.append(length - tile_size)
    return starts


def _score_straightness(tiles: list[TileMeasurement]) -> None:
    """Record how far each tile's grid orientation sits from the consensus.

    A fold rotates the grid locally as well as compressing it, so orientation
    scatter is a direct read on which part of the replica lies flat.
    """
    angles = [
        tile.result.axis_angle_deg
        for tile in tiles
        if math.isfinite(tile.result.axis_angle_deg)
    ]
    if not angles:
        return
    consensus = _circular_median(angles)
    for tile in tiles:
        tile.angle_deviation_deg = _angular_distance(
            tile.result.axis_angle_deg, consensus
        )


def _select(tiles: list[TileMeasurement]) -> list[TileMeasurement]:
    """Keep tiles whose grid is flat, sharp, square and correctly oriented."""
    passing = [tile for tile in tiles if _is_well_formed(tile)]
    if len(passing) >= _MIN_TILES_FOR_SELECTION:
        return passing

    # Nothing cleared the bar, so fall back to the best-oriented half rather
    # than reporting no measurement at all.
    ordered = sorted(tiles, key=lambda tile: _sort_key(tile))
    return ordered[: max(_MIN_TILES_FOR_SELECTION, len(ordered) // 2)]


def _is_well_formed(tile: TileMeasurement) -> bool:
    result = tile.result
    return (
        result.valid
        and math.isfinite(tile.angle_deviation_deg)
        and tile.angle_deviation_deg <= _ANGLE_TOLERANCE_DEG
        and result.spectral_concentration >= _MIN_CONCENTRATION
        and result.grid_uniformity >= _MIN_SQUARENESS
        and abs(result.axis_separation_deg - 90.0) <= _MAX_ORTHOGONALITY_ERROR_DEG
    )


def _sort_key(tile: TileMeasurement) -> tuple[float, float]:
    deviation = (
        tile.angle_deviation_deg
        if math.isfinite(tile.angle_deviation_deg)
        else float("inf")
    )
    return (deviation, -tile.result.spectral_concentration)


def _combine(tiles: list[TileMeasurement]) -> tuple[MeasurementResult, float]:
    """Inverse-variance weighted mean of the selected tiles' pitches."""
    spacings = np.array([tile.result.pixels_per_space for tile in tiles], dtype=float)
    weights = np.array([_weight(tile.result) for tile in tiles], dtype=float)

    pitch = float(np.sum(weights * spacings) / np.sum(weights))
    propagated = float(1.0 / math.sqrt(np.sum(weights))) / pitch
    spread = float(np.std(spacings, ddof=1)) if spacings.size > 1 else 0.0
    scatter = (spread / math.sqrt(spacings.size)) / pitch if pitch else float("nan")

    # Trust whichever error estimate is larger: the fits can be over-confident
    # when neighbouring tiles overlap and share pixels.
    relative = max(propagated, scatter)
    reference = max(tiles, key=lambda tile: tile.result.spectral_concentration).result

    return (
        _rebuild(reference, pitch, relative, len(tiles)),
        (spread / pitch * 100.0) if pitch else float("nan"),
    )


def _weight(result: MeasurementResult) -> float:
    sigma = result.pixels_per_space * result.relative_precision
    if not math.isfinite(sigma) or sigma <= 0.0:
        return 1.0
    return 1.0 / (sigma * sigma)


def _rebuild(
    reference: MeasurementResult, pitch: float, relative: float, tile_count: int
) -> MeasurementResult:
    nanometers = calibration.nm_per_pixel(pitch)
    valid = calibration.is_plausible(nanometers)
    total = calibration.total_relative_uncertainty(relative)
    return MeasurementResult(
        pixels_per_space=pitch if valid else float("nan"),
        nm_per_pixel=nanometers if valid else float("nan"),
        spacing_x=reference.spacing_x,
        spacing_y=reference.spacing_y,
        fft_confidence=reference.fft_confidence,
        grid_uniformity=reference.grid_uniformity,
        axis_separation_deg=reference.axis_separation_deg,
        axis_angle_deg=reference.axis_angle_deg,
        axis_count=reference.axis_count,
        spectral_concentration=reference.spectral_concentration,
        relative_precision=relative,
        nm_per_pixel_uncertainty=nanometers * total if valid else float("nan"),
        fundamental_inferred=reference.fundamental_inferred,
        valid=valid,
        warning=f"combined from {tile_count} selected tiles",
    )


def _warning(
    selected: list[TileMeasurement],
    valid: list[TileMeasurement],
    tiles: list[TileMeasurement],
) -> Optional[str]:
    notes = [f"used {len(selected)} of {len(tiles)} tiles"]
    if len(valid) < len(tiles):
        notes.append(f"{len(tiles) - len(valid)} tiles unmeasurable")
    rejected = len(valid) - len(selected)
    if rejected:
        notes.append(f"{rejected} rejected as distorted or contaminated")
    return "; ".join(notes)


def _circular_median(angles: list[float]) -> float:
    """Median orientation, respecting that grid axes wrap at 180 degrees."""
    doubled = np.radians(np.array(angles) * 2.0)
    mean = math.atan2(float(np.sin(doubled).mean()), float(np.cos(doubled).mean()))
    return math.degrees(mean) / 2.0 % 180.0


def _angular_distance(angle_a: float, angle_b: float) -> float:
    if not (math.isfinite(angle_a) and math.isfinite(angle_b)):
        return float("nan")
    difference = abs(angle_a - angle_b) % 180.0
    return min(difference, 180.0 - difference)
