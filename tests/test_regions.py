"""Region selection: measuring only the flat, undistorted part of a grating.

Every test works from a generated grating whose flat region has an exactly
known pitch, so these check that the answer is *right*, not merely stable.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.regions import measure_straightest_region
from tests.synthetic import cross_grating, folded_grating

PERIOD = 40.0
SIZE = 1024


def relative_error_percent(measured: float, truth: float) -> float:
    return abs(measured - truth) / truth * 100.0


FOLD_SEVERITIES = [(0.6, 25.0), (0.5, 30.0), (0.4, 35.0)]


def test_whole_frame_accuracy_is_erratic_across_fold_severities():
    """The defect this module exists for.

    A whole-frame transform averages flat and folded regions together. It is
    not reliably biased, which is what makes it dangerous: depending on whether
    the measurer's axis-disagreement fallback happens to trigger, the same
    grating reads anywhere from 0.01% to 0.2% off with nothing distinguishing
    the two cases. Region selection replaces that lottery with a stable answer,
    which the next test pins down.
    """
    errors = [
        relative_error_percent(
            measure_straightest_region(
                folded_grating(
                    PERIOD, size=SIZE, tilt_deg=tilt, fold_from_fraction=fraction
                )
            ).whole_frame.pixels_per_space,
            PERIOD,
        )
        for fraction, tilt in FOLD_SEVERITIES
    ]
    assert max(errors) - min(errors) > 0.1


def test_region_accuracy_is_stable_across_fold_severities():
    errors = [
        relative_error_percent(
            measure_straightest_region(
                folded_grating(
                    PERIOD, size=SIZE, tilt_deg=tilt, fold_from_fraction=fraction
                )
            ).measurement.pixels_per_space,
            PERIOD,
        )
        for fraction, tilt in FOLD_SEVERITIES
    ]
    assert max(errors) < 0.1
    assert max(errors) - min(errors) < 0.1


def test_region_selection_recovers_the_flat_region_pitch():
    folded = folded_grating(PERIOD, size=SIZE, tilt_deg=25.0)
    region = measure_straightest_region(folded)

    assert region.measurement.valid, region.warning
    assert relative_error_percent(region.measurement.pixels_per_space, PERIOD) < 0.2


def test_region_selection_beats_the_whole_frame_on_a_folded_grating():
    folded = folded_grating(PERIOD, size=SIZE, tilt_deg=25.0)
    region = measure_straightest_region(folded)

    region_error = relative_error_percent(region.measurement.pixels_per_space, PERIOD)
    frame_error = relative_error_percent(region.whole_frame.pixels_per_space, PERIOD)
    assert region_error < frame_error


@pytest.mark.parametrize("fraction,tilt", FOLD_SEVERITIES)
def test_no_selected_tile_lies_wholly_inside_the_fold(fraction, tilt):
    """Selection must localise, not just average differently.

    Tiles that straddle the boundary are allowed, since the fold ramps in
    gradually and a straddling tile can still be predominantly flat. A tile
    sitting entirely in the folded region never should be.
    """
    region = measure_straightest_region(
        folded_grating(PERIOD, size=SIZE, tilt_deg=tilt, fold_from_fraction=fraction)
    )
    fold_starts_at = SIZE * fraction

    assert region.selected_tiles, region.warning
    assert not [tile for tile in region.selected_tiles if tile.col0 >= fold_starts_at]


def test_a_clean_grating_is_left_alone():
    """Selection must not shift a frame that has nothing wrong with it."""
    clean = cross_grating(PERIOD, size=SIZE)
    region = measure_straightest_region(clean)

    assert region.measurement.valid, region.warning
    assert relative_error_percent(region.measurement.pixels_per_space, PERIOD) < 0.1
    assert abs(region.disagreement_percent) < 0.2


def test_a_clean_grating_keeps_most_of_its_tiles():
    region = measure_straightest_region(cross_grating(PERIOD, size=SIZE))
    assert len(region.selected_tiles) >= len(region.tiles) * 0.5


def test_tile_spread_reports_real_disagreement():
    """Spread must widen on a distorted frame; it is the honesty signal."""
    clean = measure_straightest_region(cross_grating(PERIOD, size=SIZE))
    folded = measure_straightest_region(folded_grating(PERIOD, size=SIZE, tilt_deg=25.0))

    assert folded.tile_spread_percent > clean.tile_spread_percent


def test_falls_back_to_the_whole_frame_when_nothing_is_measurable():
    """Noise has no tiles worth selecting, so the frame result stands."""
    noise = np.random.default_rng(0).normal(128.0, 20.0, (512, 512)).astype(np.float32)
    region = measure_straightest_region(noise)

    assert region.measurement is region.whole_frame or not region.measurement.valid
    assert region.warning is not None


@pytest.mark.parametrize("tilt", [10.0, 25.0, 40.0])
def test_recovers_the_flat_pitch_across_fold_severities(tilt):
    region = measure_straightest_region(
        folded_grating(PERIOD, size=SIZE, tilt_deg=tilt)
    )
    assert region.measurement.valid, region.warning
    assert relative_error_percent(region.measurement.pixels_per_space, PERIOD) < 0.25
