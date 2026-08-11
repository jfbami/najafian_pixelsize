"""Accuracy and edge-case tests for the FFT grid measurement.

Every test that asserts a number checks it against a *known* pitch, so a
regression shows up as a wrong measurement rather than a changed one. The
tolerances are deliberately far tighter than the grating's own tolerance: the
measurement should never be the dominant error term.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src import calibration
from src.imaging import content_region
from src.measurer import measure_array, measure_grid
from tests.synthetic import cross_grating, tissue_like, write_tiff

# The measurement itself must stay well inside the standard's 0.2% tolerance.
ACCURACY_TOLERANCE_PERCENT = 0.5


def relative_error_percent(measured: float, truth: float) -> float:
    return abs(measured - truth) / truth * 100.0


@pytest.mark.parametrize("period", [16.0, 32.0, 64.0, 107.59, 128.0, 200.0, 251.0, 300.0])
def test_measures_known_period_accurately(period):
    result = measure_array(cross_grating(period, size=2048))
    assert result.valid, result.warning
    assert relative_error_percent(result.pixels_per_space, period) < ACCURACY_TOLERANCE_PERCENT


def test_long_period_does_not_collapse_to_a_harmonic():
    """A grid period past the old fixed low-frequency mask used to read ~1/3 of
    the truth with full confidence and no warning."""
    period = 300.0
    result = measure_array(cross_grating(period, size=2048))
    assert result.valid
    for factor in (2, 3, 4, 5):
        assert not math.isclose(result.pixels_per_space, period / factor, rel_tol=0.05), (
            f"collapsed onto harmonic {factor}"
        )
    assert relative_error_percent(result.pixels_per_space, period) < ACCURACY_TOLERANCE_PERCENT


@pytest.mark.parametrize("angle", [0.0, 3.0, 7.0, 15.0, 22.5, 30.0, 45.0])
def test_accuracy_is_independent_of_grid_rotation(angle):
    result = measure_array(cross_grating(107.59, size=2048, angle_deg=angle))
    assert result.valid
    assert relative_error_percent(result.pixels_per_space, 107.59) < ACCURACY_TOLERANCE_PERCENT


@pytest.mark.parametrize("noise", [0.0, 10.0, 30.0, 60.0, 100.0])
def test_accuracy_survives_noise(noise):
    result = measure_array(cross_grating(107.59, size=2048, noise=noise))
    assert result.valid
    assert relative_error_percent(result.pixels_per_space, 107.59) < ACCURACY_TOLERANCE_PERCENT


@pytest.mark.parametrize("duty", [0.1, 0.25, 0.5, 0.75, 0.9])
def test_accuracy_survives_duty_cycle(duty):
    """A 50% duty cycle suppresses the even harmonics; the comb fit must not
    mistake the surviving odd comb for a different fundamental."""
    result = measure_array(cross_grating(107.59, size=2048, duty=duty))
    assert result.valid
    assert relative_error_percent(result.pixels_per_space, 107.59) < ACCURACY_TOLERANCE_PERCENT


@pytest.mark.parametrize("shape", [(2048, 2048), (1024, 2048), (2048, 700)])
def test_accuracy_on_non_square_frames(shape):
    rows, cols = shape
    image = cross_grating(100.0, size=cols, height=rows)
    result = measure_array(image)
    assert result.valid
    assert relative_error_percent(result.pixels_per_space, 100.0) < ACCURACY_TOLERANCE_PERCENT


def test_sixteen_bit_and_eight_bit_agree(tmp_path):
    """PIL's convert('L') clips 16-bit frames to a flat 255 field; the loader
    must read them at full depth, or every 16-bit measurement is garbage."""
    image = cross_grating(107.59, size=1024)
    eight = measure_grid(str(write_tiff(image, tmp_path / "g8.tif", bit_depth=8)))
    sixteen = measure_grid(str(write_tiff(image, tmp_path / "g16.tif", bit_depth=16)))

    assert eight.valid and sixteen.valid
    assert relative_error_percent(sixteen.pixels_per_space, 107.59) < ACCURACY_TOLERANCE_PERCENT
    assert math.isclose(eight.pixels_per_space, sixteen.pixels_per_space, rel_tol=1e-3)


@pytest.mark.parametrize(
    "image",
    [
        np.full((512, 512), 128.0, dtype=np.float32),
        np.zeros((512, 512), dtype=np.float32),
        np.tile(np.linspace(0, 255, 512, dtype=np.float32), (512, 1)),
        tissue_like(512),
    ],
    ids=["constant", "zeros", "gradient", "tissue"],
)
def test_non_grating_frames_yield_no_measurement(image):
    """Garbage in must not produce a number: these used to return confident
    nm/pixel values between 4 and 200."""
    result = measure_array(np.ascontiguousarray(image, dtype=np.float32))
    assert not result.valid
    assert math.isnan(result.nm_per_pixel)
    assert result.warning


def test_undersampled_frame_is_refused_not_guessed():
    """A thumbnail-scale grid aliases its harmonics; measuring it produced a
    400% error. It must be refused so the full TIFF is used instead."""
    result = measure_array(cross_grating(6.4, size=256, blur=0.2))
    assert not result.valid
    assert math.isnan(result.nm_per_pixel)


def test_too_few_periods_is_refused():
    result = measure_array(cross_grating(600.0, size=2048))
    assert not result.valid


def test_uncertainty_brackets_the_true_value():
    """The reported uncertainty must actually cover the error, or it is
    decoration. Checked at 3 sigma across periods, angles and noise."""
    inside = 0
    trials = 0
    for seed in range(20):
        period = 60.0 + (seed % 8) * 22.0
        result = measure_array(
            cross_grating(period, size=2048, angle_deg=seed * 3.1, noise=25.0, seed=seed)
        )
        if not result.valid:
            continue
        trials += 1
        truth = calibration.nm_per_pixel(period)
        if abs(result.nm_per_pixel - truth) <= 3.0 * result.nm_per_pixel_uncertainty:
            inside += 1
    assert trials >= 15
    assert inside == trials


def test_uncertainty_is_never_below_the_standards_tolerance():
    """No result may claim better accuracy than the grating itself has."""
    result = measure_array(cross_grating(107.59, size=2048, noise=0.0))
    assert result.valid
    relative = result.nm_per_pixel_uncertainty / result.nm_per_pixel
    assert relative >= calibration.PITCH_RELATIVE_UNCERTAINTY


def test_reports_a_warning_when_a_result_is_marginal():
    result = measure_array(cross_grating(300.0, size=2048))
    assert result.valid
    assert result.warning and "periods" in result.warning


def _with_info_bar(image, bar_rows=150, at_top=False, value=255.0):
    """Append a flat info bar carrying a line of 'text', as AMT/DM do."""
    bar = np.full((bar_rows, image.shape[1]), value, dtype=np.float32)
    bar[bar_rows // 3 : bar_rows // 3 + 12, 20:400] = 0.0
    return np.ascontiguousarray(
        np.vstack([bar, image] if at_top else [image, bar])
    )


@pytest.mark.parametrize("at_top", [False, True], ids=["bottom", "top"])
@pytest.mark.parametrize("period", [21.0, 64.0], ids=["fine", "coarse"])
def test_info_bar_is_cropped_before_measuring(at_top, period):
    """A burned-in info bar puts a hard full-width edge across one axis of the
    transform. On a real 1024x1194 frame with a 21px grid it made the two axes
    fit 21.1px and 298.5px, and the measurement was refused."""
    with_bar = _with_info_bar(cross_grating(period, size=1024), at_top=at_top)

    rows, cols = content_region(with_bar)
    result = measure_array(np.ascontiguousarray(with_bar[rows, cols]))

    assert result.valid, result.warning
    assert relative_error_percent(result.pixels_per_space, period) < ACCURACY_TOLERANCE_PERCENT


def test_an_info_bar_does_not_shift_the_measurement():
    """A burned-in bar must not change the answer, by either defence.

    This originally asserted that an uncropped bar made the frame unmeasurable,
    which is what the crop was built for. Making the fundamental search robust
    to a single dominant spurious peak then fixed the same failure at its root,
    so the bar no longer defeats the fit even uncropped. Both paths must now
    agree, and both must be right.
    """
    with_bar = _with_info_bar(cross_grating(21.0, size=1024))
    rows, cols = content_region(with_bar)

    uncropped = measure_array(with_bar)
    cropped = measure_array(np.ascontiguousarray(with_bar[rows, cols]))

    assert cropped.valid and uncropped.valid
    assert relative_error_percent(cropped.pixels_per_space, 21.0) < ACCURACY_TOLERANCE_PERCENT
    assert math.isclose(cropped.pixels_per_space, uncropped.pixels_per_space, rel_tol=5e-3)


def test_info_bar_crop_matches_the_clean_frame():
    grid = cross_grating(64.0, size=1024)
    rows, cols = content_region(_with_info_bar(grid))
    cropped = measure_array(np.ascontiguousarray(_with_info_bar(grid)[rows, cols]))
    clean = measure_array(grid)
    assert cropped.pixels_per_space == pytest.approx(clean.pixels_per_space, rel=1e-3)


def test_declared_content_size_beats_the_heuristic():
    """When the metadata records the imaged field ('##fv3 2512 2304'), use it
    directly rather than inferring the bar from pixel statistics."""
    grid = cross_grating(64.0, size=512)
    with_bar = _with_info_bar(grid, bar_rows=100)
    rows, cols = content_region(with_bar, declared=(512, 512))
    assert (rows.start, rows.stop) == (0, 512)
    assert (cols.start, cols.stop) == (0, 512)


def test_declared_content_size_is_ignored_when_impossible():
    grid = cross_grating(64.0, size=512)
    rows, cols = content_region(grid, declared=(9999, 9999))
    assert (rows.stop - rows.start) == 512


def test_clean_frame_is_not_cropped():
    """Cropping must never trim a frame that has no info bar."""
    grid = cross_grating(107.59, size=1024)
    rows, cols = content_region(grid)
    assert (rows.start, rows.stop) == (0, 1024)
    assert (cols.start, cols.stop) == (0, 1024)


def test_crop_refuses_to_eat_the_image():
    """A mostly-flat frame must not be cropped down to nothing."""
    almost_blank = np.full((512, 512), 200.0, dtype=np.float32)
    almost_blank[250:262] = 0.0
    rows, cols = content_region(almost_blank)
    assert (rows.stop - rows.start) >= 256
    assert (cols.stop - cols.start) >= 256


def test_repeatability_across_image_regions():
    """The precision claim in the README, as an assertion."""
    image = cross_grating(107.59, size=2048, noise=10.0)
    tile = 800
    positions = [(0, 0), (0, 2048 - tile), (2048 - tile, 0), (2048 - tile, 2048 - tile),
                 ((2048 - tile) // 2, (2048 - tile) // 2)]
    values = []
    for top, left in positions:
        result = measure_array(np.ascontiguousarray(image[top:top + tile, left:left + tile]))
        assert result.valid
        values.append(result.nm_per_pixel)
    spread = float(np.std(values) / np.mean(values))
    assert spread < 0.01, f"region-to-region spread {spread:.4%} exceeds 1%"


# --------------------------------------------------------------------------
# fundamental selection
# --------------------------------------------------------------------------

def _peak(period, angle, magnitude):
    from src.measurer import _Peak

    # row and col are only used for spectrum bookkeeping, not by the solver.
    return _Peak(
        frequency=1.0 / period, angle=angle, magnitude=magnitude, row=0, col=0
    )


def _contaminated_axis():
    """The exact peak set from real frame 2001-_05337.tif, axis at 158 deg.

    A clean 20.51px comb (20.51, 10.24, 6.82, 4.97) plus one stray peak at
    51.47px from the diagonal moire banding across that frame. 51.47 is 2.51x
    the true pitch, so it is a harmonic of nothing.
    """
    return [
        _peak(10.242, 158.37, 4171482),
        _peak(20.511, 158.24, 2875617),
        _peak(6.823, 158.45, 1769832),
        _peak(51.472, 149.30, 328203),
        _peak(4.971, 144.53, 321804),
        _peak(4.971, 172.29, 313645),
    ]


def test_a_stray_low_frequency_peak_does_not_capture_the_fit():
    """Seeding from the lowest observed frequency made that one peak a single
    point of failure: the stray 51.47px moire peak anchored the fit there,
    explained only 46% of the comb, and a measurable frame was refused."""
    from src.measurer import _solve_axis

    solution = _solve_axis(_contaminated_axis(), span=1024.0, cutoff=0.004)

    assert solution.spacing == pytest.approx(20.5, abs=0.3)
    assert solution.reliable
    assert solution.inlier_fraction > 0.8


def test_the_stray_peak_is_excluded_rather_than_fitted():
    from src.measurer import _solve_axis

    solution = _solve_axis(_contaminated_axis(), span=1024.0, cutoff=0.004)
    assert solution.dominant_order <= 2


def test_a_clean_comb_is_unaffected_by_the_new_search():
    """The fundamental must still win outright when nothing is contaminating."""
    from src.measurer import _solve_axis

    clean = [
        _peak(20.0, 158.0, 4000000),
        _peak(10.0, 158.0, 2000000),
        _peak(6.667, 158.0, 900000),
        _peak(5.0, 158.0, 400000),
    ]
    solution = _solve_axis(clean, span=1024.0, cutoff=0.004)

    assert solution.spacing == pytest.approx(20.0, rel=1e-3)
    assert solution.reliable
    assert not solution.fundamental_inferred
