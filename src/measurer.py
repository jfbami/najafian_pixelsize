"""Sub-pixel measurement of calibration grid spacing via 2D FFT.

A diffraction-grating image is periodic, so its Fourier transform shows a comb
of peaks at integer multiples of the grid's fundamental spatial frequency. The
strongest peak is often a harmonic rather than the fundamental, so a naive
"brightest peak" reading is wrong by an integer factor. This module detects the
whole harmonic comb along each grid axis and solves for the fundamental by
least squares, which both removes the harmonic ambiguity and sharpens precision
(higher harmonics resolve the frequency more finely).

Two ways the fundamental can be absent from the observed peaks, both of which
produce a *silently* wrong answer off by an integer factor if unhandled:

1. A 50%-duty grating has no even harmonics, so the visible comb can start at
   3*f0 if f0 itself is suppressed.
2. Any low-frequency mask that rejects illumination gradients also rejects the
   fundamental once the grid period grows past the mask radius.

Both are handled by not assuming the lowest observed peak is the fundamental:
the comb is explained by the largest f0 = f_lowest / k (k bounded, k = 1 first)
that makes *every* observed peak an integer multiple. The low-frequency mask is
additionally scaled to the image size, so it rejects gradients rather than long
grid periods.

Every result carries an explicit `valid` flag and an uncertainty. A frame that
cannot be measured returns NaN rather than a plausible-looking number.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import calibration
from .imaging import load_grayscale
from .models import MeasurementResult

_PEAK_COUNT = 24                # candidate peaks extracted from the spectrum
_PEAK_EXCLUSION_RADIUS = 6      # spectrum pixels masked around an accepted peak
_MIN_CYCLES_ACROSS_IMAGE = 3.0  # below this a "period" is an illumination gradient
_AXIS_ANGLE_TOLERANCE = 15.0    # degrees; peaks within this share a grid axis
_RELATIVE_PEAK_FLOOR = 0.04     # ignore comb peaks dimmer than this fraction of max
_SQUARENESS_WARN_THRESHOLD = 0.05
_ORTHOGONALITY_WARN_DEGREES = 8.0
_PROMINENCE_SATURATION = 60.0   # peak/median ratio mapped to confidence 1.0

# Harmonic-comb solving.
_MAX_MISSING_FUNDAMENTAL_ORDER = 8   # search f0 = f_lowest / k for k = 1 .. this
_HARMONIC_TOLERANCE = 0.12           # |n - f/f0| accepted as an integer multiple
_MAX_HARMONIC_ORDER = 24             # orders above this carry no usable grating energy
_MIN_INLIER_FRACTION = 0.50          # comb weight that must fit the fitted fundamental
_WEAK_INLIER_FRACTION = 0.80         # below this the comb is contaminated; warn

# A grating's own fundamental carries more energy than any of its harmonics, at
# every duty cycle. If the strongest peak sits at a high order instead, the fit
# has locked onto a sub-harmonic -- the signature of an undersampled frame,
# where aliased harmonics fold back onto the pattern's longer repeat period and
# form a comb that is perfectly self-consistent but has the wrong pitch.
_MAX_DOMINANT_ORDER = 2

# Validity bounds on the recovered spacing.
#
# The lower bound is set by harmonic aliasing, not by Nyquist on the fundamental:
# a square-wave grating of period D folds every harmonic above order D/2 back to
# a frequency that is *not* an integer multiple of the fundamental. Those folded
# peaks look like comb members and corrupt the fit. Requiring D >= 12 px keeps
# the first ~6 harmonics unaliased, which is where a blurred grating's energy
# lives. Frames below this are refused rather than measured -- notably Drive
# thumbnails, which are for detection only.
_MIN_RESOLVABLE_SPACING = 12.0       # pixels; below this the comb is alias-corrupted
_MARGINAL_SPACING = 30.0             # pixels; measurable but precision-limited
_MIN_PERIODS_ACROSS_IMAGE = 4.0      # need this many periods to trust a spacing
_MARGINAL_PERIODS_ACROSS_IMAGE = 10.0

# Geometry beyond which the frame is not a cross-grating at all.
_ORTHOGONALITY_REJECT_DEGREES = 20.0
_SQUARENESS_REJECT = 0.70

_ENERGY_WINDOW_RADIUS = 2            # bins summed per peak for concentration


@dataclass
class _Peak:
    frequency: float   # cycles per pixel
    angle: float       # degrees in [0, 180)
    magnitude: float
    row: int
    col: int


@dataclass
class _AxisSolution:
    spacing: float
    relative_uncertainty: float
    comb_size: int
    inlier_fraction: float
    dominant_order: int
    fundamental_inferred: bool
    reliable: bool


def measure_grid(image_path: str) -> MeasurementResult:
    return measure_array(load_grayscale(image_path))


def measure_array(image: np.ndarray) -> MeasurementResult:
    """Measure an already-loaded grayscale frame (used by tests and tiling)."""
    spectrum = _windowed_power_spectrum(image)
    height, width = image.shape
    span = float(min(height, width))
    cutoff = _low_frequency_cutoff(width, height)

    peaks = _extract_peaks(spectrum, width, height)
    confidence = _confidence(spectrum, peaks)
    concentration = _spectral_concentration(spectrum, peaks, width, height)
    axes = _cluster_into_axes(peaks)

    if len(axes) < 2:
        return _single_axis_result(axes, confidence, concentration, span, cutoff)

    primary = _solve_axis(axes[0], span, cutoff)
    secondary = _solve_axis(axes[1], span, cutoff)
    squareness = _dominant_peak_squareness(axes[0], axes[1])
    spacing_agreement = _squareness(primary.spacing, secondary.spacing)
    separation = _angular_distance(axes[0][0].angle, axes[1][0].angle)

    if spacing_agreement < 1.0 - _SQUARENESS_WARN_THRESHOLD:
        pixels_per_space = primary.spacing
        relative_uncertainty = primary.relative_uncertainty
    else:
        pixels_per_space = (primary.spacing + secondary.spacing) / 2.0
        relative_uncertainty = _combined_uncertainty(primary, secondary)

    warnings = _warnings(
        separation, spacing_agreement, primary, secondary, pixels_per_space, span
    )
    # Detection and measurement use different squareness notions on purpose:
    # `squareness` is scale-robust and decides whether this looks like a grid at
    # all; `spacing_agreement` compares the two fitted pitches and decides
    # whether the numbers are trustworthy. A thumbnail can pass the first and
    # fail the second, which is exactly the intended outcome.
    valid = (
        _is_valid(pixels_per_space, span)
        and primary.reliable
        and abs(separation - 90.0) <= _ORTHOGONALITY_REJECT_DEGREES
        and spacing_agreement >= _SQUARENESS_REJECT
    )

    return _build_result(
        pixels_per_space=pixels_per_space,
        relative_uncertainty=relative_uncertainty,
        spacing_primary=primary.spacing,
        spacing_secondary=secondary.spacing,
        confidence=confidence,
        concentration=concentration,
        squareness=squareness,
        separation=separation,
        axis_count=2,
        fundamental_inferred=primary.fundamental_inferred or secondary.fundamental_inferred,
        valid=valid,
        warnings=warnings,
    )


def _build_result(
    *,
    pixels_per_space: float,
    relative_uncertainty: float,
    spacing_primary: float,
    spacing_secondary: float,
    confidence: float,
    concentration: float,
    squareness: float,
    separation: float,
    axis_count: int,
    fundamental_inferred: bool,
    valid: bool,
    warnings: list[str],
) -> MeasurementResult:
    nanometers = calibration.nm_per_pixel(pixels_per_space) if valid else float("nan")
    if valid and not calibration.is_plausible(nanometers):
        warnings.append(
            f"nm/pixel {nanometers:.4f} outside the plausible range "
            f"{calibration.PLAUSIBLE_NM_PER_PIXEL}"
        )
        valid = False
        nanometers = float("nan")

    total_relative = calibration.total_relative_uncertainty(relative_uncertainty)
    return MeasurementResult(
        pixels_per_space=pixels_per_space if valid else float("nan"),
        nm_per_pixel=nanometers,
        spacing_x=spacing_primary,
        spacing_y=spacing_secondary,
        fft_confidence=confidence,
        grid_uniformity=squareness,
        axis_separation_deg=separation,
        axis_count=axis_count,
        spectral_concentration=concentration,
        relative_precision=relative_uncertainty,
        nm_per_pixel_uncertainty=(
            nanometers * total_relative if valid and math.isfinite(nanometers) else float("nan")
        ),
        fundamental_inferred=fundamental_inferred,
        valid=valid,
        warning="; ".join(warnings) if warnings else None,
    )


def _windowed_power_spectrum(image: np.ndarray) -> np.ndarray:
    """Hann-windowed magnitude spectrum, computed in single precision.

    float32 keeps a 4k x 4k frame's transform near 128 MB instead of the
    ~1 GB a float64 promotion would cost.
    """
    window_rows = np.hanning(image.shape[0]).astype(np.float32)
    window_cols = np.hanning(image.shape[1]).astype(np.float32)
    windowed = image.astype(np.float32, copy=True)
    windowed *= window_rows[:, None]
    windowed *= window_cols[None, :]
    transform = np.fft.fft2(windowed).astype(np.complex64, copy=False)
    return np.abs(np.fft.fftshift(transform)).astype(np.float32, copy=False)


def _low_frequency_cutoff(width: int, height: int) -> float:
    """Reject structures slower than a few cycles across the frame.

    Scaled to the image rather than fixed in cycles/pixel: a fixed cutoff
    silently masks the *fundamental* of any grid whose period exceeds its
    radius, which turns a long-period grid into an integer-factor error.
    """
    return _MIN_CYCLES_ACROSS_IMAGE / float(min(width, height))


def _extract_peaks(spectrum: np.ndarray, width: int, height: int) -> list[_Peak]:
    work = np.where(_searchable_region(spectrum.shape, width, height), spectrum, 0.0)
    center_row, center_col = spectrum.shape[0] // 2, spectrum.shape[1] // 2

    peaks: list[_Peak] = []
    for _ in range(_PEAK_COUNT):
        row, col = np.unravel_index(int(np.argmax(work)), work.shape)
        if work[row, col] <= 0.0:
            break
        refined_row = row + _parabolic_offset(_column_neighborhood(spectrum, row, col))
        refined_col = col + _parabolic_offset(_row_neighborhood(spectrum, row, col))
        frequency_x = (refined_col - center_col) / width
        frequency_y = (refined_row - center_row) / height
        frequency = math.hypot(frequency_x, frequency_y)
        if frequency > 0.0:
            peaks.append(
                _Peak(
                    frequency=frequency,
                    angle=math.degrees(math.atan2(frequency_y, frequency_x)) % 180.0,
                    magnitude=float(spectrum[row, col]),
                    row=int(row),
                    col=int(col),
                )
            )
        _suppress(work, row, col, _PEAK_EXCLUSION_RADIUS)
    return peaks


def _searchable_region(
    shape: tuple[int, int], width: int, height: int
) -> np.ndarray:
    rows = np.arange(shape[0])[:, None]
    cols = np.arange(shape[1])[None, :]
    center_row, center_col = shape[0] // 2, shape[1] // 2
    upper_half = (rows < center_row) | ((rows == center_row) & (cols > center_col))
    frequency = np.hypot((cols - center_col) / width, (rows - center_row) / height)
    return upper_half & (frequency > _low_frequency_cutoff(width, height))


def _suppress(spectrum: np.ndarray, row: int, col: int, radius: int) -> None:
    row_lo, row_hi = max(0, row - radius), row + radius + 1
    col_lo, col_hi = max(0, col - radius), col + radius + 1
    spectrum[row_lo:row_hi, col_lo:col_hi] = 0.0


def _cluster_into_axes(peaks: list[_Peak]) -> list[list[_Peak]]:
    """Group peaks into the (up to two) grid axes by orientation."""
    order = sorted(range(len(peaks)), key=lambda i: peaks[i].magnitude, reverse=True)
    axes: list[list[_Peak]] = []
    used: set[int] = set()
    while len(axes) < 2:
        seeds = [i for i in order if i not in used]
        if not seeds:
            break
        seed_angle = peaks[seeds[0]].angle
        members = [
            i
            for i in seeds
            if _angular_distance(peaks[i].angle, seed_angle) <= _AXIS_ANGLE_TOLERANCE
        ]
        axes.append([peaks[i] for i in members])
        used.update(members)
    return axes


def _angular_distance(angle_a: float, angle_b: float) -> float:
    difference = abs(angle_a - angle_b) % 180.0
    return min(difference, 180.0 - difference)


def _solve_axis(axis_peaks: list[_Peak], span: float, cutoff: float) -> _AxisSolution:
    """Recover the fundamental spacing of one grid axis from its comb.

    The fit is robust rather than all-or-nothing: peaks that are not integer
    multiples of the fundamental (aliased high harmonics, neighbouring texture)
    are treated as outliers and excluded, and the weight fraction that did fit
    is reported so a contaminated comb can be flagged instead of silently
    dragging the answer.
    """
    peak_floor = max(peak.magnitude for peak in axis_peaks) * _RELATIVE_PEAK_FLOOR
    comb = [peak for peak in axis_peaks if peak.magnitude >= peak_floor]

    seed, inferred = _fundamental_seed(comb, span, cutoff)
    inliers, orders, inlier_fraction = _select_inliers(comb, seed)
    if not inliers:
        return _AxisSolution(
            float("nan"), float("nan"), len(comb), 0.0, 0, inferred, False
        )

    dominant_order = max(zip(inliers, orders), key=lambda pair: pair[0].magnitude)[1]

    numerator = sum(p.magnitude * n * p.frequency for p, n in zip(inliers, orders))
    denominator = sum(p.magnitude * n * n for p, n in zip(inliers, orders))
    solved = numerator / denominator if denominator > 0.0 else 0.0
    if solved <= 0.0:
        return _AxisSolution(
            float("nan"), float("nan"), len(comb), inlier_fraction,
            dominant_order, inferred, False,
        )

    spacing = 1.0 / solved
    return _AxisSolution(
        spacing=spacing,
        relative_uncertainty=_frequency_uncertainty(inliers, orders, solved),
        comb_size=len(inliers),
        inlier_fraction=inlier_fraction,
        dominant_order=dominant_order,
        fundamental_inferred=inferred,
        reliable=(
            _is_valid(spacing, span)
            and inlier_fraction >= _MIN_INLIER_FRACTION
            and dominant_order <= _MAX_DOMINANT_ORDER
        ),
    )


def _fundamental_seed(
    comb: list[_Peak], span: float, cutoff: float
) -> tuple[float, bool]:
    """Seed the fit with the lowest observed comb frequency.

    A grating's fundamental is always present in its own spectrum -- a 50% duty
    cycle suppresses the *even* harmonics, never the first -- so the only way
    the fundamental can be absent from the observed peaks is if the
    low-frequency mask removed it. Sub-multiples are therefore considered only
    when they would land inside that mask, which makes the inference impossible
    to trigger spuriously on a normally sampled grid.
    """
    frequencies = sorted(peak.frequency for peak in comb)
    lowest = frequencies[0]
    if len(frequencies) < 2 or lowest > cutoff * _MAX_MISSING_FUNDAMENTAL_ORDER:
        return lowest, False

    for k in range(2, _MAX_MISSING_FUNDAMENTAL_ORDER + 1):
        candidate = lowest / k
        if candidate >= cutoff:
            continue  # this fundamental would have been visible; it is not missing
        if 1.0 / candidate > span / _MIN_PERIODS_ACROSS_IMAGE:
            break  # implied period no longer fits the frame
        _, orders, fraction = _select_inliers(comb, candidate)
        if fraction >= _WEAK_INLIER_FRACTION and _greatest_common_order(orders) == 1:
            return candidate, True
    return lowest, False


def _select_inliers(
    comb: list[_Peak], fundamental: float
) -> tuple[list[_Peak], list[int], float]:
    """Split the comb into integer multiples of `fundamental` and outliers."""
    if fundamental <= 0.0:
        return [], [], 0.0

    inliers: list[_Peak] = []
    orders: list[int] = []
    inlier_weight = 0.0
    total_weight = 0.0
    for peak in comb:
        total_weight += peak.magnitude
        ratio = peak.frequency / fundamental
        order = round(ratio)
        if 1 <= order <= _MAX_HARMONIC_ORDER and abs(ratio - order) <= _HARMONIC_TOLERANCE:
            inliers.append(peak)
            orders.append(order)
            inlier_weight += peak.magnitude
    fraction = inlier_weight / total_weight if total_weight > 0.0 else 0.0
    return inliers, orders, fraction


def _greatest_common_order(orders: list[int]) -> int:
    """GCD of the harmonic orders; >1 means a larger fundamental explains them."""
    common = 0
    for order in orders:
        common = math.gcd(common, order)
    return common


def _frequency_uncertainty(
    comb: list[_Peak], orders: list[int], solved: float
) -> float:
    """Relative 1-sigma on the fundamental from the comb fit residuals.

    With a single peak there is no residual to measure, so fall back to a
    conservative half-bin-equivalent bound rather than claiming zero error.
    """
    if len(comb) < 2:
        return _HARMONIC_TOLERANCE / max(1, orders[0])

    weights = [peak.magnitude for peak in comb]
    residuals = [
        weight * (peak.frequency - order * solved) ** 2
        for peak, order, weight in zip(comb, orders, weights)
    ]
    denominator = sum(w * n * n for w, n in zip(weights, orders))
    if denominator <= 0.0:
        return float("nan")
    variance = sum(residuals) / ((len(comb) - 1) * denominator)
    return math.sqrt(max(variance, 0.0)) / solved


def _combined_uncertainty(primary: _AxisSolution, secondary: _AxisSolution) -> float:
    """Uncertainty of the mean of two axis spacings, floored by their spread."""
    values = [primary.relative_uncertainty, secondary.relative_uncertainty]
    if not all(math.isfinite(value) for value in values):
        return float("nan")
    averaged = math.hypot(*values) / 2.0
    spread = abs(primary.spacing - secondary.spacing) / (
        2.0 * (primary.spacing + secondary.spacing) / 2.0
    )
    return max(averaged, spread / math.sqrt(2.0))


def _is_valid(spacing: float, span: float) -> bool:
    return (
        math.isfinite(spacing)
        and spacing >= _MIN_RESOLVABLE_SPACING
        and spacing <= span / _MIN_PERIODS_ACROSS_IMAGE
    )


def _confidence(spectrum: np.ndarray, peaks: list[_Peak]) -> float:
    """Peak prominence as a z-score against the whole spectrum.

    Retained unchanged because the deployed detection thresholds were validated
    against this definition. It saturates at 1.0 for any well-exposed grating,
    so it separates "some strong periodicity" from "none" but does not grade
    grid quality -- `_spectral_concentration` is the graded metric.
    """
    if not peaks:
        return 0.0
    spread = float(spectrum.std())
    if spread == 0.0:
        return 0.0
    median = float(np.median(spectrum))
    strongest = max(peak.magnitude for peak in peaks)
    z_score = (strongest - median) / spread
    return float(np.clip(z_score / _PROMINENCE_SATURATION, 0.0, 1.0))


def _spectral_concentration(
    spectrum: np.ndarray, peaks: list[_Peak], width: int, height: int
) -> float:
    """Fraction of non-DC spectral energy sitting in the detected peaks.

    Scale-invariant, unlike the z-score confidence: a grating concentrates its
    energy into a handful of comb bins, while tissue spreads it broadly. This
    is what actually distinguishes a grid from textured tissue at any image
    size, so it transfers between thumbnails and full-resolution frames.
    """
    if not peaks:
        return 0.0
    searchable = _searchable_region(spectrum.shape, width, height)
    energy = np.square(spectrum, dtype=np.float64)
    total = float(energy[searchable].sum())
    if total <= 0.0:
        return 0.0

    captured = 0.0
    radius = _ENERGY_WINDOW_RADIUS
    for peak in peaks:
        row_lo, row_hi = max(0, peak.row - radius), peak.row + radius + 1
        col_lo, col_hi = max(0, peak.col - radius), peak.col + radius + 1
        captured += float(energy[row_lo:row_hi, col_lo:col_hi].sum())
    return float(min(captured / total, 1.0))


def _squareness(spacing_a: float, spacing_b: float) -> float:
    if not (math.isfinite(spacing_a) and math.isfinite(spacing_b)):
        return 0.0
    larger = max(spacing_a, spacing_b)
    if larger <= 0.0:
        return 0.0
    return 1.0 - abs(spacing_a - spacing_b) / larger


def _dominant_peak_squareness(
    axis_a: list[_Peak], axis_b: list[_Peak]
) -> float:
    """How equal the two axes' periods are, from their strongest peaks alone.

    Derived from raw peak frequencies rather than the fitted fundamentals,
    because the comb fit needs a well-sampled grid and this feature is used to
    *detect* grids on thumbnails, where it is not. A grating's strongest peak is
    its fundamental on both axes, so their frequencies match whatever the image
    scale; deriving squareness from the fits instead collapsed it to ~0.2 on a
    thumbnail and made correctly-detected grids look like tissue.
    """
    frequency_a = max(axis_a, key=lambda peak: peak.magnitude).frequency
    frequency_b = max(axis_b, key=lambda peak: peak.magnitude).frequency
    larger = max(frequency_a, frequency_b)
    if larger <= 0.0:
        return 0.0
    return 1.0 - abs(frequency_a - frequency_b) / larger


def _single_axis_result(
    axes: list[list[_Peak]],
    confidence: float,
    concentration: float,
    span: float,
    cutoff: float,
) -> MeasurementResult:
    if not axes:
        return MeasurementResult(
            pixels_per_space=float("nan"),
            nm_per_pixel=float("nan"),
            spacing_x=float("nan"),
            spacing_y=float("nan"),
            fft_confidence=confidence,
            grid_uniformity=0.0,
            axis_separation_deg=0.0,
            axis_count=0,
            spectral_concentration=concentration,
            relative_precision=float("nan"),
            nm_per_pixel_uncertainty=float("nan"),
            fundamental_inferred=False,
            valid=False,
            warning="no periodic structure detected",
        )

    solution = _solve_axis(axes[0], span, cutoff)
    warnings = ["only one grid axis detected; result is single-direction"]
    if solution.fundamental_inferred:
        warnings.append("fundamental inferred from harmonics")
    if solution.comb_size < 2:
        warnings.append("single spectral peak; harmonic order unverified")

    return _build_result(
        pixels_per_space=solution.spacing,
        relative_uncertainty=solution.relative_uncertainty,
        spacing_primary=solution.spacing,
        spacing_secondary=float("nan"),
        confidence=confidence,
        concentration=concentration,
        squareness=0.0,
        separation=0.0,
        axis_count=1,
        fundamental_inferred=solution.fundamental_inferred,
        valid=solution.reliable and _is_valid(solution.spacing, span),
        warnings=warnings,
    )


def _warnings(
    separation: float,
    spacing_agreement: float,
    primary: _AxisSolution,
    secondary: _AxisSolution,
    pixels_per_space: float,
    span: float,
) -> list[str]:
    warnings: list[str] = []
    if abs(separation - 90.0) > _ORTHOGONALITY_WARN_DEGREES:
        warnings.append(f"grid axes are {separation:.1f} deg apart (expected ~90)")
    if spacing_agreement < 1.0 - _SQUARENESS_WARN_THRESHOLD:
        warnings.append(
            f"axis spacings disagree: {primary.spacing:.2f}px vs "
            f"{secondary.spacing:.2f}px; using dominant axis only"
        )
    if primary.fundamental_inferred or secondary.fundamental_inferred:
        warnings.append(
            "fundamental was not directly observed and was inferred from the "
            "harmonic comb; verify the spacing against the frame"
        )
    if min(primary.comb_size, secondary.comb_size) < 2:
        warnings.append("an axis showed a single spectral peak; harmonic order unverified")
    worst_order = max(primary.dominant_order, secondary.dominant_order)
    if worst_order > _MAX_DOMINANT_ORDER:
        warnings.append(
            f"strongest spectral peak is harmonic {worst_order} of the fitted "
            f"fundamental; the frame is undersampled and the pitch is unreliable"
        )
    weakest_fit = min(primary.inlier_fraction, secondary.inlier_fraction)
    if weakest_fit < _WEAK_INLIER_FRACTION:
        warnings.append(
            f"only {weakest_fit * 100:.0f}% of comb energy fits the fitted "
            f"fundamental; spectrum may be contaminated or aliased"
        )
    if not _is_valid(pixels_per_space, span):
        warnings.append(
            f"spacing {pixels_per_space:.2f}px outside the measurable range "
            f"[{_MIN_RESOLVABLE_SPACING:.0f}, {span / _MIN_PERIODS_ACROSS_IMAGE:.0f}]px "
            f"for a {span:.0f}px frame"
        )
    elif pixels_per_space < _MARGINAL_SPACING:
        warnings.append(
            f"spacing {pixels_per_space:.1f}px is close to the aliasing limit; "
            f"measure on the full-resolution frame if this came from a thumbnail"
        )
    elif span / pixels_per_space < _MARGINAL_PERIODS_ACROSS_IMAGE:
        warnings.append(
            f"only {span / pixels_per_space:.1f} grid periods across the frame; "
            f"precision is limited"
        )
    if abs(separation - 90.0) > _ORTHOGONALITY_REJECT_DEGREES:
        warnings.append(f"axes {separation:.1f} deg apart are not a cross-grating")
    elif spacing_agreement < _SQUARENESS_REJECT:
        warnings.append(
            f"the two axes fitted pitches that differ by "
            f"{(1.0 - spacing_agreement) * 100:.0f}%; no trustworthy single pitch"
        )
    return warnings


def _row_neighborhood(spectrum: np.ndarray, row: int, col: int) -> np.ndarray:
    if col == 0 or col == spectrum.shape[1] - 1:
        return np.array([])
    return spectrum[row, col - 1 : col + 2]


def _column_neighborhood(spectrum: np.ndarray, row: int, col: int) -> np.ndarray:
    if row == 0 or row == spectrum.shape[0] - 1:
        return np.array([])
    return spectrum[row - 1 : row + 2, col]


def _parabolic_offset(values: np.ndarray) -> float:
    """Sub-pixel peak offset from the vertex of a parabola through 3 samples."""
    if values.size < 3:
        return 0.0
    left, center, right = float(values[0]), float(values[1]), float(values[2])
    denominator = left - 2.0 * center + right
    if denominator == 0.0:
        return 0.0
    offset = 0.5 * (left - right) / denominator
    return offset if abs(offset) <= 1.0 else 0.0
