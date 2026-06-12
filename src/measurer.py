"""Sub-pixel measurement of calibration grid spacing via 2D FFT.

A diffraction-grating image is periodic, so its Fourier transform shows a comb
of peaks at integer multiples of the grid's fundamental spatial frequency. The
strongest peak is often a harmonic rather than the fundamental, so a naive
"brightest peak" reading is wrong by an integer factor. This module instead
detects the whole harmonic comb along each grid axis and solves for the
fundamental by least squares, which both removes the harmonic ambiguity and
sharpens precision (higher harmonics resolve the frequency more finely).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image

from . import calibration
from .models import MeasurementResult

Image.MAX_IMAGE_PIXELS = None

_PEAK_COUNT = 16                # candidate peaks extracted from the spectrum
_PEAK_EXCLUSION_RADIUS = 6      # spectrum pixels masked around an accepted peak
_LOW_FREQUENCY_CUTOFF = 0.004   # cycles/pixel; ignore slow illumination gradients
_AXIS_ANGLE_TOLERANCE = 15.0    # degrees; peaks within this share a grid axis
_RELATIVE_PEAK_FLOOR = 0.04     # ignore comb peaks dimmer than this fraction of max
_SQUARENESS_WARN_THRESHOLD = 0.05
_ORTHOGONALITY_WARN_DEGREES = 8.0
_PROMINENCE_SATURATION = 60.0   # peak/median ratio mapped to confidence 1.0


@dataclass
class _Peak:
    frequency: float   # cycles per pixel
    angle: float       # degrees in [0, 180)
    magnitude: float


def measure_grid(image_path: str) -> MeasurementResult:
    image = _load_grayscale(image_path)
    spectrum = _windowed_power_spectrum(image)
    height, width = image.shape

    peaks = _extract_peaks(spectrum, width, height)
    confidence = _confidence(spectrum, peaks)
    axes = _cluster_into_axes(peaks)

    if len(axes) < 2:
        return _single_axis_result(axes, confidence)

    spacing_primary = _fundamental_spacing(axes[0])
    spacing_secondary = _fundamental_spacing(axes[1])
    squareness = _squareness(spacing_primary, spacing_secondary)
    separation = _angular_distance(axes[0][0].angle, axes[1][0].angle)

    pixels_per_space = (spacing_primary + spacing_secondary) / 2.0
    if squareness < 1.0 - _SQUARENESS_WARN_THRESHOLD:
        pixels_per_space = spacing_primary

    return MeasurementResult(
        pixels_per_space=pixels_per_space,
        nm_per_pixel=calibration.nm_per_pixel(pixels_per_space),
        spacing_x=spacing_primary,
        spacing_y=spacing_secondary,
        fft_confidence=confidence,
        grid_uniformity=squareness,
        axis_separation_deg=separation,
        axis_count=2,
        warning=_warning(axes, spacing_primary, spacing_secondary, squareness),
    )


def _load_grayscale(image_path: str) -> np.ndarray:
    with Image.open(image_path) as handle:
        return np.asarray(handle.convert("L"), dtype=np.float32)


def _windowed_power_spectrum(image: np.ndarray) -> np.ndarray:
    window = np.outer(np.hanning(image.shape[0]), np.hanning(image.shape[1]))
    transform = np.fft.fft2(image * window)
    return np.abs(np.fft.fftshift(transform))


def _extract_peaks(spectrum: np.ndarray, width: int, height: int) -> list[_Peak]:
    work = np.where(_searchable_region(spectrum.shape, width, height), spectrum, 0.0)
    center_row, center_col = spectrum.shape[0] // 2, spectrum.shape[1] // 2

    peaks: list[_Peak] = []
    for _ in range(_PEAK_COUNT):
        row, col = np.unravel_index(int(np.argmax(work)), work.shape)
        if work[row, col] == 0.0:
            break
        refined_row = row + _parabolic_offset(_column_neighborhood(spectrum, row, col))
        refined_col = col + _parabolic_offset(_row_neighborhood(spectrum, row, col))
        frequency_x = (refined_col - center_col) / width
        frequency_y = (refined_row - center_row) / height
        peaks.append(
            _Peak(
                frequency=math.hypot(frequency_x, frequency_y),
                angle=math.degrees(math.atan2(frequency_y, frequency_x)) % 180.0,
                magnitude=float(spectrum[row, col]),
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
    return upper_half & (frequency > _LOW_FREQUENCY_CUTOFF)


def _suppress(spectrum: np.ndarray, row: int, col: int, radius: int) -> None:
    row_lo, row_hi = max(0, row - radius), row + radius + 1
    col_lo, col_hi = max(0, col - radius), col + radius + 1
    spectrum[row_lo:row_hi, col_lo:col_hi] = 0.0


def _cluster_into_axes(peaks: list[_Peak]) -> list[list[_Peak]]:
    """Group peaks into the (up to two) grid axes by orientation."""
    remaining = sorted(peaks, key=lambda peak: peak.magnitude, reverse=True)
    axes: list[list[_Peak]] = []
    while remaining and len(axes) < 2:
        seed = remaining[0]
        members = [p for p in remaining if _angular_distance(p.angle, seed.angle) <= _AXIS_ANGLE_TOLERANCE]
        axes.append(members)
        remaining = [p for p in remaining if p not in members]
    return axes


def _angular_distance(angle_a: float, angle_b: float) -> float:
    difference = abs(angle_a - angle_b) % 180.0
    return min(difference, 180.0 - difference)


def _fundamental_spacing(axis_peaks: list[_Peak]) -> float:
    """Solve f_i = n_i * f0 by magnitude-weighted least squares over the comb."""
    peak_floor = max(peak.magnitude for peak in axis_peaks) * _RELATIVE_PEAK_FLOOR
    comb = [peak for peak in axis_peaks if peak.magnitude >= peak_floor]
    fundamental_frequency = min(peak.frequency for peak in comb)

    numerator = 0.0
    denominator = 0.0
    for peak in comb:
        harmonic = max(1, round(peak.frequency / fundamental_frequency))
        weight = peak.magnitude
        numerator += weight * harmonic * peak.frequency
        denominator += weight * harmonic * harmonic

    solved_frequency = numerator / denominator
    return 1.0 / solved_frequency


def _confidence(spectrum: np.ndarray, peaks: list[_Peak]) -> float:
    if not peaks:
        return 0.0
    spread = float(spectrum.std())
    if spread == 0.0:
        return 0.0
    median = float(np.median(spectrum))
    strongest = max(peak.magnitude for peak in peaks)
    z_score = (strongest - median) / spread
    return float(np.clip(z_score / _PROMINENCE_SATURATION, 0.0, 1.0))


def _squareness(spacing_a: float, spacing_b: float) -> float:
    larger = max(spacing_a, spacing_b)
    if larger == 0.0:
        return 0.0
    return 1.0 - abs(spacing_a - spacing_b) / larger


def _single_axis_result(
    axes: list[list[_Peak]], confidence: float
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
            warning="no periodic structure detected",
        )
    spacing = _fundamental_spacing(axes[0])
    return MeasurementResult(
        pixels_per_space=spacing,
        nm_per_pixel=calibration.nm_per_pixel(spacing),
        spacing_x=spacing,
        spacing_y=float("nan"),
        fft_confidence=confidence,
        grid_uniformity=0.0,
        axis_separation_deg=0.0,
        axis_count=1,
        warning="only one grid axis detected; result is single-direction",
    )


def _warning(
    axes: list[list[_Peak]],
    spacing_primary: float,
    spacing_secondary: float,
    squareness: float,
) -> str | None:
    separation = _angular_distance(axes[0][0].angle, axes[1][0].angle)
    if abs(separation - 90.0) > _ORTHOGONALITY_WARN_DEGREES:
        return f"grid axes are {separation:.1f} deg apart (expected ~90)"
    if squareness < 1.0 - _SQUARENESS_WARN_THRESHOLD:
        return (
            f"axis spacings disagree: {spacing_primary:.2f}px vs "
            f"{spacing_secondary:.2f}px; using dominant axis only"
        )
    return None


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
    return 0.5 * (left - right) / denominator
