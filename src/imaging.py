"""Bit-depth-aware loading of microscope TIFFs into a float array.

`PIL.Image.convert("L")` is lossy in a way that silently destroys 16-bit EM
frames: converting mode "I;16" to "L" clips every value above 255, so a
16-bit grating image becomes a flat field of 255 and the FFT measurement
returns a plausible-looking but meaningless number. Integer and float modes
are therefore read through the array interface, which preserves the full
range; only true 8-bit and colour modes go through `convert`.

The FFT measurement depends only on ratios within the spectrum, so no
rescaling is applied - the native intensity range is carried through.
"""

from __future__ import annotations

import re
from typing import Optional

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

_IMAGE_DESCRIPTION_TAG = 270
# "##fv3\r<width>\r<height>" - the imaged field, excluding the info bar.
_CONTENT_SIZE_PATTERN = re.compile(r"##fv\d+\s*[\r\n]\s*(\d+)\s*[\r\n]\s*(\d+)")

# Modes whose sample values exceed what "L" can hold; read them directly.
_WIDE_MODES = frozenset({"I", "I;16", "I;16B", "I;16L", "I;16N", "I;32", "I;32B", "F"})

# Info-bar detection.
_BAR_FLATNESS = 0.7        # fraction of a line at one value for it to be bar
# A bar line is also nearly uniform. Modal share alone is not enough: a
# high-contrast grating is mostly dark background with thin bright lines, so
# over 70% of its pixels sit at the background value while the line still has a
# standard deviation of 89. Requiring low variance as well stops real image
# content being trimmed away as if it were bar.
_BAR_MAX_RELATIVE_STD = 0.05
_BAR_VALUE_TOLERANCE = 1.0 # intensity units counted as "the same value"
_BAR_TEXT_GAP = 60         # lines of text a bar may contain without splitting
_MIN_CONTENT_FRACTION = 0.5  # never crop away more than half an axis


def load_grayscale(image_path: str, crop_info_bar: bool = True) -> np.ndarray:
    """Load the first page of a TIFF as a 2-D float32 array.

    The burned-in info bar many acquisition programs append is cropped by
    default: it is not part of the imaged field, and its hard edge against the
    grid dominates the transform along one axis.
    """
    with Image.open(image_path) as handle:
        handle.seek(0)
        array = _to_array(handle)
        declared = _declared_content_size(handle)

    if array.ndim == 3:
        array = array.mean(axis=2)
    if array.ndim != 2:
        raise ValueError(f"expected a 2-D image, got shape {array.shape}")
    if min(array.shape) < 2:
        raise ValueError(f"image too small to measure: {array.shape}")

    array = np.ascontiguousarray(array, dtype=np.float32)
    if crop_info_bar:
        rows, cols = content_region(array, declared)
        array = np.ascontiguousarray(array[rows, cols])
    return array


def _declared_content_size(handle: Image.Image) -> Optional[tuple[int, int]]:
    """Imaged field size from the '##fv3 <width> <height>' metadata marker.

    The acquisition software records the field it actually exposed, separately
    from the saved file size, which includes the info bar. When it is present
    this is exact and beats inferring the bar from pixel statistics.
    """
    raw = handle.tag_v2.get(_IMAGE_DESCRIPTION_TAG, "")
    if isinstance(raw, bytes):
        raw = raw.decode("latin-1", errors="ignore")
    match = _CONTENT_SIZE_PATTERN.search(str(raw))
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def content_region(
    image: np.ndarray, declared: Optional[tuple[int, int]] = None
) -> tuple[slice, slice]:
    """Locate the imaged field, excluding any burned-in info/data bar.

    Acquisition software (AMT, DigitalMicrograph, TIA) appends a strip carrying
    the file name, scale bar and instrument settings. It is a flat background
    with a little text, and the step between it and the image is a hard edge
    spanning the full width -- which puts a strong ridge across one axis of the
    transform and can dominate the real grid comb. On a real 1024x1194 grating
    frame the bar made the two axes fit 21.1 px and 298.5 px, and the
    measurement was refused.

    `declared` is the field size the metadata reports, when available; it is
    exact and is used directly. Otherwise the bar is inferred: a bar row is
    almost entirely one value, which separates it sharply from imaged content.
    Small gaps are bridged so the bar's text lines do not split it, and only
    edge-anchored strips are trimmed, so flat regions inside the field are
    never removed.
    """
    if declared is not None:
        width, height = declared
        if 0 < height <= image.shape[0] and 0 < width <= image.shape[1]:
            return slice(0, height), slice(0, width)
    return _axis_slice(image, axis=1), _axis_slice(image, axis=0)


def _axis_slice(image: np.ndarray, axis: int) -> slice:
    flat = _flat_fraction(image, axis=axis)
    is_bar = (flat > _BAR_FLATNESS) & _is_uniform(image, axis)
    is_bar = _bridge_gaps(is_bar, _BAR_TEXT_GAP)
    length = is_bar.size

    start = 0
    while start < length and is_bar[start]:
        start += 1
    end = length
    while end > start and is_bar[end - 1]:
        end -= 1

    # Never trim so much that the crop stops being the image.
    if end - start < length * _MIN_CONTENT_FRACTION:
        return slice(0, length)
    return slice(start, end)


def _is_uniform(image: np.ndarray, axis: int) -> np.ndarray:
    """Which lines are nearly flat compared with the frame's typical line."""
    spread = image.std(axis=axis)
    reference = float(np.median(spread))
    if reference <= 0.0:
        return np.zeros(spread.shape, dtype=bool)
    return spread <= reference * _BAR_MAX_RELATIVE_STD


def _flat_fraction(image: np.ndarray, axis: int) -> np.ndarray:
    """Per-line fraction of pixels sitting at that line's median value."""
    median = np.median(image, axis=axis, keepdims=True)
    return np.mean(np.abs(image - median) <= _BAR_VALUE_TOLERANCE, axis=axis)


def _bridge_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    """Fill short runs of False that sit between two True runs.

    The info bar's text lines are not flat, so without this the bar would be
    detected as several strips and only the outermost would be trimmed.
    """
    filled = mask.copy()
    length = filled.size
    index = 0
    while index < length:
        if filled[index]:
            index += 1
            continue
        gap_end = index
        while gap_end < length and not filled[gap_end]:
            gap_end += 1
        interior = index > 0 and gap_end < length
        if interior and (gap_end - index) <= max_gap:
            filled[index:gap_end] = True
        index = gap_end
    return filled


def _to_array(handle: Image.Image) -> np.ndarray:
    if handle.mode in _WIDE_MODES:
        return np.asarray(handle, dtype=np.float32)
    return np.asarray(handle.convert("L"), dtype=np.float32)


def page_count(image_path: str) -> int:
    """Number of pages in a TIFF; >1 means only the first is measured."""
    with Image.open(image_path) as handle:
        return getattr(handle, "n_frames", 1)
