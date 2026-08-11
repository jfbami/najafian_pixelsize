"""Synthetic cross-gratings with an exactly known pitch.

The real reference frames live on lab Drive accounts, so the accuracy of the
measurement chain cannot be regression-tested against them. A generated
grating has a pitch that is known exactly, which is a stronger test: it checks
the answer is *right*, not merely reproducible.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import gaussian_filter

BACKGROUND = 28.0
CONTRAST = 200.0


def cross_grating(
    period: float,
    size: int = 2048,
    height: int | None = None,
    angle_deg: float = 0.0,
    duty: float = 0.5,
    noise: float = 2.0,
    blur: float = 1.0,
    seed: int = 0,
) -> np.ndarray:
    """Two orthogonal square-wave bar sets with the given period in pixels."""
    rows = height if height is not None else size
    y, x = np.mgrid[0:rows, 0:size].astype(np.float64)
    angle = math.radians(angle_deg)
    along = x * math.cos(angle) + y * math.sin(angle)
    across = -x * math.sin(angle) + y * math.cos(angle)

    def bars(coordinate: np.ndarray) -> np.ndarray:
        return ((coordinate / period) % 1.0 < duty).astype(np.float64)

    image = bars(along) + bars(across)
    image = BACKGROUND + CONTRAST * (image / image.max())
    if blur > 0:
        image = gaussian_filter(image, blur)
    if noise > 0:
        image = image + np.random.default_rng(seed).normal(0.0, noise, image.shape)
    return np.clip(image, 0.0, 255.0).astype(np.float32)


def tissue_like(size: int = 2048, seed: int = 0) -> np.ndarray:
    """Broadband, structured, non-periodic texture standing in for tissue."""
    rng = np.random.default_rng(seed)
    image = gaussian_filter(rng.normal(128.0, 40.0, (size, size)), 3.0)
    image += gaussian_filter(rng.normal(0.0, 60.0, (size, size)), 12.0)
    return np.clip(image, 0.0, 255.0).astype(np.float32)


def write_tiff(image: np.ndarray, path, bit_depth: int = 8):
    """Save `image` as an 8- or 16-bit TIFF and return the path."""
    from PIL import Image

    if bit_depth == 16:
        scaled = (image / 255.0 * 65535.0).astype(np.uint16)
        Image.fromarray(scaled, mode="I;16").save(path)
    else:
        Image.fromarray(image.astype(np.uint8), mode="L").save(path)
    return path


def folded_grating(
    period: float,
    size: int = 1024,
    tilt_deg: float = 25.0,
    fold_from_fraction: float = 0.6,
    noise: float = 2.0,
    seed: int = 0,
) -> np.ndarray:
    """Cross grating that tilts out of the image plane past a given column.

    Models the real defect: a wrinkled replica foreshortens along the tilt
    direction only, so the folded region reads short on one axis and stays
    correct on the other. That anisotropy is what marks a tile as distorted.
    Everything left of `fold_from_fraction` has exactly `period`.
    """
    columns = np.arange(size)
    ramp = np.clip((columns - size * fold_from_fraction) / 80.0, 0.0, 1.0)
    foreshortening = 1.0 - (1.0 - math.cos(math.radians(tilt_deg))) * ramp
    phase_x = np.cumsum(1.0 / (period * foreshortening))
    phase_y = np.arange(size) / period

    horizontal = np.tile(phase_x, (size, 1))
    vertical = np.tile(phase_y[:, None], (1, size))
    image = ((horizontal % 1.0) < 0.5).astype(float)
    image += ((vertical % 1.0) < 0.5).astype(float)
    image = 28.0 + 200.0 * (image / image.max())
    image = gaussian_filter(image, 1.0)
    if noise:
        image = image + np.random.default_rng(seed).normal(0.0, noise, image.shape)
    return np.clip(image, 0.0, 255.0).astype(np.float32)
