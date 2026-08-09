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
