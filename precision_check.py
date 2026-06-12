"""Quantify measurement precision by sampling regions across the grid image.

Mirrors the manual SOP ("sample widely across the whole image") and reports the
spread of the per-region nm/pixel estimates as an empirical precision bound.
"""

from pathlib import Path

import numpy as np
from PIL import Image

from src import calibration
from src.measurer import (
    _cluster_into_axes,
    _extract_peaks,
    _fundamental_spacing,
    _windowed_power_spectrum,
)

Image.MAX_IMAGE_PIXELS = None

FRAME = Path(r"C:\Users\jfbaa\OneDrive\Documents\test\17E00231\17E00231-1\17E00231-1_115.tif")
TILE = 1600


def measure_region(region: np.ndarray) -> float:
    spectrum = _windowed_power_spectrum(region)
    height, width = region.shape
    axes = _cluster_into_axes(_extract_peaks(spectrum, width, height))
    spacings = [_fundamental_spacing(axis) for axis in axes[:2]]
    return float(np.mean(spacings))


def main() -> None:
    with Image.open(FRAME) as handle:
        image = np.asarray(handle.convert("L"), dtype=np.float32)
    height, width = image.shape

    positions = {
        "top-left": (0, 0),
        "top-right": (0, width - TILE),
        "bottom-left": (height - TILE, 0),
        "bottom-right": (height - TILE, width - TILE),
        "center": ((height - TILE) // 2, (width - TILE) // 2),
    }

    spacings = []
    print(f"Region spacing (D) over {TILE}x{TILE} tiles:")
    for name, (top, left) in positions.items():
        region = image[top : top + TILE, left : left + TILE]
        spacing = measure_region(region)
        spacings.append(spacing)
        print(f"  {name:13s}: D = {spacing:8.4f} px   ({calibration.nm_per_pixel(spacing):.4f} nm/px)")

    spacings = np.array(spacings)
    nm = np.array([calibration.nm_per_pixel(s) for s in spacings])
    print(f"\nAll tiles  : median D {np.median(spacings):.4f} px   std {spacings.std():.4f} px")

    inliers = spacings[np.abs(spacings - np.median(spacings)) < 5.0]
    nm_inliers = np.array([calibration.nm_per_pixel(s) for s in inliers])
    print(f"Inliers ({len(inliers)}/{len(spacings)}): D mean {inliers.mean():.4f} px   std {inliers.std():.4f} px   range {np.ptp(inliers):.4f} px")
    print(f"            nm/px mean {nm_inliers.mean():.4f}   std {nm_inliers.std():.4f}   range {np.ptp(nm_inliers):.4f}")
    print(f"            relative precision: {nm_inliers.std() / nm_inliers.mean() * 100:.3f}%")


if __name__ == "__main__":
    main()
