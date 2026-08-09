"""Quantify measurement precision by sampling regions across a grid image.

Mirrors the manual SOP ("sample widely across the whole image") and reports the
spread of the per-region nm/pixel estimates as an empirical precision bound.

Precision is not accuracy: this measures repeatability only. The grating's own
ruling tolerance (`calibration.PITCH_RELATIVE_UNCERTAINTY`) is a systematic
floor underneath every number here, and no amount of region averaging reduces
it.

Usage:
    python precision_check.py PATH/TO/calibration.tif [--tile 1600]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src import calibration
from src.imaging import load_grayscale
from src.measurer import measure_array

DEFAULT_TILE = 1600


def measure_region(region: np.ndarray) -> float:
    """nm/pixel for one tile, or NaN if the tile is not measurable."""
    return measure_array(np.ascontiguousarray(region)).nm_per_pixel


def tile_positions(height: int, width: int, tile: int) -> dict[str, tuple[int, int]]:
    return {
        "top-left": (0, 0),
        "top-right": (0, width - tile),
        "bottom-left": (height - tile, 0),
        "bottom-right": (height - tile, width - tile),
        "center": ((height - tile) // 2, (width - tile) // 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frame", type=Path, help="calibration TIFF to profile")
    parser.add_argument("--tile", type=int, default=DEFAULT_TILE)
    args = parser.parse_args()

    image = load_grayscale(str(args.frame))
    height, width = image.shape
    tile = min(args.tile, height, width)
    if tile < args.tile:
        print(f"note: tile reduced to {tile}px to fit a {width}x{height} frame")

    whole = measure_array(image)
    print(f"\nWhole frame: {whole.nm_per_pixel:.4f} nm/px "
          f"(D = {whole.pixels_per_space:.4f} px, valid={whole.valid})")
    if whole.warning:
        print(f"  warning: {whole.warning}")

    print(f"\nPer-region nm/pixel over {tile}x{tile} tiles:")
    values: list[float] = []
    for name, (top, left) in tile_positions(height, width, tile).items():
        value = measure_region(image[top : top + tile, left : left + tile])
        marker = "" if np.isfinite(value) else "   (not measurable)"
        print(f"  {name:13s}: {value:9.4f} nm/px{marker}")
        if np.isfinite(value):
            values.append(value)

    if len(values) < 2:
        print("\nToo few measurable regions to estimate precision.")
        return

    array = np.array(values)
    relative = float(array.std(ddof=1) / array.mean())
    print(f"\nRegions measured : {len(values)}")
    print(f"  mean           : {array.mean():.4f} nm/px")
    print(f"  std (n-1)      : {array.std(ddof=1):.4f} nm/px")
    print(f"  range          : {np.ptp(array):.4f} nm/px")
    print(f"  repeatability  : {relative * 100:.3f}%  (precision only)")
    print(f"  grating tolerance: {calibration.PITCH_RELATIVE_UNCERTAINTY * 100:.3f}%  (systematic floor)")
    print(f"  combined        : {calibration.total_relative_uncertainty(relative) * 100:.3f}%")


if __name__ == "__main__":
    main()
