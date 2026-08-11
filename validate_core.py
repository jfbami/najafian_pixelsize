"""Validate the FFT measurer and metadata parser against real frames.

Usage:
    python validate_core.py FRAME.tif [FRAME2.tif ...]
    python validate_core.py --folder PATH/TO/ACQUISITION
    python validate_core.py FRAME.tif --region

Prints the parsed metadata, the measurement with its uncertainty, and the
cross-checks against the microscope's own embedded scale.

`--region` additionally measures only the flat, undistorted tiles of the
grating and reports how far the whole-frame reading sits from them. A large
disagreement means the replica is folded or contaminated and the whole-frame
number should not be used.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src import calibration, crosscheck
from src.bootstrap import tiff_files
from src.detect import detect_calibration
from src.metadata import parse_tiff_metadata


def report(frame: Path, use_regions: bool = False) -> None:
    print(f"\n=== {frame.name} ===")
    try:
        metadata = parse_tiff_metadata(str(frame))
    except Exception as error:
        print(f"  metadata unreadable: {type(error).__name__}: {error}")
        metadata = None
    else:
        print(f"  instrument:    {metadata.instrument_id}")
        print(f"  date / time:   {metadata.acquisition_date} {metadata.acquisition_time}")
        print(f"  magnification: {metadata.magnification}")
        print(f"  embedded cal:  {metadata.pixel_cal_embedded} {metadata.unit_embedded}")
        print(f"  dimensions:    {metadata.image_width} x {metadata.image_height}")

    try:
        detection = detect_calibration(str(frame))
    except Exception as error:
        print(f"  MEASUREMENT FAILED: {type(error).__name__}: {error}")
        return

    result = detection.measurement
    print(f"  classified as: {'CALIBRATION' if detection.is_calibration else 'tissue'} "
          f"(score {detection.score:.3f})")
    print(f"  fft confidence:    {result.fft_confidence:.3f}")
    print(f"  concentration:     {result.spectral_concentration:.4f}")
    print(f"  squareness:        {result.grid_uniformity:.3f}")
    print(f"  axis separation:   {result.axis_separation_deg:.1f} deg")

    if not result.valid:
        print(f"  --> NOT MEASURABLE: {result.warning}")
        return

    print(f"  spacing primary:   {result.spacing_x:.3f} px")
    print(f"  spacing secondary: {result.spacing_y:.3f} px")
    print(f"  pixels/space (D):  {result.pixels_per_space:.3f} px")
    print(f"  --> nm/pixel:      {result.nm_per_pixel:.4f} "
          f"+/- {result.nm_per_pixel_uncertainty:.4f} "
          f"({result.relative_precision * 100:.3f}% fit, "
          f"{calibration.PITCH_RELATIVE_UNCERTAINTY * 100:.1f}% standard)")

    if metadata is not None and metadata.magnification:
        constant = result.nm_per_pixel * metadata.magnification
        print(f"  nm/px x mag:       {constant:.0f}  "
              f"(should be constant per camera; compare across frames)")
    if result.warning:
        print(f"  warning:           {result.warning}")

    if use_regions:
        _report_regions(frame)


def _report_regions(frame: Path) -> None:
    from src.imaging import load_grayscale
    from src.regions import measure_straightest_region

    region = measure_straightest_region(load_grayscale(str(frame)))
    best = region.measurement
    print(f"  --- straightest region ---")
    print(f"  tiles used:        {len(region.selected_tiles)} of {len(region.tiles)} "
          f"at {region.tile_size}px")
    if not best.valid:
        print(f"  no usable region:  {region.warning}")
        return
    print(f"  --> nm/pixel:      {best.nm_per_pixel:.4f} +/- "
          f"{best.nm_per_pixel_uncertainty:.4f}")
    print(f"  tile spread:       {region.tile_spread_percent:.3f}%")
    print(f"  whole-frame gap:   {region.disagreement_percent:+.3f}%  "
          f"(large means the replica is distorted)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frames", nargs="*", type=Path)
    parser.add_argument("--folder", type=Path, help="report on every TIFF in a folder")
    parser.add_argument(
        "--region",
        action="store_true",
        help="also measure only the flat, undistorted tiles of the grating",
    )
    args = parser.parse_args()

    frames = list(args.frames)
    if args.folder:
        frames.extend(tiff_files(args.folder))
    if not frames:
        parser.error("give at least one frame, or --folder")

    for frame in frames:
        report(frame, use_regions=args.region)


if __name__ == "__main__":
    main()
