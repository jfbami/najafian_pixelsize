"""Validate the FFT measurer and metadata parser against real frames.

Usage:
    python validate_core.py FRAME.tif [FRAME2.tif ...]
    python validate_core.py --folder PATH/TO/ACQUISITION

Prints the parsed metadata, the measurement with its uncertainty, and the
cross-checks against the microscope's own embedded scale.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src import calibration, crosscheck
from src.bootstrap import tiff_files
from src.detect import detect_calibration
from src.metadata import parse_tiff_metadata


def report(frame: Path) -> None:
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frames", nargs="*", type=Path)
    parser.add_argument("--folder", type=Path, help="report on every TIFF in a folder")
    args = parser.parse_args()

    frames = list(args.frames)
    if args.folder:
        frames.extend(tiff_files(args.folder))
    if not frames:
        parser.error("give at least one frame, or --folder")

    for frame in frames:
        report(frame)


if __name__ == "__main__":
    main()
