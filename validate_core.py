"""Validate the FFT measurer and metadata parser against local test data."""

from pathlib import Path

from src.measurer import measure_grid
from src.metadata import parse_tiff_metadata

DATA_ROOT = Path(r"C:\Users\jfbaa\OneDrive\Documents\test\17E00231\17E00231-1")
CALIBRATION_FRAME = DATA_ROOT / "17E00231-1_115.tif"
TISSUE_FRAME = DATA_ROOT / "17E00231-1_001.tif"


def report(label: str, frame: Path) -> None:
    print(f"\n=== {label}: {frame.name} ===")
    metadata = parse_tiff_metadata(str(frame))
    print(f"  instrument:    {metadata.instrument_id}")
    print(f"  date / time:   {metadata.acquisition_date} {metadata.acquisition_time}")
    print(f"  magnification: {metadata.magnification}")
    print(f"  embedded cal:  {metadata.pixel_cal_embedded} {metadata.unit_embedded}")
    print(f"  dimensions:    {metadata.image_width} x {metadata.image_height}")

    result = measure_grid(str(frame))
    print(f"  spacing primary:   {result.spacing_x:.3f} px")
    print(f"  spacing secondary: {result.spacing_y:.3f} px")
    print(f"  pixels/space (D):  {result.pixels_per_space:.3f} px")
    print(f"  --> nm/pixel:      {result.nm_per_pixel:.4f}")
    print(f"  fft confidence:    {result.fft_confidence:.3f}")
    print(f"  grid uniformity:   {result.grid_uniformity:.3f}")
    if result.warning:
        print(f"  warning:           {result.warning}")


if __name__ == "__main__":
    report("CALIBRATION", CALIBRATION_FRAME)
    report("TISSUE", TISSUE_FRAME)
