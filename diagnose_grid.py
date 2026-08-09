"""Diagnostic: inspect the true grid structure of the calibration frame.

Lists the strongest FFT peaks with their spacing and orientation, cross-checks
with a spatial autocorrelation estimate, and saves a zoomed crop so the grid
period can be verified by eye.
"""

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image

from src.imaging import load_grayscale

Image.MAX_IMAGE_PIXELS = None


def load_gray(path):
    """Bit-depth aware, so 16-bit frames are not flattened before inspection."""
    return load_grayscale(str(path))


def top_peaks(image, count=12):
    window = np.outer(np.hanning(image.shape[0]), np.hanning(image.shape[1]))
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(image * window)))
    height, width = image.shape
    cy, cx = height // 2, width // 2

    rows = np.arange(height)[:, None]
    cols = np.arange(width)[None, :]
    fx = (cols - cx) / width
    fy = (rows - cy) / height
    freq = np.hypot(fx, fy)

    upper = (rows < cy) | ((rows == cy) & (cols > cx))
    searchable = upper & (freq > 0.005)
    work = np.where(searchable, spectrum, 0.0)

    median = np.median(spectrum)
    peaks = []
    for _ in range(count):
        r, c = np.unravel_index(int(np.argmax(work)), work.shape)
        f = math.hypot((c - cx) / width, (r - cy) / height)
        if f == 0:
            break
        spacing = 1.0 / f
        angle = math.degrees(math.atan2((r - cy) / height, (c - cx) / width))
        peaks.append((spacing, angle, spectrum[r, c] / median))
        work[max(0, r - 6):r + 7, max(0, c - 6):c + 7] = 0.0
    return peaks


def autocorrelation_spacing(image):
    row_profile = image.mean(axis=0) - image.mean()
    col_profile = image.mean(axis=1) - image.mean()
    return _first_peak(_autocorr(row_profile)), _first_peak(_autocorr(col_profile))


def _autocorr(signal):
    full = np.correlate(signal, signal, mode="full")
    return full[full.size // 2:]


def _first_peak(acf, min_lag=10):
    for lag in range(min_lag, len(acf) - 1):
        if acf[lag] > acf[lag - 1] and acf[lag] > acf[lag + 1] and acf[lag] > 0:
            return lag
    return None


def save_crop(image, destination):
    """Save a centre crop, rescaled to 8 bits so 16-bit frames stay visible."""
    top, left = max(0, image.shape[0] // 2 - 200), max(0, image.shape[1] // 2 - 200)
    crop = image[top : top + 400, left : left + 400]
    span = float(crop.max() - crop.min()) or 1.0
    scaled = (crop - crop.min()) / span * 255.0
    Image.fromarray(scaled.astype(np.uint8)).save(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frame", type=Path)
    parser.add_argument("--crop-out", type=Path, default=Path("calib_crop.png"))
    args = parser.parse_args()

    image = load_gray(args.frame)
    print(f"image: {image.shape[1]} x {image.shape[0]}")

    print("\nTop FFT peaks (spacing_px, angle_deg, peak/median):")
    for spacing, angle, strength in top_peaks(image):
        print(f"  {spacing:8.3f} px   {angle:7.2f} deg   x{strength:8.1f}")

    row_lag, col_lag = autocorrelation_spacing(image)
    print(f"\nAutocorrelation first peak (row-avg profile): {row_lag} px")
    print(f"Autocorrelation first peak (col-avg profile): {col_lag} px")

    save_crop(image, args.crop_out)
    print(f"\nSaved 400x400 crop to {args.crop_out}")


if __name__ == "__main__":
    main()
