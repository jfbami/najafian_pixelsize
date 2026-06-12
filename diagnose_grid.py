"""Diagnostic: inspect the true grid structure of the calibration frame.

Lists the strongest FFT peaks with their spacing and orientation, cross-checks
with a spatial autocorrelation estimate, and saves a zoomed crop so the grid
period can be verified by eye.
"""

import math
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

FRAME = Path(r"C:\Users\jfbaa\OneDrive\Documents\test\17E00231\17E00231-1\17E00231-1_115.tif")
CROP_OUT = Path(r"C:\Users\jfbaa\OneDrive\Documents\test\calib_crop.png")


def load_gray(path):
    with Image.open(path) as handle:
        return np.asarray(handle.convert("L"), dtype=np.float32)


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


def save_crop(image):
    crop = image[800:1200, 800:1200]
    Image.fromarray(crop.astype(np.uint8)).save(CROP_OUT)


if __name__ == "__main__":
    image = load_gray(FRAME)
    print(f"image: {image.shape[1]} x {image.shape[0]}")

    print("\nTop FFT peaks (spacing_px, angle_deg, peak/median):")
    for spacing, angle, strength in top_peaks(image):
        print(f"  {spacing:8.3f} px   {angle:7.2f} deg   x{strength:8.1f}")

    row_lag, col_lag = autocorrelation_spacing(image)
    print(f"\nAutocorrelation first peak (row-avg profile): {row_lag} px")
    print(f"Autocorrelation first peak (col-avg profile): {col_lag} px")

    save_crop(image)
    print(f"\nSaved 400x400 crop to {CROP_OUT}")
