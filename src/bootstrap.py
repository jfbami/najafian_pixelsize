"""Scan a folder of TIFFs and auto-label calibration vs tissue frames via FFT.

The output CSV is the training set for the learned detector and is itself a
usable inventory of which frames are calibration grids and their measured
nm/pixel. No manual annotation is required.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterator

from .detect import detect_calibration

_FIELDNAMES = [
    "path",
    "frame",
    "is_calibration",
    "score",
    "confidence",
    "squareness",
    "axis_separation_deg",
    "axis_count",
    "pixels_per_space",
    "nm_per_pixel",
    "warning",
]


def label_folder(root: Path, output_csv: Path) -> list[dict]:
    rows = [_label_frame(path) for path in _tiff_files(root)]
    _write_csv(rows, output_csv)
    return rows


def _tiff_files(root: Path) -> Iterator[Path]:
    return sorted(root.rglob("*.tif"))


def _label_frame(path: Path) -> dict:
    detection = detect_calibration(str(path))
    measurement = detection.measurement
    return {
        "path": str(path),
        "frame": path.name,
        "is_calibration": int(detection.is_calibration),
        "score": round(detection.score, 4),
        "confidence": round(measurement.fft_confidence, 4),
        "squareness": round(measurement.grid_uniformity, 4),
        "axis_separation_deg": round(measurement.axis_separation_deg, 2),
        "axis_count": measurement.axis_count,
        "pixels_per_space": round(measurement.pixels_per_space, 4),
        "nm_per_pixel": round(measurement.nm_per_pixel, 4),
        "warning": measurement.warning or "",
    }


def _write_csv(rows: list[dict], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
