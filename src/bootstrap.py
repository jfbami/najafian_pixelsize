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

_TIFF_SUFFIXES = frozenset({".tif", ".tiff"})

_FIELDNAMES = [
    "path",
    "frame",
    "is_calibration",
    "score",
    "confidence",
    "concentration",
    "squareness",
    "axis_separation_deg",
    "axis_count",
    "pixels_per_space",
    "nm_per_pixel",
    "measurable",
    "warning",
    "error",
]


def label_folder(root: Path, output_csv: Path) -> list[dict]:
    rows = [_label_frame(path) for path in tiff_files(root)]
    _write_csv(rows, output_csv)
    return rows


def tiff_files(root: Path) -> Iterator[Path]:
    """Every TIFF under `root`, case-insensitively and including .tiff."""
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in _TIFF_SUFFIXES
    )


def _label_frame(path: Path) -> dict:
    """Label one frame. A frame that cannot be read is recorded, not fatal:
    aborting mid-run would discard the labels already computed."""
    try:
        detection = detect_calibration(str(path))
    except Exception as error:
        return _blank_row(path, f"{type(error).__name__}: {error}")

    measurement = detection.measurement
    return {
        "path": str(path),
        "frame": path.name,
        "is_calibration": int(detection.is_calibration),
        "score": round(detection.score, 4),
        "confidence": round(measurement.fft_confidence, 4),
        "concentration": round(measurement.spectral_concentration, 4),
        "squareness": round(measurement.grid_uniformity, 4),
        "axis_separation_deg": round(measurement.axis_separation_deg, 2),
        "axis_count": measurement.axis_count,
        "pixels_per_space": round(measurement.pixels_per_space, 4),
        "nm_per_pixel": round(measurement.nm_per_pixel, 4),
        "measurable": int(measurement.valid),
        "warning": measurement.warning or "",
        "error": "",
    }


def _blank_row(path: Path, error: str) -> dict:
    row = {name: "" for name in _FIELDNAMES}
    row.update({"path": str(path), "frame": path.name, "is_calibration": 0, "error": error})
    return row


def _write_csv(rows: list[dict], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
