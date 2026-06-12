"""Parser for the microscope metadata embedded in each TIFF.

The acquisition software stores a carriage-return delimited string in the TIFF
ImageDescription tag (270). The layout observed across the dataset is:

    instrument, date, time, magnification, kV, working_distance, "Imaging", ...
    XpixCal=<value>, YpixCal=<value>, Unit=<unit>, ...

The embedded XpixCal/Unit is treated as advisory only: its unit encoding has
been observed to be unreliable, so the FFT measurement is authoritative.
"""

from __future__ import annotations

import re
from typing import Optional

from PIL import Image

from .models import TiffMetadata

Image.MAX_IMAGE_PIXELS = None

_IMAGE_DESCRIPTION_TAG = 270
_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_TIME_PATTERN = re.compile(r"^\d{1,2}:\d{2}$")
_PIXEL_CAL_PATTERN = re.compile(r"XpixCal=([0-9.]+)")
_UNIT_PATTERN = re.compile(r"Unit=([^\s]+)")

# Field layout: date, then time, then magnification.
_FIELDS_FROM_DATE_TO_MAGNIFICATION = 2


def parse_tiff_metadata(image_path: str) -> TiffMetadata:
    with Image.open(image_path) as handle:
        description = _read_description(handle)
        width, height = handle.size

    fields = _split_fields(description)

    return TiffMetadata(
        magnification=_extract_magnification(fields),
        acquisition_date=_extract_date(fields),
        acquisition_time=_extract_time(fields),
        pixel_cal_embedded=_extract_pixel_cal(description),
        unit_embedded=_extract_unit(description),
        instrument_id=fields[0] if fields else None,
        image_width=width,
        image_height=height,
    )


def _read_description(handle: Image.Image) -> str:
    raw = handle.tag_v2.get(_IMAGE_DESCRIPTION_TAG, "")
    if isinstance(raw, bytes):
        raw = raw.decode("latin-1", errors="ignore")
    return str(raw).replace("\x00", "")


def _split_fields(description: str) -> list[str]:
    normalized = description.replace("\n", "\r")
    return [field.strip() for field in normalized.split("\r") if field.strip()]


def _extract_date(fields: list[str]) -> Optional[str]:
    return next((field for field in fields if _DATE_PATTERN.match(field)), None)


def _extract_time(fields: list[str]) -> Optional[str]:
    return next((field for field in fields if _TIME_PATTERN.match(field)), None)


def _extract_magnification(fields: list[str]) -> Optional[int]:
    date_index = _index_of(fields, _DATE_PATTERN)
    if date_index is None:
        return None
    magnification_index = date_index + _FIELDS_FROM_DATE_TO_MAGNIFICATION
    if magnification_index >= len(fields):
        return None
    candidate = fields[magnification_index]
    return int(candidate) if candidate.isdigit() else None


def _index_of(fields: list[str], pattern: re.Pattern) -> Optional[int]:
    for index, field in enumerate(fields):
        if pattern.match(field):
            return index
    return None


def _extract_pixel_cal(description: str) -> Optional[float]:
    match = _PIXEL_CAL_PATTERN.search(description)
    return float(match.group(1)) if match else None


def _extract_unit(description: str) -> Optional[str]:
    match = _UNIT_PATTERN.search(description)
    return match.group(1) if match else None
