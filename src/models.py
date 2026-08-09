"""Data models shared across the calibration pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class DriveFolder:
    account_name: str
    folder_id: str
    folder_name: str
    specimen_id: str
    subfolder_path: str
    frame_file_ids: list[str]
    frame_count: int


@dataclass
class ThumbnailResult:
    frame_index: int
    frame_id: str
    filename: str
    local_path: str


@dataclass
class DetectionResult:
    frame_index: int
    frame_id: str
    filename: str
    classifier_confidence: float
    fft_peak_ratio: float
    is_calibration: bool


@dataclass
class TiffMetadata:
    magnification: Optional[int]
    acquisition_date: Optional[str]
    acquisition_time: Optional[str]
    pixel_cal_embedded: Optional[float]
    unit_embedded: Optional[str]
    instrument_id: Optional[str]
    image_width: Optional[int]
    image_height: Optional[int]


@dataclass
class MeasurementResult:
    """One frame's grid measurement.

    `valid` is the gate: when it is False, `pixels_per_space` and `nm_per_pixel`
    are NaN and the frame has no usable measurement. `nm_per_pixel_uncertainty`
    combines the comb-fit precision with the grating standard's own tolerance,
    so it is the figure to quote alongside a result.
    """

    pixels_per_space: float
    nm_per_pixel: float
    spacing_x: float
    spacing_y: float
    fft_confidence: float
    grid_uniformity: float
    axis_separation_deg: float = 0.0
    axis_count: int = 0
    spectral_concentration: float = 0.0
    relative_precision: float = float("nan")
    nm_per_pixel_uncertainty: float = float("nan")
    fundamental_inferred: bool = False
    valid: bool = False
    warning: Optional[str] = None


@dataclass
class CalibrationCandidate:
    folder: DriveFolder
    frame_index: int
    frame_id: str
    filename: str
    metadata: TiffMetadata
    detector_confidence: float
    fft_peak_ratio: float


@dataclass
class CachedCalibration:
    magnification: int
    calibration_date: str
    account_name: str
    folder_id: str
    frame_index: int
    nm_per_pixel: float
    fft_confidence: float


@dataclass
class CaseResult:
    case_id: str
    subfolder_path: str
    status: str
    nm_per_pixel: Optional[float]
    calibration_frame: Optional[str]
    calibration_source: str
    calibration_date: Optional[str]
    tissue_date: Optional[str]
    date_delta_days: Optional[int]
    magnification: Optional[int]
    pixels_per_space: Optional[float]
    fft_confidence: Optional[float]
    detector_confidence: Optional[float]
    agent_notes: str
    review_reason: Optional[str] = None
