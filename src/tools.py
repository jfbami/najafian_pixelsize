"""Tool definitions and dispatcher exposed to the orchestrator agent.

Each public method on ToolBox corresponds to a tool the model can call. The
ToolBox holds the live dependencies (Drive client, state store, config) and a
small per-case cache so the model can detect frames before deciding which full
TIFF to download.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import calibration
from . import crosscheck
from . import matcher
from . import metadata as metadata_module
from .detect import Thresholds, detect_calibration
from .drive_client import DriveClient
from .measurer import measure_grid
from .models import CalibrationCandidate, CaseResult, DriveFolder, TiffMetadata
from .state import StateStore

TOOL_DEFINITIONS: list[dict] = [
    {
        "name": "list_specimen_folders",
        "description": "List all acquisition subfolders containing TIFFs for a Drive account.",
        "input_schema": {
            "type": "object",
            "properties": {"account_name": {"type": "string"}},
            "required": ["account_name"],
        },
    },
    {
        "name": "download_thumbnails",
        "description": "Download thumbnails for every frame in a folder for fast detection.",
        "input_schema": {
            "type": "object",
            "properties": {"folder_id": {"type": "string"}, "case_id": {"type": "string"}},
            "required": ["folder_id", "case_id"],
        },
    },
    {
        "name": "detect_calibration_frames",
        "description": "Score each frame's thumbnail as calibration grid vs tissue.",
        "input_schema": {
            "type": "object",
            "properties": {"case_id": {"type": "string"}},
            "required": ["case_id"],
        },
    },
    {
        "name": "download_full_tiff",
        "description": "Download the full-resolution TIFF for one frame.",
        "input_schema": {
            "type": "object",
            "properties": {"frame_id": {"type": "string"}, "case_id": {"type": "string"}},
            "required": ["frame_id", "case_id"],
        },
    },
    {
        "name": "measure_grid",
        "description": "FFT sub-pixel measurement of a calibration TIFF; returns nm/pixel.",
        "input_schema": {
            "type": "object",
            "properties": {"tiff_path": {"type": "string"}},
            "required": ["tiff_path"],
        },
    },
    {
        "name": "parse_tiff_metadata",
        "description": "Parse magnification and acquisition date from a TIFF.",
        "input_schema": {
            "type": "object",
            "properties": {"tiff_path": {"type": "string"}},
            "required": ["tiff_path"],
        },
    },
    {
        "name": "cross_check_measurement",
        "description": (
            "Independently check a measured nm/pixel against the microscope's own "
            "embedded scale and against the magnification law from already-accepted "
            "frames. Catches integer-factor (harmonic) errors. Always run this "
            "before save_result."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "tiff_path": {"type": "string"},
                "nm_per_pixel": {"type": "number"},
            },
            "required": ["tiff_path", "nm_per_pixel"],
        },
    },
    {
        "name": "choose_calibration_frame",
        "description": (
            "Apply the lab SOP to pick which calibration candidate pairs with the "
            "tissue: same magnification required, nearest acquisition date wins, "
            "and the date windows decide auto-use vs review."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "candidates": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "frame_id": {"type": "string"},
                            "magnification": {"type": "integer"},
                            "acquisition_date": {"type": "string"},
                        },
                        "required": ["frame_id"],
                    },
                },
                "tissue_magnification": {"type": "integer"},
                "tissue_date": {"type": "string"},
            },
            "required": ["candidates", "tissue_magnification"],
        },
    },
    {
        "name": "search_calibration_cross_account",
        "description": "Find cached calibration at a magnification near a date.",
        "input_schema": {
            "type": "object",
            "properties": {
                "magnification": {"type": "integer"},
                "target_date": {"type": "string"},
            },
            "required": ["magnification", "target_date"],
        },
    },
    {
        "name": "save_result",
        "description": "Persist a completed measurement to the results store.",
        "input_schema": {
            "type": "object",
            "properties": {
                "case_id": {"type": "string"},
                "nm_per_pixel": {"type": "number"},
                "calibration_frame": {"type": "string"},
                "calibration_source": {"type": "string"},
                "calibration_date": {"type": "string"},
                "tissue_date": {"type": "string"},
                "date_delta_days": {"type": "integer"},
                "magnification": {"type": "integer"},
                "pixels_per_space": {"type": "number"},
                "fft_confidence": {"type": "number"},
                "detector_confidence": {"type": "number"},
                "agent_notes": {"type": "string"},
            },
            "required": ["case_id", "nm_per_pixel", "agent_notes"],
        },
    },
    {
        "name": "flag_for_review",
        "description": "Flag a case for human review with an explanation.",
        "input_schema": {
            "type": "object",
            "properties": {
                "case_id": {"type": "string"},
                "reason": {"type": "string"},
                "agent_explanation": {"type": "string"},
                "priority": {"type": "string"},
            },
            "required": ["case_id", "reason", "agent_explanation", "priority"],
        },
    },
    {
        "name": "cleanup_temp_files",
        "description": "Delete temp files for a finished case.",
        "input_schema": {
            "type": "object",
            "properties": {"case_id": {"type": "string"}},
            "required": ["case_id"],
        },
    },
]


_DISPATCHABLE = frozenset(definition["name"] for definition in TOOL_DEFINITIONS)


class ToolBox:
    def __init__(self, drive: DriveClient, store: StateStore, config: dict, temp_dir: Path):
        self._drive = drive
        self._store = store
        self._config = config
        self._temp_dir = temp_dir
        self._thresholds = Thresholds.from_config(config)
        self._folders_by_id: dict[str, Any] = {}
        self._thumbnails_by_case: dict[str, list[dict]] = {}

    def dispatch(self, name: str, arguments: dict) -> dict:
        """Route a model tool call.

        Only names declared in TOOL_DEFINITIONS are reachable -- a bare getattr
        would also expose private helpers such as `_root_folder_id`, which
        returns a Drive folder id the model was never given. Failures are
        returned as tool results rather than raised, so one malformed call
        cannot abort an unattended multi-hour run.
        """
        if name not in _DISPATCHABLE:
            return {"error": f"unknown tool: {name}"}
        if not isinstance(arguments, dict):
            return {"error": f"tool {name} expects an object of arguments"}
        try:
            return getattr(self, name)(**arguments)
        except TypeError as error:
            return {"error": f"bad arguments for {name}: {error}"}
        except Exception as error:  # surfaced to the model, which can flag the case
            return {"error": f"{name} failed: {type(error).__name__}: {error}"}

    def list_specimen_folders(self, account_name: str) -> dict:
        root_id = self._root_folder_id(account_name)
        folders = self._drive.list_specimen_folders(account_name, root_id)
        for folder in folders:
            self._folders_by_id[folder.folder_id] = folder
        return {
            "folders": [
                {"folder_id": f.folder_id, "path": f.subfolder_path, "frames": f.frame_count}
                for f in folders
            ]
        }

    def download_thumbnails(self, folder_id: str, case_id: str) -> dict:
        folder = self._folders_by_id.get(folder_id)
        if folder is None:
            return {"error": f"folder {folder_id} not listed; call list_specimen_folders first"}
        case_dir = self._temp_dir / case_id / "thumbs"
        thumbnails = []
        for index, frame_id in enumerate(folder.frame_file_ids):
            destination = case_dir / f"{index:03d}.jpg"
            self._drive.download_thumbnail(frame_id, destination)
            thumbnails.append({"frame_index": index, "frame_id": frame_id, "path": str(destination)})
        self._thumbnails_by_case[case_id] = thumbnails
        return {"count": len(thumbnails)}

    def detect_calibration_frames(self, case_id: str) -> dict:
        thumbnails = self._thumbnails_by_case.get(case_id)
        if thumbnails is None:
            return {"error": f"no thumbnails for {case_id}; call download_thumbnails first"}

        detections = []
        for thumbnail in thumbnails:
            detection = detect_calibration(thumbnail["path"], self._thresholds)
            detections.append(
                {
                    "frame_index": thumbnail["frame_index"],
                    "frame_id": thumbnail["frame_id"],
                    "is_calibration": detection.is_calibration,
                    "score": round(detection.score, 4),
                    "confidence": round(detection.measurement.fft_confidence, 4),
                    "concentration": round(
                        detection.measurement.spectral_concentration, 4
                    ),
                }
            )
        return {
            "detections": detections,
            "note": (
                "Thumbnail scores identify which frame is the grid. They are not "
                "measurements: nm/pixel must come from measure_grid on the "
                "full-resolution TIFF."
            ),
        }

    def download_full_tiff(self, frame_id: str, case_id: str) -> dict:
        destination = self._temp_dir / case_id / "full" / f"{frame_id}.tif"
        self._drive.download_file(frame_id, destination)
        return {"tiff_path": str(destination)}

    def measure_grid(self, tiff_path: str) -> dict:
        result = measure_grid(tiff_path)
        if not result.valid:
            return {
                "valid": False,
                "nm_per_pixel": None,
                "reason": result.warning or "frame could not be measured",
                "fft_confidence": round(result.fft_confidence, 4),
                "axis_count": result.axis_count,
            }
        return {
            "valid": True,
            "nm_per_pixel": round(result.nm_per_pixel, 4),
            "nm_per_pixel_uncertainty": round(result.nm_per_pixel_uncertainty, 4),
            "relative_precision_percent": round(result.relative_precision * 100, 4),
            "pixels_per_space": round(result.pixels_per_space, 4),
            "fft_confidence": round(result.fft_confidence, 4),
            "grid_uniformity": round(result.grid_uniformity, 4),
            "axis_separation_deg": round(result.axis_separation_deg, 2),
            "fundamental_inferred": result.fundamental_inferred,
            "warning": result.warning,
        }

    def parse_tiff_metadata(self, tiff_path: str) -> dict:
        metadata = metadata_module.parse_tiff_metadata(tiff_path)
        return {
            "magnification": metadata.magnification,
            "acquisition_date": metadata.acquisition_date,
            "acquisition_time": metadata.acquisition_time,
            "image_width": metadata.image_width,
            "image_height": metadata.image_height,
            "embedded_pixel_cal": metadata.pixel_cal_embedded,
            "embedded_unit": metadata.unit_embedded,
            "embedded_note": (
                "advisory only -- the unit encoding is unreliable; use "
                "cross_check_measurement rather than reading this as a value"
            ),
        }

    def cross_check_measurement(self, tiff_path: str, nm_per_pixel: float) -> dict:
        metadata = metadata_module.parse_tiff_metadata(tiff_path)
        checks = [
            crosscheck.check_embedded_scale(
                nm_per_pixel,
                metadata.pixel_cal_embedded,
                metadata.unit_embedded,
                trusted=bool(
                    self._config.get("crosscheck", {}).get("trust_embedded_scale", False)
                ),
            ),
            crosscheck.check_magnification_law(
                nm_per_pixel, metadata.magnification, self._store.reference_frames()
            ),
        ]
        passed, summary = crosscheck.summarize(checks)
        factors = [c.suspected_factor for c in checks if c.suspected_factor]
        return {
            "passed": passed,
            "summary": summary,
            "suspected_harmonic_factor": factors[0] if factors else None,
            "checks": [
                {"name": c.name, "passed": c.passed, "detail": c.detail} for c in checks
            ],
        }

    def choose_calibration_frame(
        self,
        candidates: list[dict],
        tissue_magnification: int,
        tissue_date: str | None = None,
    ) -> dict:
        matching = self._config.get("matching", {})
        decision = matcher.choose_calibration(
            candidates=[_to_candidate(entry) for entry in candidates],
            tissue=TiffMetadata(
                magnification=tissue_magnification,
                acquisition_date=tissue_date,
                acquisition_time=None,
                pixel_cal_embedded=None,
                unit_embedded=None,
                instrument_id=None,
                image_width=None,
                image_height=None,
            ),
            auto_use_within_days=int(matching.get("auto_use_within_days", 7)),
            max_date_window_days=int(matching.get("max_date_window_days", 30)),
        )
        return {
            "frame_id": decision.candidate.frame_id if decision.candidate else None,
            "date_delta_days": decision.date_delta_days,
            "auto_usable": decision.auto_usable,
            "reason": decision.reason,
        }

    def search_calibration_cross_account(self, magnification: int, target_date: str) -> dict:
        cached = self._store.find_cached_calibration(magnification, target_date)
        if cached is None:
            return {"found": False}
        return {
            "found": True,
            "nm_per_pixel": cached.nm_per_pixel,
            "account": cached.account_name,
            "frame_index": cached.frame_index,
        }

    def save_result(self, **fields: Any) -> dict:
        value = fields.get("nm_per_pixel")
        if value is None or not calibration.is_plausible(float(value)):
            return {
                "saved": False,
                "error": (
                    f"refusing to save nm_per_pixel={value!r}: outside the plausible "
                    f"range {calibration.PLAUSIBLE_NM_PER_PIXEL}. Flag the case instead."
                ),
            }
        self._store.save_result(
            CaseResult(
                case_id=fields["case_id"],
                subfolder_path=fields.get("subfolder_path", ""),
                status="success",
                nm_per_pixel=fields["nm_per_pixel"],
                calibration_frame=fields.get("calibration_frame"),
                calibration_source=fields.get("calibration_source", "same_folder"),
                calibration_date=fields.get("calibration_date"),
                tissue_date=fields.get("tissue_date"),
                date_delta_days=fields.get("date_delta_days"),
                magnification=fields.get("magnification"),
                pixels_per_space=fields.get("pixels_per_space"),
                fft_confidence=fields.get("fft_confidence"),
                detector_confidence=fields.get("detector_confidence"),
                agent_notes=fields["agent_notes"],
            )
        )
        return {"saved": True}

    def flag_for_review(
        self, case_id: str, reason: str, agent_explanation: str, priority: str
    ) -> dict:
        self._store.flag_for_review(case_id, reason, agent_explanation, priority)
        return {"flagged": True}

    def cleanup_temp_files(self, case_id: str) -> dict:
        import shutil

        case_dir = self._temp_dir / case_id
        if case_dir.exists():
            shutil.rmtree(case_dir)
        self._thumbnails_by_case.pop(case_id, None)
        return {"cleaned": True}

    def _root_folder_id(self, account_name: str) -> str:
        for account in self._config["accounts"]:
            if account["name"] == account_name:
                return account["root_folder_id"]
        raise KeyError(f"unknown account: {account_name}")


def _to_candidate(entry: dict) -> CalibrationCandidate:
    """Build the matcher's candidate type from the model's compact description."""
    return CalibrationCandidate(
        folder=DriveFolder("", "", "", "", "", [], 0),
        frame_index=int(entry.get("frame_index", 0)),
        frame_id=str(entry.get("frame_id", "")),
        filename=str(entry.get("filename", entry.get("frame_id", ""))),
        metadata=TiffMetadata(
            magnification=entry.get("magnification"),
            acquisition_date=entry.get("acquisition_date"),
            acquisition_time=None,
            pixel_cal_embedded=None,
            unit_embedded=None,
            instrument_id=None,
            image_width=None,
            image_height=None,
        ),
        detector_confidence=float(entry.get("detector_confidence", 0.0)),
        fft_peak_ratio=float(entry.get("fft_peak_ratio", 0.0)),
    )
