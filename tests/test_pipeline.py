"""Tests for the non-imaging pipeline: constants, matching, state, detection."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src import calibration, crosscheck
from src.detect import Thresholds, classify, detect_calibration
from src.matcher import choose_calibration
from src.measurer import measure_array
from src.models import CachedCalibration, CalibrationCandidate, CaseResult, DriveFolder, TiffMetadata
from src.state import StateStore
from tests.synthetic import cross_grating, tissue_like, write_tiff


# --------------------------------------------------------------------------
# calibration constants
# --------------------------------------------------------------------------

def test_pitch_is_derived_from_the_ruling():
    assert calibration.GRID_PITCH_UM == pytest.approx(1000.0 / 2160, rel=1e-12)
    assert calibration.GRID_PITCH_UM == pytest.approx(0.463, abs=5e-5)


@pytest.mark.parametrize("spacing", [0.0, -5.0, float("nan"), float("inf")])
def test_conversion_returns_nan_rather_than_raising(spacing):
    """A bad spacing must propagate as a missing measurement, not kill a batch."""
    assert math.isnan(calibration.nm_per_pixel(spacing))


def test_conversion_round_trips():
    assert calibration.pixels_per_space(calibration.nm_per_pixel(107.59)) == pytest.approx(107.59)


def test_plausibility_bounds():
    assert calibration.is_plausible(4.3)
    assert not calibration.is_plausible(0.0)
    assert not calibration.is_plausible(float("nan"))
    assert not calibration.is_plausible(1e6)


def test_total_uncertainty_never_below_the_standard():
    assert calibration.total_relative_uncertainty(0.0) == pytest.approx(
        calibration.PITCH_RELATIVE_UNCERTAINTY
    )


# --------------------------------------------------------------------------
# detection
# --------------------------------------------------------------------------

def test_grating_detected_and_tissue_rejected(tmp_path):
    grid = write_tiff(cross_grating(107.59, size=1024), tmp_path / "grid.tif")
    tissue = write_tiff(tissue_like(1024), tmp_path / "tissue.tif")

    assert detect_calibration(str(grid)).is_calibration
    assert not detect_calibration(str(tissue)).is_calibration


def test_detection_works_on_thumbnail_sized_frames():
    """Detection must survive downscaling even though measurement cannot: the
    production path detects on Drive thumbnails and measures on the full TIFF."""
    thumbnail = cross_grating(12.8, size=256, blur=0.3)
    result = measure_array(thumbnail)
    detection = classify(result)

    assert detection.is_calibration, "grid thumbnail must still be detected"
    assert not result.valid, "thumbnail must not yield a measurement"


def test_detection_thresholds_come_from_config():
    limits = Thresholds.from_config(
        {"detection": {"confidence_threshold": 0.9, "squareness_threshold": 0.99}}
    )
    assert limits.confidence == 0.9
    assert limits.squareness == 0.99
    assert limits.orthogonality_deg == Thresholds().orthogonality_deg


def test_tissue_scores_below_grating(tmp_path):
    grid = write_tiff(cross_grating(107.59, size=1024), tmp_path / "g.tif")
    tissue = write_tiff(tissue_like(1024), tmp_path / "t.tif")
    assert detect_calibration(str(grid)).score > detect_calibration(str(tissue)).score


# --------------------------------------------------------------------------
# matching
# --------------------------------------------------------------------------

def _metadata(date, magnification=15000, width=2512, height=2496):
    return TiffMetadata(
        magnification, date, None, None, None, None, width, height
    )


def _candidate(date, name, magnification=15000, width=2512, height=2496):
    return CalibrationCandidate(
        folder=DriveFolder("a", "f", "n", "s", "p", [], 0),
        frame_index=0,
        frame_id=name,
        filename=name,
        metadata=_metadata(date, magnification, width, height),
        detector_confidence=1.0,
        fft_peak_ratio=1.0,
    )


def test_calibration_at_a_different_resolution_is_refused():
    """The lab saves at several resolutions, and nm/pixel is a property of the
    pixel grid. Pairing a 2512px calibration with a 1024px tissue frame at the
    same magnification would apply a scale wrong by the resolution ratio,
    silently and with every cross-check passing."""
    decision = choose_calibration(
        [_candidate("2026-02-17", "highres", width=2512, height=2496)],
        _metadata("2026-02-17", width=1024, height=1194),
        auto_use_within_days=7,
        max_date_window_days=30,
    )
    assert not decision.auto_usable
    assert decision.candidate is None
    # The reason must name both formats, so a reviewer can see why at a glance.
    assert "2512x2496" in decision.reason
    assert "1024x1194" in decision.reason


def test_calibration_at_the_same_resolution_still_matches():
    decision = choose_calibration(
        [_candidate("2026-02-17", "same", width=2512, height=2496)],
        _metadata("2026-02-17", width=2512, height=2496),
        auto_use_within_days=7,
        max_date_window_days=30,
    )
    assert decision.auto_usable
    assert decision.candidate.frame_id == "same"


def test_a_differing_info_bar_height_does_not_block_a_match():
    """Bar height varies between acquisitions on the same camera, so a small
    difference must not be read as a different sensor format."""
    decision = choose_calibration(
        [_candidate("2026-02-17", "shorter-bar", width=2512, height=2460)],
        _metadata("2026-02-17", width=2512, height=2496),
        auto_use_within_days=7,
        max_date_window_days=30,
    )
    assert decision.auto_usable


def test_missing_dates_do_not_crash_the_match():
    """Mixed known/unknown dates used to raise TypeError inside min()."""
    decision = choose_calibration(
        [_candidate(None, "undated"), _candidate("2026-02-17", "dated")],
        _metadata("2026-02-18"),
        auto_use_within_days=7,
        max_date_window_days=30,
    )
    assert decision.candidate.frame_id == "dated"
    assert decision.auto_usable


def test_all_dates_missing_is_flagged_not_used():
    decision = choose_calibration(
        [_candidate(None, "a"), _candidate(None, "b")],
        _metadata("2026-02-18"),
        auto_use_within_days=7,
        max_date_window_days=30,
    )
    assert not decision.auto_usable


def test_unknown_tissue_magnification_is_not_matched():
    decision = choose_calibration(
        [_candidate("2026-02-17", "a")],
        _metadata("2026-02-18", magnification=None),
        auto_use_within_days=7,
        max_date_window_days=30,
    )
    assert decision.candidate is None
    assert not decision.auto_usable


@pytest.mark.parametrize(
    "date,auto_usable",
    [("2026-02-18", True), ("2026-02-27", False), ("2026-06-01", False)],
)
def test_date_windows_control_auto_use(date, auto_usable):
    decision = choose_calibration(
        [_candidate(date, "a")], _metadata("2026-02-17"),
        auto_use_within_days=7, max_date_window_days=30,
    )
    assert decision.auto_usable is auto_usable


def test_magnification_must_match_exactly():
    decision = choose_calibration(
        [_candidate("2026-02-17", "a", magnification=8000)],
        _metadata("2026-02-17", magnification=15000),
        auto_use_within_days=7, max_date_window_days=30,
    )
    assert decision.candidate is None


# --------------------------------------------------------------------------
# cross-checks
# --------------------------------------------------------------------------

@pytest.mark.parametrize("factor", [2, 3, 4])
def test_embedded_scale_catches_integer_factor_errors(factor):
    check = crosscheck.check_embedded_scale(4.30 * factor, 4.30, "nm", trusted=True)
    assert not check.passed
    assert check.suspected_factor == factor


def test_embedded_scale_tolerates_unit_encoding():
    """The unit field is unreliable, so a pure power-of-ten offset must pass."""
    assert crosscheck.check_embedded_scale(4.30, 0.00430, "um", trusted=True).passed
    assert crosscheck.check_embedded_scale(4.30, 4300.0, "pm", trusted=True).passed


def test_embedded_scale_absent_is_not_a_failure():
    assert crosscheck.check_embedded_scale(4.30, None, None, trusted=True).passed


def test_embedded_scale_is_off_by_default():
    """This instrument writes a constant XpixCal=150.000000 at every
    magnification. Trusting it flags every correct frame as a 3x harmonic
    error, so the check must stay off unless explicitly enabled."""
    check = crosscheck.check_embedded_scale(4.2339, 150.0, "um")
    assert check.passed
    assert check.suspected_factor is None

    enabled = crosscheck.check_embedded_scale(4.2339, 150.0, "um", trusted=True)
    assert not enabled.passed, "the false positive this default protects against"


def test_magnification_law_catches_factor_errors():
    reference = [(15000, 4.30), (20000, 3.225), (10000, 6.45)]
    assert crosscheck.check_magnification_law(4.30, 15000, reference).passed
    bad = crosscheck.check_magnification_law(4.30 * 3, 15000, reference)
    assert not bad.passed
    assert bad.suspected_factor == 3


def test_magnification_law_needs_references():
    assert crosscheck.check_magnification_law(4.30, 15000, []).passed


def test_summarize_reports_failures():
    failing = crosscheck.check_embedded_scale(8.6, 4.3, "nm", trusted=True)
    passed, _ = crosscheck.summarize([failing])
    assert not passed


# --------------------------------------------------------------------------
# state store
# --------------------------------------------------------------------------

def _case(case_id="acct:folder"):
    return CaseResult(
        case_id=case_id, subfolder_path="frida/17E00231-1", status="success",
        nm_per_pixel=4.3033, calibration_frame="115.tif", calibration_source="same_folder",
        calibration_date="2026-02-17", tissue_date="2026-02-17", date_delta_days=0,
        magnification=15000, pixels_per_space=107.59, fft_confidence=1.0,
        detector_confidence=0.99, agent_notes="clean square grid",
    )


def test_result_for_an_unregistered_case_is_still_persisted(tmp_path):
    """An UPDATE-only save silently discarded results for unregistered cases."""
    store = StateStore(str(tmp_path / "s.db"))
    store.save_result(_case("never:registered"))
    assert len(store.all_results()) == 1
    assert store.is_processed("never:registered")
    store.close()


def test_reflagging_updates_rather_than_duplicates(tmp_path):
    store = StateStore(str(tmp_path / "s.db"))
    store.register_pending("a:b", "path", "a")
    store.flag_for_review("a:b", "no_calibration", "first pass", "medium")
    store.flag_for_review("a:b", "no_calibration", "second pass", "high")
    items = store.review_items()
    assert len(items) == 1
    assert items[0]["agent_explanation"] == "second pass"
    assert items[0]["priority"] == "high"
    store.close()


def test_state_survives_reopen(tmp_path):
    path = str(tmp_path / "s.db")
    store = StateStore(path)
    store.register_pending("a:b", "p", "a")
    store.save_result(_case("a:b"))
    store.close()

    reopened = StateStore(path)
    assert reopened.is_processed("a:b")
    assert reopened.pending_case_ids() == []
    reopened.close()


def test_reference_frames_feed_the_magnification_check(tmp_path):
    store = StateStore(str(tmp_path / "s.db"))
    store.cache_calibration(
        CachedCalibration(15000, "2026-02-17", "frida", "fid", 114, 4.3033, 1.0)
    )
    assert store.reference_frames() == [(15000, 4.3033)]
    store.close()


def test_resolution_check_survives_the_tool_layer():
    """The matcher's resolution rule is only real if the tool passes the
    dimensions through. An earlier version hardcoded image_width=None here,
    which left the check in place but permanently disabled."""
    from src.tools import ToolBox

    toolbox = ToolBox.__new__(ToolBox)
    toolbox._config = {"matching": {"auto_use_within_days": 7, "max_date_window_days": 30}}

    mismatched = toolbox.choose_calibration_frame(
        candidates=[
            {
                "frame_id": "highres",
                "magnification": 15000,
                "acquisition_date": "2026-02-17",
                "image_width": 2512,
                "image_height": 2496,
            }
        ],
        tissue_magnification=15000,
        tissue_date="2026-02-17",
        tissue_width=1024,
        tissue_height=1194,
    )
    assert not mismatched["auto_usable"]
    assert "2512x2496" in mismatched["reason"]

    matched = toolbox.choose_calibration_frame(
        candidates=[
            {
                "frame_id": "sameres",
                "magnification": 15000,
                "acquisition_date": "2026-02-17",
                "image_width": 1024,
                "image_height": 1194,
            }
        ],
        tissue_magnification=15000,
        tissue_date="2026-02-17",
        tissue_width=1024,
        tissue_height=1194,
    )
    assert matched["auto_usable"]
    assert matched["frame_id"] == "sameres"


def test_an_older_database_gains_new_columns(tmp_path):
    """Resuming a run written before image_width existed must not fail."""
    import sqlite3

    db_path = tmp_path / "old.db"
    legacy = sqlite3.connect(db_path)
    # The exact `cases` table the previous version wrote: everything except the
    # image_width and image_height columns added for resolution matching.
    legacy.execute(
        """
        CREATE TABLE cases (
            case_id TEXT PRIMARY KEY,
            subfolder_path TEXT NOT NULL,
            account_name TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            nm_per_pixel REAL,
            calibration_frame TEXT,
            calibration_source TEXT,
            pixels_per_space REAL,
            fft_confidence REAL,
            detector_confidence REAL,
            magnification INTEGER,
            calibration_date TEXT,
            tissue_date TEXT,
            date_delta_days INTEGER,
            agent_notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    legacy.execute(
        "INSERT INTO cases (case_id, subfolder_path, account_name) VALUES "
        "('frida:old', 'p', 'frida')"
    )
    legacy.commit()
    legacy.close()

    store = StateStore(str(db_path))
    store.save_result(
        CaseResult(
            case_id="frida:old", subfolder_path="p", status="success",
            nm_per_pixel=4.23, calibration_frame="f.tif", calibration_source="same_folder",
            calibration_date="2026-02-17", tissue_date="2026-02-17", date_delta_days=0,
            magnification=15000, pixels_per_space=109.3, fft_confidence=1.0,
            detector_confidence=1.0, agent_notes="n",
            image_width=2512, image_height=2496,
        )
    )
    row = store.all_results()[0]
    assert row["image_width"] == 2512
    assert row["image_height"] == 2496
    store.close()
