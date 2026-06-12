"""Validate StateStore: persistence, resumability, review queue, cache."""

import tempfile
from pathlib import Path

from src.models import CachedCalibration, CaseResult
from src.state import StateStore


def make_result(case_id: str) -> CaseResult:
    return CaseResult(
        case_id=case_id,
        subfolder_path="frida/17E00231-1",
        status="success",
        nm_per_pixel=4.3033,
        calibration_frame="17E00231-1_115.tif",
        calibration_source="same_folder",
        calibration_date="2026-02-17",
        tissue_date="2026-02-17",
        date_delta_days=0,
        magnification=15000,
        pixels_per_space=107.59,
        fft_confidence=1.0,
        detector_confidence=0.99,
        agent_notes="Used frame 115; clean square grid.",
    )


def main() -> None:
    db_path = str(Path(tempfile.gettempdir()) / "calib_state_test.db")
    Path(db_path).unlink(missing_ok=True)

    store = StateStore(db_path)
    store.register_pending("frida:abc", "frida/17E00231-1", "frida")
    store.register_pending("frida:def", "frida/17E00231-2", "frida")

    assert store.is_processed("frida:abc") is False
    assert set(store.pending_case_ids()) == {"frida:abc", "frida:def"}

    store.save_result(make_result("frida:abc"))
    assert store.is_processed("frida:abc") is True
    assert store.pending_case_ids() == ["frida:def"]

    store.flag_for_review("frida:def", "no_calibration", "No grid in folder or nearby.", "high")
    assert store.is_processed("frida:def") is True
    assert len(store.review_items()) == 1

    store.cache_calibration(
        CachedCalibration(
            magnification=15000,
            calibration_date="2026-02-17",
            account_name="frida",
            folder_id="fid",
            frame_index=114,
            nm_per_pixel=4.3033,
            fft_confidence=1.0,
        )
    )
    cached = store.find_cached_calibration(15000, "2026-02-17")
    assert cached is not None and abs(cached.nm_per_pixel - 4.3033) < 1e-6

    store.close()

    reopened = StateStore(db_path)
    assert reopened.is_processed("frida:abc") is True
    assert reopened.pending_case_ids() == []
    reopened.close()

    print("StateStore: all assertions passed")
    print("  - pending registration and processed-skip works")
    print("  - save_result persists and removes from pending")
    print("  - flag_for_review populates review queue")
    print("  - calibration cache stores and retrieves")
    print("  - state survives close/reopen (resumable)")


if __name__ == "__main__":
    main()
