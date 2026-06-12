"""SQLite-backed run state: results, review queue, and calibration cache.

State persistence makes the run resumable. A folder is processed at most once;
if the agent is interrupted, already-saved cases are skipped on restart.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

from .models import CachedCalibration, CaseResult

_SCHEMA_PATH = Path(__file__).resolve().parent.parent / "schema.sql"


class StateStore:
    def __init__(self, db_path: str):
        self._connection = sqlite3.connect(db_path)
        self._connection.row_factory = sqlite3.Row
        self._initialize_schema()

    def _initialize_schema(self) -> None:
        self._connection.executescript(_SCHEMA_PATH.read_text())
        self._connection.commit()

    def register_pending(self, case_id: str, subfolder_path: str, account_name: str) -> None:
        self._connection.execute(
            """
            INSERT OR IGNORE INTO cases (case_id, subfolder_path, account_name, status)
            VALUES (?, ?, ?, 'pending')
            """,
            (case_id, subfolder_path, account_name),
        )
        self._connection.commit()

    def is_processed(self, case_id: str) -> bool:
        row = self._connection.execute(
            "SELECT status FROM cases WHERE case_id = ?", (case_id,)
        ).fetchone()
        return row is not None and row["status"] in ("success", "flagged")

    def pending_case_ids(self) -> list[str]:
        rows = self._connection.execute(
            "SELECT case_id FROM cases WHERE status IN ('pending', 'error')"
        ).fetchall()
        return [row["case_id"] for row in rows]

    def save_result(self, result: CaseResult) -> None:
        self._connection.execute(
            """
            UPDATE cases SET
                status = ?, nm_per_pixel = ?, calibration_frame = ?,
                calibration_source = ?, pixels_per_space = ?, fft_confidence = ?,
                detector_confidence = ?, magnification = ?, calibration_date = ?,
                tissue_date = ?, date_delta_days = ?, agent_notes = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE case_id = ?
            """,
            (
                result.status,
                result.nm_per_pixel,
                result.calibration_frame,
                result.calibration_source,
                result.pixels_per_space,
                result.fft_confidence,
                result.detector_confidence,
                result.magnification,
                result.calibration_date,
                result.tissue_date,
                result.date_delta_days,
                result.agent_notes,
                result.case_id,
            ),
        )
        self._connection.commit()

    def flag_for_review(
        self, case_id: str, reason: str, explanation: str, priority: str
    ) -> None:
        self._connection.execute(
            """
            INSERT INTO review_queue (case_id, reason, agent_explanation, priority)
            VALUES (?, ?, ?, ?)
            """,
            (case_id, reason, explanation, priority),
        )
        self._connection.execute(
            "UPDATE cases SET status = 'flagged', updated_at = CURRENT_TIMESTAMP WHERE case_id = ?",
            (case_id,),
        )
        self._connection.commit()

    def cache_calibration(self, entry: CachedCalibration) -> None:
        self._connection.execute(
            """
            INSERT OR REPLACE INTO calibration_cache
                (magnification, calibration_date, account_name, folder_id,
                 frame_index, nm_per_pixel, fft_confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry.magnification,
                entry.calibration_date,
                entry.account_name,
                entry.folder_id,
                entry.frame_index,
                entry.nm_per_pixel,
                entry.fft_confidence,
            ),
        )
        self._connection.commit()

    def find_cached_calibration(
        self, magnification: int, calibration_date: str
    ) -> Optional[CachedCalibration]:
        row = self._connection.execute(
            """
            SELECT * FROM calibration_cache
            WHERE magnification = ? AND calibration_date = ?
            ORDER BY fft_confidence DESC LIMIT 1
            """,
            (magnification, calibration_date),
        ).fetchone()
        return _to_cached_calibration(row) if row else None

    def all_results(self) -> list[sqlite3.Row]:
        return self._connection.execute(
            "SELECT * FROM cases ORDER BY subfolder_path"
        ).fetchall()

    def review_items(self) -> list[sqlite3.Row]:
        return self._connection.execute(
            "SELECT * FROM review_queue WHERE resolved = 0 ORDER BY priority, id"
        ).fetchall()

    def close(self) -> None:
        self._connection.close()


def _to_cached_calibration(row: sqlite3.Row) -> CachedCalibration:
    return CachedCalibration(
        magnification=row["magnification"],
        calibration_date=row["calibration_date"],
        account_name=row["account_name"],
        folder_id=row["folder_id"],
        frame_index=row["frame_index"],
        nm_per_pixel=row["nm_per_pixel"],
        fft_confidence=row["fft_confidence"],
    )
