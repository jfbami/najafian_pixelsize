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

# Columns added after the first schema shipped, applied to existing databases.
_ADDED_COLUMNS = (
    ("cases", "image_width", "INTEGER"),
    ("cases", "image_height", "INTEGER"),
)


class StateStore:
    def __init__(self, db_path: str):
        self._connection = sqlite3.connect(db_path)
        self._connection.row_factory = sqlite3.Row
        self._initialize_schema()

    def _initialize_schema(self) -> None:
        self._connection.executescript(_SCHEMA_PATH.read_text())
        self._add_missing_columns()
        self._connection.commit()

    def _add_missing_columns(self) -> None:
        """Bring an existing database up to the current schema.

        `CREATE TABLE IF NOT EXISTS` leaves an older table untouched, so a
        database written before a column was added would fail every insert with
        `sqlite3.OperationalError: no such column`. Resuming a part-finished run
        is the whole point of this store, so the upgrade has to be automatic.
        """
        for table, column, column_type in _ADDED_COLUMNS:
            if column not in self._columns_of(table):
                self._connection.execute(
                    f"ALTER TABLE {table} ADD COLUMN {column} {column_type}"
                )

    def _columns_of(self, table: str) -> set[str]:
        rows = self._connection.execute(f"PRAGMA table_info({table})").fetchall()
        return {row["name"] for row in rows}

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
        """Persist a finished case.

        The row is created if it is missing: an UPDATE alone silently discards
        the result for any case the run never registered, which loses a
        completed measurement with no error anywhere.
        """
        self._connection.execute(
            """
            INSERT OR IGNORE INTO cases (case_id, subfolder_path, account_name, status)
            VALUES (?, ?, ?, 'pending')
            """,
            (result.case_id, result.subfolder_path, _account_of(result.case_id)),
        )
        cursor = self._connection.execute(
            """
            UPDATE cases SET
                status = ?, nm_per_pixel = ?, calibration_frame = ?,
                calibration_source = ?, pixels_per_space = ?, fft_confidence = ?,
                detector_confidence = ?, magnification = ?,
                image_width = ?, image_height = ?, calibration_date = ?,
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
                result.image_width,
                result.image_height,
                result.calibration_date,
                result.tissue_date,
                result.date_delta_days,
                result.agent_notes,
                result.case_id,
            ),
        )
        if cursor.rowcount != 1:
            raise RuntimeError(f"failed to persist case {result.case_id!r}")
        self._connection.commit()

    def flag_for_review(
        self, case_id: str, reason: str, explanation: str, priority: str
    ) -> None:
        """Queue a case for human review; repeat calls update rather than stack."""
        self._connection.execute(
            """
            INSERT INTO review_queue (case_id, reason, agent_explanation, priority)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(case_id, reason) DO UPDATE SET
                agent_explanation = excluded.agent_explanation,
                priority = excluded.priority,
                resolved = 0
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

    def reference_frames(self) -> list[tuple[int, float]]:
        """(magnification, nm_per_pixel) pairs for the magnification-law check."""
        rows = self._connection.execute(
            """
            SELECT magnification, nm_per_pixel FROM calibration_cache
            WHERE magnification IS NOT NULL AND nm_per_pixel IS NOT NULL
            """
        ).fetchall()
        return [(row["magnification"], row["nm_per_pixel"]) for row in rows]

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


def _account_of(case_id: str) -> str:
    """Case ids are '<account>:<folder_id>'; fall back to the whole id."""
    return case_id.split(":", 1)[0] if ":" in case_id else case_id


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
