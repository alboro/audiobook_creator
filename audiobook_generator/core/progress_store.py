from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import datetime, UTC
from pathlib import Path


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


class NormalizationProgressStore:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self):
        connection = sqlite3.connect(self.db_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        return connection

    def _initialize(self):
        with closing(self._connect()) as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS normalization_steps (
                    chapter_key TEXT NOT NULL,
                    step_index INTEGER NOT NULL,
                    step_name TEXT NOT NULL,
                    input_hash TEXT NOT NULL,
                    config_hash TEXT NOT NULL,
                    status TEXT NOT NULL,
                    output_path TEXT,
                    error_message TEXT,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (chapter_key, step_index, input_hash, config_hash)
                );

                CREATE TABLE IF NOT EXISTS normalization_units (
                    chapter_key TEXT NOT NULL,
                    step_index INTEGER NOT NULL,
                    unit_index INTEGER NOT NULL,
                    step_name TEXT NOT NULL,
                    input_hash TEXT NOT NULL,
                    config_hash TEXT NOT NULL,
                    status TEXT NOT NULL,
                    output_path TEXT,
                    error_message TEXT,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (chapter_key, step_index, unit_index, input_hash, config_hash)
                );
                """
            )
            connection.commit()

    def get_step_record(self, *, chapter_key, step_index, input_hash, config_hash):
        with closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT *
                FROM normalization_steps
                WHERE chapter_key = ?
                  AND step_index = ?
                  AND input_hash = ?
                  AND config_hash = ?
                """,
                (chapter_key, step_index, input_hash, config_hash),
            ).fetchone()
        return row

    def get_unit_record(self, *, chapter_key, step_index, unit_index, input_hash, config_hash):
        with closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT *
                FROM normalization_units
                WHERE chapter_key = ?
                  AND step_index = ?
                  AND unit_index = ?
                  AND input_hash = ?
                  AND config_hash = ?
                """,
                (chapter_key, step_index, unit_index, input_hash, config_hash),
            ).fetchone()
        return row

    def upsert_step(
        self,
        *,
        chapter_key,
        step_index,
        step_name,
        input_hash,
        config_hash,
        status,
        output_path=None,
        error_message=None,
    ):
        with closing(self._connect()) as connection:
            connection.execute(
                """
                INSERT INTO normalization_steps (
                    chapter_key,
                    step_index,
                    step_name,
                    input_hash,
                    config_hash,
                    status,
                    output_path,
                    error_message,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(chapter_key, step_index, input_hash, config_hash)
                DO UPDATE SET
                    step_name = excluded.step_name,
                    status = excluded.status,
                    output_path = excluded.output_path,
                    error_message = excluded.error_message,
                    updated_at = excluded.updated_at
                """,
                (
                    chapter_key,
                    step_index,
                    step_name,
                    input_hash,
                    config_hash,
                    status,
                    output_path,
                    error_message,
                    utc_now(),
                ),
            )
            connection.commit()

    def upsert_unit(
        self,
        *,
        chapter_key,
        step_index,
        unit_index,
        step_name,
        input_hash,
        config_hash,
        status,
        output_path=None,
        error_message=None,
    ):
        with closing(self._connect()) as connection:
            connection.execute(
                """
                INSERT INTO normalization_units (
                    chapter_key,
                    step_index,
                    unit_index,
                    step_name,
                    input_hash,
                    config_hash,
                    status,
                    output_path,
                    error_message,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(chapter_key, step_index, unit_index, input_hash, config_hash)
                DO UPDATE SET
                    step_name = excluded.step_name,
                    status = excluded.status,
                    output_path = excluded.output_path,
                    error_message = excluded.error_message,
                    updated_at = excluded.updated_at
                """,
                (
                    chapter_key,
                    step_index,
                    unit_index,
                    step_name,
                    input_hash,
                    config_hash,
                    status,
                    output_path,
                    error_message,
                    utc_now(),
                ),
            )
            connection.commit()
