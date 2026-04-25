# Copyright (c) 2026 alboro <alboro@users.noreply.github.com>
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See: https://polyformproject.org/licenses/noncommercial/1.0.0/

"""SQLite store for sentence text version history.

Tracks the chain of edits for each sentence, keyed by content-hash.
Audio synthesis status is derived entirely from the filesystem (file
presence in ``wav/chunks/<chapter_key>/<hash>.<ext>``).

DB location: <output_folder>/wav/_state/audio_chunks.sqlite3
"""
from __future__ import annotations

import logging
import sqlite3
from contextlib import closing
from datetime import datetime, UTC
from pathlib import Path
from typing import List, Optional

from audiobook_generator.utils.sentence_hash import sentence_hash as _sentence_hash  # noqa: F401 – re-exported


logger = logging.getLogger(__name__)


STATUS_CHECKED = "checked"
STATUS_DISPUTED = "disputed"
STATUS_RESOLVED = "resolved"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


class AudioChunkStore:
    """Tracks per-sentence text version history in a WAL-mode SQLite database."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _connect(self):
        conn = sqlite3.connect(str(self.db_path), timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _initialize(self):
        with closing(self._connect()) as conn:
            # Create table if it doesn't exist yet
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS sentence_text_versions (
                    sentence_hash    TEXT NOT NULL PRIMARY KEY,
                    sentence_text    TEXT NOT NULL,
                    replaced_by_hash TEXT,
                    created_at       TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_text_versions_replaced_by
                    ON sentence_text_versions (replaced_by_hash);

                CREATE TABLE IF NOT EXISTS chunk_cache (
                    chapter_key   TEXT NOT NULL,
                    sentence_hash TEXT NOT NULL,
                    original_text TEXT NOT NULL,
                    transcription TEXT,
                    raw_transcription TEXT,
                    similarity    REAL,
                    checked_at    TEXT,
                    status        TEXT,
                    PRIMARY KEY (chapter_key, sentence_hash)
                );
                """
            )
            conn.commit()

            # ── Migration: drop legacy columns (version_index, run_id) ──────────
            cols = {row[1] for row in conn.execute("PRAGMA table_info(sentence_text_versions)")}
            logger.debug("AudioChunkStore sentence_text_versions columns: %s", cols)
            needs_migration = bool(cols & {"version_index", "run_id"})
            if needs_migration:
                logger.info("Migrating sentence_text_versions: dropping legacy columns version_index, run_id")
                conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS sentence_text_versions_new (
                        sentence_hash    TEXT NOT NULL PRIMARY KEY,
                        sentence_text    TEXT NOT NULL,
                        replaced_by_hash TEXT,
                        created_at       TEXT NOT NULL
                    );

                    INSERT OR IGNORE INTO sentence_text_versions_new
                        (sentence_hash, sentence_text, replaced_by_hash, created_at)
                    SELECT sentence_hash, sentence_text, replaced_by_hash, created_at
                    FROM sentence_text_versions;

                    DROP TABLE sentence_text_versions;
                    ALTER TABLE sentence_text_versions_new RENAME TO sentence_text_versions;

                    CREATE INDEX IF NOT EXISTS idx_text_versions_replaced_by
                        ON sentence_text_versions (replaced_by_hash);
                    """
                )
                conn.commit()
                logger.info("AudioChunkStore migration complete")

            legacy_disputed_exists = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'disputed_chunks'"
            ).fetchone()
            if legacy_disputed_exists:
                disputed_cols = {row[1] for row in conn.execute("PRAGMA table_info(disputed_chunks)")}
                raw_expr = (
                    "CASE WHEN raw_transcription IS NOT NULL THEN raw_transcription "
                    "WHEN transcription NOT LIKE '[manual]%' THEN transcription ELSE NULL END"
                    if "raw_transcription" in disputed_cols
                    else "CASE WHEN transcription NOT LIKE '[manual]%' THEN transcription ELSE NULL END"
                )
                status_expr = (
                    f"CASE WHEN resolved = 1 THEN '{STATUS_RESOLVED}' ELSE '{STATUS_DISPUTED}' END"
                    if "resolved" in disputed_cols
                    else f"'{STATUS_DISPUTED}'"
                )
                logger.info("Migrating legacy disputed_chunks rows into chunk_cache")
                conn.execute(
                    f"""
                    INSERT OR IGNORE INTO chunk_cache
                        (chapter_key, sentence_hash, original_text, transcription, raw_transcription, similarity, checked_at, status)
                    SELECT
                        chapter_key,
                        sentence_hash,
                        original_text,
                        transcription,
                        {raw_expr},
                        similarity,
                        checked_at,
                        {status_expr}
                    FROM disputed_chunks
                    """
                )
                conn.commit()

    def _upsert_chunk_cache(
        self,
        *,
        chapter_key: str,
        sentence_hash: str,
        original_text: str,
        transcription: str | None,
        raw_transcription: str | None,
        similarity: float | None,
        status: str,
        keep_resolved: bool = False,
    ) -> None:
        now = _utc_now()
        with closing(self._connect()) as conn:
            if keep_resolved:
                status_update = (
                    f"CASE WHEN chunk_cache.status = '{STATUS_RESOLVED}' "
                    f"THEN '{STATUS_RESOLVED}' ELSE excluded.status END"
                )
            else:
                status_update = "excluded.status"
            conn.execute(
                f"""
                INSERT INTO chunk_cache
                    (chapter_key, sentence_hash, original_text, transcription, raw_transcription, similarity, checked_at, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(chapter_key, sentence_hash) DO UPDATE SET
                    original_text = excluded.original_text,
                    transcription = excluded.transcription,
                    raw_transcription = excluded.raw_transcription,
                    similarity = excluded.similarity,
                    checked_at = excluded.checked_at,
                    status = {status_update}
                """,
                (chapter_key, sentence_hash, original_text, transcription, raw_transcription, similarity, now, status),
            )
            conn.commit()

    # ------------------------------------------------------------------
    # Sentence text version history
    # ------------------------------------------------------------------

    def save_sentence_version(
        self,
        sentence_hash: str,
        sentence_text: str,
        replaced_by_hash: Optional[str] = None,
    ) -> bool:
        """Save sentence text for this hash (INSERT OR IGNORE — one record per hash).

        If replaced_by_hash is provided, updates the chain link.
        Returns True when a new version row was inserted.
        """
        now = _utc_now()
        with closing(self._connect()) as conn:
            cur = conn.execute(
                """
                INSERT OR IGNORE INTO sentence_text_versions
                    (sentence_hash, sentence_text, replaced_by_hash, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (sentence_hash, sentence_text, replaced_by_hash, now),
            )
            inserted = cur.rowcount > 0
            if replaced_by_hash is not None:
                conn.execute(
                    "UPDATE sentence_text_versions SET replaced_by_hash = ? WHERE sentence_hash = ? AND replaced_by_hash IS NULL",
                    (replaced_by_hash, sentence_hash),
                )
            conn.commit()
            return inserted

    def get_sentence_predecessors(self, current_hash: str) -> List[sqlite3.Row]:
        """Return all versions that were replaced by current_hash (predecessor chain).

        Ordered newest → oldest (by created_at DESC).
        """
        with closing(self._connect()) as conn:
            return conn.execute(
                """
                SELECT * FROM sentence_text_versions
                WHERE replaced_by_hash = ?
                ORDER BY created_at DESC
                """,
                (current_hash,),
            ).fetchall()

    def get_latest_sentence_text(self, sentence_hash: str) -> Optional[str]:
        """Return the text for this hash, or None if not found."""
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT sentence_text FROM sentence_text_versions WHERE sentence_hash = ?",
                (sentence_hash,),
            ).fetchone()
        return row["sentence_text"] if row else None

    # ------------------------------------------------------------------
    # Chunk cache / disputed state
    # ------------------------------------------------------------------

    def save_checked_chunk(
        self,
        chapter_key: str,
        sentence_hash: str,
        original_text: str,
        transcription: str,
        similarity: float,
        raw_transcription: str | None = None,
    ) -> None:
        """Save a successfully checked chunk so future audio_check runs can reuse it."""
        raw_transcription = transcription if raw_transcription is None else raw_transcription
        self._upsert_chunk_cache(
            chapter_key=chapter_key,
            sentence_hash=sentence_hash,
            original_text=original_text,
            transcription=transcription,
            raw_transcription=raw_transcription,
            similarity=similarity,
            status=STATUS_CHECKED,
        )

    def save_disputed_chunk(
        self,
        chapter_key: str,
        sentence_hash: str,
        original_text: str,
        transcription: str,
        similarity: float,
        raw_transcription: str | None = None,
    ) -> None:
        """Save a disputed chunk entry and keep legacy resolved decisions intact."""
        raw_transcription = transcription if raw_transcription is None else raw_transcription
        self._upsert_chunk_cache(
            chapter_key=chapter_key,
            sentence_hash=sentence_hash,
            original_text=original_text,
            transcription=transcription,
            raw_transcription=raw_transcription,
            similarity=similarity,
            status=STATUS_DISPUTED,
            keep_resolved=True,
        )
        logger.info(
            "Recorded/updated disputed chunk chapter=%s hash=%s similarity=%.2f",
            chapter_key, sentence_hash[:8], similarity,
        )

    def get_cached_transcription_entry(
        self,
        chapter_key: str,
        sentence_hash: str,
    ) -> Optional[sqlite3.Row]:
        """Return cached transcription metadata for a chunk, if present."""
        with closing(self._connect()) as conn:
            return conn.execute(
                """
                SELECT transcription, raw_transcription, checked_at, status
                FROM chunk_cache
                WHERE chapter_key = ? AND sentence_hash = ?
                """,
                (chapter_key, sentence_hash),
            ).fetchone()

    def get_cached_raw_transcription(
        self,
        chapter_key: str,
        sentence_hash: str,
    ) -> Optional[str]:
        """Return reusable raw transcription for a chunk, if one exists."""
        row = self.get_cached_transcription_entry(chapter_key, sentence_hash)
        if not row:
            return None
        raw = row["raw_transcription"] or None
        if raw:
            return raw
        transcription = row["transcription"] or ""
        if transcription.startswith("[manual]"):
            return None
        return transcription or None

    def mark_missing_audio_disputed(
        self,
        chapter_key: str,
        sentence_hash: str,
        original_text: str,
        note: str = "[manual] audio deleted",
    ) -> None:
        """Mark a chunk as disputed because its audio file was deleted manually.

        Unlike ``save_disputed_chunk()``, this action is explicit user intent and
        must reopen the chunk even if it was previously marked resolved.
        """
        self._upsert_chunk_cache(
            chapter_key=chapter_key,
            sentence_hash=sentence_hash,
            original_text=original_text,
            transcription=note,
            raw_transcription=None,
            similarity=0.0,
            status=STATUS_DISPUTED,
        )
        logger.info(
            "Marked missing-audio chunk as disputed chapter=%s hash=%s",
            chapter_key, sentence_hash[:8],
        )

    def get_disputed_chunks(self, chapter_key: str) -> List[sqlite3.Row]:
        """Return all disputed chunks for a chapter, ordered by similarity DESC."""
        with closing(self._connect()) as conn:
            return conn.execute(
                """
                SELECT * FROM chunk_cache
                WHERE chapter_key = ? AND status = ?
                ORDER BY similarity DESC
                """,
                (chapter_key, STATUS_DISPUTED),
            ).fetchall()

    def resolve_disputed_chunk(self, chapter_key: str, sentence_hash: str) -> None:
        """Mark a disputed chunk as resolved."""
        with closing(self._connect()) as conn:
            cur = conn.execute(
                """
                UPDATE chunk_cache
                SET status = ?
                WHERE chapter_key = ? AND sentence_hash = ?
                """,
                (STATUS_RESOLVED, chapter_key, sentence_hash),
            )
            conn.commit()
            logger.debug(
                "Resolved disputed chunk chapter=%s hash=%s updated=%d",
                chapter_key, sentence_hash[:8], cur.rowcount,
            )
