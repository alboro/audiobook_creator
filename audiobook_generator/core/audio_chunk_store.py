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

    _CHUNK_CACHE_REFERENCE_COLUMNS = {
        "reference_check_score": "REAL",
        "reference_check_threshold": "REAL",
        "reference_check_status": "TEXT",
        "reference_check_payload": "TEXT",
    }

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
                    reference_check_score REAL,
                    reference_check_threshold REAL,
                    reference_check_status TEXT,
                    reference_check_payload TEXT,
                    checked_at    TEXT,
                    status        TEXT,
                    PRIMARY KEY (chapter_key, sentence_hash)
                );

                CREATE TABLE IF NOT EXISTS chunk_auto_deletions (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    chapter_key   TEXT NOT NULL,
                    sentence_hash TEXT NOT NULL,
                    deleted_at    TEXT NOT NULL,
                    similarity    REAL,
                    threshold     REAL
                );

                CREATE INDEX IF NOT EXISTS idx_auto_deletions
                    ON chunk_auto_deletions (sentence_hash);
                """
            )
            conn.commit()

            self._ensure_chunk_cache_reference_columns(conn)

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

    def _ensure_chunk_cache_reference_columns(self, conn) -> None:
        """Add reference-check columns to legacy chunk_cache tables if needed."""
        chunk_cols = {row[1] for row in conn.execute("PRAGMA table_info(chunk_cache)")}
        added: list[str] = []
        for col_name, col_type in self._CHUNK_CACHE_REFERENCE_COLUMNS.items():
            if col_name not in chunk_cols:
                conn.execute(f"ALTER TABLE chunk_cache ADD COLUMN {col_name} {col_type}")
                added.append(col_name)
        if added:
            conn.commit()
            logger.info(
                "Migrated chunk_cache reference-check columns: %s",
                ", ".join(added),
            )

    def _ensure_checker_column_in_conn(self, conn, checker_name: str) -> None:
        """Add checker_<name>_passed column to chunk_cache if it does not exist yet.

        The column stores per-checker pass/fail results:
          NULL  = checker has never been run for this chunk
          1     = checker passed (not disputed)
          0     = checker failed (disputed)
        """
        col = f"checker_{checker_name}_passed"
        existing = {row[1] for row in conn.execute("PRAGMA table_info(chunk_cache)")}
        if col not in existing:
            conn.execute(f"ALTER TABLE chunk_cache ADD COLUMN {col} INTEGER")
            conn.commit()
            logger.debug("Added checker column %r to chunk_cache", col)

    # ------------------------------------------------------------------
    # Per-checker result persistence
    # ------------------------------------------------------------------

    def save_checker_result(
        self,
        chapter_key: str,
        sentence_hash: str,
        checker_name: str,
        passed: bool,
    ) -> None:
        """Store pass/fail result for a single checker in its dedicated column.

        The column ``checker_<name>_passed`` is created automatically if it does
        not yet exist.  Only rows already present in ``chunk_cache`` are updated;
        if no row exists the call is silently a no-op.
        """
        col = f"checker_{checker_name}_passed"
        with closing(self._connect()) as conn:
            self._ensure_checker_column_in_conn(conn, checker_name)
            conn.execute(
                f"UPDATE chunk_cache SET {col} = ? WHERE chapter_key = ? AND sentence_hash = ?",
                (1 if passed else 0, chapter_key, sentence_hash),
            )
            conn.commit()

    def get_all_checker_passed_columns(
        self,
        chapter_key: str,
        sentence_hash: str,
    ) -> dict[str, Optional[bool]]:
        """Return ``{checker_name: passed}`` for every ``checker_*_passed`` column.

        Values:
          ``True``  – checker passed for this chunk
          ``False`` – checker failed (flagged as disputed)
          ``None``  – column exists but was never populated for this chunk
        """
        with closing(self._connect()) as conn:
            all_cols = [row[1] for row in conn.execute("PRAGMA table_info(chunk_cache)")]
            checker_cols = [
                c for c in all_cols
                if c.startswith("checker_") and c.endswith("_passed")
            ]
            if not checker_cols:
                return {}
            row = conn.execute(
                f"SELECT {', '.join(checker_cols)} FROM chunk_cache "
                "WHERE chapter_key = ? AND sentence_hash = ?",
                (chapter_key, sentence_hash),
            ).fetchone()
            if not row:
                return {}
            out: dict[str, Optional[bool]] = {}
            for col in checker_cols:
                name = col[8:-7]  # strip "checker_" (8) and "_passed" (7)
                val = row[col]
                out[name] = (val == 1) if val is not None else None
            return out

    def get_chunk_cache_full_row(
        self,
        chapter_key: str,
        sentence_hash: str,
    ) -> Optional[sqlite3.Row]:
        """Return the full ``chunk_cache`` row for a (chapter_key, sentence_hash) pair."""
        with closing(self._connect()) as conn:
            return conn.execute(
                "SELECT * FROM chunk_cache WHERE chapter_key = ? AND sentence_hash = ?",
                (chapter_key, sentence_hash),
            ).fetchone()

    # ------------------------------------------------------------------

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
        reference_check_score: float | None = None,
        reference_check_threshold: float | None = None,
        reference_check_status: str | None = None,
        reference_check_payload: str | None = None,
        keep_resolved: bool = False,
    ) -> None:
        now = _utc_now()
        with closing(self._connect()) as conn:
            self._ensure_chunk_cache_reference_columns(conn)
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
                    (
                        chapter_key, sentence_hash, original_text,
                        transcription, raw_transcription, similarity,
                        reference_check_score, reference_check_threshold,
                        reference_check_status, reference_check_payload,
                        checked_at, status
                    )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(chapter_key, sentence_hash) DO UPDATE SET
                    original_text = excluded.original_text,
                    transcription = excluded.transcription,
                    raw_transcription = excluded.raw_transcription,
                    similarity = excluded.similarity,
                    reference_check_score = excluded.reference_check_score,
                    reference_check_threshold = excluded.reference_check_threshold,
                    reference_check_status = excluded.reference_check_status,
                    reference_check_payload = excluded.reference_check_payload,
                    checked_at = excluded.checked_at,
                    status = {status_update}
                """,
                (
                    chapter_key,
                    sentence_hash,
                    original_text,
                    transcription,
                    raw_transcription,
                    similarity,
                    reference_check_score,
                    reference_check_threshold,
                    reference_check_status,
                    reference_check_payload,
                    now,
                    status,
                ),
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
        reference_check_score: float | None = None,
        reference_check_threshold: float | None = None,
        reference_check_status: str | None = None,
        reference_check_payload: str | None = None,
        force_status: bool = False,
    ) -> None:
        """Save an automatically checked chunk that passed the current checker set."""
        raw_transcription = transcription if raw_transcription is None else raw_transcription
        self._upsert_chunk_cache(
            chapter_key=chapter_key,
            sentence_hash=sentence_hash,
            original_text=original_text,
            transcription=transcription,
            raw_transcription=raw_transcription,
            similarity=similarity,
            reference_check_score=reference_check_score,
            reference_check_threshold=reference_check_threshold,
            reference_check_status=reference_check_status,
            reference_check_payload=reference_check_payload,
            status=STATUS_CHECKED,
            keep_resolved=not force_status,  # force_status=True allows rewriting any prior verdict
        )

    def save_disputed_chunk(
        self,
        chapter_key: str,
        sentence_hash: str,
        original_text: str,
        transcription: str,
        similarity: float,
        raw_transcription: str | None = None,
        reference_check_score: float | None = None,
        reference_check_threshold: float | None = None,
        reference_check_status: str | None = None,
        reference_check_payload: str | None = None,
        force_status: bool = False,
    ) -> None:
        """Store a chunk that the checker pipeline has deemed disputed.

        Marks ``status = 'disputed'`` so the Review UI can surface it without
        re-running audio_check.  An existing ``resolved`` status (user-approved)
        is never overwritten.
        """
        raw_transcription = transcription if raw_transcription is None else raw_transcription
        self._upsert_chunk_cache(
            chapter_key=chapter_key,
            sentence_hash=sentence_hash,
            original_text=original_text,
            transcription=transcription,
            raw_transcription=raw_transcription,
            similarity=similarity,
            reference_check_score=reference_check_score,
            reference_check_threshold=reference_check_threshold,
            reference_check_status=reference_check_status,
            reference_check_payload=reference_check_payload,
            status=STATUS_DISPUTED,
            keep_resolved=not force_status,
        )
        logger.info(
            "Marked disputed chunk chapter=%s hash=%s similarity=%.2f",
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

    def get_disputed_chunks(self, chapter_key: str, threshold: float = 0.70) -> List[dict]:
        """Return chunks that are disputed for this chapter.

        Only rows whose main ``status`` is ``'disputed'`` are returned.
        The *threshold* argument is kept for backward compatibility but is
        intentionally ignored here: checker-specific fields such as
        ``similarity`` and ``reference_check_status`` are diagnostic metadata,
        not an alternate source of disputed state for the Review UI.
        """
        with closing(self._connect()) as conn:
            rows = conn.execute(
                """
                SELECT * FROM chunk_cache
                WHERE chapter_key = ?
                  AND status = ?
                ORDER BY similarity ASC
                """,
                (chapter_key, STATUS_DISPUTED),
            ).fetchall()
            return [dict(r) for r in rows]

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

    # ------------------------------------------------------------------
    # Auto-retry deletion tracking
    # ------------------------------------------------------------------

    def record_auto_deletion(
        self,
        chapter_key: str,
        sentence_hash: str,
        similarity: float,
        threshold: float,
    ) -> None:
        """Record that an audio chunk was automatically deleted for re-synthesis.

        Each call appends one row; the row count serves as the retry counter.
        """
        with closing(self._connect()) as conn:
            conn.execute(
                """
                INSERT INTO chunk_auto_deletions
                    (chapter_key, sentence_hash, deleted_at, similarity, threshold)
                VALUES (?, ?, ?, ?, ?)
                """,
                (chapter_key, sentence_hash, _utc_now(), similarity, threshold),
            )
            conn.commit()
        logger.debug(
            "Recorded auto-deletion chapter=%s hash=%s sim=%.2f threshold=%.2f",
            chapter_key, sentence_hash[:8], similarity, threshold,
        )

    def get_auto_deletion_count(self, sentence_hash: str) -> int:
        """Return how many times this chunk was automatically deleted for re-synthesis."""
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS cnt FROM chunk_auto_deletions WHERE sentence_hash = ?",
                (sentence_hash,),
            ).fetchone()
        return int(row["cnt"]) if row else 0

    def get_all_failed_chunks(self, threshold: float) -> List[sqlite3.Row]:
        """Return all disputed chunk_cache rows that are not resolved.

        Only the main ``status`` drives retry selection.
        The *threshold* argument is kept for backward compatibility but is
        intentionally ignored: per-checker metrics remain informative only.
        """
        with closing(self._connect()) as conn:
            return conn.execute(
                """
                SELECT chapter_key, sentence_hash, similarity, original_text
                FROM chunk_cache
                WHERE status = ?
                ORDER BY chapter_key, similarity ASC
                """,
                (STATUS_DISPUTED,),
            ).fetchall()
