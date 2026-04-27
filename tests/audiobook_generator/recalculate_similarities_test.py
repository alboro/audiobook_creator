# Copyright (c) 2026 alboro <alboro@users.noreply.github.com>
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See: https://polyformproject.org/licenses/noncommercial/1.0.0/

"""Tests for recalculate_chunk_similarities script."""
from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path


def _make_db(tmp: Path, rows: list[tuple]) -> Path:
    """Create a chunk_cache DB in <tmp>/wav/_state/audio_chunks.sqlite3."""
    db_path = tmp / "wav" / "_state" / "audio_chunks.sqlite3"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE chunk_cache (
            chapter_key TEXT, sentence_hash TEXT, original_text TEXT,
            transcription TEXT, raw_transcription TEXT, similarity REAL,
            checked_at TEXT, status TEXT,
            PRIMARY KEY (chapter_key, sentence_hash)
        )
        """
    )
    conn.executemany(
        "INSERT INTO chunk_cache VALUES (?,?,?,?,?,?,?,?)", rows
    )
    conn.commit()
    conn.close()
    return db_path


def _read_db(db_path: Path) -> dict[str, dict]:
    """Return {sentence_hash: {similarity, transcription}} for all rows."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT sentence_hash, similarity, transcription, status FROM chunk_cache"
    ).fetchall()
    conn.close()
    return {r["sentence_hash"]: dict(r) for r in rows}


def _try_build():
    """Return (recalculate_fn, skip_reason)."""
    try:
        from audiobook_generator.core.recalculate_chunk_similarities import recalculate
        from audiobook_generator.core.recalculate_chunk_similarities import _build_normalizer
        _build_normalizer("ru-RU")   # will raise if deps missing
        return recalculate, None
    except Exception as exc:
        return None, str(exc)


class RecalculateSimilaritiesTest(unittest.TestCase):

    def setUp(self):
        self._recalculate, self._skip = _try_build()

    def _skip_if_unavailable(self):
        if self._skip:
            self.skipTest(f"normalizer unavailable: {self._skip}")

    # ------------------------------------------------------------------

    def test_digit_transcription_gets_higher_similarity_after_recalc(self):
        """
        Original text has year in word form; raw_transcription has digits (Whisper output).
        Old similarity was computed without pre_compare on transcription → very low.
        After recalculate the similarity must increase substantially.
        """
        self._skip_if_unavailable()
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            db_path = _make_db(tmp, [
                (
                    "ch1", "h_date",
                    "пятого августа тысяча семьсот девяносто четвёртого года",
                    "5 августа 1794 года",  # transcription (old, not expanded)
                    "5 августа 1794 года",  # raw_transcription (Whisper)
                    0.21,                   # old similarity (too low — false disputed)
                    "2026-01-01", "checked",
                ),
            ])

            counters = self._recalculate(tmp, language="ru-RU", dry_run=False)
            self.assertEqual(counters["updated"], 1)
            self.assertEqual(counters["skipped"], 0)

            after = _read_db(db_path)["h_date"]
            self.assertGreater(
                after["similarity"], 0.85,
                f"Expected sim > 0.85 after recalc, got {after['similarity']:.3f}",
            )

    def test_already_correct_row_not_updated(self):
        """A row with matching text and correct similarity must be left unchanged."""
        self._skip_if_unavailable()
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            # Use sim=1.0 — identical texts normalise to the same string → sim=1.0
            _make_db(tmp, [
                (
                    "ch1", "h_ok",
                    "Привет мир",
                    "Привет мир",
                    "Привет мир",
                    1.0,
                    "2026-01-01", "checked",
                ),
            ])

            counters = self._recalculate(tmp, language="ru-RU", dry_run=False)
            self.assertEqual(counters["unchanged"], 1, "Perfect-match row should be unchanged")

    def test_manual_transcription_skipped(self):
        """Rows with '[manual]' prefix in raw_transcription must be skipped."""
        self._skip_if_unavailable()
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            db_path = _make_db(tmp, [
                (
                    "ch1", "h_manual",
                    "Оригинальный текст",
                    "[manual] ручная правка",
                    "[manual] ручная правка",
                    0.0,
                    "2026-01-01", "resolved",
                ),
            ])

            before = _read_db(db_path)["h_manual"]["similarity"]
            counters = self._recalculate(tmp, language="ru-RU", dry_run=False)
            after = _read_db(db_path)["h_manual"]["similarity"]

            self.assertEqual(counters["skipped"], 1)
            self.assertAlmostEqual(before, after, places=6, msg="Manual row must not be modified")

    def test_resolved_status_preserved_after_recalc(self):
        """Status 'resolved' must never be changed even after similarity update."""
        self._skip_if_unavailable()
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            db_path = _make_db(tmp, [
                (
                    "ch1", "h_resolved",
                    "пятый год",
                    "5 год",
                    "5 год",
                    0.30,
                    "2026-01-01", "resolved",
                ),
            ])

            self._recalculate(tmp, language="ru-RU", dry_run=False)
            after = _read_db(db_path)["h_resolved"]
            self.assertEqual(after["status"], "resolved", "Status must remain 'resolved'")

    def test_dry_run_does_not_write(self):
        """With dry_run=True the DB must not be modified."""
        self._skip_if_unavailable()
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            db_path = _make_db(tmp, [
                (
                    "ch1", "h_dry",
                    "пятого августа тысяча семьсот девяносто четвёртого года",
                    "5 августа 1794 года",
                    "5 августа 1794 года",
                    0.21,
                    "2026-01-01", "checked",
                ),
            ])

            counters = self._recalculate(tmp, language="ru-RU", dry_run=True)
            # reported as "would update"
            self.assertEqual(counters["updated"], 1)

            # DB untouched
            after = _read_db(db_path)["h_dry"]
            self.assertAlmostEqual(after["similarity"], 0.21, places=4,
                                   msg="dry_run must not write to DB")

    def test_missing_db_raises(self):
        """FileNotFoundError when the DB does not exist."""
        from audiobook_generator.core.recalculate_chunk_similarities import recalculate
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str) / "nonexistent_folder"
            with self.assertRaises(FileNotFoundError):
                recalculate(tmp)

    def test_counters_total_equals_row_count(self):
        """Sum of counters must equal total rows in the DB."""
        self._skip_if_unavailable()
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            _make_db(tmp, [
                ("ch1", "h1", "пятого августа тысяча семьсот девяносто четвёртого года",
                 "5 августа 1794 года", "5 августа 1794 года", 0.21, "2026-01-01", "checked"),
                ("ch1", "h2", "Привет мир", "Привет мир", "Привет мир",
                 0.99, "2026-01-01", "checked"),
                ("ch1", "h3", "Текст", "[manual] правка", "[manual] правка",
                 0.0, "2026-01-01", "resolved"),
            ])

            counters = self._recalculate(tmp, language="ru-RU", dry_run=False)
            total = counters["updated"] + counters["unchanged"] + counters["skipped"]
            self.assertEqual(total, 3, f"Expected 3 rows total, counters={counters}")

