# Copyright (c) 2026 alboro <alboro@users.noreply.github.com>
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See: https://polyformproject.org/licenses/noncommercial/1.0.0/

"""FastAPI tests for the Review UI disputed-chunk pipeline.

Current contract:
  - /api/settings exposes ``audio_check_threshold`` for the whisper checker.
  - /api/disputed returns only rows whose main ``status`` is ``disputed``.
  - similarity / reference-check metadata are returned for diagnostics only.
  - /api/disputed/resolve flips the main status to ``resolved`` and removes the
    row from subsequent disputed queries.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from audiobook_generator.core.audio_chunk_store import AudioChunkStore
from audiobook_generator.ui.review_server import app


_CHAPTER_KEY = "0001_Test_Chapter"


def _make_store(output_dir: Path) -> AudioChunkStore:
    db_path = output_dir / "wav" / "_state" / "audio_chunks.sqlite3"
    return AudioChunkStore(db_path)


def _seed_chunks(store: AudioChunkStore) -> None:
    """Insert checked/disputed/resolved rows with varied diagnostic metadata."""
    store.save_checked_chunk(
        chapter_key=_CHAPTER_KEY,
        sentence_hash="hash_checked_low",
        original_text="Низкое сходство, но статус checked.",
        transcription="Низкое сходство, но статус checked.",
        similarity=0.11,
    )
    store.save_disputed_chunk(
        chapter_key=_CHAPTER_KEY,
        sentence_hash="hash_disputed",
        original_text="Это спорный чанк.",
        transcription="Это спорный чанк.",
        similarity=0.97,
        reference_check_score=1.25,
        reference_check_threshold=0.85,
        reference_check_status="suspicious",
        reference_check_payload='{"score":1.25}',
    )
    store.save_disputed_chunk(
        chapter_key=_CHAPTER_KEY,
        sentence_hash="hash_resolved",
        original_text="Раньше спорный, теперь утверждён вручную.",
        transcription="Раньше спорный, теперь утверждён вручную.",
        similarity=0.42,
    )
    store.resolve_disputed_chunk(_CHAPTER_KEY, "hash_resolved")


class DisputedThresholdE2ETest(unittest.TestCase):
    """FastAPI endpoint tests for disputed-chunk status handling."""

    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="eta_disputed_")
        self.output_dir = Path(self._tmp) / "book_output"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.store = _make_store(self.output_dir)
        _seed_chunks(self.store)

        cfg = MagicMock()
        cfg.audio_check_threshold = 0.94
        cfg.audio_folder = None
        app.state.review_config = cfg

        self.client = TestClient(app)

    def tearDown(self):
        app.state.review_config = None
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_settings_returns_configured_threshold(self):
        resp = self.client.get("/api/settings")
        self.assertEqual(resp.status_code, 200)
        self.assertAlmostEqual(resp.json()["audio_check_threshold"], 0.94, places=4)

    def test_disputed_returns_only_main_disputed_status(self):
        resp = self.client.get("/api/disputed", params={
            "dir": str(self.output_dir),
            "chapter_key": _CHAPTER_KEY,
            "threshold": 0.94,
        })
        self.assertEqual(resp.status_code, 200)
        rows = resp.json()
        hashes = {r["hash"] for r in rows}
        self.assertEqual(hashes, {"hash_disputed"})

    def test_disputed_ignores_threshold_parameter(self):
        r1 = self.client.get("/api/disputed", params={
            "dir": str(self.output_dir),
            "chapter_key": _CHAPTER_KEY,
            "threshold": 0.10,
        }).json()
        r2 = self.client.get("/api/disputed", params={
            "dir": str(self.output_dir),
            "chapter_key": _CHAPTER_KEY,
            "threshold": 0.99,
        }).json()

        self.assertEqual({r["hash"] for r in r1}, {"hash_disputed"})
        self.assertEqual({r["hash"] for r in r2}, {"hash_disputed"})

    def test_disputed_row_fields_include_diagnostics(self):
        rows = self.client.get("/api/disputed", params={
            "dir": str(self.output_dir),
            "chapter_key": _CHAPTER_KEY,
            "threshold": 0.94,
        }).json()
        self.assertTrue(rows)
        row = rows[0]
        for field in (
            "hash", "original_text", "transcription", "similarity",
            "reference_check_score", "reference_check_threshold",
            "reference_check_status", "reference_check_payload",
            "checked_at", "status", "resolved",
        ):
            self.assertIn(field, row)
        self.assertEqual(row["status"], "disputed")
        self.assertFalse(row["resolved"])

    def test_resolve_excludes_chunk_from_subsequent_disputed_queries(self):
        resolve_resp = self.client.post("/api/disputed/resolve", json={
            "dir": str(self.output_dir),
            "chapter_key": _CHAPTER_KEY,
            "hash": "hash_disputed",
        })
        self.assertEqual(resolve_resp.status_code, 200)
        self.assertEqual(resolve_resp.json()["status"], "ok")

        disputed = self.client.get("/api/disputed", params={
            "dir": str(self.output_dir),
            "chapter_key": _CHAPTER_KEY,
            "threshold": 0.94,
        }).json()
        self.assertEqual(disputed, [])

    def test_resolve_nonexistent_hash_is_idempotent(self):
        resp = self.client.post("/api/disputed/resolve", json={
            "dir": str(self.output_dir),
            "chapter_key": _CHAPTER_KEY,
            "hash": "does_not_exist",
        })
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["status"], "ok")


if __name__ == "__main__":
    unittest.main()
