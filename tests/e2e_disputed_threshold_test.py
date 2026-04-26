# Copyright (c) 2026 alboro <alboro@users.noreply.github.com>
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See: https://polyformproject.org/licenses/noncommercial/1.0.0/

"""End-to-end test that locks down the full disputed-threshold pipeline.

The tested flow (as described in comments / README):

    INI: audio_check_threshold = 0.94
      ↓
    /api/settings → { audio_check_threshold: 0.94 }
      ↓
    boot(): disputeThreshold = 0.94  ← $watch fired
      ↓
    _loadDisputed(chapterKey, threshold=0.94)
      ↓
    SQL: similarity < 0.94 AND status != 'resolved'
      ↓
    disputedMap = { hash1: {...sim=0.76}, hash2: {...sim=0.89}, ... }
      ↓
    isDisputed(hash) → disputedMap[hash]
      ↓
    Play Disputed воспроизводит только реально спорные записи

This test covers the *server side* of the pipeline via FastAPI TestClient:
  - /api/settings reads threshold from app.state.review_config
  - /api/disputed filters by similarity < threshold AND status != 'resolved'
  - /api/disputed/resolve changes status → 'resolved' (excludes from subsequent queries)
  - Threshold changes (different ?threshold= values) work without re-running audio_check
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from audiobook_generator.core.audio_chunk_store import AudioChunkStore
from audiobook_generator.ui.review_server import app


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

_CHAPTER_KEY = "0001_Test_Chapter"


def _make_store(output_dir: Path) -> AudioChunkStore:
    db_path = output_dir / "wav" / "_state" / "audio_chunks.sqlite3"
    return AudioChunkStore(db_path)


def _seed_chunks(store: AudioChunkStore) -> dict[str, float]:
    """Insert three chunks with known similarities and return {hash: similarity}."""
    chunks = {
        "hash_low":    ("Низкое сходство.", 0.76),
        "hash_medium": ("Среднее сходство.", 0.89),
        "hash_high":   ("Высокое сходство.", 0.95),
    }
    for h, (text, sim) in chunks.items():
        store.save_checked_chunk(
            chapter_key=_CHAPTER_KEY,
            sentence_hash=h,
            original_text=text,
            transcription=text,
            similarity=sim,
        )
    return {h: sim for h, (_, sim) in chunks.items()}


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------

class DisputedThresholdE2ETest(unittest.TestCase):
    """FastAPI endpoint tests for the disputed-threshold pipeline."""

    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="eta_disputed_")
        self.output_dir = Path(self._tmp) / "book_output"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Seed the DB with three chunks covering low / medium / high similarity.
        self.store = _make_store(self.output_dir)
        self.similarities = _seed_chunks(self.store)

        # Point the FastAPI app at our test config so /api/settings returns 0.94.
        cfg = MagicMock()
        cfg.audio_check_threshold = 0.94
        cfg.audio_folder = None
        app.state.review_config = cfg

        self.client = TestClient(app)

    def tearDown(self):
        app.state.review_config = None
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)

    # ------------------------------------------------------------------
    # /api/settings
    # ------------------------------------------------------------------

    def test_settings_returns_configured_threshold(self):
        """GET /api/settings must reflect the value from app.state.review_config."""
        resp = self.client.get("/api/settings")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertAlmostEqual(data["audio_check_threshold"], 0.94, places=4)

    def test_settings_falls_back_to_default_when_no_config(self):
        """When app.state.review_config is absent, threshold must default to 0.70."""
        app.state.review_config = None
        # Patch load_merged_ini to return an empty dict so no per-project INI interferes.
        with unittest.mock.patch(
            "audiobook_generator.ui.review_server.load_merged_ini",
            return_value={},
        ):
            resp = self.client.get("/api/settings")
        self.assertEqual(resp.status_code, 200)
        self.assertAlmostEqual(resp.json()["audio_check_threshold"], 0.70, places=4)

    # ------------------------------------------------------------------
    # /api/disputed — dynamic threshold
    # ------------------------------------------------------------------

    def test_disputed_threshold_094_excludes_high_similarity(self):
        """similarity=0.95 must NOT appear when threshold=0.94."""
        resp = self.client.get("/api/disputed", params={
            "dir": str(self.output_dir),
            "chapter_key": _CHAPTER_KEY,
            "threshold": 0.94,
        })
        self.assertEqual(resp.status_code, 200)
        rows = resp.json()
        hashes = {r["hash"] for r in rows}
        self.assertIn("hash_low", hashes,
                      "hash_low (0.76) must be disputed at threshold 0.94")
        self.assertIn("hash_medium", hashes,
                      "hash_medium (0.89) must be disputed at threshold 0.94")
        self.assertNotIn("hash_high", hashes,
                         "hash_high (0.95) must NOT be disputed at threshold 0.94")

    def test_disputed_threshold_080_excludes_medium_and_high(self):
        """At threshold=0.80 only hash_low (0.76) should be returned."""
        resp = self.client.get("/api/disputed", params={
            "dir": str(self.output_dir),
            "chapter_key": _CHAPTER_KEY,
            "threshold": 0.80,
        })
        self.assertEqual(resp.status_code, 200)
        rows = resp.json()
        hashes = {r["hash"] for r in rows}
        self.assertIn("hash_low", hashes)
        self.assertNotIn("hash_medium", hashes)
        self.assertNotIn("hash_high", hashes)

    def test_disputed_threshold_changes_without_rerun(self):
        """Changing ?threshold dynamically must update results in the same request."""
        r94 = self.client.get("/api/disputed", params={
            "dir": str(self.output_dir),
            "chapter_key": _CHAPTER_KEY,
            "threshold": 0.94,
        }).json()
        r80 = self.client.get("/api/disputed", params={
            "dir": str(self.output_dir),
            "chapter_key": _CHAPTER_KEY,
            "threshold": 0.80,
        }).json()

        hashes_94 = {r["hash"] for r in r94}
        hashes_80 = {r["hash"] for r in r80}

        # At 0.94 we see more rows than at 0.80
        self.assertGreater(len(hashes_94), len(hashes_80))
        # hash_medium appears at 0.94 but not at 0.80
        self.assertIn("hash_medium", hashes_94)
        self.assertNotIn("hash_medium", hashes_80)

    def test_disputed_row_fields(self):
        """Each disputed row must contain the documented fields."""
        resp = self.client.get("/api/disputed", params={
            "dir": str(self.output_dir),
            "chapter_key": _CHAPTER_KEY,
            "threshold": 0.94,
        })
        rows = resp.json()
        self.assertTrue(rows, "Expected at least one disputed row")
        row = rows[0]
        for field in ("hash", "original_text", "transcription", "similarity",
                      "checked_at", "status", "resolved"):
            self.assertIn(field, row, f"Field '{field}' missing from disputed row")
        self.assertFalse(row["resolved"],
                         "Unresolved rows must have resolved=False")

    # ------------------------------------------------------------------
    # /api/disputed/resolve
    # ------------------------------------------------------------------

    def test_resolve_excludes_chunk_from_subsequent_disputed_queries(self):
        """After resolving hash_low it must vanish from /api/disputed results."""
        # Resolve hash_low
        resolve_resp = self.client.post("/api/disputed/resolve", json={
            "dir": str(self.output_dir),
            "chapter_key": _CHAPTER_KEY,
            "hash": "hash_low",
        })
        self.assertEqual(resolve_resp.status_code, 200)
        self.assertEqual(resolve_resp.json()["status"], "ok")

        # Query at threshold that covers hash_low (sim=0.76 < 0.94)
        disputed = self.client.get("/api/disputed", params={
            "dir": str(self.output_dir),
            "chapter_key": _CHAPTER_KEY,
            "threshold": 0.94,
        }).json()
        hashes = {r["hash"] for r in disputed}
        self.assertNotIn("hash_low", hashes,
                         "Resolved chunk must not appear in disputed list")
        # hash_medium still present
        self.assertIn("hash_medium", hashes)

    def test_resolved_chunk_stays_excluded_at_any_threshold(self):
        """A resolved chunk must not re-appear even if threshold encompasses its similarity."""
        self.store.resolve_disputed_chunk(_CHAPTER_KEY, "hash_low")

        for threshold in (0.80, 0.90, 0.94, 0.99):
            with self.subTest(threshold=threshold):
                rows = self.client.get("/api/disputed", params={
                    "dir": str(self.output_dir),
                    "chapter_key": _CHAPTER_KEY,
                    "threshold": threshold,
                }).json()
                hashes = {r["hash"] for r in rows}
                self.assertNotIn(
                    "hash_low", hashes,
                    f"Resolved hash_low must not appear at threshold={threshold}",
                )

    def test_resolve_nonexistent_hash_is_idempotent(self):
        """Resolving a hash absent from the DB must succeed silently (idempotent).

        The server performs an UPDATE that may touch zero rows, but that is not
        an error — the UI just wants the item marked resolved.  If the DB file
        does not exist at all, the endpoint returns 404; if it exists but the
        hash is not found the operation is a no-op and still returns 200.
        """
        resp = self.client.post("/api/disputed/resolve", json={
            "dir": str(self.output_dir),
            "chapter_key": _CHAPTER_KEY,
            "hash": "does_not_exist",
        })
        # DB exists → the UPDATE is a no-op, endpoint still returns 200
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["status"], "ok")

    # ------------------------------------------------------------------
    # Full pipeline scenario: settings → disputed → resolve → re-query
    # ------------------------------------------------------------------

    def test_full_pipeline_settings_disputed_resolve(self):
        """End-to-end: read threshold from /api/settings, fetch disputed,
        resolve one entry, confirm re-query reflects the change."""
        # 1. Fetch threshold from server
        threshold = self.client.get("/api/settings").json()["audio_check_threshold"]
        self.assertAlmostEqual(threshold, 0.94, places=4)

        # 2. Load disputed with that threshold
        initial_disputed = self.client.get("/api/disputed", params={
            "dir": str(self.output_dir),
            "chapter_key": _CHAPTER_KEY,
            "threshold": threshold,
        }).json()
        initial_hashes = {r["hash"] for r in initial_disputed}
        self.assertEqual(initial_hashes, {"hash_low", "hash_medium"},
                         f"Unexpected disputed set: {initial_hashes}")

        # 3. Verify similarities via the response
        by_hash = {r["hash"]: r for r in initial_disputed}
        self.assertAlmostEqual(by_hash["hash_low"]["similarity"], 0.76, places=3)
        self.assertAlmostEqual(by_hash["hash_medium"]["similarity"], 0.89, places=3)

        # 4. Resolve hash_low (user clicked "Resolve" in Review UI)
        self.client.post("/api/disputed/resolve", json={
            "dir": str(self.output_dir),
            "chapter_key": _CHAPTER_KEY,
            "hash": "hash_low",
        })

        # 5. Re-load disputed — only hash_medium remains
        after_resolve = self.client.get("/api/disputed", params={
            "dir": str(self.output_dir),
            "chapter_key": _CHAPTER_KEY,
            "threshold": threshold,
        }).json()
        after_hashes = {r["hash"] for r in after_resolve}
        self.assertEqual(after_hashes, {"hash_medium"},
                         f"After resolve, expected only hash_medium, got: {after_hashes}")


if __name__ == "__main__":
    unittest.main()

