# Copyright (c) 2026 alboro <alboro@users.noreply.github.com>
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See: https://polyformproject.org/licenses/noncommercial/1.0.0/

"""Tests for AudioChunkStore sentence version history."""

import sqlite3
import tempfile
from pathlib import Path

from audiobook_generator.core.audio_chunk_store import AudioChunkStore


def test_save_and_retrieve_sentence_versions():
    """Test saving sentence versions and retrieving via hash chain."""
    with tempfile.TemporaryDirectory() as tmp:
        store = AudioChunkStore(str(Path(tmp) / "test.db"))

        hash_a = "abc123def456"
        hash_b = "fffaaa000111"

        # Save first version (no predecessor)
        store.save_sentence_version(hash_a, "Hello world.")
        # Save second version, replacing first
        store.save_sentence_version(hash_a, "Hello world.", replaced_by_hash=hash_b)
        store.save_sentence_version(hash_b, "Hello world! How are you?")

        # Predecessors of hash_b should include hash_a
        preds = store.get_sentence_predecessors(hash_b)
        assert len(preds) == 1
        assert preds[0]["sentence_hash"] == hash_a
        assert preds[0]["sentence_text"] == "Hello world."

        # No predecessors of hash_a
        assert store.get_sentence_predecessors(hash_a) == []


def test_get_latest_sentence_text():
    """Test getting text for a known hash."""
    with tempfile.TemporaryDirectory() as tmp:
        store = AudioChunkStore(str(Path(tmp) / "test.db"))

        hash_x = "xyz789"

        # No version yet
        assert store.get_latest_sentence_text(hash_x) is None

        store.save_sentence_version(hash_x, "Hello from hash_x.")
        assert store.get_latest_sentence_text(hash_x) == "Hello from hash_x."


def test_version_chain_replacement():
    """Test that version chain is correctly updated when a sentence is replaced."""
    with tempfile.TemporaryDirectory() as tmp:
        store = AudioChunkStore(str(Path(tmp) / "test.db"))

        old_hash = "hash_old_111"
        new_hash = "hash_new_222"

        # Save original text with old_hash
        store.save_sentence_version(old_hash, "Original text.")
        # Replace with new_hash
        store.save_sentence_version(old_hash, "Original text.", replaced_by_hash=new_hash)
        store.save_sentence_version(new_hash, "Modified text!")

        # Check that new_hash points to old_hash as predecessor
        preds = store.get_sentence_predecessors(new_hash)
        assert len(preds) == 1
        assert preds[0]["sentence_hash"] == old_hash
        assert preds[0]["sentence_text"] == "Original text."


def test_mark_missing_audio_disputed_creates_disputed_entry():
    """Manual audio deletion should create a disputed entry with similarity=0."""
    with tempfile.TemporaryDirectory() as tmp:
        store = AudioChunkStore(str(Path(tmp) / "test.db"))

        chapter_key = "0001_Test"
        s_hash = "hash_missing_audio"
        text = "Some sentence text."

        store.mark_missing_audio_disputed(chapter_key, s_hash, text)

        rows = store.get_disputed_chunks(chapter_key)
        assert len(rows) == 1
        assert rows[0]["sentence_hash"] == s_hash
        assert rows[0]["original_text"] == text
        assert rows[0]["transcription"] == "[manual] audio deleted"
        assert rows[0]["raw_transcription"] is None
        assert rows[0]["similarity"] == 0.0
        assert rows[0]["status"] == "disputed"


def test_mark_missing_audio_disputed_reopens_resolved_entry():
    """Manual deletion must reopen even a previously resolved disputed entry."""
    with tempfile.TemporaryDirectory() as tmp:
        store = AudioChunkStore(str(Path(tmp) / "test.db"))

        chapter_key = "0001_Test"
        s_hash = "hash_resolved_reopen"
        text = "Some sentence text."

        store.save_disputed_chunk(chapter_key, s_hash, text, "old transcription", 0.42)
        store.resolve_disputed_chunk(chapter_key, s_hash)
        store.mark_missing_audio_disputed(chapter_key, s_hash, text)

        rows = store.get_disputed_chunks(chapter_key)
        assert len(rows) == 1
        assert rows[0]["sentence_hash"] == s_hash
        assert rows[0]["status"] == "disputed"
        assert rows[0]["transcription"] == "[manual] audio deleted"
        assert rows[0]["raw_transcription"] is None
        assert rows[0]["similarity"] == 0.0


def test_save_checked_chunk_creates_reusable_cache_entry():
    with tempfile.TemporaryDirectory() as tmp:
        store = AudioChunkStore(str(Path(tmp) / "test.db"))

        store.save_checked_chunk(
            "0001_Test",
            "hash_checked",
            "Исходный текст.",
            "whisper text",
            0.97,
            raw_transcription="raw whisper text",
        )

        row = store.get_cached_transcription_entry("0001_Test", "hash_checked")
        assert row is not None
        assert row["status"] == "checked"
        assert row["raw_transcription"] == "raw whisper text"


def test_save_disputed_chunk_stores_raw_transcription_and_cache_lookup():
    with tempfile.TemporaryDirectory() as tmp:
        store = AudioChunkStore(str(Path(tmp) / "test.db"))

        store.save_disputed_chunk(
            "0001_Test",
            "hash_cached_raw",
            "Исходный текст.",
            "Показать в UI",
            0.25,
            raw_transcription="raw whisper text",
        )

        rows = store.get_disputed_chunks("0001_Test")
        assert rows[0]["raw_transcription"] == "raw whisper text"
        assert store.get_cached_raw_transcription("0001_Test", "hash_cached_raw") == "raw whisper text"


def test_get_cached_raw_transcription_falls_back_to_legacy_transcription_column():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "legacy.db"
        conn = sqlite3.connect(db_path)
        conn.execute(
            """
            CREATE TABLE disputed_chunks (
                chapter_key   TEXT NOT NULL,
                sentence_hash TEXT NOT NULL,
                original_text TEXT NOT NULL,
                transcription TEXT NOT NULL,
                similarity    REAL NOT NULL,
                checked_at    TEXT NOT NULL,
                resolved      INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (chapter_key, sentence_hash)
            )
            """
        )
        conn.execute(
            """
            INSERT INTO disputed_chunks
                (chapter_key, sentence_hash, original_text, transcription, similarity, checked_at, resolved)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            ("0001_Test", "hash_legacy", "Текст", "legacy whisper", 0.3, "2026-04-25T12:00:00+00:00", 0),
        )
        conn.commit()
        conn.close()

        store = AudioChunkStore(str(db_path))
        row = store.get_cached_transcription_entry("0001_Test", "hash_legacy")
        assert row["raw_transcription"] == "legacy whisper"
        assert row["status"] == "disputed"
        assert store.get_cached_raw_transcription("0001_Test", "hash_legacy") == "legacy whisper"


if __name__ == "__main__":
    test_save_and_retrieve_sentence_versions()
    test_get_latest_sentence_text()
    test_version_chain_replacement()
    print("All tests passed!")
