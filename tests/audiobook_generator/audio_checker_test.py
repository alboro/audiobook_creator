# Copyright (c) 2026 alboro <alboro@users.noreply.github.com>
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See: https://polyformproject.org/licenses/noncommercial/1.0.0/

import tempfile
from pathlib import Path

from audiobook_generator.core.audio_checker import AudioChecker
from audiobook_generator.core.audio_chunk_store import AudioChunkStore


def test_check_one_file_reuses_fresh_cached_raw_transcription():
    with tempfile.TemporaryDirectory() as tmp:
        output_dir = Path(tmp)
        audio_path = output_dir / "wav" / "chunks" / "0001_Test" / "hash_cached.wav"
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        audio_path.write_bytes(b"fake wav")

        store = AudioChunkStore(output_dir / "wav" / "_state" / "audio_chunks.sqlite3")
        store.save_checked_chunk(
            "0001_Test",
            "hash_cached",
            "Привет, мир.",
            "Привет мир",
            0.99,
            raw_transcription="Привет мир",
        )

        checker = AudioChecker(output_folder=output_dir, threshold=0.5)
        checker._pre_compare = None
        transcribe_called = False

        def fake_transcribe(_path):
            nonlocal transcribe_called
            transcribe_called = True
            return "не должно вызываться"

        checker._transcribe = fake_transcribe
        counters = {"checked": 0, "disputed": 0, "skipped": 0}

        checker._check_one_file(
            audio_path,
            "0001_Test",
            "hash_cached",
            "Привет, мир.",
            store,
            counters,
        )

        assert transcribe_called is False
        assert counters == {"checked": 1, "disputed": 0, "skipped": 0}


def test_check_one_file_persists_checked_chunk_cache_after_successful_run():
    with tempfile.TemporaryDirectory() as tmp:
        output_dir = Path(tmp)
        store = AudioChunkStore(output_dir / "wav" / "_state" / "audio_chunks.sqlite3")

        audio_path = output_dir / "wav" / "chunks" / "0001_Test" / "hash_checked_after_run.wav"
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        audio_path.write_bytes(b"fake wav")

        checker = AudioChecker(output_folder=output_dir, threshold=0.5)
        checker._pre_compare = None

        def fake_transcribe(_path):
            return "Привет мир"

        checker._transcribe = fake_transcribe
        counters = {"checked": 0, "disputed": 0, "skipped": 0}

        checker._check_one_file(
            audio_path,
            "0001_Test",
            "hash_checked_after_run",
            "Привет, мир.",
            store,
            counters,
        )

        assert counters == {"checked": 1, "disputed": 0, "skipped": 0}
        row = store.get_cached_transcription_entry("0001_Test", "hash_checked_after_run")
        assert row is not None
        assert row["status"] == "checked"
        assert row["raw_transcription"] == "Привет мир"

