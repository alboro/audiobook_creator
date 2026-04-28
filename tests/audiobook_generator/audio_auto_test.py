# Copyright (c) 2026 alboro <alboro@users.noreply.github.com>
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See: https://polyformproject.org/licenses/noncommercial/1.0.0/

"""Tests for audio_auto mode and auto-deletion tracking in AudioChunkStore."""
from __future__ import annotations

import io
import struct
import tempfile
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, call

import pytest

from audiobook_generator.core.audio_chunk_store import AudioChunkStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_store(tmp_path: Path) -> AudioChunkStore:
    db = tmp_path / "wav" / "_state" / "audio_chunks.sqlite3"
    return AudioChunkStore(db)


def _pcm_wav(samples=(100, -100), framerate=22050) -> bytes:
    buf = io.BytesIO()
    data = struct.pack(f'<{len(samples)}h', *samples)
    with wave.open(buf, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(framerate)
        w.writeframes(data)
    return buf.getvalue()


# ===========================================================================
# AudioChunkStore auto-deletion tracking
# ===========================================================================

class TestAutoDeleteTracking:
    def test_record_auto_deletion_increments_count(self, tmp_path):
        """Each record_auto_deletion call increments the count by one."""
        store = _make_store(tmp_path)
        assert store.get_auto_deletion_count("abc123") == 0

        store.record_auto_deletion("0001_Ch", "abc123", 0.50, 0.78)
        assert store.get_auto_deletion_count("abc123") == 1

        store.record_auto_deletion("0001_Ch", "abc123", 0.55, 0.78)
        assert store.get_auto_deletion_count("abc123") == 2

    def test_count_is_per_hash_independent(self, tmp_path):
        """Counts are independent per sentence hash."""
        store = _make_store(tmp_path)
        store.record_auto_deletion("0001_Ch", "hash_a", 0.60, 0.78)
        store.record_auto_deletion("0001_Ch", "hash_a", 0.62, 0.78)
        store.record_auto_deletion("0001_Ch", "hash_b", 0.70, 0.78)

        assert store.get_auto_deletion_count("hash_a") == 2
        assert store.get_auto_deletion_count("hash_b") == 1
        assert store.get_auto_deletion_count("hash_c") == 0

    def test_count_spans_multiple_chapters(self, tmp_path):
        """A hash that was deleted in different chapter contexts still counts."""
        store = _make_store(tmp_path)
        # Same hash, different chapters (could happen with repeated text)
        store.record_auto_deletion("0001_Ch", "shared_hash", 0.60, 0.78)
        store.record_auto_deletion("0002_Ch", "shared_hash", 0.65, 0.78)
        assert store.get_auto_deletion_count("shared_hash") == 2

    def test_get_all_failed_chunks_respects_threshold(self, tmp_path):
        """get_all_failed_chunks returns only rows below the given threshold."""
        store = _make_store(tmp_path)
        store.save_checked_chunk("0001_Ch", "low_h", "text", "transc", 0.50)
        store.save_checked_chunk("0001_Ch", "high_h", "text2", "transc2", 0.90)
        store.save_checked_chunk("0001_Ch", "border_h", "textb", "transcb", 0.78)

        failed = store.get_all_failed_chunks(0.78)
        hashes = {r["sentence_hash"] for r in failed}
        assert "low_h" in hashes
        assert "high_h" not in hashes
        assert "border_h" not in hashes  # 0.78 is not < 0.78

    def test_get_all_failed_chunks_excludes_resolved(self, tmp_path):
        """get_all_failed_chunks never returns resolved chunks."""
        store = _make_store(tmp_path)
        store.save_checked_chunk("0001_Ch", "h1", "text", "transc", 0.40)
        store.resolve_disputed_chunk("0001_Ch", "h1")

        failed = store.get_all_failed_chunks(0.78)
        hashes = {r["sentence_hash"] for r in failed}
        assert "h1" not in hashes


# ===========================================================================
# _run_audio_auto integration (unit-level with mocked subcomponents)
# ===========================================================================

def _make_auto_config(tmp_path: Path, threshold=0.78, max_retry=3) -> SimpleNamespace:
    return SimpleNamespace(
        output_folder=str(tmp_path),
        mode="audio_auto",
        language="ru-RU",
        audio_check_model="small",
        audio_check_threshold=0.70,
        audio_check_device="cpu",
        audio_auto_check_threshold=threshold,
        audio_auto_retry=max_retry,
        # fields consumed by AudiobookGenerator.run()
        prepare_text=False,
        package_m4b=False,
        normalize=False,
        chunked_audio=True,
        tts="openai",
        voice_name="default",
        voice_name2=None,
        prepared_text_folder=None,
        force_new_run=False,
        current_run_index=None,
        no_prompt=True,
        log="DEBUG",
        log_file=None,
    )


class TestRunAudioAuto:
    """Tests for the _run_audio_auto function in main.py."""

    def _make_db_and_store(self, tmp_path: Path) -> tuple[Path, AudioChunkStore]:
        db_path = tmp_path / "wav" / "_state" / "audio_chunks.sqlite3"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        store = AudioChunkStore(db_path)
        return db_path, store

    def test_stops_when_all_pass_on_first_attempt(self, tmp_path):
        """If all chunks pass after first synthesis+check cycle, no deletions happen."""
        db_path, store = self._make_db_and_store(tmp_path)
        # Pre-populate with passing similarity
        store.save_checked_chunk("0001_Ch", "good_hash", "text", "transc", 0.95)

        synth_calls = []
        check_calls = []

        def fake_synth(cfg):
            synth_calls.append(1)

        def fake_check(store_arg):
            check_calls.append(1)
            return {"checked": 1, "disputed": 0, "skipped": 0}

        config = _make_auto_config(tmp_path, threshold=0.78, max_retry=5)

        with patch("main._run_audio_auto") as mock_auto:
            # Run the actual function instead of mock
            mock_auto.side_effect = None

        # Test the logic directly
        from main import _run_audio_auto
        import copy

        synth_run_count = [0]
        check_run_count = [0]

        with patch("audiobook_generator.core.audiobook_generator.AudiobookGenerator") as MockGen, \
             patch("audiobook_generator.core.audio_checker.AudioChecker") as MockChecker:

            MockGen.return_value.run.side_effect = lambda: synth_run_count.__setitem__(0, synth_run_count[0] + 1)

            mock_checker_inst = MagicMock()
            # On first check: all good (no failed chunks in DB)
            mock_checker_inst.run.side_effect = lambda s: check_run_count.__setitem__(0, check_run_count[0] + 1) or {}
            MockChecker.return_value = mock_checker_inst

            # DB has no failing chunks → loop should stop immediately after attempt 1
            _run_audio_auto(config)

        # Should have synthesised once and checked once (+ possibly final check)
        assert synth_run_count[0] >= 1
        assert check_run_count[0] >= 1
        # No auto-deletions recorded
        assert store.get_auto_deletion_count("good_hash") == 0

    def test_deletes_failed_chunk_and_records_deletion(self, tmp_path):
        """A chunk that fails the auto threshold is deleted and the deletion is recorded."""
        db_path, store = self._make_db_and_store(tmp_path)

        # Create a fake chunk file
        chunk_dir = tmp_path / "wav" / "chunks" / "0001_Ch"
        chunk_dir.mkdir(parents=True)
        bad_chunk = chunk_dir / "bad0000.wav"
        bad_chunk.write_bytes(_pcm_wav())

        # Record a failing similarity in DB
        store.save_checked_chunk("0001_Ch", "bad0000", "bad text", "wrong transc", 0.50)

        config = _make_auto_config(tmp_path, threshold=0.78, max_retry=1)

        check_call_count = [0]

        with patch("audiobook_generator.core.audiobook_generator.AudiobookGenerator") as MockGen, \
             patch("audiobook_generator.core.audio_checker.AudioChecker") as MockChecker:

            MockGen.return_value.run.return_value = None

            mock_checker_inst = MagicMock()
            def fake_run(s):
                check_call_count[0] += 1
                # No-op: similarity already in DB from setup above
                return {}
            mock_checker_inst.run.side_effect = fake_run
            MockChecker.return_value = mock_checker_inst

            from main import _run_audio_auto
            _run_audio_auto(config)

        # The bad chunk file should be deleted
        assert not bad_chunk.exists(), "Failed chunk audio file should have been deleted"

        # The deletion should be recorded in DB
        deletion_count = store.get_auto_deletion_count("bad0000")
        assert deletion_count >= 1, f"Expected at least 1 deletion recorded, got {deletion_count}"

    def test_max_retry_respected(self, tmp_path):
        """Loop stops after max_retry attempts even if chunks keep failing."""
        db_path, store = self._make_db_and_store(tmp_path)
        max_retry = 3

        chunk_dir = tmp_path / "wav" / "chunks" / "0001_Ch"
        chunk_dir.mkdir(parents=True)

        synth_count = [0]
        check_count = [0]

        config = _make_auto_config(tmp_path, threshold=0.78, max_retry=max_retry)

        with patch("audiobook_generator.core.audiobook_generator.AudiobookGenerator") as MockGen, \
             patch("audiobook_generator.core.audio_checker.AudioChecker") as MockChecker:

            def fake_synth_run():
                synth_count[0] += 1
                # Re-create the bad chunk file each time it's "re-synthesised"
                bad = chunk_dir / "stubborn00.wav"
                bad.write_bytes(_pcm_wav())
                # Record a failing similarity so it keeps failing
                store.save_checked_chunk("0001_Ch", "stubborn00", "text", "wrong", 0.50)

            MockGen.return_value.run.side_effect = fake_synth_run

            mock_checker_inst = MagicMock()
            mock_checker_inst.run.side_effect = lambda s: check_count.__setitem__(0, check_count[0] + 1) or {}
            MockChecker.return_value = mock_checker_inst

            from main import _run_audio_auto
            _run_audio_auto(config)

        # Should have run at most max_retry + 1 synthesis attempts
        assert synth_count[0] <= max_retry + 1, (
            f"Expected at most {max_retry + 1} synthesis attempts, got {synth_count[0]}"
        )

    def test_retry_count_displayed_in_store_after_multiple_failures(self, tmp_path):
        """Successive failures of the same chunk accumulate in auto_deletion_count."""
        db_path, store = self._make_db_and_store(tmp_path)

        chunk_dir = tmp_path / "wav" / "chunks" / "0001_Ch"
        chunk_dir.mkdir(parents=True)

        config = _make_auto_config(tmp_path, threshold=0.78, max_retry=3)

        with patch("audiobook_generator.core.audiobook_generator.AudiobookGenerator") as MockGen, \
             patch("audiobook_generator.core.audio_checker.AudioChecker") as MockChecker:

            def fake_synth():
                bad = chunk_dir / "repeat000.wav"
                bad.write_bytes(_pcm_wav())
                store.save_checked_chunk("0001_Ch", "repeat000", "text", "wrong", 0.50)

            MockGen.return_value.run.side_effect = fake_synth
            mock_checker_inst = MagicMock()
            mock_checker_inst.run.return_value = {}
            MockChecker.return_value = mock_checker_inst

            from main import _run_audio_auto
            _run_audio_auto(config)

        count = store.get_auto_deletion_count("repeat000")
        assert count == 3, (
            f"Expected 3 deletions (= max_retry), got {count}. "
            "Each failed attempt should be recorded."
        )

