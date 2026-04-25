# Copyright (c) 2026 alboro <alboro@users.noreply.github.com>
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See: https://polyformproject.org/licenses/noncommercial/1.0.0/

"""Tests for the smart chunk-rebuild logic in package mode (_smart_chapter_list)."""
from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import pytest

from audiobook_generator.utils.sentence_hash import sentence_hash as _shash
from audiobook_generator.core.audiobook_generator import AudiobookGenerator


def _make_config(output_folder: str, chunked_audio: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        output_folder=output_folder,
        language="ru-RU",
        voice_name2=None,
        chunked_audio=chunked_audio,
        audio_folder=None,
    )


def _make_generator(output_folder: str, **kwargs) -> AudiobookGenerator:
    cfg = _make_config(output_folder, **kwargs)
    gen = AudiobookGenerator.__new__(AudiobookGenerator)
    gen.config = cfg
    return gen


# ---------------------------------------------------------------------------
# Helper: write fake WAV bytes (need non-zero content, pydub is mocked anyway)
# ---------------------------------------------------------------------------
_FAKE_WAV = b"RIFF\x00\x00\x00\x00WAVEfmt "


def _write_chunk(chunks_dir: Path, chapter_key: str, sentence: str, ext: str = ".wav"):
    d = chunks_dir / chapter_key
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{_shash(sentence)}{ext}"
    p.write_bytes(_FAKE_WAV)
    return p


# ---------------------------------------------------------------------------
# Scenario 1: all chunks present for all chapters → rebuild + return
# ---------------------------------------------------------------------------
def test_smart_chapter_list_all_chunks_complete():
    """When all chunks exist, _smart_chapter_list rebuilds chapters and returns them."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        # text run
        run_dir = root / "text" / "001"
        run_dir.mkdir(parents=True)

        sentences_ch1 = ["First sentence.", "Second sentence."]
        sentences_ch2 = ["Third sentence.", "Fourth sentence."]

        (run_dir / "0001_Chapter_One.txt").write_text(
            " ".join(sentences_ch1), encoding="utf-8"
        )
        (run_dir / "0002_Chapter_Two.txt").write_text(
            " ".join(sentences_ch2), encoding="utf-8"
        )

        # wav folder + chunks
        wav_dir = root / "wav"
        chunks_dir = wav_dir / "chunks"
        for s in sentences_ch1:
            _write_chunk(chunks_dir, "0001_Chapter_One", s)
        for s in sentences_ch2:
            _write_chunk(chunks_dir, "0002_Chapter_Two", s)

        gen = _make_generator(str(root))

        # Patch _merge_audio_files so we don't need pydub
        merged_calls: list[tuple] = []

        def fake_merge(chunk_paths, output_path):
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_bytes(_FAKE_WAV)
            merged_calls.append((chunk_paths, output_path))

        with patch(
            "audiobook_generator.core.audiobook_generator._merge_audio_files" if False
            else "audiobook_generator.core.chunked_audio_generator._merge_audio_files",
            side_effect=fake_merge,
        ):
            result = gen._smart_chapter_list(str(wav_dir))

        assert result is not None, "Should return a list, not None"
        assert len(result) == 2, f"Expected 2 chapters, got {len(result)}"
        assert len(merged_calls) == 2, "Both chapters should have been merged"

        paths = [p for p, _ in result]
        titles = [t for _, t in result]
        assert "Chapter One" in titles
        assert "Chapter Two" in titles
        # Output paths should be inside wav/
        for p in paths:
            assert Path(p).parent == wav_dir


# ---------------------------------------------------------------------------
# Scenario 2: one chapter has complete chunks, another has only a chapter file
# ---------------------------------------------------------------------------
def test_smart_chapter_list_mixed_fallback():
    """One chapter rebuilt from chunks; other falls back to existing chapter file."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        run_dir = root / "text" / "001"
        run_dir.mkdir(parents=True)

        sentences_ch1 = ["Hello world."]
        sentences_ch2 = ["Chapter two text."]

        (run_dir / "0001_Chapter_One.txt").write_text(
            sentences_ch1[0], encoding="utf-8"
        )
        (run_dir / "0002_Chapter_Two.txt").write_text(
            sentences_ch2[0], encoding="utf-8"
        )

        wav_dir = root / "wav"
        wav_dir.mkdir(parents=True)
        chunks_dir = wav_dir / "chunks"

        # Ch1: chunks present
        _write_chunk(chunks_dir, "0001_Chapter_One", sentences_ch1[0])
        # Ch2: no chunks, but chapter file exists
        existing_ch2_file = wav_dir / "0002_Chapter_Two.wav"
        existing_ch2_file.write_bytes(_FAKE_WAV)

        gen = _make_generator(str(root))

        def fake_merge(chunk_paths, output_path):
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_bytes(_FAKE_WAV)

        with patch(
            "audiobook_generator.core.chunked_audio_generator._merge_audio_files",
            side_effect=fake_merge,
        ):
            result = gen._smart_chapter_list(str(wav_dir))

        assert result is not None
        assert len(result) == 2
        titles = [t for _, t in result]
        assert "Chapter One" in titles
        assert "Chapter Two" in titles
        # Ch2 file should be the existing one
        ch2_path = next(p for p, t in result if t == "Chapter Two")
        assert Path(ch2_path) == existing_ch2_file


# ---------------------------------------------------------------------------
# Scenario 3: a chapter has no chunks AND no chapter file → abort (empty list)
# ---------------------------------------------------------------------------
def test_smart_chapter_list_impossible_chapter():
    """When a chapter has neither chunks nor file, empty list is returned (abort)."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        run_dir = root / "text" / "001"
        run_dir.mkdir(parents=True)

        (run_dir / "0001_Lonely_Chapter.txt").write_text(
            "Some text here.", encoding="utf-8"
        )

        wav_dir = root / "wav"
        wav_dir.mkdir(parents=True)
        # No chunks, no chapter file

        gen = _make_generator(str(root))

        result = gen._smart_chapter_list(str(wav_dir))

        assert result == [], "Should return empty list signalling abort"


# ---------------------------------------------------------------------------
# Scenario 4: no text run folder → None (fall through)
# ---------------------------------------------------------------------------
def test_smart_chapter_list_no_text_run():
    """When there is no text/ run folder at all, None is returned (fall through)."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        wav_dir = root / "wav"
        wav_dir.mkdir(parents=True)

        gen = _make_generator(str(root))
        result = gen._smart_chapter_list(str(wav_dir))

        assert result is None


# ---------------------------------------------------------------------------
# Scenario 5: rebuilt chapter file replaces old stale chapter file
# ---------------------------------------------------------------------------
def test_smart_chapter_list_deletes_old_chapter_file():
    """Old chapter audio file is deleted before merging fresh from chunks."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        run_dir = root / "text" / "001"
        run_dir.mkdir(parents=True)

        sentence = "The quick brown fox."
        (run_dir / "0001_Chapter_One.txt").write_text(sentence, encoding="utf-8")

        wav_dir = root / "wav"
        chunks_dir = wav_dir / "chunks"
        _write_chunk(chunks_dir, "0001_Chapter_One", sentence)

        # Pre-existing stale chapter file
        stale = wav_dir / "0001_Chapter_One.wav"
        stale.write_bytes(b"stale")

        gen = _make_generator(str(root))

        merged_output: list[str] = []

        def fake_merge(chunk_paths, output_path):
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_bytes(b"fresh")
            merged_output.append(output_path)

        with patch(
            "audiobook_generator.core.chunked_audio_generator._merge_audio_files",
            side_effect=fake_merge,
        ):
            result = gen._smart_chapter_list(str(wav_dir))

        assert result is not None
        assert len(result) == 1
        # Verify the output file has fresh content (stale was overwritten)
        out_path = result[0][0]
        assert Path(out_path).read_bytes() == b"fresh"


# ---------------------------------------------------------------------------
# Scenario 6: _run_package_only uses smart list when chunked_audio=True
# ---------------------------------------------------------------------------
def test_run_package_only_uses_smart_chapter_list():
    """_run_package_only calls _smart_chapter_list and passes result to package_m4b."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        wav_dir = root / "wav"
        wav_dir.mkdir(parents=True)
        dummy_file = wav_dir / "0001_Chapter.wav"
        dummy_file.write_bytes(_FAKE_WAV)

        gen = _make_generator(str(root), chunked_audio=True)

        smart_result = [(str(dummy_file), "Chapter")]

        packaged: list = []

        def fake_package_m4b(**kwargs):
            packaged.append(kwargs)
            return str(root / "book.m4b")

        with patch.object(gen, "_smart_chapter_list", return_value=smart_result), \
             patch.object(gen, "_detect_audio_folder", return_value=str(wav_dir)), \
             patch("audiobook_generator.core.audiobook_generator.package_m4b", side_effect=fake_package_m4b):
            gen.config.input_file = None
            gen.config.m4b_filename = None
            gen.config.m4b_bitrate = "64k"
            gen.config.ffmpeg_path = "ffmpeg"
            gen._run_package_only()

        assert len(packaged) == 1
        assert packaged[0]["chapter_files"] == [str(dummy_file)]
        assert packaged[0]["chapter_titles"] == ["Chapter"]


# ---------------------------------------------------------------------------
# Scenario 7: _run_package_only falls back when chunked_audio=False
# ---------------------------------------------------------------------------
def test_run_package_only_skips_smart_when_not_chunked():
    """When chunked_audio=False, _smart_chapter_list is not called."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        wav_dir = root / "wav"
        wav_dir.mkdir(parents=True)
        dummy_file = wav_dir / "0001_Chapter.wav"
        dummy_file.write_bytes(_FAKE_WAV)

        gen = _make_generator(str(root), chunked_audio=False)

        packaged: list = []

        def fake_package(**kwargs):
            packaged.append(kwargs)
            return "out.m4b"

        smart_called = []

        with patch.object(gen, "_smart_chapter_list", side_effect=lambda *a: smart_called.append(1) or []), \
             patch.object(gen, "_detect_audio_folder", return_value=str(wav_dir)), \
             patch("audiobook_generator.core.audiobook_generator.package_m4b", side_effect=fake_package):
            gen.config.input_file = None
            gen.config.m4b_filename = None
            gen.config.m4b_bitrate = "64k"
            gen.config.ffmpeg_path = "ffmpeg"
            gen._run_package_only()

        assert not smart_called, "_smart_chapter_list should not be called when chunked_audio=False"
        # But package_m4b should still be called via _scan_audio_files path
        assert len(packaged) == 1

