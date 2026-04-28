# Copyright (c) 2026 alboro <alboro@users.noreply.github.com>
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See: https://polyformproject.org/licenses/noncommercial/1.0.0/

"""Tests for the smart chunk-rebuild logic in package mode (_smart_chapter_list)."""
from __future__ import annotations

import io
import struct
import tempfile
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import pytest

from audiobook_generator.utils.sentence_hash import sentence_hash as _shash
from audiobook_generator.core.audiobook_generator import AudiobookGenerator
from audiobook_generator.core.chunked_audio_generator import (
    _read_wav_frames,
    _merge_wav_files,
)


def _make_config(output_folder: str, chunked_audio: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        output_folder=output_folder,
        language="ru-RU",
        voice_name2=None,
        chunked_audio=chunked_audio,
        audio_folder=None,
        chapter_titles_file=None,
        cover_image=None,
    )


def _make_generator(output_folder: str, **kwargs) -> AudiobookGenerator:
    cfg = _make_config(output_folder, **kwargs)
    gen = AudiobookGenerator.__new__(AudiobookGenerator)
    gen.config = cfg
    return gen


# ---------------------------------------------------------------------------
# WAV file builder helpers
# ---------------------------------------------------------------------------

def _pcm16_wav_bytes(samples: list[int], nchannels: int = 1, framerate: int = 22050) -> bytes:
    """Create a valid 16-bit PCM WAV file in memory and return its bytes."""
    data = struct.pack(f'<{len(samples)}h', *samples)
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as w:
        w.setnchannels(nchannels)
        w.setsampwidth(2)
        w.setframerate(framerate)
        w.writeframes(data)
    return buf.getvalue()


def _float32_wav_bytes(samples: list[float], nchannels: int = 1, framerate: int = 22050) -> bytes:
    """Create an IEEE float-32 WAV file in memory and return its bytes."""
    n = len(samples)
    audio_data = struct.pack(f'<{n}f', *samples)
    buf = io.BytesIO()
    buf.write(b'RIFF')
    buf.write(struct.pack('<I', 36 + len(audio_data)))
    buf.write(b'WAVE')
    buf.write(b'fmt ')
    buf.write(struct.pack('<I', 16))               # fmt chunk size
    buf.write(struct.pack('<H', 3))                # format: IEEE float
    buf.write(struct.pack('<H', nchannels))
    buf.write(struct.pack('<I', framerate))
    buf.write(struct.pack('<I', framerate * 4 * nchannels))  # byte rate
    buf.write(struct.pack('<H', 4 * nchannels))    # block align
    buf.write(struct.pack('<H', 32))               # bits per sample
    buf.write(b'data')
    buf.write(struct.pack('<I', len(audio_data)))
    buf.write(audio_data)
    return buf.getvalue()


def _write_pcm16_chunk(chunks_dir: Path, chapter_key: str, sentence: str,
                       samples: list[int] | None = None, framerate: int = 22050) -> Path:
    """Write a 16-bit PCM WAV chunk for *sentence* under *chapter_key*."""
    d = chunks_dir / chapter_key
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{_shash(sentence)}.wav"
    p.write_bytes(_pcm16_wav_bytes(samples or [1000, -1000], framerate=framerate))
    return p


def _write_float32_chunk(chunks_dir: Path, chapter_key: str, sentence: str,
                         samples: list[float] | None = None, framerate: int = 22050) -> Path:
    """Write an IEEE float-32 WAV chunk for *sentence* under *chapter_key*."""
    d = chunks_dir / chapter_key
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{_shash(sentence)}.wav"
    p.write_bytes(_float32_wav_bytes(samples or [0.1, -0.1], framerate=framerate))
    return p


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


# ===========================================================================
# Tests for _read_wav_frames (handles PCM and IEEE float WAV)
# ===========================================================================

class TestReadWavFrames:
    def test_reads_pcm16_wav(self, tmp_path):
        """_read_wav_frames reads a standard 16-bit PCM WAV file."""
        samples = [100, -100, 200, -200]
        wav_file = tmp_path / "pcm.wav"
        wav_file.write_bytes(_pcm16_wav_bytes(samples, framerate=22050))

        nchannels, sampwidth, framerate, frames = _read_wav_frames(str(wav_file))

        assert nchannels == 1
        assert sampwidth == 2
        assert framerate == 22050
        assert len(frames) == len(samples) * 2  # 2 bytes per sample

    def test_reads_float32_wav_and_converts_to_pcm16(self, tmp_path):
        """_read_wav_frames reads an IEEE float-32 WAV and converts samples to int16."""
        samples = [0.5, -0.5, 0.25, -0.25]
        wav_file = tmp_path / "float32.wav"
        wav_file.write_bytes(_float32_wav_bytes(samples, framerate=24000))

        nchannels, sampwidth, framerate, frames = _read_wav_frames(str(wav_file))

        assert nchannels == 1
        assert sampwidth == 2          # converted to int16
        assert framerate == 24000
        # Should have 4 int16 samples = 8 bytes
        assert len(frames) == len(samples) * 2

        # Values should be roughly ±16383 for ±0.5
        pcm = struct.unpack(f'<{len(samples)}h', frames)
        assert pcm[0] > 0    # 0.5 → positive
        assert pcm[1] < 0    # -0.5 → negative
        assert abs(pcm[0]) > 10000  # roughly 32767 * 0.5

    def test_float32_channel_and_rate_preserved(self, tmp_path):
        """_read_wav_frames preserves nchannels and framerate from float WAV."""
        wav_file = tmp_path / "stereo.wav"
        # 4 stereo samples (2 frames of 2 channels each)
        wav_file.write_bytes(_float32_wav_bytes([0.1, -0.1, 0.2, -0.2],
                                                nchannels=2, framerate=44100))

        nchannels, sampwidth, framerate, frames = _read_wav_frames(str(wav_file))
        assert nchannels == 2
        assert framerate == 44100
        assert sampwidth == 2


# ===========================================================================
# Tests for _merge_wav_files  (ordering + float WAV support)
# ===========================================================================

class TestMergeWavFiles:
    def test_merges_pcm16_chunks_in_order(self, tmp_path):
        """_merge_wav_files concatenates PCM chunks in the passed order."""
        # Chunk 1: samples [1000, 1000], Chunk 2: samples [-1000, -1000]
        c1 = tmp_path / "c1.wav"
        c2 = tmp_path / "c2.wav"
        c1.write_bytes(_pcm16_wav_bytes([1000, 1000], framerate=22050))
        c2.write_bytes(_pcm16_wav_bytes([-1000, -1000], framerate=22050))

        out = tmp_path / "out.wav"
        _merge_wav_files([str(c1), str(c2)], str(out))

        with wave.open(str(out), 'rb') as w:
            frames = w.readframes(w.getnframes())
        samples = struct.unpack('<4h', frames)
        # First two samples from c1, last two from c2
        assert samples[0] > 0
        assert samples[1] > 0
        assert samples[2] < 0
        assert samples[3] < 0

    def test_merges_float32_chunks(self, tmp_path):
        """_merge_wav_files can merge float-32 WAV chunks (via _read_wav_frames)."""
        c1 = tmp_path / "c1.wav"
        c2 = tmp_path / "c2.wav"
        # c1: positive, c2: negative
        c1.write_bytes(_float32_wav_bytes([0.9, 0.9], framerate=22050))
        c2.write_bytes(_float32_wav_bytes([-0.9, -0.9], framerate=22050))

        out = tmp_path / "out.wav"
        _merge_wav_files([str(c1), str(c2)], str(out))

        with wave.open(str(out), 'rb') as w:
            assert w.getsampwidth() == 2  # converted to PCM16
            frames = w.readframes(w.getnframes())
        samples = struct.unpack('<4h', frames)
        assert samples[0] > 0
        assert samples[1] > 0
        assert samples[2] < 0
        assert samples[3] < 0

    def test_merge_preserves_text_order_with_float32(self, tmp_path):
        """When float-32 chunks are assembled, output is in the exact chunk_paths order."""
        # Create 5 chunks, each with a distinct positive value to identify them
        n_chunks = 5
        chunk_paths = []
        for i in range(n_chunks):
            f = tmp_path / f"c{i}.wav"
            # Each chunk: a unique amplitude level (0.1, 0.2, ..., 0.5)
            amp = (i + 1) * 0.1
            f.write_bytes(_float32_wav_bytes([amp] * 4, framerate=22050))
            chunk_paths.append(str(f))

        out = tmp_path / "out.wav"
        _merge_wav_files(chunk_paths, out_str := str(out))

        with wave.open(out_str, 'rb') as w:
            samples = struct.unpack(f'<{w.getnframes()}h', w.readframes(w.getnframes()))

        # Samples should be in strictly ascending order (chunk 0 lowest, chunk 4 highest)
        # Groups of 4 samples per chunk
        group_means = [
            sum(abs(s) for s in samples[i * 4: (i + 1) * 4]) / 4
            for i in range(n_chunks)
        ]
        # Each subsequent group should be larger (later chunk = higher amplitude)
        for j in range(n_chunks - 1):
            assert group_means[j] < group_means[j + 1], (
                f"Chunk order not preserved: group {j} mean {group_means[j]:.0f} "
                f">= group {j+1} mean {group_means[j+1]:.0f}"
            )

    def test_skip_truly_unreadable_but_keep_others(self, tmp_path):
        """A corrupt file is skipped; valid chunks around it are included."""
        good1 = tmp_path / "g1.wav"
        bad = tmp_path / "bad.wav"
        good2 = tmp_path / "g2.wav"

        good1.write_bytes(_pcm16_wav_bytes([500, 500], framerate=22050))
        bad.write_bytes(b"not_a_wav_file")
        good2.write_bytes(_pcm16_wav_bytes([-500, -500], framerate=22050))

        out = tmp_path / "out.wav"
        _merge_wav_files([str(good1), str(bad), str(good2)], str(out))

        with wave.open(str(out), 'rb') as w:
            n = w.getnframes()
            frames = w.readframes(n)
        samples = struct.unpack(f'<{n}h', frames)
        assert any(s > 0 for s in samples)
        assert any(s < 0 for s in samples)


# ===========================================================================
# Tests for _smart_chapter_list ordering (text-order preserved)
# ===========================================================================

class TestSmartChapterListOrdering:
    """Verify that _smart_chapter_list passes chunk_paths in text/sentence order."""

    def test_chunk_paths_passed_in_sentence_order(self, tmp_path):
        """chunk_paths fed to _merge_audio_files match the sentence order in the text file."""
        run_dir = tmp_path / "text" / "001"
        run_dir.mkdir(parents=True)

        # Three distinct sentences in a specific order
        s1 = "First sentence here."
        s2 = "Second sentence here."
        s3 = "Third sentence here."
        text = f"{s1} {s2} {s3}"
        (run_dir / "0001_Chapter_One.txt").write_text(text, encoding="utf-8")

        wav_dir = tmp_path / "wav"
        chunks_dir = wav_dir / "chunks"
        # Write PCM16 chunks with distinct amplitude per sentence so we can identify order
        _write_pcm16_chunk(chunks_dir, "0001_Chapter_One", s1, samples=[100, 100])
        _write_pcm16_chunk(chunks_dir, "0001_Chapter_One", s2, samples=[200, 200])
        _write_pcm16_chunk(chunks_dir, "0001_Chapter_One", s3, samples=[300, 300])

        gen = _make_generator(str(tmp_path))

        received_paths: list[list[str]] = []

        def capture_merge(chunk_paths, output_path, *args, **kwargs):
            received_paths.append(list(chunk_paths))
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_bytes(_pcm16_wav_bytes([0], framerate=22050))

        with patch(
            "audiobook_generator.core.chunked_audio_generator._merge_audio_files",
            side_effect=capture_merge,
        ):
            result = gen._smart_chapter_list(str(wav_dir))

        assert result is not None
        assert len(received_paths) == 1
        paths = received_paths[0]
        assert len(paths) == 3

        # Expected order: s1, s2, s3 (by position in text)
        expected_hashes = [_shash(s1), _shash(s2), _shash(s3)]
        actual_hashes = [Path(p).stem for p in paths]
        assert actual_hashes == expected_hashes, (
            f"Chunk order mismatch.\n"
            f"Expected: {expected_hashes}\n"
            f"Actual:   {actual_hashes}"
        )

    def test_float32_chunks_assembled_in_text_order(self, tmp_path):
        """Float-32 WAV chunks are merged in text order without any being skipped."""
        run_dir = tmp_path / "text" / "001"
        run_dir.mkdir(parents=True)

        s1 = "First float chunk."
        s2 = "Second float chunk."
        s3 = "Third float chunk."
        text = f"{s1} {s2} {s3}"
        (run_dir / "0001_Chapter_One.txt").write_text(text, encoding="utf-8")

        wav_dir = tmp_path / "wav"
        chunks_dir = wav_dir / "chunks"
        # Write float32 WAV chunks (as TTS server would produce)
        _write_float32_chunk(chunks_dir, "0001_Chapter_One", s1, samples=[0.1, 0.1])
        _write_float32_chunk(chunks_dir, "0001_Chapter_One", s2, samples=[0.2, 0.2])
        _write_float32_chunk(chunks_dir, "0001_Chapter_One", s3, samples=[0.3, 0.3])

        gen = _make_generator(str(tmp_path))

        # Let the real _merge_audio_files run (no mock) to test end-to-end
        result = gen._smart_chapter_list(str(wav_dir))

        assert result is not None, "Should return a chapter list"
        assert len(result) == 1
        out_path = result[0][0]
        assert Path(out_path).exists(), "Output WAV should exist"

        # The output should be readable as PCM WAV (float was converted)
        with wave.open(out_path, 'rb') as w:
            assert w.getsampwidth() == 2, "Should be 16-bit PCM after float conversion"
            total_frames = w.getnframes()
        # 3 chunks × 2 samples each = 6 total frames
        assert total_frames == 6, f"Expected 6 frames from 3 × 2-sample chunks, got {total_frames}"


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


def test_run_package_only_applies_title_and_cover_overrides():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        wav_dir = root / "wav"
        wav_dir.mkdir(parents=True)
        dummy_file = wav_dir / "0001_Chapter.wav"
        dummy_file.write_bytes(_FAKE_WAV)

        titles_file = root / "chapter_titles.txt"
        titles_file.write_text("Custom Chapter Title\n", encoding="utf-8")
        cover_file = root / "cover.jpg"
        cover_file.write_bytes(b"fake-jpeg-data")

        gen = _make_generator(str(root), chunked_audio=False)
        gen.config.chapter_titles_file = str(titles_file)
        gen.config.cover_image = str(cover_file)
        gen.config.input_file = None
        gen.config.m4b_filename = None
        gen.config.m4b_bitrate = "64k"
        gen.config.ffmpeg_path = "ffmpeg"

        packaged: list[dict] = []

        def fake_package(**kwargs):
            packaged.append(kwargs)
            return "out.m4b"

        with patch.object(gen, "_detect_audio_folder", return_value=str(wav_dir)), \
             patch("audiobook_generator.core.audiobook_generator.package_m4b", side_effect=fake_package):
            gen._run_package_only()

        assert len(packaged) == 1
        assert packaged[0]["chapter_titles"] == ["Custom Chapter Title"]
        assert packaged[0]["cover"] == (b"fake-jpeg-data", "image/jpeg")
