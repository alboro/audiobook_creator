from __future__ import annotations

import os
import struct
import tempfile
import unittest
from unittest.mock import MagicMock, call, patch

from audiobook_generator.core.chunked_audio_generator import (
    _atempo_filter,
    _prepare_chunks_with_tempo,
)


# ---------------------------------------------------------------------------
# _atempo_filter
# ---------------------------------------------------------------------------

class TestAtempoFilter(unittest.TestCase):
    """Unit tests for the ffmpeg atempo filter-chain builder."""

    def test_identity_returns_empty(self):
        self.assertEqual(_atempo_filter(1.0), "")

    def test_near_identity_returns_empty(self):
        self.assertEqual(_atempo_filter(1.0 + 1e-8), "")

    def test_simple_speedup(self):
        result = _atempo_filter(1.5)
        self.assertEqual(result, "atempo=1.500000")

    def test_simple_slowdown(self):
        result = _atempo_filter(0.8)
        self.assertEqual(result, "atempo=0.800000")

    def test_boundary_fast(self):
        # Exactly 2.0 — single stage
        result = _atempo_filter(2.0)
        self.assertIn("atempo=", result)
        self.assertNotIn(",", result)

    def test_boundary_slow(self):
        # Exactly 0.5 — single stage
        result = _atempo_filter(0.5)
        self.assertIn("atempo=", result)
        self.assertNotIn(",", result)

    def test_over_2_chains(self):
        # 3.0 → two stages
        result = _atempo_filter(3.0)
        stages = result.split(",")
        self.assertEqual(len(stages), 2)
        self.assertEqual(stages[0], "atempo=2.0")
        # second stage: 3.0 / 2.0 = 1.5
        self.assertIn("atempo=1.5", stages[1])

    def test_over_4_chains_three_stages(self):
        # 5.0 → three stages: 2.0, 2.0, 1.25
        result = _atempo_filter(5.0)
        stages = result.split(",")
        self.assertEqual(len(stages), 3)
        self.assertEqual(stages[0], "atempo=2.0")
        self.assertEqual(stages[1], "atempo=2.0")
        self.assertIn("atempo=1.25", stages[2])

    def test_under_0_5_chains(self):
        # 0.25 → two stages: 0.5, 0.5
        result = _atempo_filter(0.25)
        stages = result.split(",")
        self.assertEqual(len(stages), 2)
        # Both stages are atempo=0.5 (may be formatted as "0.500000")
        for s in stages:
            val = float(s.split("=")[1])
            self.assertAlmostEqual(val, 0.5)

    def test_under_0_25_chains_three_stages(self):
        # 0.125 → three stages: 0.5, 0.5, 0.5
        result = _atempo_filter(0.125)
        stages = result.split(",")
        self.assertEqual(len(stages), 3)
        for s in stages:
            val = float(s.split("=")[1])
            self.assertAlmostEqual(val, 0.5)

    def test_result_is_string(self):
        self.assertIsInstance(_atempo_filter(1.3), str)


# ---------------------------------------------------------------------------
# _prepare_chunks_with_tempo
# ---------------------------------------------------------------------------

def _make_dummy_wav(path: str, duration_samples: int = 100, sample_rate: int = 22050) -> None:
    """Write a minimal PCM WAV file (mono, 16-bit) to *path*."""
    num_channels = 1
    bits_per_sample = 16
    block_align = num_channels * bits_per_sample // 8
    byte_rate = sample_rate * block_align
    data_size = duration_samples * block_align
    with open(path, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + data_size))  # chunk size
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<I", 16))            # sub-chunk size
        f.write(struct.pack("<H", 1))             # PCM
        f.write(struct.pack("<H", num_channels))
        f.write(struct.pack("<I", sample_rate))
        f.write(struct.pack("<I", byte_rate))
        f.write(struct.pack("<H", block_align))
        f.write(struct.pack("<H", bits_per_sample))
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        f.write(b"\x00" * data_size)


class TestPrepareChunksWithTempo(unittest.TestCase):
    """Tests for _prepare_chunks_with_tempo."""

    def setUp(self):
        self._tmp = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _make_chunk(self, name: str) -> str:
        path = os.path.join(self._tmp, name)
        _make_dummy_wav(path)
        return path

    # ── No-op cases ──────────────────────────────────────────────────────────

    def test_no_tempo_configured_returns_original_list(self):
        cp1 = self._make_chunk("a.wav")
        cp2 = self._make_chunk("b.wav")
        voices_config = {"voice_a": {}, "voice_b": {}}
        result = _prepare_chunks_with_tempo(
            [cp1, cp2],
            {cp1: "voice_a", cp2: "voice_b"},
            voices_config,
            primary_voice="voice_a",
        )
        self.assertEqual(result.__class__, list)
        self.assertEqual(result, [cp1, cp2])

    def test_tempo_1_0_returns_original_list(self):
        cp = self._make_chunk("c.wav")
        result = _prepare_chunks_with_tempo(
            [cp],
            {cp: "voice_a"},
            {"voice_a": {"audio_tempo": 1.0}},
            primary_voice="voice_a",
        )
        self.assertEqual(result, [cp])

    def test_empty_chunk_list(self):
        result = _prepare_chunks_with_tempo(
            [],
            {},
            {"voice_a": {"audio_tempo": 1.2}},
            primary_voice="voice_a",
        )
        self.assertEqual(result, [])

    # ── Tempo applied ─────────────────────────────────────────────────────────

    @patch("audiobook_generator.core.chunked_audio_generator._apply_audio_tempo")
    def test_tempo_applied_to_matching_voice(self, mock_apply):
        cp1 = self._make_chunk("p.wav")
        cp2 = self._make_chunk("q.wav")
        voices_config = {"voice_primary": {}, "voice_secondary": {"audio_tempo": 1.2}}
        chunk_voice_map = {cp1: None, cp2: "voice_secondary"}

        with tempfile.TemporaryDirectory() as td:
            result = _prepare_chunks_with_tempo(
                [cp1, cp2],
                chunk_voice_map,
                voices_config,
                primary_voice="voice_primary",
                tmp_dir=td,
            )

        # cp1 (primary, no tempo) stays unchanged
        self.assertEqual(result[0], cp1)
        # cp2 (voice_secondary, tempo=1.2) was processed
        self.assertNotEqual(result[1], cp2)
        self.assertTrue(result[1].endswith(".wav"))
        # _apply_audio_tempo called once for cp2
        mock_apply.assert_called_once()
        args = mock_apply.call_args[0]
        self.assertEqual(args[0], cp2)   # src
        self.assertAlmostEqual(args[2], 1.2)  # tempo

    @patch("audiobook_generator.core.chunked_audio_generator._apply_audio_tempo")
    def test_primary_voice_tempo_via_none_key(self, mock_apply):
        """Chunk with voice=None maps to primary_voice for tempo lookup."""
        cp = self._make_chunk("r.wav")
        voices_config = {"voice_primary": {"audio_tempo": 1.3}}
        chunk_voice_map = {cp: None}  # None → primary

        with tempfile.TemporaryDirectory() as td:
            result = _prepare_chunks_with_tempo(
                [cp],
                chunk_voice_map,
                voices_config,
                primary_voice="voice_primary",
                tmp_dir=td,
            )

        self.assertNotEqual(result[0], cp)
        mock_apply.assert_called_once()
        args = mock_apply.call_args[0]
        self.assertAlmostEqual(args[2], 1.3)

    @patch("audiobook_generator.core.chunked_audio_generator._apply_audio_tempo")
    def test_multiple_chunks_same_voice_all_processed(self, mock_apply):
        chunks = [self._make_chunk(f"s{i}.wav") for i in range(3)]
        voices_config = {"voice_a": {"audio_tempo": 0.9}}
        chunk_voice_map = {cp: "voice_a" for cp in chunks}

        with tempfile.TemporaryDirectory() as td:
            result = _prepare_chunks_with_tempo(
                chunks,
                chunk_voice_map,
                voices_config,
                primary_voice="voice_a",
                tmp_dir=td,
            )

        self.assertEqual(mock_apply.call_count, 3)
        # All result paths differ from originals
        for orig, res in zip(chunks, result):
            self.assertNotEqual(orig, res)

    @patch("audiobook_generator.core.chunked_audio_generator._apply_audio_tempo")
    def test_tmp_dir_none_skips_tempo(self, mock_apply):
        """tmp_dir=None → no processing, original paths returned."""
        cp = self._make_chunk("t.wav")
        voices_config = {"voice_a": {"audio_tempo": 1.5}}
        result = _prepare_chunks_with_tempo(
            [cp],
            {cp: "voice_a"},
            voices_config,
            primary_voice="voice_a",
            tmp_dir=None,
        )
        mock_apply.assert_not_called()
        self.assertEqual(result, [cp])

    @patch("audiobook_generator.core.chunked_audio_generator._apply_audio_tempo")
    def test_output_filenames_are_unique(self, mock_apply):
        """Each processed chunk gets a unique temp filename."""
        chunks = [self._make_chunk(f"u{i}.wav") for i in range(4)]
        voices_config = {"v": {"audio_tempo": 1.1}}
        chunk_voice_map = {cp: "v" for cp in chunks}

        with tempfile.TemporaryDirectory() as td:
            result = _prepare_chunks_with_tempo(
                chunks, chunk_voice_map, voices_config, primary_voice="v", tmp_dir=td,
            )

        self.assertEqual(len(set(result)), len(result))  # all unique

    @patch("audiobook_generator.core.chunked_audio_generator._apply_audio_tempo")
    def test_mixed_voices_only_matching_processed(self, mock_apply):
        cp_pri = self._make_chunk("x1.wav")
        cp_sec = self._make_chunk("x2.wav")
        cp_pri2 = self._make_chunk("x3.wav")
        voices_config = {
            "voice_primary": {},
            "voice_secondary": {"audio_tempo": 1.4},
        }
        chunk_voice_map = {cp_pri: None, cp_sec: "voice_secondary", cp_pri2: None}

        with tempfile.TemporaryDirectory() as td:
            result = _prepare_chunks_with_tempo(
                [cp_pri, cp_sec, cp_pri2],
                chunk_voice_map,
                voices_config,
                primary_voice="voice_primary",
                tmp_dir=td,
            )

        # Only cp_sec was processed
        self.assertEqual(result[0], cp_pri)
        self.assertNotEqual(result[1], cp_sec)
        self.assertEqual(result[2], cp_pri2)
        mock_apply.assert_called_once()


if __name__ == "__main__":
    unittest.main()
