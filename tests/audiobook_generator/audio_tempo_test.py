from __future__ import annotations

import os
import struct
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from audiobook_generator.core.chunked_audio_generator import (
    _atempo_filter,
    _merge_audio_files,
    _prepare_chunks_with_tempo,
    _voices_need_tempo,
)


# ---------------------------------------------------------------------------
# Helper: write a minimal valid PCM WAV file
# ---------------------------------------------------------------------------

def _make_dummy_wav(path: str, duration_samples: int = 100, sample_rate: int = 22050) -> None:
    num_channels = 1
    bits_per_sample = 16
    block_align = num_channels * bits_per_sample // 8
    byte_rate = sample_rate * block_align
    data_size = duration_samples * block_align
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + data_size))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<I", 16))
        f.write(struct.pack("<H", 1))
        f.write(struct.pack("<H", num_channels))
        f.write(struct.pack("<I", sample_rate))
        f.write(struct.pack("<I", byte_rate))
        f.write(struct.pack("<H", block_align))
        f.write(struct.pack("<H", bits_per_sample))
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        f.write(b"\x00" * data_size)


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
        self.assertEqual(_atempo_filter(1.5), "atempo=1.500000")

    def test_simple_slowdown(self):
        self.assertEqual(_atempo_filter(0.8), "atempo=0.800000")

    def test_boundary_fast(self):
        result = _atempo_filter(2.0)
        self.assertIn("atempo=", result)
        self.assertNotIn(",", result)

    def test_boundary_slow(self):
        result = _atempo_filter(0.5)
        self.assertIn("atempo=", result)
        self.assertNotIn(",", result)

    def test_over_2_chains(self):
        stages = _atempo_filter(3.0).split(",")
        self.assertEqual(len(stages), 2)
        self.assertEqual(stages[0], "atempo=2.0")
        self.assertIn("atempo=1.5", stages[1])

    def test_over_4_chains_three_stages(self):
        stages = _atempo_filter(5.0).split(",")
        self.assertEqual(len(stages), 3)
        self.assertEqual(stages[0], "atempo=2.0")
        self.assertEqual(stages[1], "atempo=2.0")
        self.assertIn("atempo=1.25", stages[2])

    def test_under_0_5_chains(self):
        stages = _atempo_filter(0.25).split(",")
        self.assertEqual(len(stages), 2)
        for s in stages:
            self.assertAlmostEqual(float(s.split("=")[1]), 0.5)

    def test_under_0_25_chains_three_stages(self):
        stages = _atempo_filter(0.125).split(",")
        self.assertEqual(len(stages), 3)
        for s in stages:
            self.assertAlmostEqual(float(s.split("=")[1]), 0.5)


# ---------------------------------------------------------------------------
# _voices_need_tempo
# ---------------------------------------------------------------------------

class TestVoicesNeedTempo(unittest.TestCase):
    def test_empty_or_none(self):
        self.assertFalse(_voices_need_tempo(None))
        self.assertFalse(_voices_need_tempo({}))

    def test_no_audio_tempo_keys(self):
        self.assertFalse(_voices_need_tempo({"v1": {}, "v2": {"speed": 1.2}}))

    def test_only_identity_tempos(self):
        self.assertFalse(_voices_need_tempo({"v1": {"audio_tempo": 1.0}}))

    def test_one_voice_has_tempo(self):
        self.assertTrue(_voices_need_tempo({"v1": {}, "v2": {"audio_tempo": 1.25}}))

    def test_slowdown(self):
        self.assertTrue(_voices_need_tempo({"v1": {"audio_tempo": 0.85}}))


# ---------------------------------------------------------------------------
# _prepare_chunks_with_tempo
# ---------------------------------------------------------------------------

class TestPrepareChunksWithTempo(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _make_chunk(self, name: str) -> str:
        path = os.path.join(self._tmp, name)
        _make_dummy_wav(path)
        return path

    def test_no_tempo_in_voices_returns_original(self):
        cp1 = self._make_chunk("a.wav")
        cp2 = self._make_chunk("b.wav")
        result = _prepare_chunks_with_tempo(
            [cp1, cp2],
            {cp1: "v1", cp2: "v2"},
            {"v1": {}, "v2": {}},
            primary_voice="v1",
            tmp_dir=self._tmp,
        )
        self.assertEqual(result, [cp1, cp2])

    def test_tempo_1_0_returns_original(self):
        cp = self._make_chunk("c.wav")
        result = _prepare_chunks_with_tempo(
            [cp], {cp: "v"}, {"v": {"audio_tempo": 1.0}},
            primary_voice="v", tmp_dir=self._tmp,
        )
        self.assertEqual(result, [cp])

    def test_empty_chunk_list(self):
        self.assertEqual(
            _prepare_chunks_with_tempo([], {}, {"v": {"audio_tempo": 1.2}}, primary_voice="v"),
            [],
        )

    def test_tmp_dir_none_returns_original(self):
        cp = self._make_chunk("d.wav")
        result = _prepare_chunks_with_tempo(
            [cp], {cp: "v"}, {"v": {"audio_tempo": 1.5}},
            primary_voice="v", tmp_dir=None,
        )
        self.assertEqual(result, [cp])

    @patch("audiobook_generator.core.chunked_audio_generator._apply_audio_tempo")
    def test_only_matching_voice_processed(self, mock_apply):
        cp_pri = self._make_chunk("p.wav")
        cp_sec = self._make_chunk("q.wav")
        with tempfile.TemporaryDirectory() as td:
            result = _prepare_chunks_with_tempo(
                [cp_pri, cp_sec],
                {cp_pri: None, cp_sec: "v_sec"},      # cp_pri uses primary
                {"v_pri": {}, "v_sec": {"audio_tempo": 1.2}},
                primary_voice="v_pri",
                tmp_dir=td,
            )
        # cp_pri unchanged (primary has no tempo)
        self.assertEqual(result[0], cp_pri)
        # cp_sec processed (secondary has tempo=1.2)
        self.assertNotEqual(result[1], cp_sec)
        self.assertTrue(result[1].endswith(".wav"))
        mock_apply.assert_called_once()
        args = mock_apply.call_args[0]
        self.assertEqual(args[0], cp_sec)
        self.assertAlmostEqual(args[2], 1.2)

    @patch("audiobook_generator.core.chunked_audio_generator._apply_audio_tempo")
    def test_none_voice_resolves_to_primary(self, mock_apply):
        cp = self._make_chunk("r.wav")
        with tempfile.TemporaryDirectory() as td:
            result = _prepare_chunks_with_tempo(
                [cp], {cp: None},
                {"v_pri": {"audio_tempo": 1.3}},
                primary_voice="v_pri", tmp_dir=td,
            )
        self.assertNotEqual(result[0], cp)
        mock_apply.assert_called_once()
        self.assertAlmostEqual(mock_apply.call_args[0][2], 1.3)

    @patch("audiobook_generator.core.chunked_audio_generator._apply_audio_tempo")
    def test_multiple_chunks_same_voice_all_processed(self, mock_apply):
        chunks = [self._make_chunk(f"s{i}.wav") for i in range(3)]
        with tempfile.TemporaryDirectory() as td:
            result = _prepare_chunks_with_tempo(
                chunks,
                {cp: "v" for cp in chunks},
                {"v": {"audio_tempo": 0.9}},
                primary_voice="v", tmp_dir=td,
            )
        self.assertEqual(mock_apply.call_count, 3)
        for orig, res in zip(chunks, result):
            self.assertNotEqual(orig, res)

    @patch("audiobook_generator.core.chunked_audio_generator._apply_audio_tempo")
    def test_per_chunk_different_tempos(self, mock_apply):
        """Two voices with two different audio_tempo values — each chunk gets its own."""
        cp1 = self._make_chunk("t1.wav")
        cp2 = self._make_chunk("t2.wav")
        with tempfile.TemporaryDirectory() as td:
            _prepare_chunks_with_tempo(
                [cp1, cp2],
                {cp1: "primary", cp2: "secondary"},
                {"primary": {"audio_tempo": 1.25}, "secondary": {"audio_tempo": 0.95}},
                primary_voice="primary", tmp_dir=td,
            )
        self.assertEqual(mock_apply.call_count, 2)
        tempos_used = sorted(c.args[2] for c in mock_apply.call_args_list)
        self.assertAlmostEqual(tempos_used[0], 0.95)
        self.assertAlmostEqual(tempos_used[1], 1.25)


# ---------------------------------------------------------------------------
# _merge_audio_files: per-chunk tempo integration
# ---------------------------------------------------------------------------

class TestMergeAudioFilesTempo(unittest.TestCase):
    """Integration of per-chunk audio_tempo into the merge dispatcher."""

    def setUp(self):
        self._tmp = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _make_chunk(self, name: str) -> str:
        path = os.path.join(self._tmp, name)
        _make_dummy_wav(path)
        return path

    @patch("audiobook_generator.core.chunked_audio_generator._merge_wav_files")
    @patch("audiobook_generator.core.chunked_audio_generator._apply_audio_tempo")
    def test_tempo_applied_per_chunk_then_merged(self, mock_apply, mock_merge):
        cp1 = self._make_chunk("a.wav")
        cp2 = self._make_chunk("b.wav")
        out = os.path.join(self._tmp, "ch.wav")

        # Side-effect: emulate ffmpeg by creating a real WAV at dst.
        def fake_apply(src, dst, tempo, ffmpeg_path="ffmpeg"):
            _make_dummy_wav(dst)

        mock_apply.side_effect = fake_apply

        _merge_audio_files(
            [cp1, cp2],
            out,
            voices_config={"v_pri": {}, "v_sec": {"audio_tempo": 1.25}},
            chunk_voice_map={cp1: None, cp2: "v_sec"},
            primary_voice="v_pri",
        )

        mock_apply.assert_called_once()
        # cp2 was the one with v_sec tempo
        self.assertEqual(mock_apply.call_args[0][0], cp2)
        self.assertAlmostEqual(mock_apply.call_args[0][2], 1.25)
        # _merge_wav_files received: cp1 unchanged + processed cp2 (NOT cp2 itself)
        merged_paths = mock_merge.call_args[0][0]
        self.assertEqual(merged_paths[0], cp1)
        self.assertNotEqual(merged_paths[1], cp2)

    @patch("audiobook_generator.core.chunked_audio_generator._merge_wav_files")
    @patch("audiobook_generator.core.chunked_audio_generator._apply_audio_tempo")
    def test_no_tempo_passes_paths_unchanged(self, mock_apply, mock_merge):
        cp = self._make_chunk("c.wav")
        out = os.path.join(self._tmp, "ch2.wav")
        _merge_audio_files(
            [cp], out,
            voices_config={"v": {}},
            chunk_voice_map={cp: "v"},
            primary_voice="v",
        )
        mock_apply.assert_not_called()
        merged_paths = mock_merge.call_args[0][0]
        self.assertEqual(list(merged_paths), [cp])

    @patch("audiobook_generator.core.chunked_audio_generator._merge_wav_files")
    @patch("audiobook_generator.core.chunked_audio_generator._apply_audio_tempo")
    def test_missing_voices_config_skips_tempo(self, mock_apply, mock_merge):
        cp = self._make_chunk("d.wav")
        out = os.path.join(self._tmp, "ch3.wav")
        _merge_audio_files([cp], out)  # no kw args at all
        mock_apply.assert_not_called()
        self.assertEqual(list(mock_merge.call_args[0][0]), [cp])

    @patch("audiobook_generator.core.chunked_audio_generator._merge_wav_files")
    @patch("audiobook_generator.core.chunked_audio_generator._apply_audio_tempo")
    def test_temp_dir_cleaned_up_after_merge(self, mock_apply, mock_merge):
        """After _merge_audio_files returns, the temp dir with stretched files is gone."""
        cp = self._make_chunk("e.wav")
        out = os.path.join(self._tmp, "ch4.wav")

        recorded = {}

        def fake_apply(src, dst, tempo, ffmpeg_path="ffmpeg"):
            _make_dummy_wav(dst)
            recorded["dst"] = dst

        mock_apply.side_effect = fake_apply

        _merge_audio_files(
            [cp], out,
            voices_config={"v": {"audio_tempo": 1.2}},
            chunk_voice_map={cp: "v"},
            primary_voice="v",
        )
        self.assertIn("dst", recorded)
        # Temp dir & file should be gone now.
        self.assertFalse(os.path.exists(recorded["dst"]))
        self.assertFalse(os.path.exists(os.path.dirname(recorded["dst"])))

    @patch(
        "audiobook_generator.core.chunked_audio_generator._merge_wav_files",
        side_effect=RuntimeError("merge boom"),
    )
    @patch("audiobook_generator.core.chunked_audio_generator._apply_audio_tempo")
    def test_temp_dir_cleaned_up_on_merge_failure(self, mock_apply, mock_merge):
        cp = self._make_chunk("f.wav")
        out = os.path.join(self._tmp, "ch5.wav")
        recorded = {}

        def fake_apply(src, dst, tempo, ffmpeg_path="ffmpeg"):
            _make_dummy_wav(dst)
            recorded["dst"] = dst

        mock_apply.side_effect = fake_apply

        with self.assertRaises(RuntimeError):
            _merge_audio_files(
                [cp], out,
                voices_config={"v": {"audio_tempo": 1.2}},
                chunk_voice_map={cp: "v"},
                primary_voice="v",
            )
        self.assertFalse(os.path.exists(recorded["dst"]))
        self.assertFalse(os.path.exists(os.path.dirname(recorded["dst"])))


if __name__ == "__main__":
    unittest.main()
