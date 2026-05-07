from __future__ import annotations

"""Tests for _gap_preamble_trim_frames and _remove_gap_preamble_from_pcm."""

import array
import sys
import unittest

from audiobook_generator.core.chunked_audio_generator import (
    _gap_preamble_trim_frames,
    _remove_gap_preamble_from_pcm,
    _GAP_PREAMBLE_MAX_MS,
    _GAP_PREAMBLE_MIN_GAP_MS,
    _GAP_PREAMBLE_MAX_TRIM_MS,
    _GAP_PREAMBLE_WIN_MS,
)

SR = 24000    # sample rate used throughout
NCHANNELS = 1
SAMPWIDTH = 2  # int16


def _pcm16(samples: list[float]) -> array.array:
    """Convert normalised float samples [-1, 1] to int16 array."""
    arr = array.array('h', [max(-32767, min(32767, int(v * 32767))) for v in samples])
    if sys.byteorder != 'little':
        arr.byteswap()
    return arr


def _sine_block(freq_hz: float, dur_ms: int, amplitude: float = 0.3) -> list[float]:
    """Generate *dur_ms* ms of a sine wave at *freq_hz*."""
    import math
    n = int(SR * dur_ms / 1000)
    return [amplitude * math.sin(2 * math.pi * freq_hz * i / SR) for i in range(n)]


def _noise_block(dur_ms: int, amplitude: float = 0.05) -> list[float]:
    """White noise at low amplitude."""
    import random
    n = int(SR * dur_ms / 1000)
    return [amplitude * (random.random() * 2 - 1) for _ in range(n)]


def _silence_block(dur_ms: int) -> list[float]:
    """Near-zero samples (below detection threshold)."""
    n = int(SR * dur_ms / 1000)
    return [1e-6] * n  # below _GAP_PREAMBLE_SILENCE_THRESH after normalisation


def _make_samples(*blocks: list[float]) -> array.array:
    """Concatenate float blocks and return normalised int16 array."""
    combined: list[float] = []
    for b in blocks:
        combined.extend(b)
    # Normalise so the loudest block drives abs_peak > 32767 * 0.005 threshold
    peak = max(abs(v) for v in combined) if combined else 1.0
    if peak < 1e-9:
        peak = 1.0
    norm = [v / peak for v in combined]
    return _pcm16(norm)


# ---------------------------------------------------------------------------
# _gap_preamble_trim_frames
# ---------------------------------------------------------------------------

class TestGapPreambleTrimFrames(unittest.TestCase):

    # ── No-trim cases ─────────────────────────────────────────────────────────

    def test_no_trim_for_pure_speech(self):
        """Continuous speech without a gap should return 0."""
        samples = _make_samples(
            _sine_block(300, 500, amplitude=0.4),
            _sine_block(200, 500, amplitude=0.3),
        )
        result = _gap_preamble_trim_frames(samples, NCHANNELS, SR)
        self.assertEqual(result, 0)

    def test_no_trim_when_starts_with_silence(self):
        """Chunk that STARTS with silence has no preamble."""
        samples = _make_samples(
            _silence_block(200),
            _sine_block(300, 400, amplitude=0.4),
        )
        result = _gap_preamble_trim_frames(samples, NCHANNELS, SR)
        self.assertEqual(result, 0)

    def test_no_trim_when_gap_too_short(self):
        """A gap shorter than MIN_GAP_MS should not trigger trimming."""
        # gap = 50 ms < 200 ms minimum
        samples = _make_samples(
            _sine_block(300, 200, amplitude=0.4),
            _silence_block(50),
            _sine_block(200, 400, amplitude=0.4),
        )
        result = _gap_preamble_trim_frames(samples, NCHANNELS, SR)
        self.assertEqual(result, 0)

    def test_no_trim_when_nothing_after_gap(self):
        """Content before gap but only silence after — don't trim."""
        samples = _make_samples(
            _sine_block(300, 200, amplitude=0.4),
            _silence_block(800),
        )
        result = _gap_preamble_trim_frames(samples, NCHANNELS, SR)
        self.assertEqual(result, 0)

    def test_no_trim_when_gap_starts_after_max_search(self):
        """Gap that begins after MAX_MS should not be detected as preamble."""
        # preamble = 800 ms > _GAP_PREAMBLE_MAX_MS (700 ms)
        samples = _make_samples(
            _sine_block(300, 800, amplitude=0.4),
            _silence_block(300),
            _sine_block(200, 400, amplitude=0.4),
        )
        result = _gap_preamble_trim_frames(samples, NCHANNELS, SR)
        self.assertEqual(result, 0)

    def test_no_trim_for_short_audio(self):
        """Very short audio clips should return 0 safely."""
        samples = _pcm16([0.1, 0.1, 0.1, 0.1])
        result = _gap_preamble_trim_frames(samples, NCHANNELS, SR)
        self.assertEqual(result, 0)

    def test_no_trim_for_silent_file(self):
        """Essentially silent file (abs_peak near zero) should return 0."""
        n = int(SR * 0.5)
        samples = array.array('h', [0] * n)
        result = _gap_preamble_trim_frames(samples, NCHANNELS, SR)
        self.assertEqual(result, 0)

    # ── Trim cases ────────────────────────────────────────────────────────────

    def test_detects_preamble_gap_speech(self):
        """Clear [preamble → gap → speech] pattern is detected."""
        preamble_ms = 400
        gap_ms = 350
        speech_ms = 500
        samples = _make_samples(
            _sine_block(300, preamble_ms, amplitude=0.4),
            _silence_block(gap_ms),
            _sine_block(200, speech_ms, amplitude=0.4),
        )
        result = _gap_preamble_trim_frames(samples, NCHANNELS, SR)

        preamble_frames = int(SR * preamble_ms / 1000)
        gap_frames = int(SR * gap_ms / 1000)
        expected_min = preamble_frames + gap_frames  # speech starts here

        self.assertGreater(result, 0, "should detect preamble")
        # Result should be close to speech onset (within a few windows)
        win_fr = int(SR * _GAP_PREAMBLE_WIN_MS / 1000)
        self.assertAlmostEqual(result, expected_min, delta=win_fr * 3)

    def test_trim_point_is_speech_onset_not_gap_start(self):
        """The trim point must be the START of speech, not the start of the gap."""
        preamble_ms = 200
        gap_ms = 300
        samples = _make_samples(
            _sine_block(300, preamble_ms, amplitude=0.4),
            _silence_block(gap_ms),
            _sine_block(200, 500, amplitude=0.4),
        )
        result = _gap_preamble_trim_frames(samples, NCHANNELS, SR)

        preamble_frames = int(SR * preamble_ms / 1000)
        self.assertGreater(result, preamble_frames,
                           "trim should be at speech onset, past the gap")

    def test_minimum_gap_boundary(self):
        """Gap exactly at MIN_GAP_MS should still trigger trim."""
        samples = _make_samples(
            _sine_block(300, 300, amplitude=0.4),
            _silence_block(_GAP_PREAMBLE_MIN_GAP_MS),   # exactly min
            _sine_block(200, 400, amplitude=0.4),
        )
        result = _gap_preamble_trim_frames(samples, NCHANNELS, SR)
        self.assertGreater(result, 0)

    def test_trim_does_not_exceed_max_trim_ms(self):
        """Even with a very long gap, trim must not exceed MAX_TRIM_MS."""
        # gap so long that speech_start > MAX_TRIM_MS
        very_long_gap_ms = _GAP_PREAMBLE_MAX_TRIM_MS + 200
        samples = _make_samples(
            _sine_block(300, 200, amplitude=0.4),
            _silence_block(very_long_gap_ms),
            _sine_block(200, 400, amplitude=0.4),
        )
        result = _gap_preamble_trim_frames(samples, NCHANNELS, SR)
        self.assertEqual(result, 0, "speech start beyond MAX_TRIM_MS must not trim")

    def test_multichannel_stereo(self):
        """Stereo (2-channel interleaved) input is handled correctly."""
        nch = 2
        preamble_ms = 300
        gap_ms = 350  # must exceed _GAP_PREAMBLE_MIN_GAP_MS (300 ms)
        speech_ms = 400
        # Interleave L and R channels (identical content for simplicity)
        mono_blocks = (
            _sine_block(300, preamble_ms, amplitude=0.4)
            + _silence_block(gap_ms)
            + _sine_block(200, speech_ms, amplitude=0.4)
        )
        # Peak-normalise
        peak = max(abs(v) for v in mono_blocks)
        norm = [v / peak for v in mono_blocks]
        interleaved = []
        for v in norm:
            interleaved.extend([v, v])  # L = R
        stereo = array.array('h', [max(-32767, min(32767, int(v * 32767))) for v in interleaved])

        result = _gap_preamble_trim_frames(stereo, nch, SR)
        self.assertGreater(result, 0)

    def test_short_preamble_long_gap_detected(self):
        """Short 100ms preamble followed by 400ms gap is still detected."""
        samples = _make_samples(
            _sine_block(300, 100, amplitude=0.4),
            _silence_block(400),
            _sine_block(200, 600, amplitude=0.4),
        )
        result = _gap_preamble_trim_frames(samples, NCHANNELS, SR)
        self.assertGreater(result, 0)

    def test_returns_integer(self):
        """Return value must always be an integer."""
        samples = _make_samples(
            _sine_block(300, 200, amplitude=0.4),
            _silence_block(300),
            _sine_block(200, 300, amplitude=0.4),
        )
        result = _gap_preamble_trim_frames(samples, NCHANNELS, SR)
        self.assertIsInstance(result, int)


# ---------------------------------------------------------------------------
# _remove_gap_preamble_from_pcm
# ---------------------------------------------------------------------------

class TestRemoveGapPreambleFromPcm(unittest.TestCase):

    def _make_pcm(self, *blocks: list[float]) -> bytes:
        arr = _make_samples(*blocks)
        if sys.byteorder != 'little':
            arr.byteswap()
        return arr.tobytes()

    def test_removes_preamble_from_bytes(self):
        preamble_ms = 400
        gap_ms = 300
        speech_ms = 600
        data = self._make_pcm(
            _sine_block(300, preamble_ms, amplitude=0.4),
            _silence_block(gap_ms),
            _sine_block(200, speech_ms, amplitude=0.4),
        )
        original_len = len(data)
        result = _remove_gap_preamble_from_pcm(data, SAMPWIDTH, NCHANNELS, SR, fade_ms=10)
        self.assertLess(len(result), original_len, "preamble bytes should be removed")

    def test_returns_unchanged_when_no_preamble(self):
        data = self._make_pcm(
            _sine_block(300, 800, amplitude=0.4),
        )
        result = _remove_gap_preamble_from_pcm(data, SAMPWIDTH, NCHANNELS, SR, fade_ms=10)
        self.assertEqual(result, data)

    def test_passthrough_for_non_int16(self):
        """Non-16-bit sampwidth returns data unchanged."""
        data = b'\x00' * 100
        result = _remove_gap_preamble_from_pcm(data, sampwidth=4, nchannels=1, framerate=SR, fade_ms=10)
        self.assertEqual(result, data)

    def test_passthrough_for_invalid_params(self):
        data = b'\x00\x01' * 10
        self.assertEqual(
            _remove_gap_preamble_from_pcm(data, 2, 0, SR, 10), data
        )
        self.assertEqual(
            _remove_gap_preamble_from_pcm(data, 2, 1, 0, 10), data
        )


if __name__ == "__main__":
    unittest.main()
