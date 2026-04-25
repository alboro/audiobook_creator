# Copyright (c) 2026 alboro <alboro@users.noreply.github.com>
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See: https://polyformproject.org/licenses/noncommercial/1.0.0/

"""Tests for trailing-silence trimming in ChunkedAudioGenerator."""
import struct
import wave
from pathlib import Path

import pytest

from audiobook_generator.core.chunked_audio_generator import (
    SILENCE_TAIL_MS,
    _trim_trailing_silence,
)


def _make_wav(path: Path, samples: list[int], sample_rate: int = 16000) -> None:
    """Write a mono 16-bit PCM WAV file with the given sample values."""
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        data = struct.pack(f"<{len(samples)}h", *samples)
        wf.writeframes(data)


def _wav_duration_ms(path: Path) -> float:
    with wave.open(str(path), "r") as wf:
        return wf.getnframes() / wf.getframerate() * 1000


# ---------------------------------------------------------------------------
# Helpers to build test audio
# ---------------------------------------------------------------------------

def _speech_samples(duration_ms: int, sample_rate: int = 16000) -> list[int]:
    """'Speech' = loud sine-like signal at max amplitude."""
    n = int(sample_rate * duration_ms / 1000)
    return [16000 if i % 2 == 0 else -16000 for i in range(n)]


def _silence_samples(duration_ms: int, sample_rate: int = 16000) -> list[int]:
    """Silence = near-zero samples (well below -45 dBFS threshold)."""
    n = int(sample_rate * duration_ms / 1000)
    return [0] * n


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTrimTrailingSilence:
    def test_silence_is_trimmed(self, tmp_path):
        """A file with 2s speech + 2s silence should have trailing silence removed."""
        try:
            import pydub  # noqa: F401
        except ImportError:
            pytest.skip("pydub not installed")

        wav = tmp_path / "test.wav"
        samples = _speech_samples(2000) + _silence_samples(2000)
        _make_wav(wav, samples)

        original_ms = _wav_duration_ms(wav)
        assert original_ms == pytest.approx(4000, abs=50), f"Expected ~4000ms, got {original_ms}"

        _trim_trailing_silence(str(wav), tail_ms=SILENCE_TAIL_MS)

        trimmed_ms = _wav_duration_ms(wav)
        # After trim: should be speech(2000ms) + tail(200ms) = ~2200ms, well under 4000ms
        assert trimmed_ms < 3500, f"Expected silence to be trimmed, got {trimmed_ms}ms"
        assert trimmed_ms >= 2000, f"Speech content should be preserved, got {trimmed_ms}ms"

    def test_tail_is_preserved(self, tmp_path):
        """At least SILENCE_TAIL_MS of audio is kept after last speech."""
        try:
            import pydub  # noqa: F401
        except ImportError:
            pytest.skip("pydub not installed")

        wav = tmp_path / "tail.wav"
        samples = _speech_samples(1000) + _silence_samples(2000)
        _make_wav(wav, samples)

        _trim_trailing_silence(str(wav), tail_ms=SILENCE_TAIL_MS)

        trimmed_ms = _wav_duration_ms(wav)
        # Must be at least 1000ms (speech) + SILENCE_TAIL_MS (tail)
        assert trimmed_ms >= 1000 + SILENCE_TAIL_MS - 50, (
            f"Tail not preserved: expected ≥{1000 + SILENCE_TAIL_MS}ms, got {trimmed_ms}ms"
        )

    def test_no_silence_file_unchanged(self, tmp_path):
        """A file that is all speech should not be modified."""
        try:
            import pydub  # noqa: F401
        except ImportError:
            pytest.skip("pydub not installed")

        wav = tmp_path / "nosil.wav"
        samples = _speech_samples(1000)
        _make_wav(wav, samples)

        original_ms = _wav_duration_ms(wav)
        _trim_trailing_silence(str(wav), tail_ms=SILENCE_TAIL_MS)
        after_ms = _wav_duration_ms(wav)

        assert abs(after_ms - original_ms) < 50, (
            f"File should be unchanged when no trailing silence, was {original_ms}ms → {after_ms}ms"
        )

    def test_all_silence_file_unchanged(self, tmp_path):
        """A completely silent file should not crash and remain intact."""
        try:
            import pydub  # noqa: F401
        except ImportError:
            pytest.skip("pydub not installed")

        wav = tmp_path / "allsil.wav"
        samples = _silence_samples(500)
        _make_wav(wav, samples)

        original_ms = _wav_duration_ms(wav)
        _trim_trailing_silence(str(wav), tail_ms=SILENCE_TAIL_MS)  # must not crash
        after_ms = _wav_duration_ms(wav)

        assert abs(after_ms - original_ms) < 50, (
            "All-silent file should be left unchanged"
        )

    def test_missing_file_does_not_crash(self, tmp_path):
        """Non-existent file path should not raise."""
        _trim_trailing_silence(str(tmp_path / "nonexistent.wav"))  # must not raise


if __name__ == "__main__":
    import tempfile, os
    with tempfile.TemporaryDirectory() as d:
        p = Path(d)
        samples = _speech_samples(2000) + _silence_samples(2000)
        wav = p / "demo.wav"
        _make_wav(wav, samples)
        before = _wav_duration_ms(wav)
        _trim_trailing_silence(str(wav))
        after = _wav_duration_ms(wav)
        print(f"Before: {before:.0f}ms → After: {after:.0f}ms (trimmed {before-after:.0f}ms)")

