from __future__ import annotations

import array
import sys
import unittest

from audiobook_generator.core.chunked_audio_generator import _remove_start_click_from_pcm


def _pcm(samples: list[int]) -> bytes:
    data = array.array("h", samples)
    if sys.byteorder != "little":
        data.byteswap()
    return data.tobytes()


class ChunkStartDeclickTest(unittest.TestCase):
    def test_removes_detected_start_burst_and_fades_new_start(self) -> None:
        # 1 kHz makes 10 ms exactly 10 mono samples. The first samples model
        # CosyVoice's synthetic start click; useful speech begins shortly after.
        samples = [9000] * 5 + [0] * 5 + [6000] * 20
        cleaned = _remove_start_click_from_pcm(
            _pcm(samples),
            sampwidth=2,
            nchannels=1,
            framerate=1000,
            trim_ms=10,
            fade_ms=4,
        )
        out = array.array("h")
        out.frombytes(cleaned)
        if sys.byteorder != "little":
            out.byteswap()

        self.assertLess(len(out), len(samples))
        self.assertLessEqual(abs(out[0]), 100)
        self.assertEqual(out[-1], 6000)

    def test_leaves_quiet_start_unchanged(self) -> None:
        data = _pcm([100] * 10 + [6000] * 20)
        cleaned = _remove_start_click_from_pcm(
            data,
            sampwidth=2,
            nchannels=1,
            framerate=1000,
            trim_ms=10,
            fade_ms=4,
        )

        self.assertEqual(cleaned, data)


if __name__ == "__main__":
    unittest.main()
