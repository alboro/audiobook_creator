# Copyright (c) 2026 alboro <alboro@users.noreply.github.com>
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See: https://polyformproject.org/licenses/noncommercial/1.0.0/

import tempfile
from pathlib import Path
from unittest.mock import patch

from audiobook_generator.core.chunked_audio_generator import _sentence_hash, split_into_sentences


def test_audio_mode_reuses_existing_chunks():
    """Ensure chunked audio mode reuses existing chunk files (no TTS calls for already-synthesised sentences)."""
    from audiobook_generator.config.general_config import GeneralConfig
    from audiobook_generator.core.audiobook_generator import AudiobookGenerator

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        # create previous text run
        (out / "text" / "001").mkdir(parents=True)

        chapter_text = "alpha\n\nbeta"
        sentences = split_into_sentences(chapter_text, "ru")
        assert sentences == ["alpha", "beta"]

        args = type("A", (), {})()
        args.output_folder = tmp
        args.tts = "openai"
        args.language = "ru-RU"
        args.voice_name = "reference"
        args.model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
        args.worker_count = 1
        args.chapter_start = 1
        args.chapter_end = -1
        args.mode = "audio"
        args.normalize = False
        args.chunked_audio = True
        args.no_prompt = True
        args.normalize_log_changes = False

        config = GeneralConfig(args)
        gen = AudiobookGenerator(config)

        wav_dir = out / "wav"
        wav_dir.mkdir(parents=True, exist_ok=True)

        # Pre-create chunk files on disk — resume is now purely file-based.
        hashes = [_sentence_hash(s) for s in sentences]
        chapter_key = "0001_Test"
        chunk_dir = wav_dir / "chunks" / chapter_key
        chunk_dir.mkdir(parents=True, exist_ok=True)
        for s_hash in hashes:
            (chunk_dir / f"{s_hash}.wav").write_bytes(b"dummy")

        with patch("audiobook_generator.core.audiobook_generator.make_safe_filename", return_value=chapter_key):

            class DummyTTS:
                calls = 0

                def get_break_string(self):
                    return "\n\n"

                def estimate_cost(self, total_chars):
                    return 0.0

                def get_output_file_extension(self):
                    return "wav"

                def text_to_speech(self, text, out_path, tags):
                    self.calls += 1
                    raise RuntimeError("TTS should not be called when chunks already present")

                def prepare_tts_text(self, text):
                    return text

            class DummyParser:
                def get_chapters(self, break_str):
                    return [("Test", chapter_text)]

                def get_book_title(self):
                    return "Book"

                def get_book_author(self):
                    return "Author"

                def get_book_cover(self):
                    return None

            dummy_tts = DummyTTS()

            with patch("audiobook_generator.core.audiobook_generator.get_book_parser", return_value=DummyParser()), \
                 patch("audiobook_generator.core.audiobook_generator.get_tts_provider", return_value=dummy_tts):
                gen.run()

        # TTS must not be called — chunks existed on disk
        assert dummy_tts.calls == 0


if __name__ == "__main__":
    test_audio_mode_reuses_existing_chunks()

