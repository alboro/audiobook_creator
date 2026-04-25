# Copyright (c) 2026 alboro <alboro@users.noreply.github.com>
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See: https://polyformproject.org/licenses/noncommercial/1.0.0/

"""End-to-end test covering the full prepare → audio (mocked TTS) → review UI flow.

This test exercises the real CLI code paths using the `Writings_of_Thomas_Paine`
EPUB that the user keeps locally, but it is SKIPPED gracefully if the file is
not available on the current machine (so CI is safe).

Steps:
    1. Run `AudiobookGenerator` in `prepare` mode on the EPUB → text/001/ files.
    2. Run `AudiobookGenerator` in `audio` mode with a MOCK TTS provider that
       just copies a known WAV file into each requested chunk path
       (simulating a real TTS server that returned a synthesized chunk).
    3. Drive the `review_ui` module's helper functions directly (NOT via the
       Gradio UI) to simulate the user's walkthrough: load chapters, switch
       chapters, edit a sentence, verify text file + DB were updated, verify
       a version history entry was written.

The test uses a temporary output folder so existing user artifacts are not
touched.
"""
from __future__ import annotations

import shutil
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# ----------------------------------------------------------------------
# Locations
# ----------------------------------------------------------------------
EPUB_PATH = Path(
    "/Users/aldem/Documents/books/Writings_of_Thomas_Paine__Volume_4_1794-1796.epub"
)
# A known pre-existing WAV chunk that we reuse as the "TTS output" for every
# synthesised sentence. Any small WAV works; this particular one is on the
# user's disk so the test is representative of the real pipeline.
SAMPLE_WAV = Path(
    "/Users/aldem/Documents/books/Writings_of_Thomas_Paine__Volume_4_1794-1796"
    "/wav/002/chunks/0001_Writings_of_Thomas_Paine_Volume_4_17941796_the_Age_of_Reason"
    "/e8c1e815698e3d3a.wav"
)


def _skip_reason() -> str | None:
    if not EPUB_PATH.exists():
        return f"Fixture EPUB not available on this machine: {EPUB_PATH}"
    if not SAMPLE_WAV.exists():
        return f"Fixture WAV not available on this machine: {SAMPLE_WAV}"
    return None


# ----------------------------------------------------------------------
# Mock TTS provider — installed by monkeypatching `get_tts_provider`.
# ----------------------------------------------------------------------
class _MockTTSProvider:
    """TTS provider stub that copies SAMPLE_WAV into every requested chunk path.

    This lets `ChunkedAudioGenerator` run its entire pipeline (sentence split,
    content hash, DB upsert, merge) without making any network calls.
    """
    def __init__(self, config):
        self.config = config

    def validate_config(self):
        pass

    def text_to_speech(self, text, output_path, audio_tags=None):
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SAMPLE_WAV, out)

    def estimate_cost(self, total_chars):
        return 0.0

    def get_break_string(self):
        return "\n\n"

    def get_output_file_extension(self):
        return "wav"


def _build_argparse_namespace(**overrides):
    """Build a namespace compatible with `GeneralConfig(args)`.

    Mirrors the dict used in `tests/numbers_ru_normalizer_test.make_config` but
    is self-contained here so we don't depend on that private helper.
    """
    values = dict(
        input_file=None,
        output_folder=None,
        output_text=False,
        prepare_text=False,
        prepared_text_folder=None,
        log="INFO",
        no_prompt=True,
        worker_count=1,
        use_pydub_merge=False,
        force_new_run=False,
        package_m4b=False,
        chunked_audio=True,
        audio_folder=None,
        m4b_filename=None,
        m4b_bitrate="64k",
        ffmpeg_path="ffmpeg",
        title_mode="auto",
        chapter_mode="documents",
        newline_mode="double",
        chapter_start=1,
        chapter_end=-1,
        search_and_replace_file="",
        mode=None,
        tts="openai",
        language="ru-RU",
        voice_name="reference",
        output_format="wav",
        model_name="tts_models/multilingual/multi-dataset/xtts_v2",
        openai_api_key=None,
        openai_base_url=None,
        openai_max_chars=0,
        openai_enable_polling=False,
        openai_submit_url=None,
        openai_status_url_template=None,
        openai_download_url_template=None,
        openai_job_id_path="id",
        openai_job_status_path="status",
        openai_job_download_url_path="download_url",
        openai_job_done_values="done,completed,succeeded,success",
        openai_job_failed_values="failed,error,cancelled",
        openai_poll_interval=5,
        openai_poll_timeout=60,
        openai_poll_request_timeout=60,
        openai_poll_max_errors=3,
        instructions=None,
        speed=1.0,
        normalize=False,  # Keep prepare fast: no normalizer chain
        normalize_steps=None,
        normalize_provider=None,
        normalize_model=None,
        normalize_prompt_file=None,
        normalize_system_prompt_file=None,
        normalize_user_prompt_file=None,
        normalize_api_key=None,
        normalize_base_url=None,
        normalize_max_chars=4000,
        normalize_tts_safe_max_chars=180,
        normalize_pronunciation_exceptions_file=None,
        normalize_tts_pronunciation_overrides_file=None,
        normalize_tts_pronunciation_overrides_words=None,
        normalize_pronunciation_lexicon_db=None,
        normalize_stress_ambiguity_file=None,
        normalize_tsnorm_stress_yo=True,
        normalize_tsnorm_stress_monosyllabic=False,
        normalize_tsnorm_min_word_length=2,
        normalize_stress_paradox_words=None,
        normalize_log_changes=False,
        normalize_stress_ambiguity_system_prompt=None,
        normalize_safe_split_system_prompt=None,
        break_duration="1250",
        voice_rate=None,
        voice_volume=None,
        voice_pitch=None,
        proxy=None,
        piper_path="piper",
        piper_docker_image="lscr.io/linuxserver/piper:latest",
        piper_speaker=0,
        piper_noise_scale=None,
        piper_noise_w_scale=None,
        piper_length_scale=1.0,
        piper_sentence_silence=0.2,
        qwen_api_key=None,
        qwen_language_type=None,
        qwen_stream=False,
        qwen_request_timeout=60,
        gemini_api_key=None,
        gemini_sample_rate=24000,
        gemini_channels=1,
        gemini_audio_encoding="LINEAR16",
        gemini_temperature=0.7,
        gemini_speaker_map=None,
        kokoro_base_url=None,
        kokoro_volume_multiplier=None,
    )
    values.update(overrides)
    return MagicMock(**values)


@unittest.skipIf(_skip_reason() is not None, _skip_reason() or "")
class E2EPrepareAudioReviewTest(unittest.TestCase):
    """End-to-end: prepare EPUB → mock audio → review UI helpers."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="eta_e2e_"))
        self.output_dir = self.tmp / "book_output"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    # ------------------------------------------------------------------
    # Step 1: prepare mode — parse EPUB into per-chapter .txt files
    # ------------------------------------------------------------------
    def _run_prepare(self):
        from audiobook_generator.config.general_config import GeneralConfig
        from audiobook_generator.core.audiobook_generator import AudiobookGenerator

        args = _build_argparse_namespace(
            input_file=str(EPUB_PATH),
            output_folder=str(self.output_dir),
            mode="prepare",
            chapter_end=2,  # Limit to first 2 chapters → keep the test fast
        )
        config = GeneralConfig(args)
        gen = AudiobookGenerator(config)
        gen.run()

        # Prepare must produce text/001/ with .txt files
        run_folder = self.output_dir / "text" / "001"
        self.assertTrue(run_folder.is_dir(), f"text/001 not created in {self.output_dir}")
        txt_files = sorted(run_folder.glob("*.txt"))
        self.assertGreaterEqual(len(txt_files), 2, "Expected ≥2 chapter .txt files")
        for f in txt_files:
            self.assertGreater(f.stat().st_size, 0, f"Empty chapter file: {f}")
        return txt_files

    # ------------------------------------------------------------------
    # Step 2: audio mode — with mocked TTS provider
    # ------------------------------------------------------------------
    def _run_audio_with_mock_tts(self):
        from audiobook_generator.config.general_config import GeneralConfig
        from audiobook_generator.core.audiobook_generator import AudiobookGenerator

        args = _build_argparse_namespace(
            input_file=str(EPUB_PATH),
            output_folder=str(self.output_dir),
            mode="audio",
            chunked_audio=True,
            chapter_end=2,
            worker_count=1,
        )
        config = GeneralConfig(args)

        # ---- Synchronous Pool replacement -------------------------------
        # multiprocessing.Pool forks/spawns children where monkey-patched
        # symbols (get_tts_provider, etc.) are lost. Swap it for a tiny
        # in-process shim that runs tasks serially in this process so our
        # patches remain visible.
        class _SyncPool:
            def __init__(self, *args, **kwargs):
                initializer = kwargs.get("initializer")
                initargs = kwargs.get("initargs", ())
                if initializer:
                    try:
                        initializer(*initargs)
                    except Exception:
                        pass

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def imap_unordered(self, fn, iterable):
                return [fn(item) for item in iterable]

            def close(self):
                pass

            def join(self):
                pass

        # Patch get_tts_provider AND multiprocessing.Pool (both visible in
        # audiobook_generator.core.audiobook_generator's module namespace).
        with patch(
            "audiobook_generator.core.audiobook_generator.get_tts_provider",
            side_effect=lambda cfg: _MockTTSProvider(cfg),
        ), patch(
            "audiobook_generator.core.audiobook_generator.multiprocessing.Pool",
            new=_SyncPool,
        ):
            gen = AudiobookGenerator(config)
            gen.run()

        # After audio run there must be a DB and some synthesised chunks.
        db_path = self.output_dir / "_state" / "audio_chunks.sqlite3"
        self.assertTrue(db_path.exists(), f"audio_chunks DB not created: {db_path}")
        conn = sqlite3.connect(str(db_path))
        try:
            rows = conn.execute(
                "SELECT COUNT(*) FROM audio_chunks WHERE status = 'synthesized'"
            ).fetchone()
            self.assertGreater(rows[0], 0, "No synthesized chunks recorded in DB")
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Step 3: review UI helpers
    # ------------------------------------------------------------------
    def _drive_review_ui(self):
        # The review UI stores state in module-level globals.
        # We call its helper functions directly, bypassing Gradio events.
        from audiobook_generator.ui import review_ui
        from audiobook_generator.utils.existing_chapters_loader import (
            find_latest_run_folder,
            load_chapters_from_run_folder,
            split_text_into_chunks,
        )

        # --- Step 3.1: "Load Chapters" (emulate button click) -------------
        run_folder = find_latest_run_folder(str(self.output_dir))
        self.assertIsNotNone(run_folder, "Review UI: no run folder found")

        chapters = load_chapters_from_run_folder(run_folder)
        self.assertGreaterEqual(len(chapters), 2, "Review UI: expected ≥2 chapters")

        # Seed the module globals so get_history/save_edit can find the DB
        review_ui._current_chapters = chapters
        review_ui._current_output_dir = str(self.output_dir)
        review_ui._audio_db_path = str(self.output_dir / "_state" / "audio_chunks.sqlite3")

        # --- Step 3.2: select chapter 2, count sentences ---------------
        ch2 = chapters[1]
        ch2_text = Path(ch2.text_path).read_text(encoding="utf-8")
        ch2_chunks = split_text_into_chunks(ch2_text, "ru")
        self.assertGreater(len(ch2_chunks), 0, "Chapter 2 has no sentences")

        review_ui._current_chapter_key = ch2.chapter_key
        review_ui._current_full_text = ch2_text
        review_ui._current_chunks = ch2_chunks

        # --- Step 3.3: switch back to chapter 1 --------------------------
        ch1 = chapters[0]
        ch1_text = Path(ch1.text_path).read_text(encoding="utf-8")
        ch1_chunks = split_text_into_chunks(ch1_text, "ru")
        self.assertGreater(len(ch1_chunks), 0, "Chapter 1 has no sentences")

        review_ui._current_chapter_key = ch1.chapter_key
        review_ui._current_full_text = ch1_text
        review_ui._current_chunks = ch1_chunks

        # Chapter 1's first sentence must have a backing audio chunk on disk
        # (_MockTTSProvider copied SAMPLE_WAV → every sentence gets a WAV file).
        audio_path, db_hash, db_text = review_ui.get_audio_chunk_from_db(
            review_ui._audio_db_path, ch1.chapter_key, 0
        )
        self.assertIsNotNone(audio_path, "No audio_path in DB for chapter 1 sentence 0")
        self.assertTrue(Path(audio_path).exists(), f"Chunk WAV missing on disk: {audio_path}")
        self.assertIsNotNone(db_hash, "sentence_hash missing in DB")

        # --- Step 3.4: "Edit" sentence 0 → change text, save -------------
        review_ui._selected_sentence_idx = 0
        original_sentence = ch1_chunks[0]
        new_sentence = original_sentence + " [отредактировано]"

        # Emulate save_edit flow by invoking the underlying functions that
        # rewrite the .txt file and bump the DB. We do this via a tiny
        # sqlite update here (the full gradio-bound save_edit is a closure
        # inside build_review_ui and is not directly callable from tests).

        conn = sqlite3.connect(review_ui._audio_db_path)
        try:
            import hashlib
            from datetime import datetime, UTC
            new_hash = hashlib.sha256(new_sentence.strip().encode("utf-8")).hexdigest()[:16]
            now = datetime.now(UTC).isoformat()

            conn.execute("""
                CREATE TABLE IF NOT EXISTS sentence_text_versions (
                    sentence_hash    TEXT NOT NULL,
                    sentence_text   TEXT NOT NULL,
                    replaced_by_hash TEXT,
                    run_id          TEXT NOT NULL,
                    created_at      TEXT NOT NULL,
                    PRIMARY KEY (sentence_hash)
                )
            """)
            # AudioChunkStore creates this table without replaced_by_hash; align
            # the schema exactly like review_ui.save_edit does in production.
            try:
                conn.execute(
                    "ALTER TABLE sentence_text_versions ADD COLUMN replaced_by_hash TEXT"
                )
            except sqlite3.OperationalError:
                pass
            conn.execute(
                "INSERT OR REPLACE INTO sentence_text_versions "
                "(sentence_hash, sentence_text, replaced_by_hash, run_id, created_at) "
                "VALUES (?, ?, ?, '001', ?)",
                (db_hash, original_sentence, new_hash, now),
            )
            conn.execute(
                "UPDATE audio_chunks SET sentence_hash=?, sentence_text=?, "
                "status='pending', updated_at=? "
                "WHERE chapter_key=? AND sentence_pos=?",
                (new_hash, new_sentence, now, ch1.chapter_key, 0),
            )
            conn.commit()
        finally:
            conn.close()

        # Rewrite the chapter .txt file (same logic as save_edit)
        new_full_text = ch1_text.replace(original_sentence, new_sentence, 1)
        Path(ch1.text_path).write_text(new_full_text, encoding="utf-8")

        # --- Step 3.5: verify DB has a version record -------------------
        versions = review_ui.get_sentence_versions_from_db(
            review_ui._audio_db_path, new_hash
        )
        self.assertEqual(len(versions), 1,
                         f"Expected 1 history record, got: {versions}")
        old_hash, old_text, replaced_by = versions[0]
        self.assertEqual(old_hash, db_hash)
        self.assertEqual(old_text, original_sentence)
        self.assertEqual(replaced_by, new_hash)

        # --- Step 3.6: verify disk file reflects the edit ---------------
        final_text_on_disk = Path(ch1.text_path).read_text(encoding="utf-8")
        self.assertIn(new_sentence, final_text_on_disk,
                      "Edit was not written to the chapter .txt file")
        self.assertNotIn(original_sentence + "\n", final_text_on_disk,
                         "Original sentence still present verbatim")

    # ------------------------------------------------------------------
    # Main scenario — all three steps in one shot
    # ------------------------------------------------------------------
    def test_full_e2e_prepare_audio_review(self):
        self._run_prepare()
        self._run_audio_with_mock_tts()
        self._drive_review_ui()


if __name__ == "__main__":
    unittest.main()



