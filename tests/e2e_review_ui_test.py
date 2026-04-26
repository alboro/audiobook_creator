# Copyright (c) 2026 alboro <alboro@users.noreply.github.com>
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See: https://polyformproject.org/licenses/noncommercial/1.0.0/

"""End-to-end test covering the full prepare → audio (mocked TTS) → review UI flow.

Uses the Robinson Crusoe EPUB bundled with the project (examples/) and a
programmatically-generated silent WAV as the "TTS output" — no external files
required.  The test can run on any machine with the project checked out.

Steps:
    1. Run `AudiobookGenerator` in `prepare` mode on the EPUB → text/001/ files.
    2. Run `AudiobookGenerator` in `audio` mode with a MOCK TTS provider that
       writes a minimal silent WAV into each requested chunk path, simulating a
       real TTS server without any network calls.
    3. Drive the FastAPI `review_server` via TestClient to simulate the user's
       walkthrough: load chapters, inspect chunks, verify audio exists, edit a
       sentence, verify text file + DB were updated, verify a version history
       entry was written.

The test uses a temporary output folder so existing user artefacts are never
touched.
"""
from __future__ import annotations

import wave
import shutil
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# ----------------------------------------------------------------------
# Locations — all inside the project tree (no external dependencies)
# ----------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).parent.parent
EPUB_PATH = _PROJECT_ROOT / "examples" / "The_Life_and_Adventures_of_Robinson_Crusoe.epub"


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _make_dummy_wav(output_path: Path) -> None:
    """Write a minimal valid PCM WAV (100 ms of silence) to *output_path*."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample_rate = 22050
    n_frames = sample_rate // 10  # 100 ms
    with wave.open(str(output_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)          # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * n_frames)


# ----------------------------------------------------------------------
# Mock TTS provider — installed by monkeypatching `get_tts_provider`.
# ----------------------------------------------------------------------
class _MockTTSProvider:
    """TTS provider stub that writes a silent WAV into every requested chunk path.

    This lets `ChunkedAudioGenerator` run its entire pipeline (sentence split,
    content hash, DB upsert, merge) without making any network calls.
    """
    def __init__(self, config):
        self.config = config

    def validate_config(self):
        pass

    def text_to_speech(self, text, output_path, audio_tags=None):
        _make_dummy_wav(Path(output_path))

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


class E2EPrepareAudioReviewTest(unittest.TestCase):
    """End-to-end: prepare EPUB → mock audio → review UI (FastAPI TestClient)."""

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

        # After audio run there must be a DB and some sentence version records.
        db_path = self.output_dir / "wav" / "_state" / "audio_chunks.sqlite3"
        self.assertTrue(db_path.exists(), f"audio_chunks DB not created: {db_path}")
        conn = sqlite3.connect(str(db_path))
        try:
            rows = conn.execute(
                "SELECT COUNT(*) FROM sentence_text_versions"
            ).fetchone()
            self.assertGreater(rows[0], 0, "No sentence versions recorded in DB")
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Step 3: review UI endpoints via FastAPI TestClient
    # ------------------------------------------------------------------
    def _drive_review_ui(self):
        from fastapi.testclient import TestClient
        from audiobook_generator.ui.review_server import app
        from audiobook_generator.utils.sentence_hash import sentence_hash as _sentence_hash

        output_dir = str(self.output_dir)

        # Reset any stale server-side config so audio_folder is not taken
        # from a previous test run's app.state.
        app.state.review_config = None

        client = TestClient(app)

        # --- Step 3.1: GET /api/chapters → ≥2 chapters ------------------
        resp = client.get("/api/chapters", params={"dir": output_dir})
        self.assertEqual(resp.status_code, 200, f"/api/chapters failed: {resp.text}")
        chapters = resp.json()
        self.assertGreaterEqual(len(chapters), 2, f"Expected ≥2 chapters, got: {chapters}")

        ch1 = chapters[0]
        ch2 = chapters[1]

        # --- Step 3.2: GET /api/chunks for both chapters ----------------
        resp2 = client.get("/api/chunks", params={
            "dir": output_dir,
            "chapter_key": ch2["key"],
            "text_path": ch2["text_path"],
        })
        self.assertEqual(resp2.status_code, 200)
        ch2_chunks = resp2.json()
        self.assertGreater(len(ch2_chunks), 0, "Chapter 2 has no chunks")

        resp1 = client.get("/api/chunks", params={
            "dir": output_dir,
            "chapter_key": ch1["key"],
            "text_path": ch1["text_path"],
        })
        self.assertEqual(resp1.status_code, 200)
        ch1_chunks = resp1.json()
        self.assertGreater(len(ch1_chunks), 0, "Chapter 1 has no chunks")

        # --- Step 3.3: GET /api/audio → first chunk must have WAV -------
        first_chunk = ch1_chunks[0]
        resp_audio = client.get("/api/audio", params={
            "dir": output_dir,
            "chapter_key": ch1["key"],
            "hash": first_chunk["hash"],
        })
        self.assertEqual(resp_audio.status_code, 200,
                         f"Audio not found for chunk {first_chunk['hash']}: {resp_audio.text}")
        self.assertTrue(
            first_chunk["has_audio"],
            f"Chunk 0 should have audio, but has_audio=False (hash={first_chunk['hash']})",
        )

        # --- Step 3.4: POST /api/save → edit sentence 0 -----------------
        original_text = first_chunk["text"]
        new_text = original_text + " [отредактировано]"
        save_resp = client.post("/api/save", json={
            "dir": output_dir,
            "chapter_key": ch1["key"],
            "text_path": ch1["text_path"],
            "old_text": original_text,
            "new_text": new_text,
        })
        self.assertEqual(save_resp.status_code, 200,
                         f"/api/save failed: {save_resp.text}")
        save_data = save_resp.json()
        self.assertEqual(save_data["status"], "ok")
        old_hash = save_data["old_hash"]
        new_hash = save_data["new_hash"]
        self.assertEqual(old_hash, _sentence_hash(original_text))
        self.assertEqual(new_hash, _sentence_hash(new_text))

        # --- Step 3.5: GET /api/history → predecessor record saved ------
        hist_resp = client.get("/api/history", params={
            "dir": output_dir,
            "hash": new_hash,
        })
        self.assertEqual(hist_resp.status_code, 200)
        history = hist_resp.json()
        self.assertEqual(len(history), 1,
                         f"Expected 1 history record for new_hash, got: {history}")
        self.assertEqual(history[0]["hash"], old_hash)
        self.assertEqual(history[0]["text"], original_text)
        self.assertEqual(history[0]["replaced_by"], new_hash)

        # --- Step 3.6: verify disk file reflects the edit ---------------
        final_text_on_disk = Path(ch1["text_path"]).read_text(encoding="utf-8")
        self.assertIn(new_text, final_text_on_disk,
                      "Edit was not written to the chapter .txt file")
        self.assertNotIn(original_text + "\n", final_text_on_disk,
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



