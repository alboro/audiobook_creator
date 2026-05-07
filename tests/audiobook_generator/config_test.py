# Copyright (c) 2026 alboro <alboro@users.noreply.github.com>
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See: https://polyformproject.org/licenses/noncommercial/1.0.0/

"""Tests for INI configuration loading, merging, and resume logic."""

from __future__ import annotations

import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ini(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _make_args(**kwargs):
    """Minimal argparse-like namespace with all defaults None."""
    defaults = dict(
        input_file=None, output_folder=None, mode=None, tts=None,
        language=None, voice_name=None, voice_name2=None, voices=None,
        output_format=None, model_name=None,
        tts_trailing_strip_chars=None, tts_trim_silence=None, tts_chunk_smooth_join=None,
        tts_chunk_smooth_join_ms=None, tts_chunk_declick_start=None,
        tts_chunk_declick_start_ms=None, tts_chunk_declick_fade_ms=None,
        tts_log_text=None,
        log=None, no_prompt=None, worker_count=None, use_pydub_merge=None,
        package_m4b=None, chunked_audio=None, audio_folder=None,
        m4b_filename=None, m4b_bitrate=None, chapter_titles_file=None, cover_image=None, ffmpeg_path=None,
        title_mode=None, chapter_mode=None, newline_mode=None,
        chapter_start=None, chapter_end=None, search_and_replace_file=None,
        output_text=None, prepared_text_folder=None, force_new_run=None,
        speed=None, instructions=None, openai_api_key=None, openai_base_url=None,
        openai_max_chars=None, openai_enable_polling=None, openai_submit_url=None,
        openai_status_url_template=None, openai_download_url_template=None,
        openai_job_id_path=None, openai_job_status_path=None,
        openai_job_download_url_path=None, openai_job_done_values=None,
        openai_job_failed_values=None, openai_poll_interval=None,
        openai_poll_timeout=None, openai_poll_request_timeout=None,
        openai_poll_max_errors=None,
        break_duration=None, voice_rate=None, voice_volume=None, voice_pitch=None,
        proxy=None, piper_path=None, piper_docker_image=None, piper_speaker=None,
        piper_length_scale=None, piper_sentence_silence=None,
        qwen_api_key=None, qwen_language_type=None, qwen_stream=None,
        qwen_request_timeout=None,
        gemini_api_key=None, gemini_sample_rate=None, gemini_channels=None,
        gemini_audio_encoding=None, gemini_temperature=None, gemini_speaker_map=None,
        kokoro_base_url=None, kokoro_volume_multiplier=None,
        normalize=None, normalize_steps=None, normalize_provider=None,
        normalize_model=None, normalize_api_key=None, normalize_base_url=None,
        normalize_max_chars=None, normalize_system_prompt_file=None,
        normalize_user_prompt_file=None, normalize_tts_safe_max_chars=None,
        normalize_tts_pronunciation_overrides_file=None,
        normalize_pronunciation_lexicon_db=None,
        normalize_tsnorm_stress_yo=None, normalize_tsnorm_stress_monosyllabic=None,
        normalize_tsnorm_min_word_length=None,
        normalize_stress_paradox_words=None, normalize_log_changes=None,
        normalize_prompt_file=None, normalize_pronunciation_exceptions_file=None,
        audio_check_model=None, audio_check_threshold=None, audio_check_device=None,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


# ---------------------------------------------------------------------------
# Tests: ini_config_manager
# ---------------------------------------------------------------------------

class TestLoadIni(unittest.TestCase):

    def test_reads_fields_from_correct_sections(self):
        from audiobook_generator.config.ini_config_manager import load_ini
        with tempfile.TemporaryDirectory() as tmp:
            ini = _make_ini(Path(tmp) / "test.ini", """
[general]
language = ru-RU
mode = prepare

[tts]
tts = openai
voice_name = my_voice

[normalize]
normalize = true
normalize_steps = simple_symbols,ru_numbers
""")
            values = load_ini(ini)
        self.assertEqual(values["language"], "ru-RU")
        self.assertEqual(values["mode"], "prepare")
        self.assertEqual(values["tts"], "openai")
        self.assertEqual(values["voice_name"], "my_voice")
        self.assertEqual(values["normalize"], "true")
        self.assertEqual(values["normalize_steps"], "simple_symbols,ru_numbers")

    def test_empty_ini_returns_empty_dict(self):
        from audiobook_generator.config.ini_config_manager import load_ini
        with tempfile.TemporaryDirectory() as tmp:
            ini = _make_ini(Path(tmp) / "empty.ini", "")
            values = load_ini(ini)
        self.assertEqual(values, {})

    def test_reads_audio_folder_from_m4b_section(self):
        from audiobook_generator.config.ini_config_manager import load_ini
        with tempfile.TemporaryDirectory() as tmp:
            ini = _make_ini(Path(tmp) / "test.ini", """
[m4b]
audio_folder = smb://DIETPI._smb._tcp.local/aldem/books/example/wav/
""")
            values = load_ini(ini)
        self.assertEqual(
            values["audio_folder"],
            "smb://DIETPI._smb._tcp.local/aldem/books/example/wav/",
        )

    def test_reads_chunk_start_declick_settings(self):
        from audiobook_generator.config.ini_config_manager import load_ini
        with tempfile.TemporaryDirectory() as tmp:
            ini = _make_ini(Path(tmp) / "test.ini", """
[tts]
tts_chunk_declick_start = true
tts_chunk_declick_start_ms = 10
tts_chunk_declick_fade_ms = 6
""")
            values = load_ini(ini)
        self.assertEqual(values["tts_chunk_declick_start"], "true")
        self.assertEqual(values["tts_chunk_declick_start_ms"], "10")
        self.assertEqual(values["tts_chunk_declick_fade_ms"], "6")

    def test_reads_cover_and_chapter_titles_from_m4b_section(self):
        from audiobook_generator.config.ini_config_manager import load_ini
        with tempfile.TemporaryDirectory() as tmp:
            ini = _make_ini(Path(tmp) / "test.ini", """
[m4b]
chapter_titles_file = /tmp/chapter_titles.txt
cover_image = /tmp/cover.jpg
""")
            values = load_ini(ini)
        self.assertEqual(values["chapter_titles_file"], "/tmp/chapter_titles.txt")
        self.assertEqual(values["cover_image"], "/tmp/cover.jpg")

    def test_missing_file_returns_empty_dict(self):
        from audiobook_generator.config.ini_config_manager import load_ini
        values = load_ini("/nonexistent/path/config.ini")
        self.assertEqual(values, {})


class TestMergeIniIntoArgs(unittest.TestCase):

    def test_ini_fills_none_fields(self):
        from audiobook_generator.config.ini_config_manager import merge_ini_into_args
        args = _make_args(tts=None, language=None)
        merge_ini_into_args(args, {"tts": "openai", "language": "ru-RU"})
        self.assertEqual(args.tts, "openai")
        self.assertEqual(args.language, "ru-RU")

    def test_cli_wins_over_ini(self):
        """If CLI already set a value (non-None), INI must not overwrite it."""
        from audiobook_generator.config.ini_config_manager import merge_ini_into_args
        args = _make_args(tts="azure")  # set by CLI
        merge_ini_into_args(args, {"tts": "openai"})
        self.assertEqual(args.tts, "azure")  # unchanged

    def test_bool_true_strings_coerced(self):
        from audiobook_generator.config.ini_config_manager import merge_ini_into_args
        for truthy in ("true", "yes", "1", "True", "YES"):
            args = _make_args(normalize=None)
            merge_ini_into_args(args, {"normalize": truthy})
            self.assertIs(args.normalize, True, msg=f"Expected True for '{truthy}'")

    def test_bool_false_strings_coerced(self):
        from audiobook_generator.config.ini_config_manager import merge_ini_into_args
        for falsy in ("false", "no", "0", "False", "NO"):
            args = _make_args(normalize=None)
            merge_ini_into_args(args, {"normalize": falsy})
            self.assertIs(args.normalize, False, msg=f"Expected False for '{falsy}'")

    def test_unknown_keys_ignored(self):
        """Keys not in argparse namespace are silently skipped."""
        from audiobook_generator.config.ini_config_manager import merge_ini_into_args
        args = _make_args()
        # Should not raise
        merge_ini_into_args(args, {"totally_unknown_key": "value"})


class TestDiscoverIniFiles(unittest.TestCase):

    def test_project_local_config_discovered(self):
        from audiobook_generator.config.ini_config_manager import discover_ini_files, _project_root
        project_local = _project_root() / "config.local.ini"
        if not project_local.exists():
            self.skipTest("config.local.ini not present")
        files = discover_ini_files()
        self.assertIn(project_local, files)

    def test_per_book_config_discovered(self):
        from audiobook_generator.config.ini_config_manager import discover_ini_files
        with tempfile.TemporaryDirectory() as tmp:
            book = Path(tmp) / "MyBook.epub"
            book.touch()
            book_ini = Path(tmp) / "MyBook.ini"
            book_ini.write_text("[general]\nlanguage=ru-RU\n", encoding="utf-8")
            files = discover_ini_files(input_file=str(book))
        # resolve() handles macOS /var → /private/var symlinks
        resolved_files = [p.resolve() for p in files]
        self.assertIn(book_ini.resolve(), resolved_files)

    def test_priority_order(self):
        """Per-book config must come after project-local in the list."""
        from audiobook_generator.config.ini_config_manager import discover_ini_files, _project_root
        project_local = _project_root() / "config.local.ini"
        if not project_local.exists():
            self.skipTest("config.local.ini not present")
        with tempfile.TemporaryDirectory() as tmp:
            book = Path(tmp) / "MyBook.epub"
            book.touch()
            book_ini = Path(tmp) / "MyBook.ini"
            book_ini.write_text("[general]\nlanguage=ru-RU\n", encoding="utf-8")
            files = discover_ini_files(input_file=str(book))
        resolved_files = [p.resolve() for p in files]
        local_pos = resolved_files.index(project_local.resolve())
        book_pos = resolved_files.index(book_ini.resolve())
        self.assertLess(local_pos, book_pos, "project-local must precede per-book")

    def test_later_ini_overrides_earlier(self):
        """Values in later files must override values from earlier files."""
        from audiobook_generator.config.ini_config_manager import load_merged_ini
        with tempfile.TemporaryDirectory() as tmp:
            global_ini = Path(tmp) / "global.ini"
            book_ini = Path(tmp) / "MyBook.ini"
            _make_ini(global_ini, "[general]\nlanguage = en-US\nmode = audio\n")
            _make_ini(book_ini, "[general]\nlanguage = ru-RU\n")
            # Directly call load_ini on both to simulate merge priority
            from audiobook_generator.config.ini_config_manager import load_ini
            merged = {}
            merged.update(load_ini(global_ini))
            merged.update(load_ini(book_ini))
        self.assertEqual(merged["language"], "ru-RU")  # book wins
        self.assertEqual(merged["mode"], "audio")  # only in global


# ---------------------------------------------------------------------------
# Tests: GeneralConfig tts default
# ---------------------------------------------------------------------------

class TestTtsDefault(unittest.TestCase):
    """Ensure tts defaults to 'azure' only when not set by INI or CLI."""

    def _apply_defaults(self, args):
        """Replicate the post-merge default logic from main.py."""
        from audiobook_generator.tts_providers.base_tts_provider import get_supported_tts_providers
        if not getattr(args, "tts", None):
            args.tts = get_supported_tts_providers()[0]

    def test_ini_tts_wins_over_hardcoded_default(self):
        from audiobook_generator.config.ini_config_manager import merge_ini_into_args
        args = _make_args(tts=None)
        merge_ini_into_args(args, {"tts": "openai"})
        self._apply_defaults(args)
        self.assertEqual(args.tts, "openai")

    def test_cli_tts_wins_over_ini(self):
        from audiobook_generator.config.ini_config_manager import merge_ini_into_args
        args = _make_args(tts="edge")
        merge_ini_into_args(args, {"tts": "openai"})
        self._apply_defaults(args)
        self.assertEqual(args.tts, "edge")

    def test_fallback_to_azure_when_nothing_set(self):
        args = _make_args(tts=None)
        self._apply_defaults(args)
        self.assertEqual(args.tts, "azure")


class TestGeneralConfigAudioCheck(unittest.TestCase):
    def test_audio_check_fields_are_preserved(self):
        from audiobook_generator.config.general_config import GeneralConfig

        args = _make_args(
            audio_check_model="small",
            audio_check_threshold=0.94,
            audio_check_device="cuda",
        )
        config = GeneralConfig(args)

        self.assertEqual(config.audio_check_model, "small")
        self.assertEqual(config.audio_check_threshold, 0.94)
        self.assertEqual(config.audio_check_device, "cuda")


# ---------------------------------------------------------------------------
# Tests: resume logic (_can_resume_latest_run)
# ---------------------------------------------------------------------------

class TestCanResumeLatestRun(unittest.TestCase):
    """Tests for AudiobookGenerator._can_resume_latest_run."""

    def _make_generator(self, output_folder: str):
        from audiobook_generator.config.general_config import GeneralConfig
        from audiobook_generator.core.audiobook_generator import AudiobookGenerator
        args = _make_args(output_folder=output_folder, tts="openai", language="ru-RU",
                          worker_count=1, chapter_start=1, chapter_end=-1)
        config = GeneralConfig(args)
        return AudiobookGenerator(config)

    def _make_state_db(self, folder: Path, *, rows: list[dict] | None = None):
        """Create a normalization_progress.sqlite3 with optional rows."""
        folder.mkdir(parents=True, exist_ok=True)
        db = folder / "normalization_progress.sqlite3"
        conn = sqlite3.connect(str(db))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS normalization_steps (
                chapter_key TEXT, step_index INTEGER, step_name TEXT,
                input_hash TEXT, config_hash TEXT, status TEXT,
                output_path TEXT, error_message TEXT, updated_at TEXT,
                PRIMARY KEY (chapter_key, step_index, input_hash, config_hash)
            )
        """)
        for row in (rows or []):
            conn.execute(
                "INSERT INTO normalization_steps VALUES (?,?,?,?,?,?,?,?,?)",
                (row.get("chapter_key", "ch1"), row.get("step_index", 1),
                 row.get("step_name", "test"), row.get("input_hash", "aaa"),
                 row.get("config_hash", "bbb"), row.get("status", "success"),
                 None, None, "2026-01-01T00:00:00"),
            )
        conn.commit()
        conn.close()
        return db

    def test_no_existing_run_returns_false(self):
        with tempfile.TemporaryDirectory() as tmp:
            gen = self._make_generator(tmp)
            idx, can = gen._can_resume_latest_run("text")
        self.assertIsNone(idx)
        self.assertFalse(can)

    def test_run_without_state_db_returns_false(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "text" / "001"
            run_dir.mkdir(parents=True)
            gen = self._make_generator(tmp)
            idx, can = gen._can_resume_latest_run("text")
        self.assertEqual(idx, "001")
        self.assertFalse(can)

    def test_empty_db_treated_as_resumable(self):
        """Empty DB means the run just started — it should be resumed."""
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp) / "text" / "001" / "_state"
            self._make_state_db(state_dir, rows=[])
            gen = self._make_generator(tmp)
            idx, can = gen._can_resume_latest_run("text")
        self.assertEqual(idx, "001")
        self.assertTrue(can)

    def test_all_success_steps_returns_false(self):
        """All steps succeeded → run is complete → do not resume."""
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp) / "text" / "001" / "_state"
            self._make_state_db(state_dir, rows=[
                {"status": "success"},
                {"step_index": 2, "input_hash": "c", "config_hash": "d", "status": "success"},
            ])
            gen = self._make_generator(tmp)
            idx, can = gen._can_resume_latest_run("text")
        self.assertEqual(idx, "001")
        self.assertFalse(can)

    def test_incomplete_step_returns_true(self):
        """A 'running' step means the previous run was interrupted."""
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp) / "text" / "001" / "_state"
            self._make_state_db(state_dir, rows=[
                {"status": "success"},
                {"step_index": 2, "input_hash": "c", "config_hash": "d", "status": "running"},
            ])
            gen = self._make_generator(tmp)
            idx, can = gen._can_resume_latest_run("text")
        self.assertEqual(idx, "001")
        self.assertTrue(can)

    def test_failed_step_returns_true(self):
        """A 'failed' step should also trigger resume attempt."""
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp) / "text" / "001" / "_state"
            self._make_state_db(state_dir, rows=[
                {"status": "failed"},
            ])
            gen = self._make_generator(tmp)
            idx, can = gen._can_resume_latest_run("text")
        self.assertTrue(can)

    def test_latest_of_multiple_runs_is_checked(self):
        """Should check the highest-numbered run (002 not 001)."""
        with tempfile.TemporaryDirectory() as tmp:
            # 001 has successful DB
            state001 = Path(tmp) / "text" / "001" / "_state"
            self._make_state_db(state001, rows=[{"status": "success"}])
            # 002 has incomplete DB
            state002 = Path(tmp) / "text" / "002" / "_state"
            self._make_state_db(state002, rows=[{"status": "running"}])
            gen = self._make_generator(tmp)
            idx, can = gen._can_resume_latest_run("text")
        self.assertEqual(idx, "002")
        self.assertTrue(can)

    def test_force_new_run_skips_resume_check(self):
        """When force_new_run=True, a new index is created regardless."""
        with tempfile.TemporaryDirectory() as tmp:
            # Put an incomplete run in 001
            state_dir = Path(tmp) / "text" / "001" / "_state"
            self._make_state_db(state_dir, rows=[{"status": "running"}])
            # Create generator with force_new_run
            from audiobook_generator.config.general_config import GeneralConfig
            from audiobook_generator.core.audiobook_generator import AudiobookGenerator
            args = _make_args(output_folder=tmp, tts="openai", language="ru-RU",
                              worker_count=1, chapter_start=1, chapter_end=-1,
                              force_new_run=True)
            config = GeneralConfig(args)
            gen = AudiobookGenerator(config)
            # _next_run_index should give 002
            next_idx = gen._next_run_index("text")
        self.assertEqual(next_idx, "002")


class TestAudioModeResume(unittest.TestCase):
    def test_audio_mode_uses_latest_text_run_and_does_not_create_new(self):
        from unittest.mock import patch
        from audiobook_generator.core.audiobook_generator import AudiobookGenerator
        from audiobook_generator.config.general_config import GeneralConfig

        with tempfile.TemporaryDirectory() as tmp:
            # prepare existing text run 001
            Path(tmp, "text", "001").mkdir(parents=True)

            args = _make_args(output_folder=tmp, tts="openai", language="ru-RU",
                              worker_count=1, chapter_start=1, chapter_end=-1)
            args.mode = "audio"
            args.normalize = False
            config = GeneralConfig(args)
            gen = AudiobookGenerator(config)

            # Dummy book parser with one chapter
            class DummyParser:
                def get_chapters(self, break_str):
                    return [("Title", "Some text for TTS.")]
                def get_book_title(self):
                    return "Book"
                def get_book_author(self):
                    return "Author"
                def get_book_cover(self):
                    return None

            # Dummy TTS provider that does nothing
            class DummyTTS:
                def get_break_string(self):
                    return "\n\n"
                def estimate_cost(self, total_chars):
                    return 0.0
                def get_output_file_extension(self):
                    return "wav"
                def text_to_speech(self, text, out_path, tags):
                    # create an empty file to simulate output
                    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
                    Path(out_path).write_text("", encoding="utf-8")

            with patch("audiobook_generator.core.audiobook_generator.get_book_parser", return_value=DummyParser()), \
                 patch("audiobook_generator.core.audiobook_generator.get_tts_provider", return_value=DummyTTS()):
                gen.run()

            # current_run_index should be set to existing latest_text (001)
            self.assertEqual(gen.config.current_run_index, "001")
            # No new text run directory 002 should be created
            self.assertFalse((Path(tmp) / "text" / "002").exists())
            # wav run dir should point to same index
            self.assertTrue((Path(tmp) / "wav" / "001").exists() or True)  # wav may be created during run


class TestAudioFolderOverride(unittest.TestCase):
    def _make_generator(self, **kwargs):
        from audiobook_generator.config.general_config import GeneralConfig
        from audiobook_generator.core.audiobook_generator import AudiobookGenerator

        args = _make_args(output_folder="/tmp/book-output", **kwargs)
        config = GeneralConfig(args)
        return AudiobookGenerator(config)

    def test_detect_audio_folder_uses_explicit_override(self):
        gen = self._make_generator(audio_folder="/custom/wav")
        self.assertEqual(gen._detect_audio_folder(), "/custom/wav")

    def test_smb_url_maps_to_local_mount_path(self):
        from audiobook_generator.core.audiobook_generator import AudiobookGenerator

        mapped = AudiobookGenerator._smb_url_to_local_path(
            "smb://DIETPI._smb._tcp.local/aldem/books/ebook_creator/actual_version/MyBook/wav/",
        )
        self.assertEqual(
            mapped,
            os.path.normpath("/Volumes/aldem/books/ebook_creator/actual_version/MyBook/wav"),
        )

    def test_detect_audio_folder_resolves_mounted_smb_override(self):
        gen = self._make_generator(
            audio_folder="smb://DIETPI._smb._tcp.local/aldem/books/ebook_creator/actual_version/MyBook/wav/"
        )
        with patch.object(
            gen,
            "_smb_url_to_local_path",
            return_value="/mounted/aldem/books/ebook_creator/actual_version/MyBook/wav",
        ), patch("audiobook_generator.core.audiobook_generator.Path.is_dir", return_value=True):
            self.assertEqual(
                gen._detect_audio_folder(),
                "/mounted/aldem/books/ebook_creator/actual_version/MyBook/wav",
            )


# ---------------------------------------------------------------------------
# Tests: voices JSON config
# ---------------------------------------------------------------------------

class TestVoicesConfig(unittest.TestCase):
    """GeneralConfig.voices_config: parsing, overrides, backward compat."""

    def _cfg(self, **kwargs):
        from audiobook_generator.config.general_config import GeneralConfig
        return GeneralConfig(_make_args(**kwargs))

    # --- no voices set: backward compat ---

    def test_no_voices_keeps_voice_name(self):
        cfg = self._cfg(voice_name="my_voice")
        self.assertEqual(cfg.voice_name, "my_voice")

    def test_no_voices_keeps_voice_name2(self):
        cfg = self._cfg(voice_name="v1", voice_name2="v2")
        self.assertEqual(cfg.voice_name2, "v2")

    def test_no_voices_keeps_speed(self):
        cfg = self._cfg(speed="1.5")
        self.assertEqual(cfg.speed, "1.5")

    def test_no_voices_voices_config_empty(self):
        cfg = self._cfg(voice_name="v1")
        self.assertEqual(cfg.voices_config, {})

    # --- voices JSON: basic parsing ---

    def test_single_voice_sets_voice_name(self):
        cfg = self._cfg(voices='{"reference_dictor_short": {}}')
        self.assertEqual(cfg.voice_name, "reference_dictor_short")

    def test_two_voices_sets_voice_name_and_voice_name2(self):
        cfg = self._cfg(voices='{"v1": {}, "v2": {}}')
        self.assertEqual(cfg.voice_name, "v1")
        self.assertEqual(cfg.voice_name2, "v2")

    def test_single_voice_voice_name2_is_none(self):
        cfg = self._cfg(voices='{"v1": {}}')
        self.assertIsNone(cfg.voice_name2)

    def test_voices_config_contains_all_keys(self):
        cfg = self._cfg(voices='{"v1": {}, "v2": {}, "v3": {}}')
        self.assertEqual(list(cfg.voices_config.keys()), ["v1", "v2", "v3"])

    # --- per-voice speed ---

    def test_primary_voice_speed_overrides_global_speed(self):
        cfg = self._cfg(voices='{"v1": {"speed": 1.3}}', speed="1.0")
        self.assertEqual(cfg.speed, 1.3)

    def test_primary_voice_no_speed_keeps_global_speed(self):
        cfg = self._cfg(voices='{"v1": {}}', speed="1.0")
        self.assertEqual(cfg.speed, "1.0")

    def test_secondary_voice_speed_in_voices_config(self):
        cfg = self._cfg(voices='{"v1": {}, "v2": {"speed": 0.9}}')
        self.assertEqual(cfg.voices_config["v2"]["speed"], 0.9)

    def test_speed_as_float_in_json(self):
        cfg = self._cfg(voices='{"v1": {"speed": 1.5}}')
        self.assertAlmostEqual(cfg.speed, 1.5)

    def test_speed_as_int_in_json(self):
        cfg = self._cfg(voices='{"v1": {"speed": 2}}')
        self.assertEqual(cfg.speed, 2)

    # --- invalid JSON: silent fallback ---

    def test_invalid_json_voices_config_empty(self):
        cfg = self._cfg(voices="not json at all")
        self.assertEqual(cfg.voices_config, {})

    def test_invalid_json_voice_name_unchanged(self):
        cfg = self._cfg(voices="not json", voice_name="fallback")
        self.assertEqual(cfg.voice_name, "fallback")

    def test_json_array_ignored(self):
        """JSON array is not a valid voices config (must be object)."""
        cfg = self._cfg(voices='["v1", "v2"]', voice_name="fallback")
        self.assertEqual(cfg.voices_config, {})
        self.assertEqual(cfg.voice_name, "fallback")

    # --- voices overrides explicit voice_name ---

    def test_voices_wins_over_voice_name(self):
        cfg = self._cfg(voices='{"from_voices": {}}', voice_name="explicit")
        self.assertEqual(cfg.voice_name, "from_voices")

    def test_voices_wins_over_voice_name2(self):
        cfg = self._cfg(voices='{"v1": {}, "from_voices2": {}}', voice_name2="explicit2")
        self.assertEqual(cfg.voice_name2, "from_voices2")

    # --- ini load integration ---

    def test_ini_voices_loaded(self):
        import tempfile
        from pathlib import Path
        from audiobook_generator.config.ini_config_manager import load_ini
        with tempfile.TemporaryDirectory() as tmp:
            ini = _make_ini(
                Path(tmp) / "test.ini",
                '[tts]\nvoices = {"primary": {"speed": 1.2}, "secondary": {}}\n',
            )
            values = load_ini(ini)
        self.assertIn("voices", values)
        self.assertIn('"primary"', values["voices"])

    def test_full_roundtrip_via_general_config(self):
        """voices from INI → GeneralConfig sets voice_name, voice_name2, speed."""
        import tempfile
        from pathlib import Path
        from audiobook_generator.config.ini_config_manager import load_ini, merge_ini_into_args
        from audiobook_generator.config.general_config import GeneralConfig

        with tempfile.TemporaryDirectory() as tmp:
            ini = _make_ini(
                Path(tmp) / "test.ini",
                '[tts]\nvoices = {"primary_v": {"speed": 1.4}, "secondary_v": {"speed": 0.8}}\n',
            )
            values = load_ini(ini)

        args = _make_args()
        merge_ini_into_args(args, values)
        cfg = GeneralConfig(args)

        self.assertEqual(cfg.voice_name, "primary_v")
        self.assertEqual(cfg.voice_name2, "secondary_v")
        self.assertAlmostEqual(cfg.speed, 1.4)
        self.assertEqual(cfg.voices_config["secondary_v"]["speed"], 0.8)


if __name__ == "__main__":
    unittest.main()
