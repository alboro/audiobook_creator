# Copyright (c) 2026 alboro <alboro@users.noreply.github.com>
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See: https://polyformproject.org/licenses/noncommercial/1.0.0/

"""Tests for the iterative prepare-run behaviour.

When a new ``prepare`` run starts because the previous run is complete,
``prepared_text_folder`` must be auto-set to the previous run directory so
that the normalizer builds on top of the already-reviewed text rather than
re-parsing the epub from scratch.
"""
from __future__ import annotations

import os
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import MagicMock, patch

from audiobook_generator.core.audiobook_generator import AudiobookGenerator


# ---------------------------------------------------------------------------
# Minimal config stub
# ---------------------------------------------------------------------------

class _Cfg:
    def __init__(self, output_folder: str):
        self.output_folder = output_folder
        self.input_file = None
        self.prepared_text_folder = None
        self.normalize = False
        self.normalize_steps = None
        self.normalize_provider = None
        self.normalize_log_changes = False
        self.force_new_run = False
        self.chapter_start = 1
        self.chapter_end = 9999
        self.chapter_mode = None
        self.no_prompt = True
        self.prepare_text = False
        self.preview = False
        self.package_m4b = False
        self.output_text = False
        self.language = "en"
        self.current_run_index = None
        self.normalization_state_path = None
        self.mode = None


def _make_generator(output_folder: str) -> AudiobookGenerator:
    cfg = _Cfg(output_folder)
    gen = AudiobookGenerator.__new__(AudiobookGenerator)
    gen.config = cfg
    return gen


# ---------------------------------------------------------------------------
# Helpers to create fake run directories / state DBs
# ---------------------------------------------------------------------------

def _create_run_dir(output_folder: str, index: str, *, complete: bool) -> Path:
    """Create a text/NNN run directory with a state DB.

    If *complete* is True the DB marks the single chapter as 'success'.
    """
    run_dir = Path(output_folder) / "text" / index
    state_dir = run_dir / "_state"
    state_dir.mkdir(parents=True, exist_ok=True)

    db_path = state_dir / "normalization_progress.sqlite3"
    with closing(sqlite3.connect(str(db_path))) as conn:
        conn.execute(
            "CREATE TABLE normalization_steps "
            "(step_name TEXT, step_index INTEGER, chapter_key TEXT, status TEXT)"
        )
        status = "success" if complete else "pending"
        conn.execute(
            "INSERT INTO normalization_steps VALUES (?, ?, ?, ?)",
            ("tts_safe_split", 0, "chapter_1", status),
        )
        conn.commit()

    # Place a dummy chapter text file
    chapter_file = run_dir / "0001_chapter.txt"
    chapter_file.write_text("Chapter text from run " + index, encoding="utf-8")
    return run_dir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAutoIterativeRun(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self.gen = _make_generator(self._tmp)

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _run_index_setup(self, mode: str = "prepare") -> None:
        """Run only the run-index / prepared_text_folder determination logic."""
        gen = self.gen
        cfg = gen.config
        cfg.force_new_run = False
        os.makedirs(cfg.output_folder, exist_ok=True)

        if mode in ("prepare", "all"):
            if cfg.force_new_run:
                prev_index = gen._latest_run_index("text")
                run_index = gen._next_run_index("text")
                if prev_index and not cfg.prepared_text_folder:
                    prev_text_dir = str(gen._run_subdir("text") / prev_index)
                    if os.path.isdir(prev_text_dir):
                        cfg.prepared_text_folder = prev_text_dir
                        cfg._prepared_text_folder_auto = True
            else:
                latest_index, can_resume = gen._can_resume_latest_run("text")
                if can_resume:
                    run_index = latest_index
                else:
                    run_index = gen._next_run_index("text")
                    if latest_index:
                        if not cfg.prepared_text_folder:
                            prev_text_dir = str(gen._run_subdir("text") / latest_index)
                            if os.path.isdir(prev_text_dir):
                                cfg.prepared_text_folder = prev_text_dir
                                cfg._prepared_text_folder_auto = True

        cfg.current_run_index = run_index

    def test_first_run_no_prepared_folder(self):
        """First ever run: no previous text dirs → prepared_text_folder stays None."""
        self._run_index_setup()
        self.assertEqual(self.gen.config.current_run_index, "001")
        self.assertIsNone(self.gen.config.prepared_text_folder)

    def test_auto_sets_prepared_folder_when_previous_run_complete(self):
        """Previous run complete → new run created, prepared_text_folder points to prev run."""
        _create_run_dir(self._tmp, "001", complete=True)
        self._run_index_setup()

        cfg = self.gen.config
        self.assertEqual(cfg.current_run_index, "002")
        self.assertIsNotNone(cfg.prepared_text_folder)
        self.assertTrue(cfg.prepared_text_folder.endswith("001"))
        self.assertTrue(os.path.isdir(cfg.prepared_text_folder))
        self.assertTrue(getattr(cfg, "_prepared_text_folder_auto", False))

    def test_resumes_incomplete_run_no_prepared_folder_change(self):
        """Incomplete run → resume it, prepared_text_folder is NOT changed."""
        _create_run_dir(self._tmp, "001", complete=False)
        self._run_index_setup()

        cfg = self.gen.config
        self.assertEqual(cfg.current_run_index, "001")
        self.assertIsNone(cfg.prepared_text_folder)

    def test_force_new_run_sets_prepared_folder(self):
        """--force_new_run: same auto-detection should also work."""
        _create_run_dir(self._tmp, "001", complete=True)
        cfg = self.gen.config
        cfg.force_new_run = True

        gen = self.gen
        os.makedirs(cfg.output_folder, exist_ok=True)
        prev_index = gen._latest_run_index("text")
        run_index = gen._next_run_index("text")
        if prev_index and not cfg.prepared_text_folder:
            prev_text_dir = str(gen._run_subdir("text") / prev_index)
            if os.path.isdir(prev_text_dir):
                cfg.prepared_text_folder = prev_text_dir
                cfg._prepared_text_folder_auto = True
        cfg.current_run_index = run_index

        self.assertEqual(cfg.current_run_index, "002")
        self.assertIsNotNone(cfg.prepared_text_folder)
        self.assertTrue(cfg.prepared_text_folder.endswith("001"))

    def test_explicit_prepared_folder_not_overwritten(self):
        """If prepared_text_folder is already set explicitly, don't overwrite it."""
        _create_run_dir(self._tmp, "001", complete=True)
        explicit_path = os.path.join(self._tmp, "custom_text")
        os.makedirs(explicit_path, exist_ok=True)
        self.gen.config.prepared_text_folder = explicit_path
        self._run_index_setup()

        self.assertEqual(self.gen.config.prepared_text_folder, explicit_path)

    def test_three_runs_chain(self):
        """run 001 → complete → run 002 uses 001; run 002 → complete → run 003 uses 002."""
        _create_run_dir(self._tmp, "001", complete=True)
        self._run_index_setup()
        self.assertEqual(self.gen.config.current_run_index, "002")
        self.assertTrue(self.gen.config.prepared_text_folder.endswith("001"))

        # Mark 002 as complete and simulate third run
        _create_run_dir(self._tmp, "002", complete=True)
        self.gen.config.prepared_text_folder = None
        self.gen.config.current_run_index = None
        self._run_index_setup()
        self.assertEqual(self.gen.config.current_run_index, "003")
        self.assertTrue(self.gen.config.prepared_text_folder.endswith("002"))


class TestTtsSafeSplitNotDeprecated(unittest.TestCase):
    def test_tts_safe_split_not_in_deprecated_aliases(self):
        from audiobook_generator.normalizers.base_normalizer import _DEPRECATED_STEP_ALIASES
        self.assertNotIn("tts_safe_split", _DEPRECATED_STEP_ALIASES)

    def test_tts_safe_split_is_distinct_class(self):
        from audiobook_generator.normalizers.tts_safe_split_normalizer import (
            TTSSafeSplitNormalizer, TTSSafeSplitAlgorithmicNormalizer,
        )
        self.assertIsNot(TTSSafeSplitNormalizer, TTSSafeSplitAlgorithmicNormalizer)
        self.assertEqual(TTSSafeSplitNormalizer.STEP_NAME, "tts_llm_safe_split")
        self.assertEqual(TTSSafeSplitAlgorithmicNormalizer.STEP_NAME, "tts_safe_split")

    def test_tts_safe_split_resolves_without_warning(self):
        import logging
        from audiobook_generator.normalizers.base_normalizer import normalize_step_name
        with self.assertLogs("audiobook_generator.normalizers.base_normalizer", level="WARNING") as cm:
            normalize_step_name("ru_stress_ambiguity")
        for msg in cm.output:
            self.assertNotIn("tts_safe_split", msg)


if __name__ == "__main__":
    unittest.main()

