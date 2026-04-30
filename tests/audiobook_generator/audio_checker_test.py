# Copyright (c) 2026 alboro <alboro@users.noreply.github.com>
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See: https://polyformproject.org/licenses/noncommercial/1.0.0/

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from audiobook_generator.core.audio_checker import (
    AudioChecker,
    _normalize_for_compare,
    _build_pre_compare_normalizer,
)
from audiobook_generator.core.audio_chunk_store import AudioChunkStore


def test_check_one_file_reuses_fresh_cached_raw_transcription():
    with tempfile.TemporaryDirectory() as tmp:
        output_dir = Path(tmp)
        audio_path = output_dir / "wav" / "chunks" / "0001_Test" / "hash_cached.wav"
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        audio_path.write_bytes(b"fake wav")

        store = AudioChunkStore(output_dir / "wav" / "_state" / "audio_chunks.sqlite3")
        store.save_checked_chunk(
            "0001_Test",
            "hash_cached",
            "Привет, мир.",
            "Привет мир",
            0.99,
            raw_transcription="Привет мир",
        )

        checker = AudioChecker(output_folder=output_dir, threshold=0.5)
        checker._pre_compare = None
        transcribe_called = False

        def fake_transcribe(_path):
            nonlocal transcribe_called
            transcribe_called = True
            return "не должно вызываться"

        checker._transcribe = fake_transcribe
        counters = {"checked": 0, "disputed": 0, "skipped": 0}

        checker._check_one_file(
            audio_path,
            "0001_Test",
            "hash_cached",
            "Привет, мир.",
            store,
            counters,
        )

        assert transcribe_called is False
        assert counters == {"checked": 1, "disputed": 0, "skipped": 0}


def test_check_one_file_runs_reference_check_when_transcription_is_cached():
    with tempfile.TemporaryDirectory() as tmp:
        output_dir = Path(tmp)
        audio_path = output_dir / "wav" / "chunks" / "0001_Test" / "hash_cached.wav"
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        audio_path.write_bytes(b"fake wav")

        store = AudioChunkStore(output_dir / "wav" / "_state" / "audio_chunks.sqlite3")
        store.save_checked_chunk(
            "0001_Test",
            "hash_cached",
            "Приве́т, мир.",
            "Привет мир",
            0.99,
            raw_transcription="Привет мир",
        )

        checker = AudioChecker(output_folder=output_dir, threshold=0.5)
        checker._pre_compare = None
        checker._reference_check_command = "dummy-auditor"
        checker._reference_check_threshold = 0.85
        transcribe_called = False
        reference_called = False

        def fake_transcribe(_path):
            nonlocal transcribe_called
            transcribe_called = True
            return "не должно вызываться"

        def fake_reference_check(_audio_file, original_text):
            nonlocal reference_called
            reference_called = True
            assert original_text == "Приве́т, мир."
            return {
                "status": "ok",
                "score": 0.12,
                "threshold": 0.85,
                "payload": {"score": 0.12},
            }

        checker._transcribe = fake_transcribe
        checker._run_reference_check = fake_reference_check
        counters = {"checked": 0, "disputed": 0, "skipped": 0}

        checker._check_one_file(
            audio_path,
            "0001_Test",
            "hash_cached",
            "Приве́т, мир.",
            store,
            counters,
        )

        assert transcribe_called is False
        assert reference_called is True
        assert counters["checked"] == 1
        assert counters["disputed"] == 0
        assert counters["reference_checked"] == 1


def test_audio_check_prepared_text_folder_overrides_auto_run_selection():
    with tempfile.TemporaryDirectory() as tmp:
        output_dir = Path(tmp)
        run_009 = output_dir / "text" / "009"
        run_010 = output_dir / "text" / "010"
        run_009.mkdir(parents=True)
        run_010.mkdir(parents=True)
        (run_009 / "0001_Title.txt").write_text("Text.", encoding="utf-8")
        (run_010 / "0001_Title.txt").write_text("Text.", encoding="utf-8")
        (output_dir / "wav" / "chunks" / "0001_Title").mkdir(parents=True)

        config = SimpleNamespace(prepared_text_folder=str(run_010))
        checker = AudioChecker(output_folder=output_dir, config=config)

        assert checker._select_text_run_folder() == run_010


def test_check_one_file_persists_checked_chunk_cache_after_successful_run():
    with tempfile.TemporaryDirectory() as tmp:
        output_dir = Path(tmp)
        store = AudioChunkStore(output_dir / "wav" / "_state" / "audio_chunks.sqlite3")

        audio_path = output_dir / "wav" / "chunks" / "0001_Test" / "hash_checked_after_run.wav"
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        audio_path.write_bytes(b"fake wav")

        checker = AudioChecker(output_folder=output_dir, threshold=0.5)
        checker._pre_compare = None

        def fake_transcribe(_path):
            return "Привет мир"

        checker._transcribe = fake_transcribe
        counters = {"checked": 0, "disputed": 0, "skipped": 0}

        checker._check_one_file(
            audio_path,
            "0001_Test",
            "hash_checked_after_run",
            "Привет, мир.",
            store,
            counters,
        )

        assert counters == {"checked": 1, "disputed": 0, "skipped": 0}
        row = store.get_cached_transcription_entry("0001_Test", "hash_checked_after_run")
        assert row is not None
        assert row["status"] == "checked"
        assert row["raw_transcription"] == "Привет мир"


# ---------------------------------------------------------------------------
# Pre-compare normalizer helpers
# ---------------------------------------------------------------------------

def _try_build_pre_compare():
    """Return (normalizer_fn, skip_reason).  skip_reason is None when available."""
    try:
        fn = _build_pre_compare_normalizer("ru-RU")
        if fn is None:
            return None, "pre-compare normalizer returned None (missing dependencies?)"
        return fn, None
    except Exception as exc:
        return None, str(exc)


class PreCompareNormalizerTests(unittest.TestCase):
    """Unit tests for _build_pre_compare_normalizer — no audio / DB involved."""

    def setUp(self):
        self._fn, self._skip = _try_build_pre_compare()

    def _skip_if_unavailable(self):
        if self._skip:
            self.skipTest(f"pre-compare normalizer unavailable: {self._skip}")

    # ------------------------------------------------------------------
    # Basic sanity: digits in → words out
    # ------------------------------------------------------------------

    def test_year_with_noun_expanded(self):
        """'1794 года' → contains 'четвертого' (ordinal genitive of 1794, ё→е after NFD strip)."""
        self._skip_if_unavailable()
        result = self._fn("1794 года")
        norm = _normalize_for_compare(result)
        # _normalize_for_compare strips combining diacritics (NFD): ё → е
        self.assertIn("четвертого", norm, f"year not expanded: {result!r}")

    def test_full_date_expanded(self):
        """'5 августа 1794 года' → result starts with 'пятого', contains year in words."""
        self._skip_if_unavailable()
        result = self._fn("5 августа 1794 года")
        norm = _normalize_for_compare(result)
        # _normalize_for_compare strips combining diacritics (NFD): ё → е
        self.assertIn("пятого", norm, f"day not expanded: {result!r}")
        self.assertIn("августа", norm)
        self.assertIn("четвертого", norm)  # ё → е after NFD strip

    def test_standalone_cardinal_expanded(self):
        """'42' → 'сорок два'."""
        self._skip_if_unavailable()
        result = self._fn("42")
        norm = _normalize_for_compare(result)
        self.assertIn("сорок", norm, f"cardinal not expanded: {result!r}")
        self.assertIn("два", norm)


class WhisperDigitFalseDisputedTests(unittest.TestCase):
    """
    Regression tests for the false-disputed issue caused by Whisper returning
    digit form for spoken numbers.

    Scenario:
        original_text  = "пятого августа тысяча семьсот девяносто четвёртого года ..."
            (source book already has dates in word form, or ru_numbers was applied)
        whisper output = "5 августа 1794 года ..."
            (Whisper converts spoken numbers back to digits)

    Before the fix:  _normalize_for_compare strips digits → trans_norm ≈ "августа года"
                     similarity with orig_norm ≪ threshold → FALSE disputed
    After the fix:   _pre_compare expands transcription digits → word form
                     both sides match → NOT disputed
    """

    def setUp(self):
        self._fn, self._skip = _try_build_pre_compare()

    def _make_checker(self, threshold: float = 0.94) -> AudioChecker:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
        checker = AudioChecker(output_folder=output_dir, threshold=threshold)
        checker._pre_compare = self._fn  # may be None → covered by separate test
        return checker

    def _run_one_file(self, original_text: str, transcription: str, threshold: float = 0.94):
        """Run _check_one_file with a fake audio file (no actual transcription needed)."""
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            audio_path = output_dir / "wav" / "chunks" / "0001" / "hash.wav"
            audio_path.parent.mkdir(parents=True, exist_ok=True)
            audio_path.write_bytes(b"fake")

            store = AudioChunkStore(output_dir / "wav" / "_state" / "audio_chunks.sqlite3")
            checker = AudioChecker(output_folder=output_dir, threshold=threshold)
            checker._pre_compare = self._fn

            # Patch _transcribe so no real Whisper is invoked
            checker._transcribe = lambda _path: transcription

            counters = {"checked": 0, "disputed": 0, "skipped": 0}
            checker._check_one_file(audio_path, "0001", "hash", original_text, store, counters)
            return counters

    # ------------------------------------------------------------------

    def test_whisper_digits_not_false_disputed_when_original_has_words(self):
        """
        Original text in word form, Whisper returns digit form → should NOT be disputed.

        This is the exact pattern from the bug report:
          original: "Ибо пятого августа тысяча семьсот девяносто четвёртого года ..."
          whisper:  "Ибо 5 августа 1794 года ..."
        """
        if self._skip:
            self.skipTest(f"pre-compare normalizer unavailable: {self._skip}")

        original = (
            "Ибо пятого августа тысяча семьсот девяносто четвёртого года "
            "Франсуа Лантенас в ходатайстве об освобождении Пэйна писал следующее."
        )
        transcription = (
            "Ибо 5 августа 1794 года "
            "Франсуа Лантенас в ходатайстве об освобождении Пэйна писал следующее."
        )
        counters = self._run_one_file(original, transcription, threshold=0.94)
        self.assertEqual(
            counters["disputed"], 0,
            "False disputed: Whisper digit form should match word-form original after pre-compare",
        )
        self.assertEqual(counters["checked"], 1)

    def test_both_digit_form_not_disputed(self):
        """
        original_text has digits (pre-normalisation text from .txt file),
        Whisper also returns digits → both normalised to words → NOT disputed.
        """
        if self._skip:
            self.skipTest(f"pre-compare normalizer unavailable: {self._skip}")

        original = "5 августа 1794 года он писал следующее."
        transcription = "5 августа 1794 года он писал следующее."
        counters = self._run_one_file(original, transcription, threshold=0.94)
        self.assertEqual(counters["disputed"], 0)
        self.assertEqual(counters["checked"], 1)

    def test_without_pre_compare_digits_cause_low_similarity(self):
        """
        Control: WITHOUT pre_compare, digit form vs word form has very low similarity.
        Documents the broken behaviour that the fix resolves.
        """
        original_word_form = (
            "пятого августа тысяча семьсот девяносто четвёртого года"
        )
        transcription_digit_form = "5 августа 1794 года"

        orig_norm = _normalize_for_compare(original_word_form)
        trans_norm = _normalize_for_compare(transcription_digit_form)

        from difflib import SequenceMatcher
        sim = SequenceMatcher(None, orig_norm, trans_norm).ratio()
        self.assertLess(
            sim, 0.60,
            f"Expected low similarity without pre_compare, got {sim:.3f}; "
            f"orig={orig_norm!r}, trans={trans_norm!r}",
        )

    def test_with_pre_compare_digits_match_words(self):
        """
        WITH pre_compare applied to BOTH sides, same texts reach high similarity.
        """
        if self._skip:
            self.skipTest(f"pre-compare normalizer unavailable: {self._skip}")

        original_word_form = (
            "пятого августа тысяча семьсот девяносто четвёртого года"
        )
        transcription_digit_form = "5 августа 1794 года"

        orig_norm = _normalize_for_compare(self._fn(original_word_form))
        trans_norm = _normalize_for_compare(self._fn(transcription_digit_form))

        from difflib import SequenceMatcher
        sim = SequenceMatcher(None, orig_norm, trans_norm).ratio()
        self.assertGreater(
            sim, 0.90,
            f"Expected high similarity with pre_compare on both sides, got {sim:.3f}; "
            f"orig={orig_norm!r}, trans={trans_norm!r}",
        )

    def test_year_only_whisper_digit_vs_word(self):
        """'тысяча девятьсот семнадцатый год' vs Whisper '1917 год' → not disputed."""
        if self._skip:
            self.skipTest(f"pre-compare normalizer unavailable: {self._skip}")

        original = "тысяча девятьсот семнадцатый год"
        transcription = "1917 год"
        counters = self._run_one_file(original, transcription, threshold=0.94)
        self.assertEqual(counters["disputed"], 0, "Year-only false disputed")

    def test_pure_text_no_numbers_unchanged(self):
        """Text without numbers: pre_compare must not break plain Russian text."""
        if self._skip:
            self.skipTest(f"pre-compare normalizer unavailable: {self._skip}")

        text = "Привет мир это обычный текст без цифр"
        result = self._fn(text)
        norm_before = _normalize_for_compare(text)
        norm_after = _normalize_for_compare(result)
        self.assertEqual(norm_before, norm_after, "pre_compare must not touch plain text")



