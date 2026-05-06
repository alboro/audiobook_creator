# Copyright (c) 2026 alboro <alboro@users.noreply.github.com>
# Licensed under the PolyForm Noncommercial License 1.0.0.

"""Tests for TranscriptionArtifactsChecker."""

from __future__ import annotations

import types
from pathlib import Path

import pytest

from audiobook_generator.core.audio_checkers.transcription_artifacts_checker import (
    TranscriptionArtifactsChecker,
    _parse_artifacts,
)


# ---------------------------------------------------------------------------
# _parse_artifacts helper
# ---------------------------------------------------------------------------

class TestParseArtifacts:
    def test_none_returns_empty(self):
        assert _parse_artifacts(None) == []

    def test_empty_string_returns_empty(self):
        assert _parse_artifacts("") == []

    def test_whitespace_only_returns_empty(self):
        assert _parse_artifacts("   ,  ,  ") == []

    def test_single_pattern(self):
        assert _parse_artifacts("точка") == ["точка"]

    def test_multiple_patterns(self):
        assert _parse_artifacts("точка,очка") == ["точка", "очка"]

    def test_strips_whitespace(self):
        assert _parse_artifacts("  точка , очка  ") == ["точка", "очка"]

    def test_lowercases_patterns(self):
        assert _parse_artifacts("ТОЧКА,Очка") == ["точка", "очка"]

    def test_skips_empty_entries(self):
        assert _parse_artifacts("точка,,очка,") == ["точка", "очка"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(patterns: str | None = None):
    return types.SimpleNamespace(audio_checker_transcription_artifacts=patterns)


DUMMY_PATH = Path("/tmp/dummy_chunk.wav")


# ---------------------------------------------------------------------------
# TranscriptionArtifactsChecker.check()
# ---------------------------------------------------------------------------

class TestTranscriptionArtifactsCheckerCheck:

    # --- no-patterns cases ---

    def test_no_patterns_always_passes(self):
        checker = TranscriptionArtifactsChecker(_make_config(None))
        result = checker.check(DUMMY_PATH, "оригинал", "любая транскрипция", None)
        assert result.disputed is False

    def test_empty_patterns_always_passes(self):
        checker = TranscriptionArtifactsChecker(_make_config(""))
        result = checker.check(DUMMY_PATH, "оригинал", "точка очка", None)
        assert result.disputed is False

    # --- basic matching ---

    def test_exact_pattern_found_disputed(self):
        checker = TranscriptionArtifactsChecker(_make_config("точка"))
        result = checker.check(DUMMY_PATH, "оригинал", "это была точка зрения", None)
        assert result.disputed is True

    def test_substring_match_disputed(self):
        """'очка' is a substring of 'точка' and of 'очках'."""
        checker = TranscriptionArtifactsChecker(_make_config("очка"))
        result = checker.check(DUMMY_PATH, "оригинал", "в этих очках ничего нет", None)
        assert result.disputed is True

    def test_pattern_not_found_passes(self):
        checker = TranscriptionArtifactsChecker(_make_config("точка,очка"))
        result = checker.check(DUMMY_PATH, "оригинал", "совершенно нормальный текст", None)
        assert result.disputed is False

    # --- case-insensitivity ---

    def test_case_insensitive_upper(self):
        checker = TranscriptionArtifactsChecker(_make_config("точка"))
        result = checker.check(DUMMY_PATH, "оригинал", "ТОЧКА в конце", None)
        assert result.disputed is True

    def test_case_insensitive_mixed(self):
        checker = TranscriptionArtifactsChecker(_make_config("точка"))
        result = checker.check(DUMMY_PATH, "оригинал", "Точка зрения", None)
        assert result.disputed is True

    def test_pattern_uppercase_normalised(self):
        """Pattern stored as uppercase in config should be lowercased internally."""
        checker = TranscriptionArtifactsChecker(_make_config("ТОЧКА"))
        result = checker.check(DUMMY_PATH, "оригинал", "обычная точка", None)
        assert result.disputed is True

    # --- first hit wins ---

    def test_first_pattern_triggers(self):
        checker = TranscriptionArtifactsChecker(_make_config("точка,очка"))
        result = checker.check(DUMMY_PATH, "оригинал", "точка в предложении", None)
        assert result.disputed is True

    def test_second_pattern_triggers(self):
        checker = TranscriptionArtifactsChecker(_make_config("точка,очка"))
        result = checker.check(DUMMY_PATH, "оригинал", "очка не должно быть", None)
        assert result.disputed is True

    # --- empty transcription ---

    def test_empty_transcription_passes(self):
        checker = TranscriptionArtifactsChecker(_make_config("точка,очка"))
        result = checker.check(DUMMY_PATH, "оригинал", "", None)
        assert result.disputed is False

    def test_whitespace_transcription_passes(self):
        checker = TranscriptionArtifactsChecker(_make_config("точка"))
        result = checker.check(DUMMY_PATH, "оригинал", "   ", None)
        assert result.disputed is False

    # --- chunk_cache_row is ignored ---

    def test_chunk_cache_row_ignored(self):
        checker = TranscriptionArtifactsChecker(_make_config("точка"))
        row = {"similarity": 0.95, "checker_transcription_artifacts_passed": 1}
        result = checker.check(DUMMY_PATH, "оригинал", "точка в конце", row)
        assert result.disputed is True

    # --- original_text is not checked ---

    def test_original_text_not_checked(self):
        """Artifacts are only looked for in transcription, not in original_text."""
        checker = TranscriptionArtifactsChecker(_make_config("точка"))
        result = checker.check(DUMMY_PATH, "точка в оригинале", "нормальная транскрипция", None)
        assert result.disputed is False

    # --- uses_fallback_passed_column ---

    def test_uses_fallback_passed_column(self):
        assert TranscriptionArtifactsChecker.uses_fallback_passed_column is True

    # --- name ---

    def test_name(self):
        assert TranscriptionArtifactsChecker.name == "transcription_artifacts"

    # --- evaluate_from_row (inherited default) ---

    def test_evaluate_from_row_passed(self):
        cfg = _make_config("точка")
        row = {"checker_transcription_artifacts_passed": 1}
        assert TranscriptionArtifactsChecker.evaluate_from_row(row, cfg) is True

    def test_evaluate_from_row_failed(self):
        cfg = _make_config("точка")
        row = {"checker_transcription_artifacts_passed": 0}
        assert TranscriptionArtifactsChecker.evaluate_from_row(row, cfg) is False

    def test_evaluate_from_row_missing(self):
        cfg = _make_config("точка")
        row = {}
        assert TranscriptionArtifactsChecker.evaluate_from_row(row, cfg) is None

    # --- score_from_row (binary-only) ---

    def test_score_from_row_none(self):
        cfg = _make_config("точка")
        assert TranscriptionArtifactsChecker.score_from_row({}, cfg) is None

    # --- real-world CosyVoice artifact scenarios ---

    def test_cosy_voice_period_as_tochka(self):
        """CosyVoice reads '.' as 'точка' which Whisper picks up."""
        checker = TranscriptionArtifactsChecker(_make_config("точка,очка"))
        result = checker.check(
            DUMMY_PATH,
            "Конец предложения.",
            "конец предложения точка",
            None,
        )
        assert result.disputed is True

    def test_cosy_voice_period_as_ochka_partial(self):
        """Whisper may transcribe only the end of 'точка' as 'очка'."""
        checker = TranscriptionArtifactsChecker(_make_config("точка,очка"))
        result = checker.check(
            DUMMY_PATH,
            "Конец предложения.",
            "конец предложения очка",
            None,
        )
        assert result.disputed is True

    def test_normal_sentence_passes(self):
        checker = TranscriptionArtifactsChecker(_make_config("точка,очка"))
        result = checker.check(
            DUMMY_PATH,
            "Это нормальное предложение без артефактов.",
            "это нормальное предложение без артефактов",
            None,
        )
        assert result.disputed is False
