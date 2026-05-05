# Copyright (c) 2026 alboro <alboro@users.noreply.github.com>
# Licensed under the PolyForm Noncommercial License 1.0.0.

"""Tests for CosyVoiceNormalizer."""

import pytest
from unittest.mock import MagicMock

from audiobook_generator.normalizers.cosy_voice_normalizer import CosyVoiceNormalizer


def make_normalizer():
    config = MagicMock()
    return CosyVoiceNormalizer(config)


# ---------------------------------------------------------------------------
# Rule 1: capitalize first letter
# ---------------------------------------------------------------------------

class TestCapitalize:
    def test_lowercase_first_letter(self):
        n = make_normalizer()
        assert n.normalize("и вот однажды") == "- И вот однажды"

    def test_uppercase_unchanged(self):
        n = make_normalizer()
        assert n.normalize("Солнце светило ярко.") == "Солнце светило ярко."

    def test_empty_string(self):
        n = make_normalizer()
        assert n.normalize("") == ""

    def test_single_lowercase_letter(self):
        n = make_normalizer()
        assert n.normalize("а") == "- А"

    def test_single_uppercase_letter(self):
        n = make_normalizer()
        assert n.normalize("А") == "- А"


# ---------------------------------------------------------------------------
# Rule 2: single-letter leading word → prepend "- "
# ---------------------------------------------------------------------------

class TestSingleLetterLeadingWord:
    def test_conjunction_a(self):
        n = make_normalizer()
        assert n.normalize("А было дело так") == "- А было дело так"

    def test_preposition_v(self):
        n = make_normalizer()
        assert n.normalize("В начале было слово") == "- В начале было слово"

    def test_conjunction_i(self):
        n = make_normalizer()
        assert n.normalize("И тут он понял") == "- И тут он понял"

    def test_normal_word_no_dash(self):
        n = make_normalizer()
        assert n.normalize("Он пришёл домой.") == "Он пришёл домой."

    def test_two_letter_first_word_no_dash(self):
        n = make_normalizer()
        assert n.normalize("Он пошёл.") == "Он пошёл."

    def test_abbreviation_not_triggered(self):
        # "В." — period right after the letter means it's not a standalone word
        n = make_normalizer()
        assert n.normalize("В. Иванов написал.") == "В. Иванов написал."

    def test_already_has_dash_prefix(self):
        # Text starting with "— " is not a letter, rule 2 not triggered
        n = make_normalizer()
        assert n.normalize("— А вот и нет.") == "— А вот и нет."

    def test_lowercase_single_letter_both_rules(self):
        # Both rules: capitalize + prepend dash
        n = make_normalizer()
        assert n.normalize("а потом всё изменилось") == "- А потом всё изменилось"

    def test_single_letter_end_of_text(self):
        n = make_normalizer()
        assert n.normalize("А") == "- А"

    def test_no_double_dash(self):
        # Should not prepend twice if called twice
        n = make_normalizer()
        result = n.normalize("А вот так")
        assert result == "- А вот так"
        assert not result.startswith("- - ")
