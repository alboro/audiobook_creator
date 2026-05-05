# Copyright (c) 2026 alboro <alboro@users.noreply.github.com>
# Licensed under the PolyForm Noncommercial License 1.0.0.

"""Tests for TTSHardConsonantsNormalizer."""

import pytest
from unittest.mock import MagicMock

from audiobook_generator.normalizers.tts_hard_consonants_normalizer import (
    TTSHardConsonantsNormalizer,
    BUILTIN_TTS_HARD_CONSONANTS_OVERRIDES,
    _parse_inline_overrides,
)


def make_normalizer(language="ru", words=None, file=None):
    config = MagicMock()
    config.language = language
    config.normalize_pronunciation_lexicon_db = None
    config.normalize_tts_hard_consonants_words = words
    config.normalize_tts_hard_consonants_file = file
    config.normalize_tts_pronunciation_overrides_file = None
    return TTSHardConsonantsNormalizer(config)


# ---------------------------------------------------------------------------
# Composite pattern: интернет
# ---------------------------------------------------------------------------

class TestИнтернет:
    def test_basic(self):
        n = make_normalizer()
        assert n.normalize("интернет") == "интэрнэт"

    def test_capitalized(self):
        n = make_normalizer()
        assert n.normalize("Интернет") == "Интэрнэт"

    def test_all_caps(self):
        n = make_normalizer()
        assert n.normalize("ИНТЕРНЕТ") == "ИНТЭРНЭТ"

    def test_mid_sentence(self):
        n = make_normalizer()
        result = n.normalize("доступ в интернет есть")
        assert "интэрнэт" in result

    def test_not_matched_inside_word(self):
        # The composite "интернет" pattern has _WB so it won't fire,
        # but the general (?<=ин)те(?=р+consonant) sub-pattern still matches.
        # зинтернет → зинтэрнет is the expected (correct) behaviour.
        n = make_normalizer()
        assert n.normalize("зинтернет") == "зинтэрнет"

    def test_genitive(self):
        n = make_normalizer()
        # "интернете" — still starts with "интернет"
        result = n.normalize("в интернете")
        assert "интэрнэт" in result


# ---------------------------------------------------------------------------
# Composite pattern: менедж
# ---------------------------------------------------------------------------

class TestМенедж:
    def test_менеджер(self):
        n = make_normalizer()
        assert n.normalize("менеджер") == "мэнэджер"

    def test_менеджмент(self):
        n = make_normalizer()
        assert n.normalize("менеджмент") == "мэнэджмент"

    def test_capitalized(self):
        n = make_normalizer()
        assert n.normalize("Менеджер") == "Мэнэджер"

    def test_not_inside_word(self):
        n = make_normalizer()
        assert n.normalize("суперменеджер") == "суперменеджер"


# ---------------------------------------------------------------------------
# де → дэ patterns
# ---------------------------------------------------------------------------

class TestДэ:
    def test_модем(self):
        n = make_normalizer()
        assert n.normalize("модем") == "модэм"

    def test_модель(self):
        n = make_normalizer()
        assert "модэл" in n.normalize("модель")

    def test_индекс(self):
        n = make_normalizer()
        assert n.normalize("индекс") == "индэкс"

    def test_шедевр(self):
        n = make_normalizer()
        assert n.normalize("шедевр") == "шэдэвр"

    def test_тенденция(self):
        n = make_normalizer()
        assert "дэнц" in n.normalize("тенденция")


# ---------------------------------------------------------------------------
# те → тэ patterns
# ---------------------------------------------------------------------------

class TestТэ:
    def test_компьютер(self):
        n = make_normalizer()
        assert n.normalize("компьютер") == "компьютэр"

    def test_принтер(self):
        n = make_normalizer()
        assert n.normalize("принтер") == "принтэр"

    def test_контент(self):
        n = make_normalizer()
        assert n.normalize("контент") == "контэнт"

    def test_контент_not_inside(self):
        n = make_normalizer()
        # preceded by Cyrillic → no match
        assert n.normalize("мегаконтент") == "мегаконтент"

    def test_бактерия(self):
        n = make_normalizer()
        assert "тэр" in n.normalize("бактерия")

    def test_альтернатива(self):
        n = make_normalizer()
        assert "тэр" in n.normalize("альтернатива")

    def test_антенна(self):
        n = make_normalizer()
        assert n.normalize("антенна") == "антэнна"

    def test_роутер(self):
        n = make_normalizer()
        assert n.normalize("роутер") == "роутэр"

    def test_бартер(self):
        n = make_normalizer()
        assert n.normalize("бартер") == "бартэр"

    def test_синтез(self):
        n = make_normalizer()
        assert n.normalize("синтез") == "синтэз"

    def test_синтетика(self):
        n = make_normalizer()
        assert "тэт" in n.normalize("синтетика")

    def test_протез(self):
        n = make_normalizer()
        assert "тэз" in n.normalize("протез")

    def test_эстет(self):
        n = make_normalizer()
        assert "тэт" in n.normalize("эстет")

    def test_интенсивный(self):
        n = make_normalizer()
        assert "тэнс" in n.normalize("интенсивный")


# ---------------------------------------------------------------------------
# не → нэ patterns
# ---------------------------------------------------------------------------

class TestНэ:
    def test_бизнес(self):
        n = make_normalizer()
        assert n.normalize("бизнес") == "бизнэс"

    def test_кларнет(self):
        n = make_normalizer()
        assert n.normalize("кларнет") == "кларнэт"

    def test_кларнет_not_inside(self):
        n = make_normalizer()
        assert n.normalize("суперкларнет") == "суперкларнет"

    def test_энергия(self):
        n = make_normalizer()
        assert "нэрг" in n.normalize("энергия")


# ---------------------------------------------------------------------------
# ле → лэ patterns
# ---------------------------------------------------------------------------

class TestЛэ:
    def test_плеер(self):
        n = make_normalizer()
        assert n.normalize("плеер") == "плэер"


# ---------------------------------------------------------------------------
# Builtin word-level overrides (отель)
# ---------------------------------------------------------------------------

class TestОтель:
    def test_отель_basic(self):
        n = make_normalizer()
        assert n.normalize("отель") == "отэль"

    def test_отели(self):
        n = make_normalizer()
        assert n.normalize("отели") == "отэли"

    def test_отелей(self):
        n = make_normalizer()
        assert n.normalize("отелей") == "отэлей"

    def test_отель_capitalized(self):
        n = make_normalizer()
        assert n.normalize("Отель") == "Отэль"

    def test_отель_all_caps(self):
        n = make_normalizer()
        assert n.normalize("ОТЕЛЬ") == "ОТЭЛЬ"


# ---------------------------------------------------------------------------
# Inline (manual) word overrides via config
# ---------------------------------------------------------------------------

class TestInlineOverrides:
    def test_custom_word(self):
        n = make_normalizer(words="тест=тэст")
        assert n.normalize("тест") == "тэст"

    def test_multiple_words(self):
        n = make_normalizer(words="кафе=кафэ,резюме=рэзюмэ")
        assert n.normalize("кафе") == "кафэ"
        assert n.normalize("резюме") == "рэзюмэ"

    def test_override_takes_precedence_over_builtin(self):
        n = make_normalizer(words="отель=оTELь")
        # Manual override wins; any value accepted.
        result = n.normalize("отель")
        assert result == "оTELь"


# ---------------------------------------------------------------------------
# Non-Russian language: passthrough
# ---------------------------------------------------------------------------

class TestLanguageGuard:
    def test_english_passthrough(self):
        n = make_normalizer(language="en")
        text = "компьютер бизнес интернет"
        assert n.normalize(text) == text

    def test_russian_with_locale_subtag(self):
        n = make_normalizer(language="ru-RU")
        assert "тэр" in n.normalize("компьютер")


# ---------------------------------------------------------------------------
# No false positives for common words
# ---------------------------------------------------------------------------

class TestNoFalsePositives:
    def test_телефон_not_changed(self):
        # "те" in "телефон" is at word-start after nothing — not matched
        n = make_normalizer()
        assert n.normalize("телефон") == "телефон"

    def test_тема_not_changed(self):
        n = make_normalizer()
        assert n.normalize("тема") == "тема"

    def test_человек_not_changed(self):
        n = make_normalizer()
        assert n.normalize("человек") == "человек"

    def test_деревня_not_changed(self):
        n = make_normalizer()
        assert n.normalize("деревня") == "деревня"

    def test_energy_english_passthrough(self):
        n = make_normalizer(language="en-US")
        assert n.normalize("energy") == "energy"


# ---------------------------------------------------------------------------
# _parse_inline_overrides helper
# ---------------------------------------------------------------------------

class TestParseInlineOverrides:
    def test_single_pair(self):
        assert _parse_inline_overrides("тест=тэст") == {"тест": "тэст"}

    def test_multiple_pairs(self):
        result = _parse_inline_overrides("кафе=кафэ,резюме=рэзюмэ")
        assert result == {"кафе": "кафэ", "резюме": "рэзюмэ"}

    def test_empty(self):
        assert _parse_inline_overrides("") == {}
        assert _parse_inline_overrides(None) == {}

    def test_whitespace_trimmed(self):
        assert _parse_inline_overrides("  тест = тэст  ") == {"тест": "тэст"}

    def test_ignores_invalid(self):
        assert _parse_inline_overrides("notapair") == {}

    def test_keys_lowercased(self):
        result = _parse_inline_overrides("ТЕСТ=тэст")
        assert "тест" in result


# ---------------------------------------------------------------------------
# Builtin overrides completeness
# ---------------------------------------------------------------------------

class TestBuiltinOverrides:
    def test_all_hotel_forms_present(self):
        forms = [
            "отель", "отеля", "отелю", "отелем", "отеле",
            "отели", "отелей", "отелям", "отелями", "отелях",
        ]
        for form in forms:
            assert form in BUILTIN_TTS_HARD_CONSONANTS_OVERRIDES, f"Missing: {form}"

    def test_replacements_use_э(self):
        for src, rep in BUILTIN_TTS_HARD_CONSONANTS_OVERRIDES.items():
            assert "э" in rep, f"Replacement for '{src}' lacks э: '{rep}'"
