# Copyright (c) 2026 alboro <alboro@users.noreply.github.com>
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See: https://polyformproject.org/licenses/noncommercial/1.0.0/

import unittest
from unittest.mock import MagicMock

from audiobook_generator.config.general_config import GeneralConfig
from audiobook_generator.normalizers.ru_roman_numerals_normalizer import (
    RomanNumeralsRuNormalizer,
    _roman_to_int,
    _to_ordinal_ru,
)


def make_config(**overrides):
    values = dict(
        input_file="examples/test.epub",
        output_folder="output",
        output_text=False,
        prepare_text=False,
        prepared_text_folder=None,
        log="INFO",
        no_prompt=True,
        worker_count=1,
        use_pydub_merge=False,
        package_m4b=False,
        m4b_filename=None,
        m4b_bitrate="64k",
        ffmpeg_path="ffmpeg",
        title_mode="auto",
        chapter_mode="documents",
        newline_mode="double",
        chapter_start=1,
        chapter_end=-1,
        search_and_replace_file="",
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
        normalize=True,
        normalize_steps="ru_roman_numerals",
        normalize_provider="openai",
        normalize_model="gpt-4o-mini",
        normalize_prompt_file=None,
        normalize_system_prompt_file=None,
        normalize_user_prompt_file=None,
        normalize_api_key=None,
        normalize_base_url=None,
        normalize_max_chars=4000,
        normalize_tts_safe_max_chars=180,
        normalize_pronunciation_exceptions_file=None,
        normalize_tts_pronunciation_overrides_file=None,
        normalize_pronunciation_lexicon_db=None,
        normalize_stress_ambiguity_file=None,
        normalize_tsnorm_stress_yo=True,
        normalize_tsnorm_stress_monosyllabic=False,
        normalize_tsnorm_min_word_length=2,
        normalize_log_changes=False,
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
    )
    values.update(overrides)
    return GeneralConfig(MagicMock(**values))


class TestRomanToInt(unittest.TestCase):
    """Unit tests for the _roman_to_int helper."""

    def test_i_is_1(self):
        self.assertEqual(_roman_to_int("I"), 1)

    def test_iv_is_4(self):
        self.assertEqual(_roman_to_int("IV"), 4)

    def test_v_is_5(self):
        self.assertEqual(_roman_to_int("V"), 5)

    def test_ix_is_9(self):
        self.assertEqual(_roman_to_int("IX"), 9)

    def test_xiv_is_14(self):
        self.assertEqual(_roman_to_int("XIV"), 14)

    def test_xvii_is_17(self):
        self.assertEqual(_roman_to_int("XVII"), 17)

    def test_xl_is_40(self):
        self.assertEqual(_roman_to_int("XL"), 40)

    def test_xc_is_90(self):
        self.assertEqual(_roman_to_int("XC"), 90)

    def test_cd_is_400(self):
        self.assertEqual(_roman_to_int("CD"), 400)

    def test_mcmxcix_is_1999(self):
        self.assertEqual(_roman_to_int("MCMXCIX"), 1999)

    def test_invalid_vv_returns_none(self):
        self.assertIsNone(_roman_to_int("VV"))

    def test_invalid_iiii_returns_none(self):
        self.assertIsNone(_roman_to_int("IIII"))

    def test_invalid_lcd_returns_none(self):
        self.assertIsNone(_roman_to_int("LCD"))

    def test_invalid_empty_returns_none(self):
        self.assertIsNone(_roman_to_int(""))

    def test_lowercase_iv_is_4(self):
        # Validator is case-insensitive
        self.assertEqual(_roman_to_int("iv"), 4)


class TestRomanNumeralsRuNormalizerMultiChar(unittest.TestCase):
    def setUp(self):
        self.n = RomanNumeralsRuNormalizer(make_config())

    def test_standalone_xvii(self):
        self.assertEqual(self.n.normalize("XVII"), "семнадцатый")

    def test_standalone_xiv(self):
        self.assertEqual(self.n.normalize("XIV"), "четырнадцатый")

    def test_after_cyrillic_name(self):
        # "Людовик XIV" — space before XIV is not Cyrillic, so it matches
        self.assertEqual(self.n.normalize("Людовик XIV"), "Людовик четырнадцатый")

    def test_with_trailing_period(self):
        self.assertEqual(self.n.normalize("Раздел II. Основные положения"),
                         "Раздел второй. Основные положения")

    def test_with_trailing_comma(self):
        # Normalizer outputs nominative masculine — grammatical agreement not attempted
        self.assertEqual(self.n.normalize("в главе III, которая"), "в главе третий, которая")

    def test_iv(self):
        self.assertEqual(self.n.normalize("IV"), "четвёртый")

    def test_mcmxcix(self):
        self.assertEqual(self.n.normalize("MCMXCIX"), "тысяча девятьсот девяносто девятый")

    def test_adjacent_cyrillic_left_prevents_match(self):
        # "ПрограммаIII" — Cyrillic letter immediately before Roman numeral
        result = self.n.normalize("ПрограммаIII")
        self.assertIn("III", result)
        self.assertNotIn("третий", result)

    def test_adjacent_latin_right_prevents_match(self):
        # "IVx" — Latin letter immediately after Roman numeral
        result = self.n.normalize("IVx")
        self.assertIn("IVx", result)
        self.assertNotIn("четвёртый", result)

    def test_adjacent_digit_prevents_match(self):
        # "IV2" — digit immediately after
        result = self.n.normalize("IV2")
        self.assertIn("IV2", result)
        self.assertNotIn("четвёртый", result)

    def test_invalid_vv_passthrough(self):
        self.assertEqual(self.n.normalize("VV"), "VV")

    def test_invalid_lcd_passthrough(self):
        self.assertEqual(self.n.normalize("LCD"), "LCD")

    def test_cd_known_false_positive(self):
        # CD = 400, known acceptable false positive
        result = self.n.normalize("CD диск")
        self.assertIn("четырёхсотый", result)

    def test_ii_at_start(self):
        self.assertEqual(self.n.normalize("II. Введение"), "второй. Введение")

    def test_in_sentence(self):
        # Nominative masculine output — agreement with context ("главу") not attempted
        result = self.n.normalize("Смотри главу XXI для деталей.")
        self.assertIn("двадцать первый", result)

    def test_vi_alone(self):
        self.assertEqual(self.n.normalize("VI"), "шестой")

    def test_xl(self):
        self.assertEqual(self.n.normalize("XL"), "сороковой")


class TestRomanNumeralsRuNormalizerIHeading(unittest.TestCase):
    def setUp(self):
        self.n = RomanNumeralsRuNormalizer(make_config())

    def test_i_dot_at_line_start(self):
        self.assertEqual(self.n.normalize("I. Введение"), "первый. Введение")

    def test_i_comma_at_line_start(self):
        self.assertEqual(self.n.normalize("I, часть"), "первый, часть")

    def test_i_dash_at_line_start(self):
        self.assertEqual(self.n.normalize("I— начало"), "первый— начало")

    def test_i_in_multiline_text(self):
        text = "Предисловие\nI. Введение\nII. Основы"
        result = self.n.normalize(text)
        self.assertIn("первый. Введение", result)
        self.assertIn("второй. Основы", result)

    def test_i_not_at_line_start_not_converted(self):
        # Mid-sentence "I" is not touched (avoids false positives in English text)
        result = self.n.normalize("В разделе I есть подраздел.")
        # "I" is not at line start, so _ROMAN_I_HEADING won't match;
        # but _ROMAN_MULTI won't match it either (only 1 char)
        self.assertIn(" I ", result)
        self.assertNotIn("первый", result)

    def test_i_not_at_line_start_with_text_before(self):
        result = self.n.normalize("Глава I. Общие")
        # "I" here is mid-line, single char — not converted by either pattern
        self.assertIn("I", result)

    def test_standalone_i_without_punctuation_not_converted(self):
        # Single "I" at line start but no following punctuation — not converted
        result = self.n.normalize("I думаю")
        self.assertIn("I", result)


class TestRomanNumeralsRuNormalizerLanguage(unittest.TestCase):
    def test_non_russian_language_skip(self):
        n = RomanNumeralsRuNormalizer(make_config(language="en-US"))
        result = n.normalize("Chapter XIV")
        # Should be unchanged
        self.assertEqual(result, "Chapter XIV")

    def test_russian_language_processed(self):
        n = RomanNumeralsRuNormalizer(make_config(language="ru-RU"))
        result = n.normalize("XIV")
        self.assertEqual(result, "четырнадцатый")


class TestRomanNumeralsEdgeCases(unittest.TestCase):
    def setUp(self):
        self.n = RomanNumeralsRuNormalizer(make_config())

    def test_no_roman_in_plain_text(self):
        text = "Привет, мир! Всё хорошо."
        self.assertEqual(self.n.normalize(text), text)

    def test_empty_string(self):
        self.assertEqual(self.n.normalize(""), "")

    def test_multiple_numerals_in_text(self):
        result = self.n.normalize("Главы II и IV посвящены теме.")
        self.assertIn("второй", result)
        self.assertIn("четвёртый", result)

    def test_numeral_already_converted_by_ru_numbers(self):
        # After ru_numbers, "XVII глава" becomes "семнадцатый глава" — no Roman numeral left
        text = "семнадцатый глава"
        self.assertEqual(self.n.normalize(text), text)

    def test_xix_century(self):
        # "XIX" standalone (after ru_numbers already handled "XIX век")
        self.assertEqual(self.n.normalize("XIX"), "девятнадцатый")

    def test_m_is_too_short_for_multi_but_valid(self):
        # "M" is a single char, not matched by _ROMAN_MULTI (requires ≥2)
        # and not at line start before punctuation for _ROMAN_I_HEADING
        result = self.n.normalize("М. Горький")  # Cyrillic М, not Latin
        self.assertEqual(result, "М. Горький")

    def test_latin_m_single_not_converted(self):
        # Single Latin "M" in middle of text — not converted
        result = self.n.normalize("размер M одежды")
        self.assertIn("M", result)
        self.assertNotIn("тысячный", result)
