# Copyright (c) 2026 alboro <alboro@users.noreply.github.com>
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See: https://polyformproject.org/licenses/noncommercial/1.0.0/

import unittest
from unittest.mock import MagicMock

from audiobook_generator.config.general_config import GeneralConfig
from audiobook_generator.normalizers.ru_time_normalizer import (
    TimeRuNormalizer,
    _HOUR_ORD_GEN,
    _BEZ_CHETVERTI_HOUR,
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
        normalize_steps="ru_time",
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


class TestTimeRuNormalizerTables(unittest.TestCase):
    """Sanity-checks for the lookup tables themselves."""

    def test_hour_ord_gen_covers_1_to_23(self):
        self.assertEqual(set(_HOUR_ORD_GEN.keys()), set(range(1, 24)))

    def test_bez_chetverti_covers_1_to_23(self):
        self.assertEqual(set(_BEZ_CHETVERTI_HOUR.keys()), set(range(1, 24)))

    def test_hour_1_is_pervogo(self):
        self.assertEqual(_HOUR_ORD_GEN[1], "первого")

    def test_hour_12_is_dvenadtsatogo(self):
        self.assertEqual(_HOUR_ORD_GEN[12], "двенадцатого")

    def test_hour_21_is_dvadtsat_pervogo(self):
        self.assertEqual(_HOUR_ORD_GEN[21], "двадцать первого")

    def test_bez_hour_1_is_chas(self):
        # "без четверти час" — special form for 1 o'clock
        self.assertEqual(_BEZ_CHETVERTI_HOUR[1], "час")

    def test_bez_hour_12_is_dvenadtsat(self):
        self.assertEqual(_BEZ_CHETVERTI_HOUR[12], "двенадцать")

    def test_bez_hour_23_is_dvadtsat_tri(self):
        self.assertEqual(_BEZ_CHETVERTI_HOUR[23], "двадцать три")


class TestTimeRuNormalizerSpecialTimes(unittest.TestCase):
    def setUp(self):
        self.n = TimeRuNormalizer(make_config())

    def test_midnight_0_00(self):
        self.assertEqual(self.n.normalize("0:00"), "полночь")

    def test_midnight_in_sentence(self):
        result = self.n.normalize("Встреча в 0:00.")
        self.assertIn("полночь", result)

    def test_noon_12_00(self):
        self.assertEqual(self.n.normalize("12:00"), "полдень")

    def test_noon_in_sentence(self):
        result = self.n.normalize("В 12:00 начнётся обед.")
        self.assertIn("полдень", result)


class TestTimeRuNormalizerQuarterPast(unittest.TestCase):
    """H:15 → четверть <gen_ordinal(H+1)>"""

    def setUp(self):
        self.n = TimeRuNormalizer(make_config())

    def test_0_15_chetvert_pervogo(self):
        self.assertEqual(self.n.normalize("0:15"), "четверть первого")

    def test_1_15_chetvert_vtorogo(self):
        self.assertEqual(self.n.normalize("1:15"), "четверть второго")

    def test_3_15_chetvert_chetvertogo(self):
        self.assertEqual(self.n.normalize("3:15"), "четверть четвёртого")

    def test_11_15_chetvert_dvenadtsatogo(self):
        self.assertEqual(self.n.normalize("11:15"), "четверть двенадцатого")

    def test_12_15_chetvert_trinadtsatogo(self):
        self.assertEqual(self.n.normalize("12:15"), "четверть тринадцатого")

    def test_20_15_chetvert_dvadtsatogo(self):
        # 20:15 → next_hour=21 → "двадцать первого"
        self.assertEqual(self.n.normalize("20:15"), "четверть двадцать первого")

    def test_22_15_chetvert_dvadtsat_tretego(self):
        # 22:15 → next_hour=23 → "двадцать третьего"
        self.assertEqual(self.n.normalize("22:15"), "четверть двадцать третьего")

    def test_23_15_left_unchanged(self):
        # 23:15 → next_hour=24 > 23 → simple format left for ru_numbers
        self.assertEqual(self.n.normalize("23:15"), "23:15")

    def test_in_sentence(self):
        result = self.n.normalize("Он пришёл в 9:15 утра.")
        self.assertIn("четверть десятого", result)


class TestTimeRuNormalizerHalfPast(unittest.TestCase):
    """H:30 → половина <gen_ordinal(H+1)>"""

    def setUp(self):
        self.n = TimeRuNormalizer(make_config())

    def test_0_30_polovina_pervogo(self):
        self.assertEqual(self.n.normalize("0:30"), "половина первого")

    def test_1_30_polovina_vtorogo(self):
        self.assertEqual(self.n.normalize("1:30"), "половина второго")

    def test_5_30_polovina_shestogo(self):
        self.assertEqual(self.n.normalize("5:30"), "половина шестого")

    def test_12_30_polovina_trinadtsatogo(self):
        self.assertEqual(self.n.normalize("12:30"), "половина тринадцатого")

    def test_21_30_polovina_dvadtsat_vtorogo(self):
        self.assertEqual(self.n.normalize("21:30"), "половина двадцать второго")

    def test_22_30_polovina_dvadtsat_tretego(self):
        self.assertEqual(self.n.normalize("22:30"), "половина двадцать третьего")

    def test_23_30_left_unchanged(self):
        self.assertEqual(self.n.normalize("23:30"), "23:30")

    def test_in_sentence(self):
        result = self.n.normalize("Встреча в 10:30 в офисе.")
        self.assertIn("половина одиннадцатого", result)


class TestTimeRuNormalizerQuarterTo(unittest.TestCase):
    """H:45 → без четверти <cardinal(H+1)>"""

    def setUp(self):
        self.n = TimeRuNormalizer(make_config())

    def test_0_45_bez_chetverti_chas(self):
        # next_hour=1 → "час"
        self.assertEqual(self.n.normalize("0:45"), "без четверти час")

    def test_1_45_bez_chetverti_dva(self):
        self.assertEqual(self.n.normalize("1:45"), "без четверти два")

    def test_3_45_bez_chetverti_chetyre(self):
        self.assertEqual(self.n.normalize("3:45"), "без четверти четыре")

    def test_11_45_bez_chetverti_dvenadtsat(self):
        self.assertEqual(self.n.normalize("11:45"), "без четверти двенадцать")

    def test_12_45_bez_chetverti_trinadtsat(self):
        self.assertEqual(self.n.normalize("12:45"), "без четверти тринадцать")

    def test_20_45_bez_chetverti_dvadtsat_odin(self):
        # next_hour=21 → "двадцать один"
        self.assertEqual(self.n.normalize("20:45"), "без четверти двадцать один")

    def test_22_45_bez_chetverti_dvadtsat_tri(self):
        self.assertEqual(self.n.normalize("22:45"), "без четверти двадцать три")

    def test_23_45_left_unchanged(self):
        self.assertEqual(self.n.normalize("23:45"), "23:45")

    def test_in_sentence(self):
        result = self.n.normalize("Без опоздания — в 7:45.")
        self.assertIn("без четверти восемь", result)


class TestTimeRuNormalizerPassthrough(unittest.TestCase):
    """Times not handled by ru_time are left unchanged for ru_numbers."""

    def setUp(self):
        self.n = TimeRuNormalizer(make_config())

    def test_regular_time_15_30_unchanged(self):
        # Non-:00/:15/:30/:45 ... wait, 15:30 IS :30 — should be converted
        # Let's test a non-special minute value instead
        self.assertEqual(self.n.normalize("15:20"), "15:20")

    def test_regular_time_8_00_unchanged(self):
        # 8:00 is not полночь or полдень, passes through
        self.assertEqual(self.n.normalize("8:00"), "8:00")

    def test_regular_time_23_00_unchanged(self):
        self.assertEqual(self.n.normalize("23:00"), "23:00")

    def test_regular_time_6_07_unchanged(self):
        self.assertEqual(self.n.normalize("6:07"), "6:07")

    def test_regular_time_14_55_unchanged(self):
        self.assertEqual(self.n.normalize("14:55"), "14:55")

    def test_invalid_hours_not_matched(self):
        # "25:00" doesn't match the TIME_PATTERN (hours > 23)
        self.assertEqual(self.n.normalize("25:00"), "25:00")

    def test_invalid_minutes_not_matched(self):
        # "3:60" doesn't match (minutes >= 60)
        self.assertEqual(self.n.normalize("3:60"), "3:60")

    def test_no_time_in_text(self):
        text = "Привет, мир!"
        self.assertEqual(self.n.normalize(text), text)


class TestTimeRuNormalizerMultiple(unittest.TestCase):
    """Multiple time tokens in one string."""

    def setUp(self):
        self.n = TimeRuNormalizer(make_config())

    def test_two_times_in_sentence(self):
        result = self.n.normalize("Расписание: 9:15 и 10:30.")
        self.assertIn("четверть десятого", result)
        self.assertIn("половина одиннадцатого", result)

    def test_mixed_converted_and_passthrough(self):
        result = self.n.normalize("В 0:00 — полночь, в 8:00 — подъём.")
        self.assertIn("полночь", result)
        self.assertIn("8:00", result)  # 8:00 passes through to ru_numbers

    def test_noon_and_quarter(self):
        result = self.n.normalize("Обед в 12:00, конец в 12:15.")
        self.assertIn("полдень", result)
        self.assertIn("четверть тринадцатого", result)


class TestTimeRuNormalizerLanguage(unittest.TestCase):
    def test_non_russian_language_skip(self):
        n = TimeRuNormalizer(make_config(language="en-US"))
        result = n.normalize("Meet at 3:15.")
        self.assertEqual(result, "Meet at 3:15.")

    def test_russian_language_processed(self):
        n = TimeRuNormalizer(make_config(language="ru-RU"))
        self.assertEqual(n.normalize("3:15"), "четверть четвёртого")
