# Copyright (c) 2026 alboro <alboro@users.noreply.github.com>
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See: https://polyformproject.org/licenses/noncommercial/1.0.0/

import unittest
from unittest.mock import MagicMock
from pathlib import Path
import tempfile
import json

from audiobook_generator.config.general_config import GeneralConfig
from audiobook_generator.normalizers.base_normalizer import BaseNormalizer
from audiobook_generator.normalizers.ru_abbreviations_normalizer import (
    AbbreviationsRuNormalizer,
    _expand_acronym,
)
from audiobook_generator.normalizers.ru_tsnorm_normalizer import TSNormRuNormalizer
from audiobook_generator.normalizers.ru_initials_normalizer import InitialsRuNormalizer
from audiobook_generator.normalizers.ru_numbers_normalizer import NumbersRuNormalizer
from audiobook_generator.normalizers.openai_normalizer import OpenAINormalizer
from audiobook_generator.core.pipeline_runner import NormalizationPipelineRunner
from audiobook_generator.normalizers.pronunciation_lexicon_db import (
    PronunciationLexiconDB,
    build_tsnorm_pronunciation_lexicon,
)
from audiobook_generator.normalizers.ru_proper_nouns_pronunciation_normalizer import (
    ProperNounsPronunciationRuNormalizer,
)
from audiobook_generator.normalizers.ru_proper_nouns_normalizer import ProperNounsRuNormalizer
from audiobook_generator.normalizers.tts_pronunciation_overrides_normalizer import (
    TTSPronunciationOverridesNormalizer,
)
from audiobook_generator.normalizers.ru_text_utils import (
    COMBINING_ACUTE,
    load_choice_mapping_file,
    plus_stress_to_combining_acute,
)
from audiobook_generator.normalizers.simple_symbols_normalizer import SimpleSymbolsNormalizer
from audiobook_generator.normalizers.ru_stress_ambiguity_normalizer import (
    StressAmbiguityLLMNormalizer,
)
from audiobook_generator.normalizers.tts_safe_split_normalizer import TTSSafeSplitNormalizer
from audiobook_generator.normalizers.llm_support import (
    NormalizerLLMChoiceService,
    NormalizerLLMChoiceItem,
    NormalizerLLMChoiceOption,
)


def make_config(**overrides):
    values = dict(
        input_file="examples/The_Life_and_Adventures_of_Robinson_Crusoe.epub",
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
        normalize_steps="ru_numbers",
        normalize_provider="openai",
        normalize_model="gpt-5.4",
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


def build_test_lexicon(
    db_path: Path,
    *,
    word_forms: dict,
    lemmas: dict,
) -> PronunciationLexiconDB:
    database = PronunciationLexiconDB(db_path)
    build_tsnorm_pronunciation_lexicon(
        database,
        word_forms=word_forms,
        lemmas=lemmas,
    )
    return database


class DummyNormalizer(BaseNormalizer):
    def validate_config(self):
        return None

    def normalize(self, text: str, chapter_title: str = "") -> str:
        return text


class DummyChunkedNormalizer(BaseNormalizer):
    STEP_NAME = "dummy_chunked"

    def __init__(self, config: GeneralConfig):
        self.process_calls = []
        super().__init__(config)

    def validate_config(self):
        return None

    def normalize(self, text: str, chapter_title: str = "") -> str:
        return text

    def supports_chunked_resume(self) -> bool:
        return True

    def plan_processing_units(self, text: str, chapter_title: str = "") -> list[str]:
        return [part for part in text.split("|") if part]

    def process_unit(
        self,
        unit: str,
        *,
        chapter_title: str = "",
        unit_index: int,
        unit_count: int,
    ) -> str:
        self.process_calls.append((unit_index, unit))
        return unit.upper()

    def get_step_artifacts(self, text: str, chapter_title: str = "") -> dict[str, str]:
        return {"00_system_prompt.txt": "dummy system"}

    def get_unit_artifacts(
        self,
        unit: str,
        *,
        chapter_title: str = "",
        unit_index: int,
        unit_count: int,
    ) -> dict[str, str]:
        return {"01_user_prompt.txt": f"prompt for {unit}"}


class FakeLLM:
    def __init__(self, *, settings, response_text):
        self.settings = settings
        self.response_text = response_text
        self.calls = 0

    def complete(self, **kwargs):
        self.calls += 1
        return self.response_text


class TestNumbersRuNormalizer(unittest.TestCase):
    def test_plain_cardinal(self):
        normalizer = NumbersRuNormalizer(make_config())
        self.assertEqual(normalizer.normalize("У меня 3 яблока."), "У меня три яблока.")

    def test_number_sign(self):
        normalizer = NumbersRuNormalizer(make_config())
        self.assertEqual(normalizer.normalize("Смотри №5."), "Смотри номер пять.")

    def test_arabic_century(self):
        normalizer = NumbersRuNormalizer(make_config())
        self.assertEqual(normalizer.normalize("17 век"), "семнадцатый век")

    def test_roman_century(self):
        normalizer = NumbersRuNormalizer(make_config())
        self.assertEqual(normalizer.normalize("XVII век"), "семнадцатый век")

    def test_ordinal_suffix(self):
        normalizer = NumbersRuNormalizer(make_config())
        self.assertEqual(normalizer.normalize("17-й век"), "семнадцатый век")

    def test_cardinal_with_feminine_book_noun(self):
        normalizer = NumbersRuNormalizer(make_config())
        self.assertEqual(normalizer.normalize("1 глава"), "одна глава")

    def test_cardinal_with_dative_book_noun(self):
        normalizer = NumbersRuNormalizer(make_config())
        self.assertEqual(normalizer.normalize("к 1 главе"), "к одной главе")


class TestNumbersRuNormalizerYears(unittest.TestCase):
    def setUp(self):
        self.n = NumbersRuNormalizer(make_config())

    def test_year_nominative(self):
        self.assertEqual(self.n.normalize("1917 год"), "тысяча девятьсот семнадцатый год")

    def test_year_genitive(self):
        self.assertEqual(self.n.normalize("до 1917 года"), "до тысяча девятьсот семнадцатого года")

    def test_year_prepositional(self):
        self.assertEqual(self.n.normalize("в 1917 году"), "в тысяча девятьсот семнадцатом году")

    def test_year_instrumental(self):
        self.assertEqual(self.n.normalize("1917 годом"), "тысяча девятьсот семнадцатым годом")

    def test_year_range_stays_as_cardinal_range(self):
        # "1917-1920" — year range, handled by RANGE_PATTERN as cardinals
        result = self.n.normalize("1917-1920")
        self.assertIn("тысяча девятьсот семнадцать", result)
        self.assertIn("тысяча девятьсот двадцать", result)

    def test_year_with_ordinal_suffix_m(self):
        # "в 2017-м году" — ORDINAL_PATTERN handles "-м", "году" stays
        result = self.n.normalize("в 2017-м году")
        self.assertIn("две тысячи семнадцатом", result)
        self.assertIn("году", result)

    def test_year_2000(self):
        self.assertEqual(self.n.normalize("2000 год"), "двухтысячный год")


class TestNumbersRuNormalizerDates(unittest.TestCase):
    def setUp(self):
        self.n = NumbersRuNormalizer(make_config())

    def test_numeric_date_full(self):
        result = self.n.normalize("14.04.2026")
        self.assertIn("четырнадцатого", result)
        self.assertIn("апреля", result)
        self.assertIn("две тысячи двадцать шестого", result)
        self.assertIn("года", result)

    def test_numeric_date_not_confused_with_decimal(self):
        # "3.14" should NOT trigger date pattern (month 14 is invalid)
        result = self.n.normalize("3.14")
        self.assertNotIn("января", result)
        self.assertNotIn("февраля", result)

    def test_numeric_date_invalid_month_passthrough(self):
        # Month 13 is invalid — NUMERIC_DATE_PATTERN regex won't match,
        # the text is processed by DECIMAL and CARDINAL instead; no month name is expected.
        result = self.n.normalize("31.13.2026")
        for month in ("января", "февраля", "марта", "апреля", "мая", "июня",
                      "июля", "августа", "сентября", "октября", "ноября", "декабря"):
            self.assertNotIn(month, result, f"unexpected month '{month}' in '{result}'")

    def test_full_date_with_text_month_and_year(self):
        result = self.n.normalize("14 апреля 2026 года")
        self.assertIn("четырнадцатого", result)
        self.assertIn("апреля", result)
        self.assertIn("две тысячи двадцать шестого", result)

    def test_partial_date_no_year(self):
        result = self.n.normalize("5 мая")
        self.assertIn("пятого", result)
        self.assertIn("мая", result)

    def test_partial_date_first_of_january(self):
        result = self.n.normalize("1 января")
        self.assertIn("первого", result)
        self.assertIn("января", result)

    def test_partial_date_31_december(self):
        result = self.n.normalize("31 декабря")
        self.assertIn("тридцать первого", result)
        self.assertIn("декабря", result)


class TestNumbersRuNormalizerTime(unittest.TestCase):
    def setUp(self):
        self.n = NumbersRuNormalizer(make_config())

    def test_time_hours_and_minutes(self):
        self.assertEqual(self.n.normalize("15:30"), "пятнадцать тридцать")

    def test_time_zero_minutes(self):
        self.assertEqual(self.n.normalize("15:00"), "пятнадцать ноль-ноль")

    def test_time_single_digit_minutes(self):
        self.assertEqual(self.n.normalize("9:05"), "девять ноль пять")

    def test_time_midnight(self):
        self.assertEqual(self.n.normalize("0:00"), "ноль ноль-ноль")

    def test_time_invalid_hours_not_matched(self):
        # Hours > 23 don't match TIME_PATTERN, numbers are handled by CARDINAL.
        result = self.n.normalize("25:00")
        # "двадцать пять" should appear (cardinal from CARDINAL_PATTERN)
        self.assertIn("двадцать пять", result)
        # The time token was NOT converted as a time (no "ноль-ноль" phrase)
        self.assertNotEqual(result, "двадцать пять ноль-ноль")

    def test_time_invalid_minutes_not_matched(self):
        # Minutes ≥ 60 don't match TIME_PATTERN.
        result = self.n.normalize("3:60")
        # "три" should appear, but not as a properly formatted time
        self.assertIn("три", result)

    def test_time_in_sentence(self):
        result = self.n.normalize("встреча в 10:30 в офисе")
        self.assertIn("десять тридцать", result)


class TestNumbersRuNormalizerFractions(unittest.TestCase):
    def setUp(self):
        self.n = NumbersRuNormalizer(make_config())

    def test_one_half(self):
        self.assertEqual(self.n.normalize("1/2"), "половина")

    def test_one_third(self):
        self.assertEqual(self.n.normalize("1/3"), "треть")

    def test_two_thirds(self):
        self.assertEqual(self.n.normalize("2/3"), "две трети")

    def test_one_quarter(self):
        self.assertEqual(self.n.normalize("1/4"), "четверть")

    def test_three_quarters(self):
        self.assertEqual(self.n.normalize("3/4"), "три четверти")

    def test_generic_fraction_five_eighths(self):
        result = self.n.normalize("5/8")
        self.assertIn("пять", result)
        self.assertIn("восьмых", result)

    def test_generic_fraction_one_fifth(self):
        self.assertEqual(self.n.normalize("1/5"), "одна пятая")

    def test_division_by_zero_passthrough(self):
        self.assertEqual(self.n.normalize("1/0"), "1/0")

    def test_fraction_not_confused_with_path(self):
        # Word chars on either side → no match
        result = self.n.normalize("path/to")
        self.assertEqual(result, "path/to")

    def test_decimal_not_affected_by_fraction_pattern(self):
        # "3.14" → decimal, not fraction
        result = self.n.normalize("3.14")
        self.assertNotIn("/", result)
        self.assertNotIn("вторых", result)


class TestNumbersRuNormalizerCurrency(unittest.TestCase):
    def setUp(self):
        self.n = NumbersRuNormalizer(make_config())

    def test_dollars_plural(self):
        self.assertEqual(self.n.normalize("$100"), "сто долларов")

    def test_dollars_singular(self):
        self.assertEqual(self.n.normalize("$1"), "один доллар")

    def test_dollars_21(self):
        self.assertEqual(self.n.normalize("$21"), "двадцать один доллар")

    def test_dollars_genitive_plural_teens(self):
        self.assertEqual(self.n.normalize("$11"), "одиннадцать долларов")

    def test_euros(self):
        self.assertEqual(self.n.normalize("€50"), "пятьдесят евро")

    def test_rubles(self):
        # num2words(1000, lang='ru') returns "одна тысяча"
        self.assertEqual(self.n.normalize("₽1000"), "одна тысяча рублей")

    def test_rubles_2(self):
        self.assertEqual(self.n.normalize("₽2"), "два рубля")

    def test_pounds(self):
        self.assertEqual(self.n.normalize("£4"), "четыре фунта")

    def test_pounds_genitive_singular(self):
        self.assertEqual(self.n.normalize("£3"), "три фунта")


class TestNumbersRuNormalizerNewNouns(unittest.TestCase):
    """Extended ORDINAL_NOUN_FORMS and ALLOWED_CARDINAL_NOUN_LEMMAS coverage."""

    def setUp(self):
        self.n = NumbersRuNormalizer(make_config())

    def test_season_nominative(self):
        self.assertEqual(self.n.normalize("5 сезон"), "пятый сезон")

    def test_season_dative(self):
        self.assertEqual(self.n.normalize("к 3 сезону"), "к третьему сезону")

    def test_episode_genitive(self):
        self.assertEqual(self.n.normalize("из 12 эпизода"), "из двенадцатого эпизода")

    def test_series_nominative(self):
        self.assertEqual(self.n.normalize("5 серия"), "пятая серия")

    def test_appendix_nominative(self):
        # "2 приложение": ORDINAL_NOUN_PATTERN defers 1/2 to CARDINAL_WITH_NOUN (pymorphy3).
        # pymorphy3 detects nominative neuter → cardinal "два" (neuter nom) + noun kept.
        # Result is "два приложение" (number agrees but noun not inflected — known limitation).
        result = self.n.normalize("2 приложение")
        self.assertIn("приложение", result)
        self.assertTrue(
            result in ("два приложение", "второе приложение"),
            f"unexpected: {result!r}",
        )

    def test_paragraph_roman(self):
        self.assertEqual(self.n.normalize("VII параграф"), "седьмой параграф")

    def test_issue_nominative(self):
        self.assertEqual(self.n.normalize("10 выпуск"), "десятый выпуск")

    def test_lesson_nominative(self):
        self.assertEqual(self.n.normalize("3 урок"), "третий урок")

    def test_task_nominative(self):
        # "1 задание": ORDINAL_NOUN_PATTERN defers 1/2 to CARDINAL_WITH_NOUN (pymorphy3).
        # pymorphy3 detects nominative neuter → NUMBER_WITH_NOUN_FORMS[1][('n','n')] = "одно".
        self.assertEqual(self.n.normalize("1 задание"), "одно задание")

    def test_question_genitive(self):
        self.assertEqual(self.n.normalize("к 5 вопросу"), "к пятому вопросу")

    def test_cardinal_with_noun_extended_pymorphy(self):
        # With pymorphy3: "1 урок" → defer to CARDINAL_WITH_NOUN → "один урок"
        result = self.n.normalize("1 урок")
        self.assertIn("урок", result)
        # "один урок" OR "первый урок" — both are linguistically acceptable;
        # the normalizer uses ordinal for nouns in ORDINAL_NOUN_FORMS (for numbers > 2)
        # and cardinal for 1/2 if pymorphy3 recognises the noun.
        self.assertIn(result, ("один урок", "первый урок"))


class TestDeterministicRuNormalizers(unittest.TestCase):
    def test_initials_ru(self):
        normalizer = InitialsRuNormalizer(make_config(normalize_steps="ru_initials"))
        self.assertEqual(
            normalizer.normalize("вкратце о Е. Д. Калашниковой"),
            "вкратце о Е Дэ Калашниковой",
        )

    def test_initials_ru_with_extra_spaces(self):
        normalizer = InitialsRuNormalizer(make_config(normalize_steps="ru_initials"))
        self.assertEqual(
            normalizer.normalize("вкратце о Е.   Д.   Калашниковой"),
            "вкратце о Е Дэ Калашниковой",
        )

    def test_tts_pronunciation_overrides(self):
        normalizer = TTSPronunciationOverridesNormalizer(
            make_config(normalize_steps="tts_pronunciation_overrides")
        )
        self.assertEqual(
            normalizer.normalize("Отель расположен рядом."),
            "Отэль расположен рядом.",
        )

    def test_stress_words_ru(self):
        # Replaced by ru_tts_stress_paradox_guard — tested in ru_stress_paradox_guard_test.py
        pass

    def test_stress_words_ru_leaves_ambiguous_word_unchanged(self):
        # Replaced by ru_tts_stress_paradox_guard — tested in ru_stress_paradox_guard_test.py
        pass

    def test_simple_symbols_aggressive_cleanup_preserves_letters_digits_and_stress(self):
        normalizer = SimpleSymbolsNormalizer(make_config(normalize_steps="simple_symbols"))
        self.assertEqual(
            normalizer.normalize('Текст\u200b с\u00a0мусором • и™ акце́нтом… «цитата» — 42'),
            'Текст с мусором и акце́нтом... `цитата` - 42',
        )

    def test_plus_stress_to_combining_acute(self):
        self.assertEqual(
            plus_stress_to_combining_acute("б+еды"),
            f"бе{COMBINING_ACUTE}ды",
        )

    def test_load_choice_mapping_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            mapping_path = Path(temp_dir) / "ambiguities.txt"
            mapping_path.write_text(
                "беды==б+еды|бед+ы\nпоступи==п+оступи|поступ+и\n",
                encoding="utf-8",
            )
            mapping = load_choice_mapping_file(str(mapping_path))
            self.assertEqual(
                mapping["беды"],
                (f"бе{COMBINING_ACUTE}ды", f"беды{COMBINING_ACUTE}"),
            )


class TestTTSSafeSplitNormalizer(unittest.TestCase):
    def test_avoids_single_word_tail_split(self):
        normalizer = TTSSafeSplitNormalizer(
            make_config(
                normalize_steps="tts_llm_safe_split",
                normalize_tts_safe_max_chars=160,
            )
        )
        result = normalizer.normalize(
            'О, если б голова моя была водой, и очи мои - Исто́чниками, льющимися, как жидкие небеса. Тогда́ бы я дал волю могучему потоку И оплакал бы потопом род человеческий'
        )
        self.assertNotIn("род. Человеческий", result)
        self.assertIn("род человеческий", result)

    def test_short_exclamation_at_start_merged_forward_preserves_mark(self):
        """A short exclamative sentence at paragraph start ('Хм!') has no previous sentence
        to merge into, so it gets merged FORWARD with the next sentence.
        The '!' must NOT be stripped: 'Хм Об этом...' is wrong; 'Хм! Об этом...' is correct."""
        normalizer = TTSSafeSplitNormalizer(
            make_config(normalize_steps="tts_llm_safe_split", normalize_tts_safe_max_chars=180)
        )
        text = (
            "Хм! "
            "Об этом автор Паралипоменон, как ни чудесна эта история, не упоминает вовсе, "
            "хотя Илью по имени называет."
        )
        result = normalizer.normalize(text)
        self.assertIn("Хм!", result, "'Хм!' exclamation mark must be preserved when merged forward")
        self.assertNotIn("Хм Об", result, "'Хм' must not lose its '!' when merged with next sentence")

    def test_short_exclamation_merged_without_double_punctuation(self):
        """A short exclamative sentence like 'Увы!' merged with its predecessor
        must NOT get a spurious period appended: 'Увы!.' is wrong.
        Bug: rstrip('.') left '!' intact → f'...года. Увы!.' had double punctuation."""
        normalizer = TTSSafeSplitNormalizer(
            make_config(normalize_steps="tts_llm_safe_split", normalize_tts_safe_max_chars=180)
        )
        text = (
            "Обошёл бы полностью менее чем за два года. "
            "Увы! "
            "что это по сравнению с могучим океаном пространства и всемогущей силой Творца!"
        )
        result = normalizer.normalize(text)
        self.assertNotIn("Увы!.", result,
                         "'Увы!' must not have a spurious period appended when merged with prev sentence")
        self.assertIn("Увы!", result, "'Увы!' must be preserved in output")

    def test_short_sentence_not_merged_with_next_long_sentence(self):
        """Short sentence like 'Или нет.' must be merged with the PREVIOUS sentence,
        not the next long one — to avoid producing a semantically incoherent chunk
        like 'Или нет Заявления автора...' (180 chars, sounds awkward in TTS)."""
        normalizer = TTSSafeSplitNormalizer(
            make_config(
                normalize_steps="tts_llm_safe_split",
                normalize_tts_safe_max_chars=180,
            )
        )
        text = (
            "Было ли это связано с веком разума? "
            "C его выпадом против богини Приро\u0301ды? "
            "Или нет. "
            "Заявления автора и переводчика согласуются благодаря тому факту, "
            "что Пэйн подготовил рукопись со значительными добавлениями "
            "и изменениями к публикации на английском языке. "
            "Как он сам указал в предисловии ко второй части."
        )
        result = normalizer.normalize(text)
        # The bad merge: "Или нет Заявления" (sentence boundary stripped)
        self.assertNotIn("нет Заявления", result,
                         "'Или нет.' must not be merged forwards into 'Заявления автора...'")
        # The long sentence must be kept intact (171 chars < 180, should not be split)
        self.assertIn(
            "Заявления автора и переводчика согласуются благодаря тому факту, "
            "что Пэйн подготовил рукопись со значительными добавлениями "
            "и изменениями к публикации на английском языке.",
            result,
            "The 171-char sentence must not be broken at a comma",
        )
        # "Как он сам..." is a separate sentence and must be present
        self.assertIn("Как он сам указал в предисловии ко второй части.", result)

    def test_split_prefers_sentence_boundary_over_comma(self):
        """When a long sequence contains an existing sentence boundary ('. Capital'),
        the split must happen there — not at a comma inside a sentence clause.
        Bug: 'появилась, как около трёх часов' was split at comma before 'как',
        producing 'появилась.' + 'Как около трёх...'."""
        normalizer = TTSSafeSplitNormalizer(
            make_config(normalize_steps="tts_llm_safe_split", normalize_tts_safe_max_chars=180)
        )
        text = (
            '"Не прошло и шести часов с тех пор, как я её закончил, в том виде, в каком она затем появилась, '
            'как около трёх часов утра явилась стража с приказом. '
            'Подписанным двумя комитетами общественного спасения и общей безопасности, взять меня под арест". '
            'Это произошло утром двадцать восьмого декабря.'
        )
        result = normalizer.normalize(text)
        # The bad split: comma before 'как' was used as split point
        self.assertNotIn("появилась. Как", result,
                         "Must not split 'появилась, как около трёх' at the comma")
        # The sentence ending at 'приказом.' must be kept intact as one unit
        self.assertIn("как около трёх часов утра явилась стража с приказом.", result)
        # 'Подписанным' must stay together with the rest of its clause
        self.assertIn("Подписанным двумя комитетами общественного спасения", result)

    def test_no_dangling_short_preposition_at_end_of_chunk(self):
        """A space-split must not leave a short function word (≤3 chars) dangling
        at the end of the left fragment.
        Bug: long text was split at the space before 'По', producing
        '...третьего года. По.' as one TTS unit and 'Старому стилю.' as the next."""
        normalizer = TTSSafeSplitNormalizer(
            make_config(normalize_steps="tts_llm_safe_split", normalize_tts_safe_max_chars=180)
        )
        text = (
            '"Я передаю Мерлену де Тионвилю экземпляр последней работы Тэ-Пэйна, бывшего нашего коллеги, '
            'находящегося под стражей со времени декрета. Исключившего иностранцев из национального '
            'представительства. Эта книга была написана автором в начале девяносто третьего года. '
            'По старому стилю. Я взял на себя её перевод до революции против священников, и она была '
            'опубликована по-французски примерно в то же время. Кутон, которому я её послал, по-видимому, '
            'был недоволен мною за то, что я перевёл эту работу".'
        )
        result = normalizer.normalize(text)
        # 'По старому стилю' must NOT be split as 'По.' + 'Старому стилю.'
        self.assertNotIn("По. Старому", result,
                         "Must not split 'По старому стилю' at the space after 'По'")
        # 'Кутон, которому' must stay together
        self.assertIn("Кутон, которому", result,
                      "'Кутон, которому' must not be split")

    def test_no_split_before_conjunction_without_preceding_punctuation(self):
        """Pattern for 'и/а/но/однако' must only fire after a comma or semicolon.
        Bug: 'Из Ве́тхого и Но́вого Заве́тов' was split to 'Из Ве́тхого. И Но́вого',
        and 'характера и в ложном изображении' became 'характера. И в ложном'.
        Neither 'и' is preceded by a comma — the split must not happen."""
        normalizer = TTSSafeSplitNormalizer(
            make_config(normalize_steps="tts_llm_safe_split", normalize_tts_safe_max_chars=180)
        )
        # Case 1: 'и' without preceding comma connects two adjectives
        text1 = (
            'Из Ве\u0301тхого и Но\u0301вого Заве\u0301тов, говорят они, мы берём только то, '
            'что полезно, главным образом нравственное учение и духовную практику.'
        )
        r1 = normalizer.normalize(text1)
        self.assertNotIn("Ве\u0301тхого. И", r1,
                         "'Из Ве́тхого и Но́вого' must not be split before 'И'")
        self.assertIn("Ве\u0301тхого и Но\u0301вого", r1)

        # Case 2: 'и' without preceding comma connects two prepositional phrases
        text2 = (
            'Очернении его характера и в ложном изображении всех обстоятельств и событий его жизни, '
            'не станет ли для потомства, спустя тысячу семьсот лет.'
        )
        r2 = normalizer.normalize(text2)
        self.assertNotIn("характера. И в ложном", r2,
                         "'характера и в ложном' must not be split before 'И'")
        self.assertNotIn("Спустя тысячу семьсот", r2,
                         "'спустя тысячу семьсот' must not be capitalized from a wrong split")

    def test_no_split_inside_sentence_when_left_ends_with_short_sentence_word(self):
        """When two sentences are merged by sentencex into >180 chars, the split must
        happen at the sentence boundary ('. Capital'), even if the left sentence
        ends with a short word like 'и'.
        Bug: '...схваченным и. Следовательно, не намеревался...' (192 chars total)
        was hard-split at position 180 producing 'Следовательно. Не намеревался'
        because the dangling-short-word check rejected '. Следовательно' split
        (left ended with 'и.' ≤ 3 chars)."""
        normalizer = TTSSafeSplitNormalizer(
            make_config(normalize_steps="tts_llm_safe_split", normalize_tts_safe_max_chars=180)
        )
        text = (
            "А то, что он был предан, или, иными словами, схвачен по доносу одного "
            "из своих последователей, показывает, что он не намеревался быть схваченным и. "
            "Следовательно, не намеревался быть распятым."
        )
        result = normalizer.normalize(text)
        self.assertNotIn("Следовательно. Не", result,
                         "Must not hard-split at 180 chars when a sentence boundary exists")
        self.assertIn("Следовательно, не намеревался", result)

    def test_no_split_inside_quoted_dialogue_after_vocative(self):
        """Splitting at a comma inside direct speech ('Авени́р, чей сын этот юноша?')
        produces 'Авени́р. Чей сын этот юноша?' — the comma is a vocative separator,
        not a sentence boundary.  The right side starts with lowercase 'чей', so the
        split must be rejected and a better point chosen."""
        normalizer = TTSSafeSplitNormalizer(
            make_config(normalize_steps="tts_llm_safe_split", normalize_tts_safe_max_chars=180)
        )
        text = (
            'В пятьдесят пятом стихе этой главы сказано: он сказал Авени́ру, '
            'начальнику войска. `Авени́р, чей сын этот юноша?` И Авени́р сказал: '
            '`Да живёт душа твоя, о царь, не знаю`. И царь сказал. '
            '`Разузна́й, чей сын этот отрок`.'
        )
        result = normalizer.normalize(text)
        self.assertNotIn("Авени́р. Чей", result,
                         "Must not split at vocative comma inside quoted dialogue")
        self.assertIn("Авени́р, чей", result)

    def test_no_split_before_и_more_того(self):
        """'рукописи, и более того' must NOT be split to 'рукописи. И более того'.
        A comma before 'и более того' (adverbial phrase) must not trigger a split."""
        normalizer = TTSSafeSplitNormalizer(
            make_config(normalize_steps="tts_llm_safe_split", normalize_tts_safe_max_chars=180)
        )
        text = (
            'Тем, кто знаком с переводами Лантенаса, нет нужды напоминать, что он был слишком '
            'буквальным переводчиком, чтобы отступать от лежавшей перед ним рукописи, и более того, '
            'он не решился изменить её даже в одном случае, который будет рассмотрен далее, хотя это '
            'изменение было явно необходимо.'
        )
        result = normalizer.normalize(text)
        self.assertNotIn("рукописи. И более того", result,
                         "'рукописи, и более того' must not be split into a new sentence")
        self.assertIn("рукописи, и более того", result)


class TestSharedNormalizerLLMSupport(unittest.TestCase):
    def test_choice_batch_planner_groups_multiple_items(self):
        llm = DummyNormalizer(
            make_config(
                normalize_steps="simple_symbols",
                normalize_base_url="http://127.0.0.1:1234/v1",
                normalize_max_chars=6000,
            )
        ).get_normalizer_llm()
        service = NormalizerLLMChoiceService(llm)
        items = [
            NormalizerLLMChoiceItem(
                item_id=f"item-{index}",
                source_text="беды",
                context=(
                    "После беды пришли новые беды, и рассказчик снова возвращается к этому слову "
                    "в длинном, но вполне обычном книжном предложении."
                ),
                note="Choose the stress or pronunciation variant that best fits this sentence.",
                options=(
                    NormalizerLLMChoiceOption("original", "беды"),
                    NormalizerLLMChoiceOption("variant_1", f"бе{COMBINING_ACUTE}ды"),
                    NormalizerLLMChoiceOption("variant_2", f"беды{COMBINING_ACUTE}"),
                ),
            )
            for index in range(6)
        ]
        batches = service.plan_batches(items)
        self.assertLess(len(batches), len(items))
        self.assertGreater(len(batches[0]), 1)

    def test_default_user_prompt_is_plain_text_only(self):
        normalizer = DummyNormalizer(
            make_config(
                normalize_steps="simple_symbols",
                normalize_base_url="http://127.0.0.1:1234/v1",
            )
        )
        llm = normalizer.get_normalizer_llm()
        self.assertEqual(llm.settings.user_prompt_template, "{text}")
        self.assertEqual(
            llm.render_user_prompt(chapter_title="Глава 1", text="Привет, мир."),
            "Привет, мир.",
        )

    def test_llm_runtime_is_shared_for_same_config(self):
        config = make_config(
            normalize_steps="simple_symbols",
            normalize_base_url="http://127.0.0.1:1234/v1",
        )
        first = DummyNormalizer(config)
        second = DummyNormalizer(config)
        self.assertIs(first.get_normalizer_llm(), second.get_normalizer_llm())
        self.assertTrue(first.has_normalizer_llm())

    def test_openai_small_chunk_merge(self):
        merged = OpenAINormalizer._merge_small_chunks(
            ["A" * 3900, "B" * 900, "C" * 3800],
            4000,
        )
        self.assertEqual(len(merged), 2)
        self.assertIn("B" * 900, merged[0])

    def test_choice_response_parser_accepts_json_fences(self):
        parsed = NormalizerLLMChoiceService.parse_choice_response(
            """```json
            {"selections":[{"id":"item-1","option_id":"phonetic"}]}
            ```"""
        )
        self.assertEqual(parsed, {"item-1": "phonetic"})

    def test_choice_response_parser_accepts_cacheable_and_custom_text(self):
        parsed = NormalizerLLMChoiceService.parse_choice_response_objects(
            """{
              "selections": [
                {
                  "id": "item-1",
                  "custom_text": "То́мас Пэйн",
                  "cacheable": true,
                  "reason": "Stable surname pronunciation"
                }
              ]
            }"""
        )
        self.assertEqual(parsed["item-1"].custom_text, "То́мас Пэйн")
        self.assertTrue(parsed["item-1"].cacheable)
        self.assertEqual(parsed["item-1"].reason, "Stable surname pronunciation")

    def test_choice_service_reuses_cache(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = MagicMock(
                max_chars=4000,
                choice_cache_path=str(Path(temp_dir) / "choice_cache.json"),
                model="gpt-5.4",
                provider="openai",
                base_url="http://127.0.0.1:1234/v1",
            )
            response = json.dumps(
                {
                    "selections": [
                        {
                            "id": "item-1",
                            "custom_text": "То́мас Пэйн",
                            "cacheable": True,
                            "reason": "Stable pronunciation",
                        }
                    ]
                },
                ensure_ascii=False,
            )
            llm = FakeLLM(settings=settings, response_text=response)
            service = NormalizerLLMChoiceService(llm)
            item = NormalizerLLMChoiceItem(
                item_id="item-1",
                source_text="Томас Пейн",
                context="Томас Пейн писал эссе.",
                note=None,
                options=(
                    NormalizerLLMChoiceOption("original", "Томас Пейн"),
                    NormalizerLLMChoiceOption("guided", "То́мас Пэйн"),
                ),
            )

            first = service.choose_batch([item], target_language="ru-RU")
            second = service.choose_batch([item], target_language="ru-RU")

            self.assertEqual(first["item-1"].custom_text, "То́мас Пэйн")
            self.assertEqual(second["item-1"].source, "cache")
            self.assertEqual(llm.calls, 1)


class TestProperNounsRuNormalizer(unittest.TestCase):
    def test_accents_internal_proper_nouns(self):
        normalizer = ProperNounsRuNormalizer(make_config(normalize_steps="ru_proper_names"))
        normalizer.backend = lambda text: {
            "Фицджеральда": "Фицджера́льда",
            "Калашниковой": "Кала́шниковой",
            "Нью-Йорка": "Нью-Йо́рка",
        }.get(text, text)
        self.assertEqual(
            normalizer.normalize("Раньше уже говорилось вкратце о Е. Д. Калашниковой и Фицджеральда из Нью-Йорка."),
            "Раньше уже говорилось вкратце о Е. Д. Кала́шниковой и Фицджера́льда из Нью-Йо́рка.",
        )

    def test_skips_sentence_start_words(self):
        normalizer = ProperNounsRuNormalizer(make_config(normalize_steps="ru_proper_names"))
        normalizer.backend = lambda text: text + COMBINING_ACUTE
        self.assertEqual(
            normalizer.normalize("Для ясности автор этой книжки не лингвист. Но Фицджеральда он помнит."),
            "Для ясности автор этой книжки не лингвист. Но Фицджеральда́ он помнит.",
        )

    def test_does_not_add_second_stress_to_prestressed_word(self):
        """If the source text already has a stress mark (combining acute), the normalizer
        must NOT add another one.  This was a bug: 'Права́ми' (stress on last 'а') was
        converted to 'Пра́ва́ми' (two stress marks) because the regex did not include
        combining diacritics in the word character class."""
        normalizer = ProperNounsRuNormalizer(make_config(normalize_steps="ru_proper_names"))
        # Real tsnorm backend would add "пра́" stress to "Правами" → double stress without fix
        normalizer.backend = lambda text: "Пра\u0301вами" if text == "Правами" else text
        text = 'его "\u041f\u0440\u0430\u0432\u0430\u0301\u043c\u0438 \u0447\u0435\u043b\u043e\u0432\u0435\u043a\u0430"'  # Права́ми
        result = normalizer.normalize(text)
        import unicodedata
        accent_count = sum(1 for c in result if unicodedata.combining(c))
        self.assertEqual(accent_count, 1,
                         f"Expected exactly 1 accent in result, got {accent_count}: {repr(result)}")
        self.assertIn("\u041f\u0440\u0430\u0432\u0430\u0301\u043c\u0438", result,
                      "Original 'Права́ми' must be preserved unchanged")

    def test_paradox_guard_overrides_tsnorm_stress_in_proper_names(self):
        """ru_proper_names must not produce wrong stress for paradox words.
        Bug: tsnorm put 'То́лстым' (stress on о), paradox guard not applied in step 7,
        step 9 used stale cache → wrong stress reached TTS.
        Fix: paradox guard is applied at the end of ProperNounsRuNormalizer.normalize().
        """
        ACUTE = "\u0301"
        # Config lists 'Толсты́м' as paradox word (canonical stress on ы)
        config = make_config(
            normalize_steps="ru_proper_names",
            normalize_stress_paradox_words=f"Толсты\u0301м",
        )
        normalizer = ProperNounsRuNormalizer(config)
        # Simulate tsnorm putting wrong stress on о
        normalizer.backend = lambda word: f"То{ACUTE}лстым" if word == "Толстым" else word
        text = "написанном графом Льво́м Толстым в газету"
        result = normalizer.normalize(text)
        self.assertIn(f"Толсты{ACUTE}м", result,
                      "Paradox guard must enforce 'Толсты́м' even in step 7")
        self.assertNotIn(f"То{ACUTE}лстым", result,
                         "Wrong tsnorm stress 'То́лстым' must be overridden by paradox guard")

    def test_no_double_stress_on_царя_and_отца(self):
        """Words like 'Царя́' and 'Отца́' that already carry a stress mark must not
        receive a second one from the ru_proper_names normalizer.
        Bug: the word regex stopped at the combining-accent char, matching only
        'Царя' (without accent) as a candidate, then adding 'ца́рь' stress on top."""
        normalizer = ProperNounsRuNormalizer(make_config(normalize_steps="ru_proper_names"))
        # Backend that would add stress to bare "Царя" or "Отца"
        def backend(word):
            mapping = {"Царя": "Царя\u0301", "Отца": "Отца\u0301"}
            return mapping.get(word, word)
        normalizer.backend = backend
        import unicodedata
        ACUTE = "\u0301"
        text = f"на Царя{ACUTE} царей и Отца{ACUTE} человечества"
        result = normalizer.normalize(text)
        # No double accents anywhere in result
        for i in range(len(result) - 2):
            if unicodedata.combining(result[i + 1]) and unicodedata.combining(result[i + 2]):
                self.fail(
                    f"Double accent found in: {repr(result[max(0,i-4):i+8])}"
                )


class TestProperNounsPronunciationRuNormalizer(unittest.TestCase):
    def test_builds_pronunciation_variants_and_applies_selected_options(self):
        normalizer = ProperNounsPronunciationRuNormalizer(
            make_config(
                normalize_steps="ru_llm_proper_nouns_pronunciation",
                normalize_base_url="http://127.0.0.1:1234/v1",
            )
        )

        def fake_choose_batch(items, **kwargs):
            selections = {}
            for item in items:
                if item.source_text == "Томас Пейн":
                    selections[item.item_id] = "guided"
                elif item.source_text == "Лев Толстой":
                    selections[item.item_id] = "guided"
                else:
                    selections[item.item_id] = "original"
            return selections

        normalizer.choice_service.choose_batch = fake_choose_batch
        result = normalizer.normalize(
            'Томас Пейн писал о вере, а позже Лев Толстой спорил с газетой "Таймс".'
        )

        self.assertIn("Пэйн", result)
        self.assertIn(f"Лев Толсто{COMBINING_ACUTE}й", result)
        self.assertIn(f"То{COMBINING_ACUTE}мас Пэйн", result)
        self.assertIn('"Таймс"', result)

    def test_prompt_artifacts_include_context_and_options(self):
        normalizer = ProperNounsPronunciationRuNormalizer(
            make_config(
                normalize_steps="ru_llm_proper_nouns_pronunciation",
                normalize_base_url="http://127.0.0.1:1234/v1",
            )
        )
        text = 'В лондонскую газету "Таймс" писал Томас Пейн.'
        units = normalizer.plan_processing_units(text, chapter_title="Test")
        self.assertTrue(units)
        artifacts = normalizer.get_unit_artifacts(
            units[0],
            chapter_title="Test",
            unit_index=1,
            unit_count=len(units),
        )
        self.assertIn("Томас Пейн", artifacts["01_choice_user_prompt.txt"])
        self.assertIn("Пэйн", artifacts["01_choice_user_prompt.txt"])

    def test_skips_generic_leading_sentence_word_in_candidate(self):
        normalizer = ProperNounsPronunciationRuNormalizer(
            make_config(
                normalize_steps="ru_llm_proper_nouns_pronunciation",
                normalize_base_url="http://127.0.0.1:1234/v1",
            )
        )
        units = normalizer.plan_processing_units("Когда Моисей сказал народу.", chapter_title="Test")
        self.assertEqual(len(units), 1)
        artifacts = normalizer.get_unit_artifacts(
            units[0],
            chapter_title="Test",
            unit_index=1,
            unit_count=1,
        )
        self.assertIn('"source_text": "Моисей"', artifacts["01_choice_user_prompt.txt"])
        self.assertIn('"context": "Когда Моисей сказал народу"', artifacts["01_choice_user_prompt.txt"])

    def test_collapses_double_stress_marks_from_backend(self):
        normalizer = ProperNounsPronunciationRuNormalizer(
            make_config(
                normalize_steps="ru_llm_proper_nouns_pronunciation",
                normalize_base_url="http://127.0.0.1:1234/v1",
            )
        )
        normalizer.backend = lambda text: {
            "Новый": f"Но{COMBINING_ACUTE}вый",
            "Завет": f"За{COMBINING_ACUTE}ве{COMBINING_ACUTE}т",
        }.get(text, text)
        self.assertEqual(
            normalizer._accent_phrase("Новый Завет"),
            f"Но{COMBINING_ACUTE}вый Заве{COMBINING_ACUTE}т",
        )

    def test_post_step_artifacts_include_selection_stats(self):
        normalizer = ProperNounsPronunciationRuNormalizer(
            make_config(
                normalize_steps="ru_llm_proper_nouns_pronunciation",
                normalize_base_url="http://127.0.0.1:1234/v1",
            )
        )
        normalizer.choice_service.choose_batch = lambda items, **kwargs: {
            items[0].item_id: MagicMock(
                option_id="guided",
                custom_text=None,
                cacheable=True,
                reason="Stable pronunciation",
                source="llm",
                has_custom_text=False,
                resolved_option_id=lambda: "guided",
            )
        }
        # Use a proper noun NOT in builtin hints so LLM path is exercised
        result = normalizer.normalize("В тексте о Наполеоне Бонапарте написано многое.")
        artifacts = normalizer.get_post_step_artifacts(
            input_text="В тексте о Наполеоне Бонапарте написано многое.",
            output_text=result,
            chapter_title="Test",
        )
        self.assertIn("changed_candidates", artifacts["93_selection_stats.json"])
        self.assertIn("Наполеоне Бонапарте", artifacts["92_selection_report.txt"])


class TestStressAmbiguityLLMNormalizer(unittest.TestCase):
    def test_selects_contextual_variants_for_same_surface_form(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "lexicon.sqlite3"
            build_test_lexicon(
                db_path,
                word_forms={
                    "беды": [
                        {
                            "word_form": "беды",
                            "stress_pos": [1],
                            "form_tags": "genitive singular",
                            "lemma": "беда",
                        },
                        {
                            "word_form": "беды",
                            "stress_pos": [3],
                            "form_tags": "nominative plural",
                            "lemma": "беда",
                        },
                    ]
                },
                lemmas={"беда": {"pos": ["NOUN"], "rank": 1}},
            )
            normalizer = StressAmbiguityLLMNormalizer(
                make_config(
                    normalize_steps="ru_llm_stress_ambiguity",
                    normalize_base_url="http://127.0.0.1:1234/v1",
                    normalize_pronunciation_lexicon_db=str(db_path),
                )
            )

            def fake_complete(*, user_prompt, **kwargs):
                # item 0001 (first «беды») → 1.2 (variant_2 = nom.pl.)
                # item 0002 (second «беды») → 2.1 (variant_1 = gen.sg.)
                return "1.2\n2.1"

            normalizer.choice_service.llm.complete = fake_complete
            result = normalizer.normalize("После беды пришли новые беды.")
            self.assertEqual(
                result,
                f"После беды{COMBINING_ACUTE} пришли новые бе{COMBINING_ACUTE}ды.",
            )

    def test_skips_already_accented_word(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "lexicon.sqlite3"
            build_test_lexicon(
                db_path,
                word_forms={
                    "беды": [
                        {
                            "word_form": "беды",
                            "stress_pos": [1],
                            "form_tags": "genitive singular",
                            "lemma": "беда",
                        },
                        {
                            "word_form": "беды",
                            "stress_pos": [3],
                            "form_tags": "nominative plural",
                            "lemma": "беда",
                        },
                    ]
                },
                lemmas={"беда": {"pos": ["NOUN"], "rank": 1}},
            )
            normalizer = StressAmbiguityLLMNormalizer(
                make_config(
                    normalize_steps="ru_llm_stress_ambiguity",
                    normalize_base_url="http://127.0.0.1:1234/v1",
                    normalize_pronunciation_lexicon_db=str(db_path),
                )
            )
            units = normalizer.plan_processing_units(
                f"После беды{COMBINING_ACUTE} пришли новые бе{COMBINING_ACUTE}ды.",
                chapter_title="Test",
            )
            self.assertEqual(units, [])

    def test_post_step_artifacts_include_reports(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "lexicon.sqlite3"
            build_test_lexicon(
                db_path,
                word_forms={
                    "беды": [
                        {
                            "word_form": "беды",
                            "stress_pos": [1],
                            "form_tags": "genitive singular",
                            "lemma": "беда",
                        },
                        {
                            "word_form": "беды",
                            "stress_pos": [3],
                            "form_tags": "nominative plural",
                            "lemma": "беда",
                        },
                    ]
                },
                lemmas={"беда": {"pos": ["NOUN"], "rank": 1}},
            )
            normalizer = StressAmbiguityLLMNormalizer(
                make_config(
                    normalize_steps="ru_llm_stress_ambiguity",
                    normalize_base_url="http://127.0.0.1:1234/v1",
                    normalize_pronunciation_lexicon_db=str(db_path),
                )
            )
            normalizer.choice_service.llm.complete = lambda *, user_prompt, **kwargs: "1.2"
            result = normalizer.normalize("После беды.")
            artifacts = normalizer.get_post_step_artifacts(
                input_text="После беды.",
                output_text=result,
                chapter_title="Test",
            )
            self.assertIn("changed_candidates", artifacts["93_selection_stats.json"])
            self.assertIn("После беды", artifacts["92_selection_report.txt"])
            self.assertIn("lexicon_entries:", artifacts["92_selection_report.txt"])


class TestNormalizationPipelineRunner(unittest.TestCase):
    def test_resumes_chunked_units_and_saves_prompt_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = make_config(output_folder=temp_dir, normalize_steps="dummy_chunked",
                                 normalize_log_changes=True)
            artifact_dir = Path(temp_dir) / "_chapter_artifacts" / "0001_Test"
            first = DummyChunkedNormalizer(config)
            runner = NormalizationPipelineRunner(config=config, artifact_dir=artifact_dir)
            normalized, trace = runner.run(first, "alpha|beta", "Test")

            self.assertEqual(normalized, "ALPHA\n\nBETA")
            self.assertEqual(first.process_calls, [(1, "alpha"), (2, "beta")])
            self.assertEqual(trace, [("dummy_chunked", "ALPHA\n\nBETA")])
            self.assertTrue((artifact_dir / "_normalizer_steps" / "01_dummy_chunked" / "00_system_prompt.txt").is_file())
            self.assertTrue((artifact_dir / "_normalizer_steps" / "01_dummy_chunked" / "chunks" / "0001" / "01_user_prompt.txt").is_file())
            self.assertTrue((artifact_dir / "_normalizer_steps" / "01_dummy_chunked" / "90_changes.md").is_file())
            self.assertTrue((artifact_dir / "_normalizer_steps" / "01_dummy_chunked" / "91_changes.diff").is_file())
            self.assertTrue((artifact_dir / "00_normalizer_change_summary.md").is_file())

            second = DummyChunkedNormalizer(config)
            second_runner = NormalizationPipelineRunner(config=config, artifact_dir=artifact_dir)
            resumed, resumed_trace = second_runner.run(second, "alpha|beta", "Test")

            self.assertEqual(resumed, "ALPHA\n\nBETA")
            self.assertEqual(resumed_trace, [("dummy_chunked", "ALPHA\n\nBETA")])
            self.assertEqual(second.process_calls, [])


class TestTSNormRuNormalizerStressRuExploration(unittest.TestCase):
    """Exploration tests for TODO 4: stress_ru step using tsnorm.

    Confirms that ``tsnorm_ru`` (aliased as ``stress_ru``) already handles:
    - Stress placement for unambiguous single-stress words.
    - Ё restoration (е→ё) when ``stress_yo=True``.
    - Leaves genuinely ambiguous homographs ("замок") untouched.
    - Works through the pipeline registration under the "stress_ru" alias.
    """

    def setUp(self):
        self.n = TSNormRuNormalizer(
            make_config(
                normalize_steps="ru_tsnorm",
                normalize_tsnorm_stress_yo=True,
                normalize_tsnorm_stress_monosyllabic=False,
                normalize_tsnorm_min_word_length=2,
            )
        )

    def test_places_stress_on_unambiguous_word(self):
        result = self.n.normalize("любовь")
        self.assertIn(COMBINING_ACUTE, result)
        # "любо́вь" — combining acute after "о"
        self.assertTrue(result.startswith("любо"))

    def test_places_stress_on_verb_form(self):
        # "бере́т" — third-person singular of брать
        result = self.n.normalize("берет")
        self.assertIn(COMBINING_ACUTE, result)

    def test_leaves_homograph_without_stress(self):
        # "замок" is ambiguous (за́мок=castle, замо́к=padlock)
        result = self.n.normalize("замок")
        self.assertNotIn(COMBINING_ACUTE, result)

    def test_yo_restoration_in_sentence(self):
        # "е" → "ё" when stress_yo=True
        result = self.n.normalize("е написано без точки")
        self.assertIn("ё", result)

    def test_stress_in_sentence(self):
        result = self.n.normalize("Одно из чудес")
        # "чуде́с" should be stressed (тsnorm + stress_overrides)
        self.assertIn(COMBINING_ACUTE, result)

    def test_skips_non_russian_language(self):
        n_en = TSNormRuNormalizer(
            make_config(
                normalize_steps="ru_tsnorm",
                language="en-US",
                normalize_tsnorm_stress_yo=True,
                normalize_tsnorm_stress_monosyllabic=False,
                normalize_tsnorm_min_word_length=2,
            )
        )
        text = "love is great"
        self.assertEqual(n_en.normalize(text), text)

    def test_registered_as_stress_ru_alias(self):
        from audiobook_generator.normalizers.base_normalizer import normalize_step_name
        from audiobook_generator.normalizers.ru_tsnorm_normalizer import TSNormRuNormalizer
        self.assertEqual(normalize_step_name("ru_tsnorm"), TSNormRuNormalizer.STEP_NAME)

    def test_step_name(self):
        self.assertEqual(self.n.get_step_name(), "ru_tsnorm")


class TestAbbreviationsRuNormalizer(unittest.TestCase):
    """Tests for AbbreviationsRuNormalizer (TODO 3 evaluation of runorm / saarus72)."""

    def setUp(self):
        self.n = AbbreviationsRuNormalizer(make_config(normalize_steps="ru_abbreviations"))

    # --- Helper: acronym expansion ---

    def test_expand_acronym_helper_usa(self):
        self.assertEqual(_expand_acronym("США"), "эс-ша-а")

    def test_expand_acronym_helper_un(self):
        self.assertEqual(_expand_acronym("ООН"), "о-о-эн")

    def test_expand_acronym_helper_rf(self):
        self.assertEqual(_expand_acronym("РФ"), "эр-эф")

    # --- Deterministic abbreviation table ---

    def test_i_t_d(self):
        self.assertEqual(self.n.normalize("Купили хлеб и т.д."), "Купили хлеб и так далее")

    def test_i_t_p(self):
        self.assertEqual(self.n.normalize("еда и т.п."), "еда и тому подобное")

    def test_i_pr(self):
        self.assertEqual(self.n.normalize("одежда и пр."), "одежда и прочее")

    def test_i_dr(self):
        self.assertEqual(self.n.normalize("страны и др."), "страны и другие")

    def test_t_e(self):
        self.assertEqual(self.n.normalize("т.е. это значит"), "то есть это значит")

    def test_napr(self):
        self.assertEqual(self.n.normalize("напр. Москва"), "например Москва")

    def test_sm(self):
        result = self.n.normalize("см. главу 3")
        self.assertIn("смотрите", result)

    def test_tys(self):
        result = self.n.normalize("5 тыс. человек")
        self.assertIn("тысяч", result)

    def test_mln(self):
        result = self.n.normalize("3 млн. жителей")
        self.assertIn("миллионов", result)

    def test_mlrd(self):
        result = self.n.normalize("1 млрд. долларов")
        self.assertIn("миллиардов", result)

    # --- ALL-CAPS acronym expansion ---

    def test_acronym_usa_in_sentence(self):
        result = self.n.normalize("США подписали договор")
        # Should be letter-by-letter, NOT "США"
        self.assertNotIn("США", result)
        self.assertIn("эс", result)

    def test_acronym_rf(self):
        result = self.n.normalize("РФ направила ноту")
        self.assertNotIn("РФ", result)
        self.assertIn("эр", result)

    def test_acronym_two_letter(self):
        result = self.n.normalize("ФСБ проводит проверку")
        self.assertNotIn("ФСБ", result)

    def test_normal_word_not_expanded(self):
        # Single uppercase letter or mixed-case words should not be expanded
        result = self.n.normalize("Россия и Москва")
        self.assertIn("Россия", result)
        self.assertIn("Москва", result)

    # --- ALL-CAPS regular words must NOT be letter-expanded ---

    def test_allcaps_regular_word_amerikanskoy_not_expanded(self):
        """АМЕРИКАНСКОЙ — a regular Russian word in ALL-CAPS (heading/emphasis), not an acronym.
        It must pass through unchanged, not be expanded letter-by-letter."""
        result = self.n.normalize("АМЕРИКАНСКОЙ Революции")
        self.assertIn("АМЕРИКАНСКОЙ", result,
                      "АМЕРИКАНСКОЙ is a real word, not an acronym — must not be expanded")

    def test_allcaps_regular_word_zaveta_not_expanded(self):
        """ЗАВЕТА — a real Russian word in ALL-CAPS. Must not be letter-expanded."""
        result = self.n.normalize("ЗАВЕТА")
        self.assertIn("ЗАВЕТА", result,
                      "ЗАВЕТА is a real word, not an acronym — must not be expanded")

    def test_allcaps_regular_word_revolyutsii_not_expanded(self):
        """РЕВОЛЮЦИИ — a real Russian word in ALL-CAPS. Must not be letter-expanded."""
        result = self.n.normalize("РЕВОЛЮЦИИ народа")
        self.assertIn("РЕВОЛЮЦИИ", result,
                      "РЕВОЛЮЦИИ is a real word, not an acronym — must not be expanded")

    def test_prestressed_word_not_double_stressed(self):
        """A word that already has a stress mark must not gain a second one.

        If the source text or a preceding normalizer adds 'Аме́рика́нской' (double stress),
        ru_abbreviations must not worsen it.  Equally, a normally stressed word
        'Америка́нской' must come out unchanged.
        """
        ACUTE = "\u0301"
        single = "Аме" + ACUTE + "риканской"
        result = self.n.normalize(single)
        self.assertEqual(result, single, "Single-stressed word must not be changed")

        double = "Аме" + ACUTE + "рика" + ACUTE + "нской"
        result2 = self.n.normalize(double)
        self.assertEqual(result2, double, "Already-double-stressed word must not be changed")

    # --- Language guard ---

    def test_skips_non_russian_language(self):
        n_en = AbbreviationsRuNormalizer(
            make_config(normalize_steps="ru_abbreviations", language="en-US")
        )
        self.assertEqual(n_en.normalize("США и т.д."), "США и т.д.")

    # --- Pipeline registration ---

    def test_step_name(self):
        self.assertEqual(self.n.get_step_name(), "ru_abbreviations")

    def test_registered_in_pipeline(self):
        from audiobook_generator.normalizers.base_normalizer import normalize_step_name
        from audiobook_generator.normalizers.ru_abbreviations_normalizer import AbbreviationsRuNormalizer
        self.assertEqual(normalize_step_name("ru_abbreviations"), AbbreviationsRuNormalizer.STEP_NAME)

    def test_allcaps_multiword_name_not_letter_expanded(self):
        """Two (or more) consecutive ALL-CAPS words form an emphasised proper name,
        not an acronym sequence.  Bug: 'ТОМАС ПЭЙН' was expanded to
        'тэ-о-эм-а-эс пэ-э-ий-эн'.  Neither word must be letter-expanded.
        Note: ru_abbreviations runs before stress normalizers, so no combining
        accents are present yet when this step fires."""
        result = self.n.normalize("ТОМАС ПЭЙН написал трактат.")
        self.assertNotIn("тэ-", result, "ТОМАС must not be letter-expanded when next to ПЭЙН")
        self.assertNotIn("пэ-э", result, "ПЭЙН must not be letter-expanded when next to ТОМАС")
        self.assertIn("ТОМАС", result)
        self.assertIn("ПЭЙН", result)

    def test_allcaps_multiword_name_not_letter_expanded_three_words(self):
        """Three consecutive ALL-CAPS words: ЛЕВ НИКОЛАЕВИЧ ТОЛСТОЙ must not be expanded.
        ЛЕВ has 1 vowel (≤ 2) so it would normally be expanded; the adjacent caps context saves it."""
        result = self.n.normalize("ЛЕВ НИКОЛАЕВИЧ ТОЛСТОЙ жил в Ясной Поляне.")
        self.assertNotIn("эл-е-вэ", result, "ЛЕВ must not be letter-expanded in multiword caps name")
        self.assertIn("ЛЕВ", result)

    def test_allcaps_multiword_name_with_comma_not_expanded(self):
        """ALL-CAPS words separated by a comma must not be letter-expanded.
        Bug: 'РУКА, СОЗДАВШАЯ НАС, БОЖЕСТВЕННА' → 'эр-у-ка-а, СОЗДАВШАЯ...'
        because the adjacent-caps check only looked for a space, not comma+space."""
        result = self.n.normalize("РУКА, СОЗДАВШАЯ НАС, БОЖЕСТВЕННА.")
        self.assertNotIn("эр-", result, "РУКА must not be letter-expanded before comma + ALL-CAPS")
        self.assertIn("РУКА", result)

    def test_single_isolated_allcaps_acronym_still_expanded(self):
        """A standalone 2-letter acronym like РФ must still be expanded."""
        result = self.n.normalize("Конституция РФ определяет права.")
        self.assertIn("эр-эф", result, "РФ must still be expanded when isolated")


class TestLLMNormalizersGracefulSkip(unittest.TestCase):
    """LLM-based normalizers must not raise when no LLM is configured.
    They should construct successfully and return text unchanged from normalize()."""

    def _no_llm_config(self, step: str):
        """Config with no LLM credentials — normalize_base_url and api_key both absent."""
        return make_config(
            normalize_steps=step,
            normalize_base_url=None,
            normalize_api_key=None,
        )

    # --- ru_stress_ambiguity ---

    def test_ru_stress_ambiguity_constructs_without_llm(self):
        from audiobook_generator.normalizers.ru_stress_ambiguity_normalizer import StressAmbiguityLLMNormalizer
        # Must NOT raise
        n = StressAmbiguityLLMNormalizer(self._no_llm_config("ru_llm_stress_ambiguity"))
        self.assertIsNotNone(n)

    def test_ru_stress_ambiguity_normalize_returns_text_unchanged(self):
        from audiobook_generator.normalizers.ru_stress_ambiguity_normalizer import StressAmbiguityLLMNormalizer
        n = StressAmbiguityLLMNormalizer(self._no_llm_config("ru_llm_stress_ambiguity"))
        text = "Если вспомнить нравственное уродство."
        self.assertEqual(n.normalize(text), text)

    def test_ru_stress_ambiguity_plan_units_returns_empty(self):
        from audiobook_generator.normalizers.ru_stress_ambiguity_normalizer import StressAmbiguityLLMNormalizer
        n = StressAmbiguityLLMNormalizer(self._no_llm_config("ru_llm_stress_ambiguity"))
        units = n.plan_processing_units("Текст главы.")
        self.assertEqual(units, [])

    # --- ru_proper_nouns_pronunciation ---

    def test_ru_proper_nouns_pronunciation_constructs_without_llm(self):
        from audiobook_generator.normalizers.ru_proper_nouns_pronunciation_normalizer import ProperNounsPronunciationRuNormalizer
        n = ProperNounsPronunciationRuNormalizer(self._no_llm_config("ru_llm_proper_nouns_pronunciation"))
        self.assertIsNotNone(n)

    def test_ru_proper_nouns_pronunciation_normalize_applies_builtin_hints_without_llm(self):
        from audiobook_generator.normalizers.ru_proper_nouns_pronunciation_normalizer import ProperNounsPronunciationRuNormalizer
        n = ProperNounsPronunciationRuNormalizer(self._no_llm_config("ru_llm_proper_nouns_pronunciation"))
        # Builtin hint: "пейн" → "пэйн"; stress hint for "толстым" → "толсты́м"
        result = n.normalize("Томас Пейн написал трактат.")
        self.assertIn("Пэйн", result, "Builtin pronunciation hint must be applied even without LLM")

    def test_ru_proper_nouns_pronunciation_applies_tolstym_stress_without_llm(self):
        """BUILTIN_PROPER_NOUN_STRESS_HINTS must correct 'Толстым' even without LLM.
        tsnorm may put stress on 'о' (То́лстым) for the adjective/noun interpretation;
        the builtin hint enforces 'Толсты́м' (surname instrumental)."""
        from audiobook_generator.normalizers.ru_proper_nouns_pronunciation_normalizer import ProperNounsPronunciationRuNormalizer, COMBINING_ACUTE
        n = ProperNounsPronunciationRuNormalizer(self._no_llm_config("ru_llm_proper_nouns_pronunciation"))
        # Simulate tsnorm applying wrong stress to Толстым (stress on 'о')
        wrong = f"То{COMBINING_ACUTE}лстым"
        result = n.normalize(f"написанное {wrong}.")
        self.assertIn(f"Толсты{COMBINING_ACUTE}м", result,
                      "Stress on 'ы' must be enforced for surname 'Толстым'")

    def test_ru_proper_nouns_pronunciation_plan_units_returns_empty(self):
        from audiobook_generator.normalizers.ru_proper_nouns_pronunciation_normalizer import ProperNounsPronunciationRuNormalizer
        n = ProperNounsPronunciationRuNormalizer(self._no_llm_config("ru_llm_proper_nouns_pronunciation"))
        units = n.plan_processing_units("Текст.")
        self.assertEqual(units, [])

    # --- openai ---

    def test_openai_normalizer_constructs_without_llm(self):
        from audiobook_generator.normalizers.openai_normalizer import OpenAINormalizer
        n = OpenAINormalizer(self._no_llm_config("openai"))
        self.assertIsNotNone(n)

    def test_openai_normalizer_normalize_returns_text_unchanged(self):
        from audiobook_generator.normalizers.openai_normalizer import OpenAINormalizer
        n = OpenAINormalizer(self._no_llm_config("openai"))
        text = "Некоторый текст для озвучивания."
        self.assertEqual(n.normalize(text), text)

    # --- Chain with LLM steps and no LLM ---

    def test_chain_with_llm_steps_no_llm_passes_through(self):
        """Full chain including LLM steps must not raise and must return text."""
        from audiobook_generator.normalizers.base_normalizer import get_normalizer
        config = self._no_llm_config(
            "simple_symbols,ru_llm_stress_ambiguity,ru_llm_proper_nouns_pronunciation"
        )
        config.normalize = True
        normalizer = get_normalizer(config)
        text = "Томас Пейн написал «Век разума»."
        result = normalizer.normalize(text)
        # LLM steps skipped; simple_symbols may clean up quotes, but no crash
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)


if __name__ == "__main__":
    unittest.main()
