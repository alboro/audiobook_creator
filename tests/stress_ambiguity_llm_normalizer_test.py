# Copyright (c) 2026 alboro <alboro@users.noreply.github.com>
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See: https://polyformproject.org/licenses/noncommercial/1.0.0/

"""
Tests for StressAmbiguityLLMNormalizer.

Two test suites:
  - TestStressAmbiguityWithSyntheticDb  — fast offline tests using a
    hand-crafted SQLite lexicon.
  - TestStressAmbiguityWithRealDb       — integration tests against the real
    tsnorm lexicon DB (built once, then cached in .cache/).

Covered words and their expected behaviour
------------------------------------------
тела   → AMBIGUOUS in tsnorm (те́ла gen.sg. vs тела́ acc.pl.)
         stress_ambiguity_llm MUST detect and resolve it.

пика   → NOT ambiguous in tsnorm (only пи́ка).
         stress_ambiguity_llm cannot help; use tsnorm_ru instead.

стража → NOT ambiguous in tsnorm (only стра́жа).
         stress_ambiguity_llm cannot help; use tsnorm_ru instead.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from audiobook_generator.config.general_config import GeneralConfig
from audiobook_generator.normalizers.pronunciation_lexicon_db import (
    PronunciationLexiconDB,
    build_tsnorm_pronunciation_lexicon,
    ensure_pronunciation_lexicon_db,
)
from audiobook_generator.normalizers.ru_text_utils import COMBINING_ACUTE
from audiobook_generator.normalizers.ru_stress_ambiguity_normalizer import (
    StressAmbiguityLLMNormalizer,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def make_config(**overrides) -> GeneralConfig:
    values = dict(
        input_file="examples/The_Life_and_Adventures_of_Robinson_Crusoe.epub",
        output_folder="output",
        preview=False,
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
        normalize_steps="ru_llm_stress_ambiguity",
        normalize_provider="openai",
        normalize_model="gpt-5.4",
        normalize_prompt_file=None,
        normalize_system_prompt_file=None,
        normalize_user_prompt_file=None,
        normalize_api_key=None,
        # Fake URL → makes LLM "available" without real credentials
        normalize_base_url="http://127.0.0.1:1234/v1",
        normalize_max_chars=4000,
        normalize_tts_safe_max_chars=180,
        normalize_pronunciation_exceptions_file=None,
        normalize_tts_pronunciation_overrides_file=None,
        normalize_pronunciation_lexicon_db=None,
        normalize_stress_ambiguity_file=None,
        normalize_tsnorm_stress_yo=True,
        normalize_tsnorm_stress_monosyllabic=False,
        normalize_tsnorm_min_word_length=2,
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
    build_tsnorm_pronunciation_lexicon(database, word_forms=word_forms, lemmas=lemmas)
    return database


# ---------------------------------------------------------------------------
# Synthetic word_forms / lemmas for «тела»
# тЕла = родительный падеж ед.ч. «тело»  (stress_pos=[1] → те́ла)
# телА  = винительный падеж мн.ч. «тело» (stress_pos=[4] → тела́)
# ---------------------------------------------------------------------------
TELA_WORD_FORMS: dict = {
    "тела": [
        {
            "word_form": "тела",
            "stress_pos": [1],
            "form_tags": "genitive singular",
            "lemma": "тело",
        },
        {
            "word_form": "тела",
            "stress_pos": [4],
            "form_tags": "accusative plural",
            "lemma": "тело",
        },
    ],
}
TELA_LEMMAS: dict = {"тело": {"pos": ["NOUN"], "rank": 1}}

# Expected spoken forms after DB build
TELA_GEN = f"те{COMBINING_ACUTE}ла"   # тЕла — genitive singular
TELA_ACC = f"тела{COMBINING_ACUTE}"   # телА  — accusative plural


# ---------------------------------------------------------------------------
# Suite 1: fast offline tests with a hand-crafted DB
# ---------------------------------------------------------------------------

class TestStressAmbiguityWithSyntheticDb(unittest.TestCase):
    """
    These tests use a tiny SQLite database containing only «тела».
    No network calls; run entirely offline.
    """

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        db_path = Path(self._tmpdir.name) / "lexicon.sqlite3"
        build_test_lexicon(db_path, word_forms=TELA_WORD_FORMS, lemmas=TELA_LEMMAS)
        config = make_config(normalize_pronunciation_lexicon_db=str(db_path))
        self.normalizer = StressAmbiguityLLMNormalizer(config)

    def tearDown(self):
        self._tmpdir.cleanup()

    # ------------------------------------------------------------------
    # Candidate detection
    # ------------------------------------------------------------------

    def test_tela_is_detected_as_ambiguous_candidate(self):
        text = (
            "Если вспомнить нравственное уродство Кутона, даже большее, "
            "чем уродство его тела, и ту готовность."
        )
        candidates = self.normalizer._collect_candidates(text)
        source_texts = [c.source_text.lower() for c in candidates]
        self.assertIn(
            "тела",
            source_texts,
            f"Слово «тела» должно быть определено как неоднозначное. "
            f"Найденные кандидаты: {source_texts}",
        )

    def test_tela_candidate_options_contain_te_la(self):
        text = "уродство его тела"
        candidates = self.normalizer._collect_candidates(text)
        tela = next((c for c in candidates if c.source_text.lower() == "тела"), None)
        self.assertIsNotNone(tela, "Кандидат «тела» не найден")
        option_texts = {opt.text for opt in tela.options}
        self.assertIn(TELA_GEN, option_texts, f"Вариант тЕла не в опциях: {option_texts}")

    def test_tela_candidate_options_contain_tela_a(self):
        text = "уродство его тела"
        candidates = self.normalizer._collect_candidates(text)
        tela = next((c for c in candidates if c.source_text.lower() == "тела"), None)
        self.assertIsNotNone(tela)
        option_texts = {opt.text for opt in tela.options}
        self.assertIn(TELA_ACC, option_texts, f"Вариант телА не в опциях: {option_texts}")

    def test_tela_candidate_has_at_least_three_options(self):
        # options = ["original", "variant_1 (тЕла)", "variant_2 (телА)"]
        text = "уродство его тела"
        candidates = self.normalizer._collect_candidates(text)
        tela = next((c for c in candidates if c.source_text.lower() == "тела"), None)
        self.assertIsNotNone(tela)
        self.assertGreaterEqual(
            len(tela.options), 3,
            f"Ожидалось ≥3 опции (original + 2 варианта ударения): {tela.options}",
        )

    # ------------------------------------------------------------------
    # plan_processing_units: items sent to the LLM
    # ------------------------------------------------------------------

    def test_plan_units_includes_tela_item(self):
        text = "уродство его тела"
        units = self.normalizer.plan_processing_units(text)
        self.assertGreaterEqual(len(units), 1)
        all_source_texts = []
        for unit in units:
            payload = json.loads(unit)
            all_source_texts.extend(item["source_text"].lower() for item in payload["items"])
        self.assertIn("тела", all_source_texts)

    def test_plan_units_tela_item_has_both_stress_variants(self):
        text = "уродство его тела"
        units = self.normalizer.plan_processing_units(text)
        option_texts: list[str] = []
        for unit in units:
            payload = json.loads(unit)
            for item in payload["items"]:
                if item["source_text"].lower() == "тела":
                    option_texts.extend(opt["text"] for opt in item["options"])
        self.assertIn(TELA_GEN, option_texts, f"тЕла не в опциях LLM-пакета: {option_texts}")
        self.assertIn(TELA_ACC, option_texts, f"телА не в опциях LLM-пакета: {option_texts}")

    # ------------------------------------------------------------------
    # merge_processed_units: applying LLM choices
    # ------------------------------------------------------------------

    def _plan_and_apply(self, text: str, override_option_id: str | None = None) -> str:
        """
        Plan → build a hand-crafted fake LLM response → merge.

        If override_option_id is given, force that option for «тела»;
        all other words get «original».
        """
        units = self.normalizer.plan_processing_units(text)
        self.assertGreaterEqual(len(units), 1, "plan_processing_units returned no units")

        fake_processed: list[str] = []
        for unit in units:
            payload = json.loads(unit)
            selections = []
            for item in payload["items"]:
                if override_option_id and item["source_text"].lower() == "тела":
                    opt_id = override_option_id
                else:
                    opt_id = "original"
                selections.append({
                    "id": item["id"],
                    "option_id": opt_id,
                    "custom_text": "",
                    "cacheable": False,
                    "reason": "test",
                })
            fake_processed.append(json.dumps({"selections": selections}))

        return self.normalizer.merge_processed_units(fake_processed)

    def _variant_option_id_for(self, text: str, spoken_form: str) -> str:
        """Return the option_id that corresponds to *spoken_form* in the тела item."""
        units = self.normalizer.plan_processing_units(text)
        for unit in units:
            payload = json.loads(unit)
            for item in payload["items"]:
                if item["source_text"].lower() == "тела":
                    for opt in item["options"]:
                        if opt["text"] == spoken_form:
                            return opt["id"]
        raise AssertionError(f"Option {spoken_form!r} not found in any unit")

    def test_merge_applies_te_la_stress_when_chosen(self):
        """When LLM selects тЕла variant, the output must contain те́ла."""
        text = "уродство его тела и ту готовность"
        opt_id = self._variant_option_id_for(text, TELA_GEN)
        result = self._plan_and_apply(text, override_option_id=opt_id)
        self.assertIn(
            TELA_GEN,
            result,
            f"Ожидалось {TELA_GEN!r} в результате, получили: {result!r}",
        )

    def test_merge_applies_tela_a_stress_when_chosen(self):
        """When LLM selects телА variant, the output must contain тела́."""
        text = "уродство его тела и ту готовность"
        opt_id = self._variant_option_id_for(text, TELA_ACC)
        result = self._plan_and_apply(text, override_option_id=opt_id)
        self.assertIn(
            TELA_ACC,
            result,
            f"Ожидалось {TELA_ACC!r} в результате, получили: {result!r}",
        )

    def test_merge_leaves_text_unchanged_when_original_selected(self):
        text = "уродство его тела и ту готовность"
        result = self._plan_and_apply(text, override_option_id="original")
        self.assertEqual(result, text)

    def test_word_already_accented_is_skipped(self):
        """A word that already carries a combining acute must not become a candidate."""
        text = f"уродство его те{COMBINING_ACUTE}ла"
        candidates = self.normalizer._collect_candidates(text)
        tela_candidates = [c for c in candidates if c.source_text.lower().startswith("те")]
        self.assertEqual(
            len(tela_candidates),
            0,
            "Уже помеченное ударением слово не должно быть кандидатом",
        )


# ---------------------------------------------------------------------------
# Suite 2: integration tests against the real tsnorm lexicon
# ---------------------------------------------------------------------------

class TestStressAmbiguityWithRealDb(unittest.TestCase):
    """
    Tests against the real tsnorm SQLite lexicon (built once, cached in .cache/).

    First run may take a few seconds while the DB is populated.
    Subsequent runs are fast (cached SQLite file).
    """

    @classmethod
    def setUpClass(cls):
        cls.db = ensure_pronunciation_lexicon_db()

    # ------------------------------------------------------------------
    # тела — should be AMBIGUOUS
    # ------------------------------------------------------------------

    def test_real_db_tela_is_ambiguous(self):
        entries = self.db.lookup_ambiguous_entries("тела")
        self.assertGreaterEqual(
            len(entries),
            2,
            "В реальной БД «тела» должна иметь ≥2 записи с разными произношениями",
        )

    def test_real_db_tela_has_te_la_genitive_variant(self):
        """тЕла — родительный падеж ед.ч. от «тело»."""
        forms = self.db.lookup_spoken_forms("тела", only_ambiguous=True)
        self.assertIn(
            TELA_GEN,
            forms,
            f"Вариант тЕла (ген.ед.) не найден в БД. Формы: {forms}",
        )

    def test_real_db_tela_has_tela_a_accusative_variant(self):
        """телА — винительный падеж мн.ч. от «тело»."""
        forms = self.db.lookup_spoken_forms("тела", only_ambiguous=True)
        self.assertIn(
            TELA_ACC,
            forms,
            f"Вариант телА (вин.мн.) не найден в БД. Формы: {forms}",
        )

    # ------------------------------------------------------------------
    # пика — should NOT be ambiguous (only пи́ка in DB)
    # Requires tsnorm_ru normalizer to insert explicit stress mark.
    # ------------------------------------------------------------------

    def test_real_db_pika_is_not_ambiguous(self):
        """
        «пика» имеет лишь одно произношение — пи́ка.
        stress_ambiguity_llm не может помочь с этим словом.

        Если TTS читает «пикА» вместо «пИка», причина не в неоднозначности,
        а в том, что TTS просто не знает ударения.
        Решение: нормалайзер tsnorm_ru вставит пи́ка в текст явно.
        """
        entries = self.db.lookup_ambiguous_entries("пика")
        self.assertEqual(
            entries,
            (),
            "Ожидалось, что «пика» НЕ является неоднозначной в БД. "
            "Если это изменилось — обнови тест.",
        )

    def test_real_db_pika_correct_spoken_form_is_known(self):
        """Правильная форма пи́ка известна БД — можно использовать через tsnorm_ru."""
        forms = self.db.lookup_spoken_forms("пика")
        self.assertIn(
            f"пи{COMBINING_ACUTE}ка",
            forms,
            f"Ожидалась форма пи́ка в БД, найдено: {forms}",
        )

    # ------------------------------------------------------------------
    # стража — should NOT be ambiguous (only стра́жа in DB)
    # ------------------------------------------------------------------

    def test_real_db_strazha_is_not_ambiguous(self):
        """
        «стража» имеет лишь одно произношение — стра́жа.
        stress_ambiguity_llm не может помочь.
        Решение: tsnorm_ru.
        """
        entries = self.db.lookup_ambiguous_entries("стража")
        self.assertEqual(
            entries,
            (),
            "Ожидалось, что «стража» НЕ является неоднозначной в БД.",
        )

    def test_real_db_strazha_correct_spoken_form_is_known(self):
        """Правильная форма стра́жа известна БД."""
        forms = self.db.lookup_spoken_forms("стража")
        self.assertIn(
            f"стра{COMBINING_ACUTE}жа",
            forms,
            f"Ожидалась форма стра́жа в БД, найдено: {forms}",
        )

    # ------------------------------------------------------------------
    # Algorithm: normalizer correctly detects / does not detect each word
    # ------------------------------------------------------------------

    def _make_normalizer(self) -> StressAmbiguityLLMNormalizer:
        config = make_config(normalize_pronunciation_lexicon_db=str(self.db.path))
        return StressAmbiguityLLMNormalizer(config)

    def test_normalizer_detects_tela_in_full_sentence(self):
        """
        Исходное предложение из задачи:
        «…уродство его тела, и ту готовность…»
        stress_ambiguity_llm должен обнаружить «тела» и предложить оба варианта.
        """
        normalizer = self._make_normalizer()
        text = (
            "Если вспомнить нравственное уродство Кутона, даже большее, "
            "чем уродство его тела, и ту готовность, с какой смерть назначалась "
            "за самое отвлечённое мнение."
        )
        candidates = normalizer._collect_candidates(text)
        source_texts = [c.source_text.lower() for c in candidates]
        self.assertIn(
            "тела",
            source_texts,
            f"«тела» не найдено как кандидат. Найдено: {source_texts}",
        )

    def test_normalizer_tela_options_contain_both_variants_real_db(self):
        normalizer = self._make_normalizer()
        text = "уродство его тела"
        candidates = normalizer._collect_candidates(text)
        tela = next((c for c in candidates if c.source_text.lower() == "тела"), None)
        self.assertIsNotNone(tela, "Кандидат «тела» не найден")
        option_texts = {opt.text for opt in tela.options}
        self.assertIn(TELA_GEN, option_texts, f"тЕла не в опциях: {option_texts}")
        self.assertIn(TELA_ACC, option_texts, f"телА не в опциях: {option_texts}")

    def test_normalizer_pika_not_detected_as_candidate(self):
        """
        «пика» не должна стать кандидатом stress_ambiguity_llm.
        Для правильного ударения (пИка) в предложении «достигла своего пика»
        нужен normalizer tsnorm_ru, который вставит пи́ка явно.
        """
        normalizer = self._make_normalizer()
        text = (
            "Когда ярость против духовенства достигла своего пика "
            "в декретах против него от девятнадцатого и двадцать шестого марта."
        )
        candidates = normalizer._collect_candidates(text)
        source_texts = [c.source_text.lower() for c in candidates]
        self.assertNotIn(
            "пика",
            source_texts,
            "«пика» не должна быть кандидатом stress_ambiguity_llm — "
            "она не неоднозначна в БД. Используй tsnorm_ru для явного ударения.",
        )

    def test_normalizer_strazha_not_detected_as_candidate(self):
        """
        «стража» не должна стать кандидатом stress_ambiguity_llm.
        Используй tsnorm_ru.
        """
        normalizer = self._make_normalizer()
        text = (
            "Когда ярость против духовенства достигла своего пика "
            "в декретах против него. Там стража встречала всех."
        )
        candidates = normalizer._collect_candidates(text)
        source_texts = [c.source_text.lower() for c in candidates]
        self.assertNotIn(
            "стража",
            source_texts,
            "«стража» не должна быть кандидатом stress_ambiguity_llm. "
            "Используй tsnorm_ru для явного ударения.",
        )


if __name__ == "__main__":
    unittest.main()

