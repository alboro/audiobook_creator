# Copyright (c) 2026 alboro <alboro@users.noreply.github.com>
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See: https://polyformproject.org/licenses/noncommercial/1.0.0/

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from audiobook_generator.config.general_config import GeneralConfig
from audiobook_generator.normalizers.pronunciation_lexicon_db import (
    PronunciationLexiconDB,
    build_tsnorm_pronunciation_lexicon,
)
from audiobook_generator.normalizers.ru_text_utils import COMBINING_ACUTE
from audiobook_generator.normalizers.ru_stress_ambiguity_normalizer import (
    StressAmbiguityLLMNormalizer,
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
        normalize_steps="ru_llm_stress_ambiguity",
        normalize_provider="openai",
        normalize_model="gpt-5.4",
        normalize_prompt_file=None,
        normalize_system_prompt_file=None,
        normalize_user_prompt_file=None,
        normalize_api_key=None,
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
    build_tsnorm_pronunciation_lexicon(
        database,
        word_forms=word_forms,
        lemmas=lemmas,
    )
    return database


class TestPronunciationLexiconDB(unittest.TestCase):
    def test_builds_entries_from_tsnorm_data(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "lexicon.sqlite3"
            database = build_test_lexicon(
                db_path,
                word_forms={
                    "собака": [
                        {
                            "word_form": "собака",
                            "stress_pos": [3],
                            "form_tags": "canonical",
                            "lemma": "собака",
                        }
                    ],
                    "томаса": [
                        {
                            "word_form": "томаса",
                            "stress_pos": [1],
                            "form_tags": "genitive singular",
                            "lemma": "томас",
                        }
                    ],
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
                    ],
                },
                lemmas={
                    "собака": {"pos": ["NOUN"], "rank": 1},
                    "томас": {"pos": ["PNOUN"], "rank": 1},
                    "беда": {"pos": ["NOUN"], "rank": 1},
                },
            )

            sobaka = database.lookup("собака")[0]
            self.assertEqual(sobaka.spoken_form, f"соба{COMBINING_ACUTE}ка")
            self.assertFalse(sobaka.is_proper_name)

            tomasa = database.lookup("томаса")[0]
            self.assertEqual(tomasa.lemma, "томас")
            self.assertEqual(tomasa.pos, "PNOUN")
            self.assertTrue(tomasa.is_proper_name)

            variants = database.lookup_spoken_forms("беды", only_ambiguous=True)
            self.assertEqual(
                variants,
                (f"бе{COMBINING_ACUTE}ды", f"беды{COMBINING_ACUTE}"),
            )

    def test_lookup_ambiguous_entries_requires_multiple_spoken_forms(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "lexicon.sqlite3"
            database = build_test_lexicon(
                db_path,
                word_forms={
                    "слово": [
                        {
                            "word_form": "слово",
                            "stress_pos": [2],
                            "form_tags": "canonical",
                            "lemma": "слово",
                        }
                    ],
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
                    ],
                },
                lemmas={
                    "слово": {"pos": ["NOUN"], "rank": 1},
                    "беда": {"pos": ["NOUN"], "rank": 1},
                },
            )

            self.assertEqual(database.lookup_ambiguous_entries("слово"), ())
            self.assertEqual(len(database.lookup_ambiguous_entries("беды")), 2)


class TestStressAmbiguityLLMWithLexiconDB(unittest.TestCase):
    def test_surface_form_with_single_spoken_form_is_not_candidate(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "lexicon.sqlite3"
            build_test_lexicon(
                db_path,
                word_forms={
                    "слова": [
                        {
                            "word_form": "слова",
                            "stress_pos": [4],
                            "form_tags": "nominative plural",
                            "lemma": "слово",
                        }
                    ]
                },
                lemmas={"слово": {"pos": ["NOUN"], "rank": 1}},
            )
            normalizer = StressAmbiguityLLMNormalizer(
                make_config(
                    normalize_pronunciation_lexicon_db=str(db_path),
                )
            )

            units = normalizer.plan_processing_units(
                "Но необходимо взвесить приведенные слова.",
                chapter_title="Test",
            )

            self.assertEqual(units, [])

    def test_surface_form_with_multiple_spoken_forms_becomes_candidate(self):
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
                    normalize_pronunciation_lexicon_db=str(db_path),
                )
            )

            units = normalizer.plan_processing_units(
                "После беды пришли новые беды.",
                chapter_title="Test",
            )

            self.assertGreaterEqual(len(units), 1)
            option_texts: list[str] = []
            for unit in units:
                payload = json.loads(unit)
                for item in payload["items"]:
                    if item["source_text"].lower() != "беды":
                        continue
                    option_texts.extend(option["text"] for option in item["options"])
            self.assertIn(f"бе{COMBINING_ACUTE}ды", option_texts)
            self.assertIn(f"беды{COMBINING_ACUTE}", option_texts)

    def test_ignores_legacy_mapping_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "lexicon.sqlite3"
            build_test_lexicon(
                db_path,
                word_forms={
                    "слово": [
                        {
                            "word_form": "слово",
                            "stress_pos": [2],
                            "form_tags": "canonical",
                            "lemma": "слово",
                        }
                    ]
                },
                lemmas={"слово": {"pos": ["NOUN"], "rank": 1}},
            )
            mapping_path = Path(temp_dir) / "legacy_ambiguities.txt"
            mapping_path.write_text(
                "замок==з+амок|зам+ок\n",
                encoding="utf-8",
            )
            normalizer = StressAmbiguityLLMNormalizer(
                make_config(
                    normalize_pronunciation_lexicon_db=str(db_path),
                    normalize_stress_ambiguity_file=str(mapping_path),
                )
            )

            units = normalizer.plan_processing_units(
                "Старинный замок молчал.",
                chapter_title="Test",
            )

            self.assertEqual(units, [])

    def test_candidates_manifest_contains_only_db_derived_cases_with_metadata(self):
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
                    ],
                    "замок": [
                        {
                            "word_form": "замок",
                            "stress_pos": [1],
                            "form_tags": "castle",
                            "lemma": "замок",
                        },
                        {
                            "word_form": "замок",
                            "stress_pos": [4],
                            "form_tags": "lock",
                            "lemma": "замок",
                        },
                    ],
                    "муки": [
                        {
                            "word_form": "муки",
                            "stress_pos": [1],
                            "form_tags": "torment",
                            "lemma": "мука",
                        },
                        {
                            "word_form": "муки",
                            "stress_pos": [3],
                            "form_tags": "flour",
                            "lemma": "мука",
                        },
                    ],
                    "поступи": [
                        {
                            "word_form": "поступи",
                            "stress_pos": [1],
                            "form_tags": "noun plural",
                            "lemma": "поступь",
                        },
                        {
                            "word_form": "поступи",
                            "stress_pos": [6],
                            "form_tags": "imperative",
                            "lemma": "поступить",
                        },
                    ],
                },
                lemmas={
                    "беда": {"pos": ["NOUN"], "rank": 1},
                    "замок": {"pos": ["NOUN"], "rank": 1},
                    "мука": {"pos": ["NOUN"], "rank": 1},
                    "поступь": {"pos": ["NOUN"], "rank": 1},
                    "поступить": {"pos": ["VERB"], "rank": 1},
                },
            )
            mapping_path = Path(temp_dir) / "legacy_ambiguities.txt"
            mapping_path.write_text(
                "слова==сл+ова|слов+а\n",
                encoding="utf-8",
            )
            normalizer = StressAmbiguityLLMNormalizer(
                make_config(
                    normalize_pronunciation_lexicon_db=str(db_path),
                    normalize_stress_ambiguity_file=str(mapping_path),
                )
            )

            artifacts = normalizer.get_step_artifacts(
                "Беды, замок, муки и поступи.",
                chapter_title="Test",
            )
            manifest = json.loads(artifacts["02_candidates.json"])

            self.assertEqual(
                [item["source_text"].lower() for item in manifest],
                ["беды", "замок", "муки", "поступи"],
            )
            self.assertTrue(all(item["lexicon_entries"] for item in manifest))
            self.assertTrue(
                all(
                    {"lemma", "pos", "grammemes", "is_proper_name", "source", "spoken_form"}
                    <= set(item["lexicon_entries"][0].keys())
                    for item in manifest
                )
            )
            pronunciation_info = json.loads(artifacts["03_pronunciation_lexicon.json"])
            self.assertTrue(pronunciation_info["legacy_stress_ambiguity_file_ignored"])

    def test_plan_processing_units_batches_multiple_db_ambiguities(self):
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
                    ],
                    "замок": [
                        {
                            "word_form": "замок",
                            "stress_pos": [1],
                            "form_tags": "castle",
                            "lemma": "замок",
                        },
                        {
                            "word_form": "замок",
                            "stress_pos": [4],
                            "form_tags": "lock",
                            "lemma": "замок",
                        },
                    ],
                    "муки": [
                        {
                            "word_form": "муки",
                            "stress_pos": [1],
                            "form_tags": "torment",
                            "lemma": "мука",
                        },
                        {
                            "word_form": "муки",
                            "stress_pos": [3],
                            "form_tags": "flour",
                            "lemma": "мука",
                        },
                    ],
                    "поступи": [
                        {
                            "word_form": "поступи",
                            "stress_pos": [1],
                            "form_tags": "noun plural",
                            "lemma": "поступь",
                        },
                        {
                            "word_form": "поступи",
                            "stress_pos": [6],
                            "form_tags": "imperative",
                            "lemma": "поступить",
                        },
                    ],
                },
                lemmas={
                    "беда": {"pos": ["NOUN"], "rank": 1},
                    "замок": {"pos": ["NOUN"], "rank": 1},
                    "мука": {"pos": ["NOUN"], "rank": 1},
                    "поступь": {"pos": ["NOUN"], "rank": 1},
                    "поступить": {"pos": ["VERB"], "rank": 1},
                },
            )
            normalizer = StressAmbiguityLLMNormalizer(
                make_config(
                    normalize_pronunciation_lexicon_db=str(db_path),
                    normalize_max_chars=6000,
                )
            )

            units = normalizer.plan_processing_units(
                "Беды и замок. Муки и поступи.",
                chapter_title="Test",
            )

            self.assertEqual(len(units), 1)
            payload = json.loads(units[0])
            self.assertEqual(len(payload["items"]), 4)


if __name__ == "__main__":
    unittest.main()
