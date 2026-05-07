# Copyright (c) 2026 alboro <alboro@users.noreply.github.com>
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See: https://polyformproject.org/licenses/noncommercial/1.0.0/

"""
Tests for SileroStressNormalizer (ru_silero_stress).

Unit tests use a minimal mock accentor so the heavy silero model is not
loaded.  A single integration-test class uses the real silero model and is
skipped automatically when silero-stress is not installed.
"""

from __future__ import annotations

import importlib
import unittest
from unittest.mock import MagicMock, patch

from audiobook_generator.config.general_config import GeneralConfig
from audiobook_generator.normalizers.ru_text_utils import COMBINING_ACUTE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_config(**overrides):
    """Return a minimal GeneralConfig for normalizer tests."""
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
        normalize_steps="ru_silero_stress",
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
        normalize_stress_paradox_words=None,
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


def _make_mock_accentor(stress_map: dict[str, str]):
    """Build a callable mock accentor that applies *stress_map* word-by-word.

    *stress_map* maps lowercase plain words to their silero ``+``-stressed
    forms, e.g. ``{"части": "ч+асти", "один": "од+ин"}``.
    Words not present in the map are returned unchanged.
    """
    import re
    _ru_re = re.compile(r"[А-Яа-яЁё]+")

    def _accentor(text: str) -> str:
        def _replace(m: re.Match) -> str:
            word = m.group(0)
            return stress_map.get(word.lower(), word)
        return _ru_re.sub(_replace, text)

    return _accentor


def _make_mock_homosolver(homodict: dict, yohomodict: dict | None = None):
    hs = MagicMock()
    hs.homodict = homodict
    hs.yohomodict = yohomodict or {}
    return hs


def _make_mock_full_accentor(
    homodict: dict[str, list[str]],
    stress_map: dict[str, str],
    yohomodict: dict | None = None,
):
    """Return a mock accentor with the given homodict and callable behaviour."""
    accentor = _make_mock_accentor(stress_map)
    accentor_obj = MagicMock()
    accentor_obj.side_effect = accentor
    accentor_obj.homosolver = _make_mock_homosolver(homodict, yohomodict)
    accentor_obj.__call__ = lambda self_, text: accentor(text)

    # make it truly callable
    class _FakeAccentor:
        def __init__(self):
            self.homosolver = _make_mock_homosolver(homodict, yohomodict)

        def __call__(self, text):
            return accentor(text)

    return _FakeAccentor()


# ---------------------------------------------------------------------------
# Module reset helper (clears the module-level singleton between tests)
# ---------------------------------------------------------------------------

def _reset_singleton():
    import audiobook_generator.normalizers.ru_silero_stress_normalizer as mod
    mod._accentor_instance = None
    mod._accentor_loaded = False


# ---------------------------------------------------------------------------
# Unit tests — _stress_homographs_in_segment
# ---------------------------------------------------------------------------

class TestStressHomographsInSegment(unittest.TestCase):
    """Tests for the core per-segment processing function."""

    def setUp(self):
        from audiobook_generator.normalizers.ru_silero_stress_normalizer import (
            _stress_homographs_in_segment,
        )
        self._fn = _stress_homographs_in_segment

        from audiobook_generator.normalizers.ru_tts_stress_paradox_guard import (
            TTSStressParadoxGuard,
        )
        self._empty_guard = TTSStressParadoxGuard([])

    # -- basic stress application --

    def test_homograph_gets_stressed(self):
        homodict = {"части": ["ч+асти", "част+и"]}
        accentor = _make_mock_full_accentor(homodict, {"части": "ч+асти"})
        result = self._fn("Части дела", accentor, homodict, self._empty_guard)
        self.assertIn(f"а{COMBINING_ACUTE}", result)  # ча́сти

    def test_stressed_form_correct_value(self):
        homodict = {"части": ["ч+асти", "част+и"]}
        accentor = _make_mock_full_accentor(homodict, {"части": "ч+асти"})
        result = self._fn("части дела", accentor, homodict, self._empty_guard)
        self.assertEqual(result, f"ча{COMBINING_ACUTE}сти дела")

    def test_non_homograph_untouched(self):
        """'земля' is not in homodict → must remain unchanged."""
        homodict = {"один": ["од+ин", "+один"]}
        accentor = _make_mock_full_accentor(homodict, {"один": "од+ин"})
        result = self._fn("земля прекрасна", accentor, homodict, self._empty_guard)
        self.assertEqual(result, "земля прекрасна")

    def test_already_stressed_word_untouched(self):
        """Word with existing stress mark must not be re-stressed."""
        stressed = f"ча{COMBINING_ACUTE}сти"
        homodict = {"части": ["ч+асти", "част+и"]}
        accentor = _make_mock_full_accentor(homodict, {"части": "ч+асти"})
        text = f"{stressed} дела"
        result = self._fn(text, accentor, homodict, self._empty_guard)
        self.assertEqual(result, text)

    # -- case preservation --

    def test_uppercase_word_preserves_case(self):
        homodict = {"один": ["од+ин", "+один"]}
        accentor = _make_mock_full_accentor(homodict, {"один": "од+ин"})
        result = self._fn("ОДИН из них", accentor, homodict, self._empty_guard)
        # preserved upper: ОДИ́Н
        self.assertIn(f"И{COMBINING_ACUTE}", result.upper())

    def test_titlecase_word_preserves_case(self):
        homodict = {"один": ["од+ин", "+один"]}
        accentor = _make_mock_full_accentor(homodict, {"один": "од+ин"})
        result = self._fn("Один из них", accentor, homodict, self._empty_guard)
        self.assertTrue(result[0].isupper())

    # -- stress at word start ('+' before first vowel) --

    def test_stress_before_first_vowel(self):
        """e.g. homodict: 'один' → '+один' (stress on initial 'о')."""
        homodict = {"один": ["од+ин", "+один"]}
        # mock returns '+один' (stress on о)
        accentor = _make_mock_full_accentor(homodict, {"один": "+один"})
        result = self._fn("один из них", accentor, homodict, self._empty_guard)
        # о́дин
        self.assertEqual(result, f"о{COMBINING_ACUTE}дин из них")

    # -- paradox guard --

    def test_paradox_word_skipped(self):
        from audiobook_generator.normalizers.ru_tts_stress_paradox_guard import (
            TTSStressParadoxGuard,
        )
        guard = TTSStressParadoxGuard(["части"])
        homodict = {"части": ["ч+асти", "част+и"]}
        accentor = _make_mock_full_accentor(homodict, {"части": "ч+асти"})
        result = self._fn("части дела", accentor, homodict, guard)
        self.assertEqual(result, "части дела")  # unchanged

    # -- segment with no Russian words --

    def test_empty_segment_returned_as_is(self):
        homodict = {"один": ["од+ин", "+один"]}
        accentor = _make_mock_full_accentor(homodict, {})
        result = self._fn("", accentor, homodict, self._empty_guard)
        self.assertEqual(result, "")

    def test_segment_with_no_russian_returned_as_is(self):
        homodict = {"один": ["од+ин", "+один"]}
        accentor = _make_mock_full_accentor(homodict, {})
        result = self._fn("1+2=3 (ok)", accentor, homodict, self._empty_guard)
        self.assertEqual(result, "1+2=3 (ok)")

    # -- multiple homographs in one segment --

    def test_multiple_homographs_in_sentence(self):
        homodict = {
            "один": ["од+ин", "+один"],
            "части": ["ч+асти", "част+и"],
        }
        stress_map = {"один": "од+ин", "части": "ч+асти"}
        accentor = _make_mock_full_accentor(homodict, stress_map)
        # "части" (not "частей") so it IS in homodict
        result = self._fn("один из части", accentor, homodict, self._empty_guard)
        self.assertIn(f"и{COMBINING_ACUTE}", result)   # оди́н
        self.assertIn(f"а{COMBINING_ACUTE}", result)   # ча́сти

    def test_two_homographs_both_stressed(self):
        homodict = {
            "один": ["од+ин", "+один"],
            "дела": ["дел+а", "д+ела"],
        }
        stress_map = {"один": "од+ин", "дела": "дел+а"}
        accentor = _make_mock_full_accentor(homodict, stress_map)
        result = self._fn("один из дела", accentor, homodict, self._empty_guard)
        # оди́н, дела́
        self.assertEqual(result, f"оди{COMBINING_ACUTE}н из дела{COMBINING_ACUTE}")

    # -- alignment mismatch falls back gracefully --

    def test_alignment_mismatch_returns_unchanged(self):
        """If silero returns a different token count the segment is not modified."""
        homodict = {"части": ["ч+асти", "част+и"]}

        class _BrokenAccentor:
            homosolver = _make_mock_homosolver(homodict)

            def __call__(self, text):
                # Returns extra word → token count mismatch
                return text + " лишнее"

        result = self._fn("части дела", _BrokenAccentor(), homodict, self._empty_guard)
        self.assertEqual(result, "части дела")

    # -- accentor exception falls back gracefully --

    def test_accentor_exception_returns_unchanged(self):
        homodict = {"части": ["ч+асти", "част+и"]}

        class _ErrorAccentor:
            homosolver = _make_mock_homosolver(homodict)

            def __call__(self, text):
                raise RuntimeError("silero crashed")

        result = self._fn("части дела", _ErrorAccentor(), homodict, self._empty_guard)
        self.assertEqual(result, "части дела")


# ---------------------------------------------------------------------------
# Unit tests — SileroStressNormalizer.normalize
# ---------------------------------------------------------------------------

class TestSileroStressNormalizerNormalize(unittest.TestCase):
    """Tests for the full normalize() method, mocking the module singleton."""

    def _make_normalizer_with_mock(
        self,
        homodict: dict,
        stress_map: dict,
        yohomodict: dict | None = None,
        paradox_words: str | None = None,
    ):
        """Instantiate normalizer with a mock accentor injected."""
        cfg = make_config(normalize_stress_paradox_words=paradox_words)
        from audiobook_generator.normalizers.ru_silero_stress_normalizer import (
            SileroStressNormalizer,
        )
        normalizer = SileroStressNormalizer(cfg)
        mock_accentor = _make_mock_full_accentor(homodict, stress_map, yohomodict)
        normalizer._accentor = mock_accentor
        normalizer._homodict = {**mock_accentor.homosolver.yohomodict,
                                 **mock_accentor.homosolver.homodict}
        return normalizer

    # -- basic round-trip --

    def test_normalize_stresses_homograph(self):
        n = self._make_normalizer_with_mock(
            homodict={"части": ["ч+асти", "част+и"]},
            stress_map={"части": "ч+асти"},
        )
        result = n.normalize("части дела")
        self.assertEqual(result, f"ча{COMBINING_ACUTE}сти дела")

    def test_normalize_leaves_non_homograph_unchanged(self):
        n = self._make_normalizer_with_mock(
            homodict={"один": ["од+ин", "+один"]},
            stress_map={"один": "од+ин"},
        )
        result = n.normalize("земля прекрасна")
        self.assertEqual(result, "земля прекрасна")

    # -- chunk_eof splitting --

    def test_chunk_eof_splits_into_independent_segments(self):
        """Each [chunk_eof]-delimited segment is processed independently."""
        homodict = {"части": ["ч+асти", "част+и"]}
        stress_map = {"части": "ч+асти"}
        n = self._make_normalizer_with_mock(homodict=homodict, stress_map=stress_map)
        text = "части дела[chunk_eof]другие части"
        result = n.normalize(text)
        # Both occurrences of "части" should be stressed
        parts = result.split("[chunk_eof]")
        self.assertEqual(len(parts), 2)
        self.assertIn(COMBINING_ACUTE, parts[0])
        self.assertIn(COMBINING_ACUTE, parts[1])

    def test_chunk_eof_marker_preserved_in_output(self):
        homodict = {"один": ["од+ин", "+один"]}
        n = self._make_normalizer_with_mock(homodict=homodict, stress_map={"один": "од+ин"})
        text = "один[chunk_eof]два"
        result = n.normalize(text)
        self.assertIn("[chunk_eof]", result)

    # -- sentence-boundary splitting (.!?) --

    def test_period_splits_into_independent_segments(self):
        """Period-terminated sentences are each processed as an independent segment."""
        homodict = {"части": ["ч+асти", "част+и"]}
        n = self._make_normalizer_with_mock(
            homodict=homodict,
            stress_map={"части": "ч+асти"},
        )
        text = "части дела. другие части."
        result = n.normalize(text)
        # Both occurrences of "части" should receive a stress mark
        self.assertEqual(result.count(COMBINING_ACUTE), 2)
        # Period must be preserved in the output
        self.assertIn(".", result)

    def test_mixed_chunk_eof_and_period_boundaries(self):
        """[chunk_eof] and period boundaries both create independent segments."""
        homodict = {"части": ["ч+асти", "част+и"]}
        n = self._make_normalizer_with_mock(
            homodict=homodict,
            stress_map={"части": "ч+асти"},
        )
        text = "части.[chunk_eof]другие части."
        result = n.normalize(text)
        # At least one stress mark must be present
        self.assertIn(COMBINING_ACUTE, result)
        # [chunk_eof] marker must survive reconstruction
        self.assertIn("[chunk_eof]", result)

    # -- language guard --

    def test_non_russian_language_returns_unchanged(self):
        cfg = make_config(language="en-US")
        from audiobook_generator.normalizers.ru_silero_stress_normalizer import (
            SileroStressNormalizer,
        )
        n = SileroStressNormalizer(cfg)
        text = "один из них"
        self.assertEqual(n.normalize(text), text)

    # -- silero not available (accentor is None) --

    def test_silero_unavailable_returns_text_unchanged(self):
        cfg = make_config()
        from audiobook_generator.normalizers.ru_silero_stress_normalizer import (
            SileroStressNormalizer,
        )
        n = SileroStressNormalizer(cfg)
        n._accentor = None
        n._homodict = {}
        # Patch _ensure_accentor to always return None
        n._accentor = None

        import audiobook_generator.normalizers.ru_silero_stress_normalizer as mod
        orig_load = mod._load_silero_accentor

        def _no_accentor():
            return None

        mod._load_silero_accentor = _no_accentor
        try:
            result = n.normalize("один из них")
            self.assertEqual(result, "один из них")
        finally:
            mod._load_silero_accentor = orig_load

    # -- paradox words --

    def test_paradox_word_not_stressed(self):
        n = self._make_normalizer_with_mock(
            homodict={"один": ["од+ин", "+один"]},
            stress_map={"один": "од+ин"},
            paradox_words="один",
        )
        result = n.normalize("один из них")
        self.assertEqual(result, "один из них")

    # -- idempotency --

    def test_already_stressed_word_idempotent(self):
        stressed = f"ча{COMBINING_ACUTE}сти"
        n = self._make_normalizer_with_mock(
            homodict={"части": ["ч+асти", "част+и"]},
            stress_map={"части": "ч+асти"},
        )
        result = n.normalize(stressed)
        self.assertEqual(result, stressed)

    # -- empty / whitespace text --

    def test_empty_text_returns_empty(self):
        n = self._make_normalizer_with_mock(homodict={}, stress_map={})
        self.assertEqual(n.normalize(""), "")

    def test_whitespace_only_text_unchanged(self):
        n = self._make_normalizer_with_mock(homodict={}, stress_map={})
        self.assertEqual(n.normalize("   \n  "), "   \n  ")

    # -- yohomodict integration --

    def test_yohomodict_word_stressed(self):
        """Words in yohomodict (ё/е alternation) should also be stressed."""
        homodict = {}
        yohomodict = {"мел": ["м+ёл", "м+ел"]}
        n = self._make_normalizer_with_mock(
            homodict=homodict,
            stress_map={"мел": "м+ёл"},
            yohomodict=yohomodict,
        )
        result = n.normalize("он мел пол")
        # м+ёл → мё́л
        self.assertIn("ё", result)


# ---------------------------------------------------------------------------
# Unit tests — NORMALIZER_REGISTRY
# ---------------------------------------------------------------------------

class TestNormalizerRegistration(unittest.TestCase):

    def test_ru_silero_stress_in_registry(self):
        from audiobook_generator.normalizers.base_normalizer import NORMALIZER_REGISTRY
        self.assertIn("ru_silero_stress", NORMALIZER_REGISTRY)

    def test_registry_entry_points_to_correct_class(self):
        from audiobook_generator.normalizers.base_normalizer import NORMALIZER_REGISTRY
        module_path, class_name = NORMALIZER_REGISTRY["ru_silero_stress"]
        self.assertEqual(class_name, "SileroStressNormalizer")

    def test_class_is_importable(self):
        from audiobook_generator.normalizers.ru_silero_stress_normalizer import (
            SileroStressNormalizer,
        )
        self.assertIsNotNone(SileroStressNormalizer)

    def test_step_name_constant(self):
        from audiobook_generator.normalizers.ru_silero_stress_normalizer import (
            SileroStressNormalizer,
        )
        self.assertEqual(SileroStressNormalizer.STEP_NAME, "ru_silero_stress")


# ---------------------------------------------------------------------------
# Integration tests — real silero model
# ---------------------------------------------------------------------------

try:
    from silero_stress.accentor import load_accentor as _check_silero  # type: ignore
    _SILERO_AVAILABLE = True
except ImportError:
    _SILERO_AVAILABLE = False


@unittest.skipUnless(_SILERO_AVAILABLE, "silero-stress not installed")
class TestSileroStressIntegration(unittest.TestCase):
    """End-to-end tests using the real silero accentor model."""

    @classmethod
    def setUpClass(cls):
        _reset_singleton()
        # Pre-load the model once for the whole test class.
        from audiobook_generator.normalizers.ru_silero_stress_normalizer import (
            _load_silero_accentor,
            _build_homodict,
        )
        cls._accentor = _load_silero_accentor()
        cls._homodict = _build_homodict(cls._accentor)

    def _normalizer(self, **cfg_overrides):
        from audiobook_generator.normalizers.ru_silero_stress_normalizer import (
            SileroStressNormalizer,
        )
        cfg = make_config(**cfg_overrides)
        n = SileroStressNormalizer(cfg)
        n._accentor = self._accentor
        n._homodict = self._homodict
        return n

    # -- known homographs from silero homodict --

    def test_chasti_genit_first_syllable(self):
        """'Части дела' — 'части' is in homodict, so it must receive a stress mark."""
        n = self._normalizer()
        result = n.normalize("Части дела")
        # silero chooses a form (first or second syllable) — we only assert stress IS placed
        self.assertIn(COMBINING_ACUTE, result)

    def test_dela_plural_last_syllable(self):
        """'Дела шли хорошо' → 'дела́'."""
        n = self._normalizer()
        result = n.normalize("Дела шли хорошо")
        self.assertIn(f"а{COMBINING_ACUTE}", result)

    def test_odin_unstressed_not_in_homodict_variant(self):
        """'один из них' — 'один' is in homodict, should be stressed."""
        n = self._normalizer()
        result = n.normalize("один из них")
        self.assertIn(COMBINING_ACUTE, result)

    def test_storon_stressed(self):
        """'стороны' is in homodict."""
        n = self._normalizer()
        result = n.normalize("Стороны конфликта договорились")
        self.assertIn(COMBINING_ACUTE, result)

    def test_uzhe_stressed(self):
        """'уже' is in homodict."""
        n = self._normalizer()
        result = n.normalize("Уже светало")
        self.assertIn(COMBINING_ACUTE, result)

    def test_propast_stressed(self):
        """'пропасть' is in homodict ('провал' vs 'исчезнуть')."""
        n = self._normalizer()
        result = n.normalize("Он смотрел в пропасть")
        self.assertIn(COMBINING_ACUTE, result)

    # -- non-homographs must NOT be stressed --

    def test_zemla_not_in_homodict_not_stressed(self):
        """'земля' is not in silero's homodict → no stress mark added."""
        n = self._normalizer()
        result = n.normalize("земля прекрасна")
        self.assertNotIn(COMBINING_ACUTE, result)

    def test_ona_not_in_homodict_not_stressed(self):
        """'она' is not in homodict → must not receive stress mark."""
        n = self._normalizer()
        result = n.normalize("она пришла домой")
        self.assertNotIn(COMBINING_ACUTE, result)

    def test_togo_not_in_homodict_not_stressed(self):
        """'того' is not in homodict → the word 'того' must not receive a stress mark.
        Use a sentence where only 'того' is the candidate (no other homodict words)."""
        n = self._normalizer()
        # "вместо того" — 'вместо' is not in homodict, 'того' is not in homodict
        result = n.normalize("вместо того чтобы")
        self.assertNotIn(COMBINING_ACUTE, result)

    def test_vera_not_stressed(self):
        """'вера' is not in homodict."""
        n = self._normalizer()
        result = n.normalize("вера в бога")
        self.assertNotIn(COMBINING_ACUTE, result)

    # -- already stressed words are preserved as-is --

    def test_prestressed_word_unchanged(self):
        stressed = f"ча{COMBINING_ACUTE}сти"
        n = self._normalizer()
        result = n.normalize(f"{stressed} дела")
        self.assertTrue(result.startswith(stressed))

    # -- homodict coverage sanity --

    def test_homodict_not_empty(self):
        self.assertGreater(len(self._homodict), 1000)

    def test_chasti_in_homodict(self):
        self.assertIn("части", self._homodict)

    def test_ona_not_in_homodict(self):
        self.assertNotIn("она", self._homodict)

    def test_togo_not_in_homodict(self):
        self.assertNotIn("того", self._homodict)


if __name__ == "__main__":
    unittest.main()
