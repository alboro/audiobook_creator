# Copyright (c) 2026 alboro <alboro@users.noreply.github.com>
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See: https://polyformproject.org/licenses/noncommercial/1.0.0/

"""Tests for TTSSafeSplitNormalizer and its module-level helpers."""
from __future__ import annotations

import json
import re
import unittest
from unittest.mock import MagicMock, patch

from audiobook_generator.normalizers.tts_safe_split_normalizer import (
    TTSSafeSplitNormalizer,
    TTSSafeSplitAlgorithmicNormalizer,
    _make_batches,
    _parse_split_response,
    _rejoin_sentences,
    _split_text_preserve_separators,
    _merge_broken_backtick_sentences,
    DEFAULT_SAFE_SPLIT_SYSTEM_PROMPT,
    _get_safe_split_prompt,
)


# ---------------------------------------------------------------------------
# Helpers for building a minimal config-like object
# ---------------------------------------------------------------------------

class _Cfg:
    language = "ru-RU"
    normalize_tts_safe_max_chars = 80
    normalize_tts_safe_comma_as_period = False
    normalize_max_chars = 4000
    normalize_model = "gpt-test"
    normalize_safe_split_system_prompt = None
    normalize_provider = "openai"
    normalize_system_prompt_file = None
    normalize_prompt_file = None
    normalize_user_prompt_file = None
    normalize_api_key = None
    normalize_base_url = None
    output_folder = None
    prepared_text_folder = None
    _normalizer_llm_runtime = None

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def _make_normalizer(max_chars=80, safe_split_prompt=None) -> TTSSafeSplitNormalizer:
    cfg = _Cfg(
        normalize_tts_safe_max_chars=max_chars,
        normalize_safe_split_system_prompt=safe_split_prompt,
    )
    return TTSSafeSplitNormalizer(cfg)


def _make_algo_normalizer(max_chars=80, comma_as_period=False) -> TTSSafeSplitAlgorithmicNormalizer:
    cfg = _Cfg(
        normalize_tts_safe_max_chars=max_chars,
        normalize_tts_safe_comma_as_period=comma_as_period,
    )
    return TTSSafeSplitAlgorithmicNormalizer(cfg)


# ---------------------------------------------------------------------------
# _split_text_preserve_separators
# ---------------------------------------------------------------------------

class TestSplitTextPreserveSeparators(unittest.TestCase):
    def test_single_sentence(self):
        sentences, separators = _split_text_preserve_separators("Привет мир.")
        self.assertEqual(sentences, ["Привет мир."])
        self.assertEqual(len(separators), 1)

    def test_two_sentences(self):
        sentences, separators = _split_text_preserve_separators("Первое. Второе.")
        self.assertEqual(sentences, ["Первое.", "Второе."])
        self.assertEqual(len(separators), 2)

    def test_three_sentences(self):
        text = "Раз. Два. Три."
        sentences, separators = _split_text_preserve_separators(text)
        self.assertEqual(sentences, ["Раз.", "Два.", "Три."])

    def test_rejoin_roundtrip(self):
        text = "Первое предложение. Второе предложение. Третье."
        sentences, separators = _split_text_preserve_separators(text)
        rejoined = _rejoin_sentences(sentences, separators)
        self.assertEqual(rejoined, text)


# ---------------------------------------------------------------------------
# _parse_split_response
# ---------------------------------------------------------------------------

class TestParseSplitResponse(unittest.TestCase):
    def test_valid_response(self):
        response = json.dumps([
            {"id": "s-0", "parts": ["Часть первая.", "Часть вторая."]},
            {"id": "s-3", "parts": ["Только одна часть."]},
        ])
        result = _parse_split_response(response)
        self.assertEqual(result["s-0"], ["Часть первая.", "Часть вторая."])
        self.assertEqual(result["s-3"], ["Только одна часть."])

    def test_markdown_fence_stripped(self):
        response = "```json\n" + json.dumps([{"id": "s-1", "parts": ["A.", "B."]}]) + "\n```"
        result = _parse_split_response(response)
        self.assertEqual(result["s-1"], ["A.", "B."])

    def test_empty_response(self):
        self.assertEqual(_parse_split_response(""), {})

    def test_invalid_json_raises(self):
        with self.assertRaises(Exception):
            _parse_split_response("not json at all !!!")

    def test_skips_entries_without_id(self):
        response = json.dumps([{"parts": ["X."]}])
        result = _parse_split_response(response)
        self.assertEqual(result, {})


# ---------------------------------------------------------------------------
# _make_batches
# ---------------------------------------------------------------------------

class TestMakeBatches(unittest.TestCase):
    def _item(self, idx: int, sentence_len: int = 100) -> dict:
        return {
            "idx": idx,
            "id": f"s-{idx}",
            "sentence": "A" * sentence_len,
            "context_before": "",
            "context_after": "",
        }

    def test_single_batch_when_small(self):
        items = [self._item(i, 50) for i in range(5)]
        batches = _make_batches(items, system_prompt="SYS", budget=10000)
        self.assertEqual(len(batches), 1)
        self.assertEqual(len(batches[0]), 5)

    def test_splits_into_multiple_batches(self):
        # Each item is 500 chars, budget=600 → each batch holds 1 item
        items = [self._item(i, 450) for i in range(4)]
        batches = _make_batches(items, system_prompt="", budget=600)
        # Each item alone (450+80=530) fits within 600; two items would be 1060 > 600
        self.assertEqual(len(batches), 4)

    def test_empty_items(self):
        self.assertEqual(_make_batches([], system_prompt="SYS", budget=1000), [])


# ---------------------------------------------------------------------------
# _get_safe_split_prompt
# ---------------------------------------------------------------------------

class TestGetSafeSplitPrompt(unittest.TestCase):
    def test_returns_default_when_no_config(self):
        cfg = _Cfg()
        self.assertEqual(
            _get_safe_split_prompt(cfg, 180),
            DEFAULT_SAFE_SPLIT_SYSTEM_PROMPT.replace("%max_chars%", "180"),
        )

    def test_returns_custom_from_config(self):
        cfg = _Cfg(normalize_safe_split_system_prompt="My custom prompt")
        self.assertEqual(_get_safe_split_prompt(cfg, 180), "My custom prompt")

    def test_strips_whitespace(self):
        cfg = _Cfg(normalize_safe_split_system_prompt="  prompt with spaces  ")
        self.assertEqual(_get_safe_split_prompt(cfg, 180), "prompt with spaces")

    def test_max_chars_substitution(self):
        cfg = _Cfg(normalize_safe_split_system_prompt="Limit is %max_chars% chars.")
        self.assertEqual(_get_safe_split_prompt(cfg, 150), "Limit is 150 chars.")


# ---------------------------------------------------------------------------
# TTSSafeSplitNormalizer — algorithmic splitting (no LLM)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# TTSSafeSplitNormalizer — algorithmic helpers (tts_safe_split mode)
# ---------------------------------------------------------------------------

class TestTTSSafeSplitNormalizerAlgorithmic(unittest.TestCase):
    """Tests for TTSSafeSplitAlgorithmicNormalizer (tts_safe_split step — no LLM)."""

    def setUp(self):
        self.n = _make_algo_normalizer(max_chars=80)

    def _algo_normalize(self, text: str) -> str:
        return self.n.normalize(text)

    def test_step_name(self):
        self.assertEqual(self.n.get_step_name(), "tts_safe_split")

    def test_short_text_unchanged(self):
        text = "Короткий текст."
        result = self._algo_normalize(text)
        self.assertEqual(result, "Короткий текст.")

    def test_long_sentence_split(self):
        text = (
            "Он спросил: куда идти? "
            "Никто не знал ответа на этот простой вопрос, и все молчали."
        )
        result = self._algo_normalize(text)
        self.assertIn("куда", result)
        self.assertIn("Никто", result)

    def test_multiple_paragraphs_preserved(self):
        text = "Первый абзац.\n\nВторой абзац."
        result = self._algo_normalize(text)
        self.assertIn("\n\n", result)

    def test_empty_string(self):
        self.assertEqual(self._algo_normalize(""), "")

    def test_semicolon_replaced_with_period(self):
        n = _make_algo_normalizer(max_chars=200)
        result = n._preprocess_punctuation("Первая часть; вторая часть.")
        self.assertNotIn(";", result)
        self.assertIn(".", result)

    def test_comma_as_period_disabled_by_default(self):
        n = _make_algo_normalizer(max_chars=200)
        result = n._preprocess_punctuation("Первое, второе, третье.")
        self.assertIn(",", result)

    def test_comma_as_period_enabled(self):
        n = _make_algo_normalizer(max_chars=200, comma_as_period=True)
        result = n._preprocess_punctuation("Первое, второе, третье.")
        self.assertNotIn(",", result)


class TestNormalizeLLMMode(unittest.TestCase):
    """normalize() always delegates to _llm_split_long_sentences."""

    def test_llm_normalizer_step_name(self):
        n = _make_normalizer(max_chars=60)
        self.assertEqual(n.get_step_name(), "tts_llm_safe_split")

    def test_normalize_calls_llm_split(self):
        n = _make_normalizer(max_chars=60)
        with patch.object(n, "_llm_split_long_sentences", return_value="ok") as mock_split:
            result = n.normalize("some text")
        mock_split.assert_called_once()
        self.assertEqual(result, "ok")



# ---------------------------------------------------------------------------
# TTSSafeSplitNormalizer — LLM pass
# ---------------------------------------------------------------------------

class TestTTSSafeSplitNormalizerLLM(unittest.TestCase):
    def _normalizer_with_mock_llm(self, llm_response: str, max_chars: int = 60) -> TTSSafeSplitNormalizer:
        n = _make_normalizer(max_chars=max_chars)
        mock_llm = MagicMock()
        mock_llm.is_available = True
        mock_llm.complete.return_value = llm_response
        n.config._normalizer_llm_runtime = mock_llm
        return n

    def test_llm_splits_long_sentence(self):
        long_sent = "Очень длинное предложение, которое нужно разбить на части для синтеза речи."  # >60 chars
        # New id format: p{para}s{sent}
        llm_response = json.dumps([
            {"id": "p0s0", "parts": ["Очень длинное предложение,", "которое нужно разбить на части для синтеза речи."]}
        ])
        n = self._normalizer_with_mock_llm(llm_response, max_chars=60)

        with patch.object(n, "has_normalizer_llm", return_value=True):
            result = n._llm_split_long_sentences(long_sent + " Короткое.")

        self.assertIn("Очень длинное предложение", result)
        self.assertIn("которое нужно разбить", result)

    def test_llm_not_called_when_all_short(self):
        n = _make_normalizer(max_chars=200)
        mock_llm = MagicMock()
        mock_llm.is_available = True
        n.config._normalizer_llm_runtime = mock_llm

        with patch.object(n, "has_normalizer_llm", return_value=True):
            result = n._llm_split_long_sentences("Короткое предложение.")

        mock_llm.complete.assert_not_called()
        self.assertEqual(result, "Короткое предложение.")

    def test_llm_unavailable_returns_text_as_is(self):
        n = _make_normalizer(max_chars=60)
        with patch.object(n, "has_normalizer_llm", return_value=False):
            text = "А" * 100
            result = n._llm_split_long_sentences(text)
        self.assertEqual(result, text)

    def test_llm_exception_falls_back_gracefully(self):
        n = _make_normalizer(max_chars=60)
        mock_llm = MagicMock()
        mock_llm.is_available = True
        mock_llm.complete.side_effect = RuntimeError("LLM down")
        n.config._normalizer_llm_runtime = mock_llm

        long_text = "А" * 100
        with patch.object(n, "has_normalizer_llm", return_value=True):
            result = n._llm_split_long_sentences(long_text)
        # Should return original text unchanged
        self.assertEqual(result, long_text)

    def test_bulk_batching_calls_llm_multiple_times(self):
        """When items don't fit into one budget, _call_llm_bulk splits into multiple LLM calls."""
        n = _make_normalizer(max_chars=60)
        n.config.normalize_max_chars = 200  # very small budget → forces batching

        mock_llm = MagicMock()
        mock_llm.is_available = True

        def side_effect(*, user_prompt, system_prompt, model, temperature):
            # Find the JSON array in the user prompt and echo ids back
            start = user_prompt.rfind("[")
            data = json.loads(user_prompt[start:])
            return json.dumps([{"id": item["id"], "parts": [item["sentence"]]} for item in data])

        mock_llm.complete.side_effect = side_effect
        n.config._normalizer_llm_runtime = mock_llm

        items = [
            {"para": 0, "sent": i, "id": f"p0s{i}", "sentence": "X" * 80,
             "context_before": "", "context_after": ""}
            for i in range(3)
        ]
        with patch.object(n, "has_normalizer_llm", return_value=True):
            result = n._call_llm_bulk(items, chapter_title="ch")

        self.assertEqual(len(result), 3)

    def test_paragraph_breaks_preserved(self):
        """LLM pass must not merge paragraphs — \\n\\n breaks must survive."""
        short1 = "Первый абзац, короткое предложение."
        long_sent = "Х" * 70  # > 60 chars, will be sent to LLM
        short2 = "Третий абзац, тоже короткое."
        text = f"{short1}\n\n{long_sent}\n\n{short2}"

        llm_response = json.dumps([
            {"id": "p1s0", "parts": ["Часть один.", "Часть два."]}
        ])
        n = self._normalizer_with_mock_llm(llm_response, max_chars=60)
        with patch.object(n, "has_normalizer_llm", return_value=True):
            result = n._llm_split_long_sentences(text)

        self.assertIn("\n\n", result, "paragraph breaks must be preserved")
        # All three paragraphs' content must survive
        self.assertIn("Первый абзац", result)
        self.assertIn("Часть один", result)
        self.assertIn("Третий абзац", result)


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------------
# _merge_broken_backtick_sentences
# ---------------------------------------------------------------------------

class TestMergeBrokenBacktickSentences(unittest.TestCase):
    """Unit tests for the backtick-repair helper."""

    def test_repairs_split_inside_backtick_quote(self):
        # sentencex artefact: [`Тьма!`, `\` Из...`]
        sentences = ["`Тьма!", "` Из Ветхого."]
        result = _merge_broken_backtick_sentences(sentences)
        self.assertEqual(result, ["`Тьма!`", "Из Ветхого."])

    def test_no_change_when_backtick_balanced(self):
        sentences = ["`Тьма!`", "Из Ветхого."]
        result = _merge_broken_backtick_sentences(sentences)
        self.assertEqual(result, ["`Тьма!`", "Из Ветхого."])

    def test_no_change_when_no_backticks(self):
        sentences = ["Обычное предложение.", "Ещё одно."]
        result = _merge_broken_backtick_sentences(sentences)
        self.assertEqual(result, ["Обычное предложение.", "Ещё одно."])

    def test_lone_closing_backtick_sentence_consumed(self):
        # next_sent is just `` ` `` with nothing after
        sentences = ["`Тьма!", "`"]
        result = _merge_broken_backtick_sentences(sentences)
        self.assertEqual(result, ["`Тьма!`"])

    def test_multiple_sentences_only_broken_pair_repaired(self):
        sentences = ["Начало.", "`Тьма!", "` Из Ветхого.", "Конец."]
        result = _merge_broken_backtick_sentences(sentences)
        self.assertEqual(result, ["Начало.", "`Тьма!`", "Из Ветхого.", "Конец."])


# ---------------------------------------------------------------------------
# Algorithmic normalizer — backtick quote splitting bug (regression test)
# ---------------------------------------------------------------------------

class TestAlgoNormalizerBacktickQuote(unittest.TestCase):
    """
    Regression: sentencex splits backtick-quoted words like `Тьма!` into
    two fragments: ['`Тьма!', '` Из...'] — the closing backtick ends up
    at the start of the next sentence.
    After the fix, the closing backtick must be re-attached and no sentence
    starting with a bare '`' must appear in the output.

    Input (single paragraph):
        Христе́, а именно что он был человеком, воскликнули.
        `Тьма!` Из Ве́тхого и Но́вого Заве́тов, говорят они, мы берём только то,
        что полезно, главным образом нравственное учение...
        Нравственные идеи духоборцев таковы.

    Expected output sentences (before short-sentence merge):
        1. Христе́, а именно что он был человеком, воскликнули.
        2. `Тьма!`                       ← quote kept intact
        3. Из Ве́тхого и Но́вого...       ← NO leading backtick
        4. Нравственные идеи духоборцев таковы.

    Because `Тьма!` (8 chars) < MIN_TTS_SAFE_CHARS (12), it will be merged
    with the previous sentence by the short-sentence merge step.  The final
    result may therefore look like:
        Христе́, а именно что он был человеком, воскликнули. `Тьма!`.
    The key invariants the test checks:
        - No fragment of the form "` Из" (stray closing backtick before content).
        - The text "Тьма" is present somewhere in the result.
        - The text "Из Ве́тхого" is present without a leading backtick.
        - Paragraph structure (only one paragraph here) is not broken.
    """

    _TEXT = (
        "Христе́, а именно что он был человеком, воскликнули. "
        "`Тьма!` Из Ве́тхого и Но́вого Заве́тов, говорят они, мы берём только то, "
        "что полезно, главным образом нравственное учение... "
        "Нравственные идеи духоборцев таковы."
    )

    def setUp(self):
        self.n = _make_algo_normalizer(max_chars=180)

    def test_no_stray_closing_backtick_fragment(self):
        """The artifact '` Из' must not appear anywhere in the result."""
        result = self.n.normalize(self._TEXT)
        self.assertNotIn("` Из", result,
                         f"Stray closing backtick found in result:\n{result}")

    def test_tьма_present(self):
        result = self.n.normalize(self._TEXT)
        self.assertIn("Тьма", result)

    def test_vetkhogo_present_without_leading_backtick(self):
        result = self.n.normalize(self._TEXT)
        # Must NOT start a sentence/chunk with a bare backtick before "Из"
        self.assertNotIn("` Из", result)
        self.assertIn("Из Ве́тхого", result)

    def test_nravstvennye_present(self):
        result = self.n.normalize(self._TEXT)
        self.assertIn("Нравственные идеи духоборцев", result)

    def test_no_extra_paragraph_breaks(self):
        """Single input paragraph must not become multiple paragraphs."""
        result = self.n.normalize(self._TEXT)
        self.assertNotIn("\n\n", result)


if __name__ == "__main__":
    unittest.main()


