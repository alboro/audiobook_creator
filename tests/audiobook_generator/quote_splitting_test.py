# Copyright (c) 2026 alboro <alboro@users.noreply.github.com>
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See: https://polyformproject.org/licenses/noncommercial/1.0.0/

"""Tests for quoted-speech block detection and splitting in ChunkedAudioGenerator."""
import pytest

from audiobook_generator.core.chunked_audio_generator import (
    _is_fully_quoted,
    _find_quoted_span,
    split_sentences_with_voices,
    split_into_sentences,
)
from audiobook_generator.utils.chunk_boundaries import CHUNK_EOF_TAG
from audiobook_generator.utils.existing_chapters_loader import split_text_into_chunks


# ---------------------------------------------------------------------------
# _is_fully_quoted
# ---------------------------------------------------------------------------

class TestIsFullyQuoted:
    def test_russian_guillemets(self):
        assert _is_fully_quoted("«Привет, мир.»") is not None

    def test_typographic_double_quotes(self):
        assert _is_fully_quoted("\u201cHello world.\u201d") is not None

    def test_straight_double_quotes(self):
        assert _is_fully_quoted('"Hello world."') is not None

    def test_with_trailing_punctuation(self):
        # Closing quote followed by period — still counts as fully quoted
        assert _is_fully_quoted('"Hello world".') is not None

    def test_not_quoted_plain_sentence(self):
        assert _is_fully_quoted("Это обычное предложение.") is None

    def test_not_quoted_starts_with_quote_but_no_close(self):
        assert _is_fully_quoted("«Незакрытая кавычка") is None

    def test_not_quoted_mixed_content_after_close(self):
        # Text after closing quote is not punctuation
        assert _is_fully_quoted("«Цитата» и ещё текст после") is None

    def test_empty_string(self):
        assert _is_fully_quoted("") is None


# ---------------------------------------------------------------------------
# split_sentences_with_voices — the Paine quote from the user's example
# ---------------------------------------------------------------------------

PAINE_QUOTE = (
    "\u201cНе прошло и шести часов с тех пор, как я её закончил, "
    "в том виде, в каком она затем появилась, как около трёх часов утра "
    "явилась стража с приказом. "
    "Подписанным двумя комитетами общественного спасения и общей безопасности, "
    "взять меня под арест\u201d."
)

class TestSplitSentencesWithVoices:
    def test_no_voice2_single_chunk(self):
        """Without voice2, the quoted block stays as one chunk (standard sentencex behaviour)."""
        pairs = split_sentences_with_voices(PAINE_QUOTE, "ru", voice2=None)
        # Sentencex may still split on the period inside, but we don't force it
        texts = [t for t, _ in pairs]
        voices = [v for _, v in pairs]
        assert all(v is None for v in voices), "All voices should be None without voice2"

    def test_with_voice2_splits_into_multiple_chunks(self):
        """With voice2 set, the quoted block is split into sub-sentences."""
        pairs = split_sentences_with_voices(PAINE_QUOTE, "ru", voice2="voice_character")
        texts = [t for t, _ in pairs]
        voices = [v for _, v in pairs]
        # Should produce MORE than 1 chunk from a two-sentence block
        assert len(texts) >= 2, (
            f"Expected ≥2 chunks from quoted two-sentence block, got {len(texts)}: {texts}"
        )

    def test_with_voice2_assigns_voice2_to_quoted_chunks(self):
        """All sub-chunks from the quoted block should carry voice2."""
        pairs = split_sentences_with_voices(PAINE_QUOTE, "ru", voice2="voice_character")
        voices = [v for _, v in pairs]
        assert all(v == "voice_character" for v in voices), (
            f"Expected all voice2, got: {voices}"
        )

    def test_mixed_narrator_and_quoted(self):
        """Known limitation: when narrator text and quoted speech are in the same
        paragraph (sentencex sees them as one sentence), the current implementation
        cannot assign voice2 to the quoted part alone.

        This test documents the current behaviour — all voice=None.
        Full inline-quote detection is a future enhancement.
        """
        text = (
            "Том Пейн написал в своих мемуарах: "
            "\u201cЯ был арестован. Меня бросили в тюрьму.\u201d "
            "Так закончилась его свобода."
        )
        pairs = split_sentences_with_voices(text, "ru", voice2="v2")
        voices = [v for _, v in pairs]
        # Current implementation: whole thing comes as one sentence → no voice2 assigned.
        # When/if inline-quote detection is added, this test should be updated.
        assert all(v is None for v in voices), (
            f"Expected no voice2 for mixed narrator+quote sentence (known limitation), got: {voices}"
        )

    def test_trailing_period_outside_closing_quote_preserved(self):
        """Period after the closing quote must be appended to the last inner sub-sentence."""
        # Structure: «Sentence one. Sentence two». — period is OUTSIDE the closing »
        text = "«Первое предложение. Второе предложение»."
        pairs = split_sentences_with_voices(text, "ru", voice2="v2")
        texts = [t for t, _ in pairs]
        assert len(texts) >= 2, f"Expected ≥2 sub-sentences, got: {texts}"
        last_text = texts[-1]
        assert last_text.endswith("."), (
            f"Last sub-sentence should end with '.', got: {last_text!r}"
        )

    def test_no_quote_text_unaffected(self):
        """Plain text without quotes is unaffected by voice2 setting."""
        text = "Первое предложение. Второе предложение. Третье."
        pairs_no_v2 = split_sentences_with_voices(text, "ru", voice2=None)
        pairs_v2 = split_sentences_with_voices(text, "ru", voice2="v2")
        texts_no_v2 = [t for t, _ in pairs_no_v2]
        texts_v2 = [t for t, _ in pairs_v2]
        voices_v2 = [v for _, v in pairs_v2]
        assert texts_no_v2 == texts_v2, "Texts should be identical when no quotes present"
        assert all(v is None for v in voices_v2), "No voice2 should be assigned without quotes"


# ---------------------------------------------------------------------------
# split_text_into_chunks (review UI function) — must mirror synthesis splitting
# ---------------------------------------------------------------------------

class TestSplitTextIntoChunks:
    def test_paine_quote_splits_into_multiple_chunks(self):
        """The Paine quote (two sentences in quotes) must appear as ≥2 chunks in UI."""
        chunks = split_text_into_chunks(PAINE_QUOTE, "ru")
        assert len(chunks) >= 2, (
            f"Review UI shows this as {len(chunks)} chunk(s), expected ≥2:\n{chunks}"
        )

    def test_plain_two_sentences(self):
        chunks = split_text_into_chunks("Первое предложение. Второе предложение.", "ru")
        assert len(chunks) >= 1  # sentencex may or may not split — just no crash

    def test_chunk_eof_splits_without_reaching_display_text(self):
        chunks = split_text_into_chunks(f"Первая часть{CHUNK_EOF_TAG} Вторая часть.", "ru")
        assert chunks == ["Первая часть", "Вторая часть."]


class TestSplitIntoSentencesChunkEof:
    def test_chunk_eof_is_boundary_and_removed(self):
        sentences = split_into_sentences(f"Первая часть{CHUNK_EOF_TAG} Вторая часть.", "ru")
        assert sentences == ["Первая часть", "Вторая часть."]


# ---------------------------------------------------------------------------
# _find_quoted_span — cross-boundary quoted block detection
# ---------------------------------------------------------------------------

class TestFindQuotedSpan:
    def test_span_found_across_two_sentences(self):
        """Opening quote in s[0], closing quote in s[1] — span is [0, 2)."""
        sentences = [
            "\u201cНе прошло и шести часов, появилась,",
            "как около трёх часов утра явилась стража\u201d.",
        ]
        assert _find_quoted_span(sentences, 0) == 2

    def test_no_span_when_already_fully_quoted(self):
        """Sentence already contains close quote → _is_fully_quoted handles it, not this."""
        sentences = ["\u201cПолностью закрытая цитата\u201d.", "Следующее."]
        # close char IS present in s[0][1:] → returns None
        assert _find_quoted_span(sentences, 0) is None

    def test_no_span_when_no_close_quote_anywhere(self):
        """No closing quote found → returns None."""
        sentences = ["\u201cОткрытая цитата без закрытия", "Следующее предложение."]
        assert _find_quoted_span(sentences, 0) is None

    def test_no_span_for_plain_sentence(self):
        """Sentence without opening quote → returns None."""
        sentences = ["Обычное предложение.", "Другое предложение."]
        assert _find_quoted_span(sentences, 0) is None

    def test_span_of_three_sentences(self):
        """Opening in s[0], closing in s[2] — span is [0, 3)."""
        sentences = [
            "\u201cПервая часть,",
            "вторая часть,",
            "третья часть\u201d.",
        ]
        assert _find_quoted_span(sentences, 0) == 3

    def test_no_span_when_close_followed_by_text(self):
        """Close char followed by non-punctuation text — not a clean close → None."""
        sentences = [
            "\u201cНачало цитаты,",
            "\u201d и продолжение нарратора.",
        ]
        assert _find_quoted_span(sentences, 0) is None


# ---------------------------------------------------------------------------
# Regression: [chunk_eof] inside a quoted block — voice2 must still be assigned
# ---------------------------------------------------------------------------

# Exact text from the user bug report (typographic " " quotes, chunk_eof inside)
PAINE_QUOTE_CHUNK_EOF = (
    "\u201c"
    "Не прошло и шести часов с тех пор, как я её закончил, "
    "в том виде, в каком она затем появилась,"
    + CHUNK_EOF_TAG
    + " как около трёх часов утра явилась стража с приказом взять меня под арест, "
    "подписанным двумя комитетами общественного спасения и общей безопасности"
    "\u201d."
)

# Full paragraph context — mirrors the real-world usage
PAINE_PARAGRAPH_CHUNK_EOF = (
    "В предисловии Пэйна ко второй части «Века разума» он пишет о себе, "
    "что заканчивал первую часть ближе к концу тысяча семьсот девяносто третьего года. "
    + PAINE_QUOTE_CHUNK_EOF
    + " Это произошло утром двадцать восьмого декабря."
)


class TestChunkEofInsideQuotedBlock:
    def test_chunk_eof_quoted_block_produces_chunks(self):
        """[chunk_eof] inside a quoted block: split_sentences_with_voices must return ≥2 pairs."""
        pairs = split_sentences_with_voices(PAINE_QUOTE_CHUNK_EOF, "ru", voice2="v2")
        assert len(pairs) >= 2, (
            f"Expected ≥2 pairs from quoted block split by [chunk_eof], got {len(pairs)}: {pairs}"
        )

    def test_chunk_eof_quoted_block_all_voice2(self):
        """All sentence pairs produced from a chunk_eof-split quoted block must carry voice2."""
        pairs = split_sentences_with_voices(PAINE_QUOTE_CHUNK_EOF, "ru", voice2="v2")
        voices = [v for _, v in pairs]
        assert all(v == "v2" for v in voices), (
            f"Expected all voice2='v2', got voices={voices}"
        )

    def test_paragraph_context_quoted_parts_are_voice2(self):
        """In full paragraph context the quoted parts get voice2, narrator parts do not."""
        pairs = split_sentences_with_voices(PAINE_PARAGRAPH_CHUNK_EOF, "ru", voice2="v2")
        # At least two pairs must have voice2 (the two halves of the split quote)
        voice2_pairs = [(t, v) for t, v in pairs if v == "v2"]
        assert len(voice2_pairs) >= 2, (
            f"Expected ≥2 voice2 pairs in paragraph, got {voice2_pairs}"
        )
        # Narrator sentences (before/after the quote) must have no voice override
        none_pairs = [(t, v) for t, v in pairs if v is None]
        assert len(none_pairs) >= 1, (
            f"Expected ≥1 narrator (voice=None) pair in paragraph, got none.\nAll pairs: {pairs}"
        )

    def test_no_voice2_no_change(self):
        """Without voice2, the [chunk_eof] inside quote just produces plain splits, no voice."""
        pairs = split_sentences_with_voices(PAINE_QUOTE_CHUNK_EOF, "ru", voice2=None)
        voices = [v for _, v in pairs]
        assert all(v is None for v in voices), (
            f"Expected all None voices without voice2, got: {voices}"
        )


if __name__ == "__main__":
    # Quick smoke-run
    chunks = split_text_into_chunks(PAINE_QUOTE, "ru")
    print(f"Chunks ({len(chunks)}):")
    for i, c in enumerate(chunks, 1):
        print(f"  {i}. {c}")

