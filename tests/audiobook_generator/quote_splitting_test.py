# Copyright (c) 2026 alboro <alboro@users.noreply.github.com>
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See: https://polyformproject.org/licenses/noncommercial/1.0.0/

"""Tests for quoted-speech block detection and splitting in ChunkedAudioGenerator."""
import pytest

from audiobook_generator.core.chunked_audio_generator import (
    _is_fully_quoted,
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


if __name__ == "__main__":
    # Quick smoke-run
    chunks = split_text_into_chunks(PAINE_QUOTE, "ru")
    print(f"Chunks ({len(chunks)}):")
    for i, c in enumerate(chunks, 1):
        print(f"  {i}. {c}")

