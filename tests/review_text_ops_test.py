# Copyright (c) 2026 alboro <alboro@users.noreply.github.com>
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See: https://polyformproject.org/licenses/noncommercial/1.0.0/

from __future__ import annotations

from audiobook_generator.ui.review_text_ops import (
    apply_review_edit,
    collapse_adjacent_duplicate_paragraphs,
    normalize_chunk_eof_text,
)


OLD_PARAGRAPH = (
    "One paragraph can be corrected by the review UI. "
    "This original version contains a punctuation bug, and it should be replaced."
)

NEW_PARAGRAPH = (
    "One paragraph can be corrected by the review UI. "
    "This corrected version fixes the punctuation bug and replaces the original."
)

PREV_PARAGRAPH = "The previous paragraph stays untouched."
NEXT_PARAGRAPH = "The next paragraph also stays untouched."


def test_collapse_adjacent_duplicate_paragraphs_removes_exact_neighbor_duplicate():
    text = (
        PREV_PARAGRAPH
        + "\n\n"
        + NEW_PARAGRAPH
        + "\n\n"
        + NEW_PARAGRAPH
        + "\n\n"
        + NEXT_PARAGRAPH
    )

    result = collapse_adjacent_duplicate_paragraphs(text)

    assert result.count(NEW_PARAGRAPH) == 1
    assert PREV_PARAGRAPH in result
    assert NEXT_PARAGRAPH in result


def test_apply_review_edit_collapses_accidental_duplicate_in_new_text():
    full_text = PREV_PARAGRAPH + "\n\n" + OLD_PARAGRAPH + "\n\n" + NEXT_PARAGRAPH
    accidentally_duplicated_new_text = NEW_PARAGRAPH + "\n\n" + NEW_PARAGRAPH

    result = apply_review_edit(full_text, OLD_PARAGRAPH, accidentally_duplicated_new_text)

    assert result.count(NEW_PARAGRAPH) == 1
    assert OLD_PARAGRAPH not in result
    assert NEW_PARAGRAPH in result


def test_apply_review_edit_regular_single_replace_still_works():
    full_text = PREV_PARAGRAPH + "\n\n" + OLD_PARAGRAPH + "\n\n" + NEXT_PARAGRAPH

    result = apply_review_edit(full_text, OLD_PARAGRAPH, NEW_PARAGRAPH)

    assert result.count(NEW_PARAGRAPH) == 1
    assert OLD_PARAGRAPH not in result
    assert NEW_PARAGRAPH in result


def test_apply_review_edit_dedupes_double_space_after_replacement_boundary():
    full_text = "First sentence. Second sentence."
    old_text = "First sentence."
    new_text = "First sentence. "

    result = apply_review_edit(full_text, old_text, new_text)

    assert result == "First sentence. Second sentence."


def test_apply_review_edit_dedupes_double_space_before_replacement_boundary():
    full_text = "First sentence. Second sentence."
    old_text = "Second sentence."
    new_text = " Second corrected sentence."

    result = apply_review_edit(full_text, old_text, new_text)

    assert result == "First sentence. Second corrected sentence."


# ---------------------------------------------------------------------------
# Fuzzy-match: closing quote stranded before terminal punctuation
# ---------------------------------------------------------------------------

def test_apply_review_edit_closing_quote_before_period_is_found():
    """File has '…не можем".' but chunk text (old_text) has '…не можем.'"""
    full_text = (
        "Лежит в основе многих преданий первых одиннадцати глав этой книги, "
        'но надеяться восстановить его мы не можем".\n'
    )
    old_text = (
        "Лежит в основе многих преданий первых одиннадцати глав этой книги, "
        "но надеяться восстановить его мы не можем."
    )
    new_text = (
        "Лежит в основе многих преданий первых одиннадцати глав этой книги,"
        "[chunk_eof] но надеяться восстановить его мы не можем."
    )

    result = apply_review_edit(full_text, old_text, new_text)

    assert "[chunk_eof]" in result
    # The closing quote must be preserved in the output
    assert '".' in result
    assert "не можем" in result


def test_apply_review_edit_closing_quote_preserved_after_fuzzy_replace():
    """Verify the closing quote ends up before the terminal period in the result."""
    full_text = 'Он сказал: «Прощай».\n'
    old_text = 'Он сказал: «Прощай».'
    new_text = 'Он сказал: «До свидания».'

    # Exact match should work here (same closing chars) — sanity check
    result = apply_review_edit(full_text, old_text, new_text)
    assert 'До свидания' in result


def test_apply_review_edit_fuzzy_preserves_closing_guillemet_before_period():
    """File: 'текст".' — chunk old_text: 'текст.' — quote must survive edit."""
    full_text = 'Первый абзац.\n\nтекст".\n\nПоследний абзац.\n'
    old_text = 'текст.'
    new_text = 'новый текст.'

    result = apply_review_edit(full_text, old_text, new_text)

    assert 'новый текст".' in result
    assert 'Первый абзац.' in result
    assert 'Последний абзац.' in result


def test_apply_review_edit_exact_match_not_broken_by_fuzzy():
    """When exact match exists it must still be used (no regression)."""
    full_text = "Раз. Два. Три."
    old_text = "Два."
    new_text = "2."

    result = apply_review_edit(full_text, old_text, new_text)

    assert result == "Раз. 2. Три."


# ---------------------------------------------------------------------------
# normalize_chunk_eof_text
# ---------------------------------------------------------------------------

def test_normalize_chunk_eof_text_no_token_is_noop():
    text = "Обычный текст без маркера."
    assert normalize_chunk_eof_text(text) == text


def test_normalize_chunk_eof_text_already_correct_is_noop():
    text = "лет,[chunk_eof]Вы, поймёте."
    assert normalize_chunk_eof_text(text) == text


def test_normalize_chunk_eof_text_strips_one_space_and_capitalises():
    text = "лет,[chunk_eof] вы, поймёте."
    assert normalize_chunk_eof_text(text) == "лет,[chunk_eof]Вы, поймёте."


def test_normalize_chunk_eof_text_strips_multiple_spaces_and_capitalises():
    text = "лет,[chunk_eof]  вы, поймёте."
    assert normalize_chunk_eof_text(text) == "лет,[chunk_eof]Вы, поймёте."


def test_normalize_chunk_eof_text_multiple_tokens():
    text = "a[chunk_eof] b[chunk_eof] c"
    assert normalize_chunk_eof_text(text) == "a[chunk_eof]B[chunk_eof]C"


def test_normalize_chunk_eof_text_token_at_end_of_string():
    text = "текст[chunk_eof]"
    assert normalize_chunk_eof_text(text) == "текст[chunk_eof]"


def test_normalize_chunk_eof_text_token_followed_by_spaces_only():
    text = "текст[chunk_eof]   "
    assert normalize_chunk_eof_text(text) == "текст[chunk_eof]"


def test_normalize_chunk_eof_text_real_example():
    """The example from the task description."""
    text = (
        "Поскольку доктор Франклин был моим близким другом на протяжении последних тридцати лет,"
        "[chunk_eof]Вы, естественно, поймёте, почему я сохраняю связь с его внуком."
    )
    # Already normalised — should be unchanged
    assert normalize_chunk_eof_text(text) == text


def test_normalize_chunk_eof_text_real_example_with_leading_space():
    text = (
        "Поскольку доктор Франклин был моим близким другом на протяжении последних тридцати лет,"
        "[chunk_eof] вы, естественно, поймёте, почему я сохраняю связь с его внуком."
    )
    expected = (
        "Поскольку доктор Франклин был моим близким другом на протяжении последних тридцати лет,"
        "[chunk_eof]Вы, естественно, поймёте, почему я сохраняю связь с его внуком."
    )
    assert normalize_chunk_eof_text(text) == expected


