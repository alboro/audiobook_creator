# Copyright (c) 2026 alboro <alboro@users.noreply.github.com>
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See: https://polyformproject.org/licenses/noncommercial/1.0.0/

from __future__ import annotations

from audiobook_generator.ui.review_text_ops import (
    apply_review_edit,
    collapse_adjacent_duplicate_paragraphs,
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
