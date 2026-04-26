# Copyright (c) 2026 alboro <alboro@users.noreply.github.com>
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See: https://polyformproject.org/licenses/noncommercial/1.0.0/

from __future__ import annotations

import re


def collapse_adjacent_duplicate_paragraphs(text: str) -> str:
    """Collapse exact adjacent duplicate paragraphs.

    This is a safety net for review/edit flows: if the same paragraph ends up
    duplicated right next to itself (typically due to a bad client-side edit or
    stale state), keep only one copy.

    Paragraph boundaries are blank-line based. Internal paragraph formatting is
    preserved; only adjacent exact duplicates are removed.
    """
    if not text:
        return text

    had_trailing_newline = text.endswith("\n")
    blocks = re.split(r"\n\s*\n+", text.strip())
    if not blocks:
        return text

    deduped: list[str] = []
    prev_norm: str | None = None
    for block in blocks:
        norm = block.strip()
        if not norm:
            continue
        if norm == prev_norm:
            continue
        deduped.append(block)
        prev_norm = norm

    result = "\n\n".join(deduped)
    if had_trailing_newline:
        result += "\n"
    return result


def _dedupe_horizontal_whitespace_at_edit_boundary(
    before: str,
    replacement: str,
    after: str,
) -> str:
    """Avoid accidental double spaces introduced exactly at edit boundaries.

    Review chunks are sentence-based, and the sentence splitter may keep the
    leading space of the following sentence in ``after``. If the edited text
    also ends with a space, a simple ``replace`` produces ``".  Next"``.

    We only normalize horizontal whitespace at the two join points:
    ``before|replacement`` and ``replacement|after``. Internal spacing inside
    the replacement text is left untouched.
    """
    if before and replacement and before[-1] in " \t" and replacement[:1] in {" ", "\t"}:
        replacement = replacement.lstrip(" \t")
    if replacement and after and replacement[-1] in " \t" and after[:1] in {" ", "\t"}:
        replacement = replacement.rstrip(" \t")
    return before + replacement + after


def apply_review_edit(full_text: str, old_text: str, new_text: str) -> str:
    """Apply one review edit safely to a chapter text.

    Replaces the first occurrence of ``old_text`` with ``new_text`` and then
    removes accidental adjacent duplicate paragraphs.
    """
    start = full_text.find(old_text)
    if start < 0:
        raise ValueError("Original text not found in file (may have been modified)")

    end = start + len(old_text)
    before = full_text[:start]
    after = full_text[end:]
    updated = _dedupe_horizontal_whitespace_at_edit_boundary(before, new_text, after)
    return collapse_adjacent_duplicate_paragraphs(updated)

