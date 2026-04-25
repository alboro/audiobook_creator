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


def apply_review_edit(full_text: str, old_text: str, new_text: str) -> str:
    """Apply one review edit safely to a chapter text.

    Replaces the first occurrence of ``old_text`` with ``new_text`` and then
    removes accidental adjacent duplicate paragraphs.
    """
    if old_text not in full_text:
        raise ValueError("Original text not found in file (may have been modified)")

    updated = full_text.replace(old_text, new_text, 1)
    return collapse_adjacent_duplicate_paragraphs(updated)

