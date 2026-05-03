# Copyright (c) 2026 alboro <alboro@users.noreply.github.com>
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See: https://polyformproject.org/licenses/noncommercial/1.0.0/

from __future__ import annotations

import re

# Characters that a sentence splitter may strand between the last word and
# the terminal punctuation mark (e.g. closing guillemet, curly/straight quote).
_CLOSING_CHARS = r'["\'»\u201c\u201d\u2018\u2019\)\]\}]'
_TERMINAL_PUNCT = r'[.!?]'
_TERMINAL_RE = re.compile(r'(' + _CLOSING_CHARS + r'*)(' + _TERMINAL_PUNCT + r')$')


def _split_terminal(text: str):
    """Return ``(body, closing_chars, terminal_punct)`` by stripping the final
    sentence-ending punctuation (and any closing chars that precede it in the
    *source* text).  If there is no terminal punctuation returns
    ``(text, '', '')``.
    """
    m = _TERMINAL_RE.search(text)
    if not m:
        return text, '', ''
    body = text[:m.start()]
    return body, m.group(1), m.group(2)


def _fuzzy_find_with_trailing_closers(full_text: str, old_text: str):
    """Find *old_text* tolerating extra closing chars before the terminal punct.

    E.g. ``old_text`` = ``"…не можем."``  will match
    ``"…не можем"."`` inside *full_text*.

    Returns ``(start, matched_span, extra_closing_chars)`` or ``(-1, '', '')``.
    """
    body, _old_closers, punct = _split_terminal(old_text)
    if not punct:
        return -1, '', ''
    closer_pat = '[' + re.escape('"\'»\u201c\u201d\u2018\u2019)\\]{}') + ']*'
    pattern = re.escape(body) + closer_pat + re.escape(punct)
    fm = re.search(pattern, full_text)
    if not fm:
        return -1, '', ''
    matched = fm.group(0)
    # Closing chars that are present in the file but absent in old_text
    extra = matched[len(body):-len(punct)]
    return fm.start(), matched, extra


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

    Fuzzy fallback: when the source file contains closing quote/bracket chars
    *before* the terminal punctuation that the sentence-splitter stripped from
    the chunk (e.g. ``не можем".`` vs chunk ``не можем.``), the match still
    succeeds and those extra characters are preserved in the output.
    """
    start = full_text.find(old_text)
    if start >= 0:
        end = start + len(old_text)
        actual_replacement = new_text
    else:
        fuzzy_start, matched_old, extra_closers = _fuzzy_find_with_trailing_closers(
            full_text, old_text
        )
        if fuzzy_start < 0:
            raise ValueError("Original text not found in file (may have been modified)")
        start = fuzzy_start
        end = start + len(matched_old)
        # Re-insert the extra closing chars before the terminal punct of new_text
        if extra_closers:
            new_body, _nc, new_punct = _split_terminal(new_text)
            actual_replacement = new_body + extra_closers + (new_punct or "")
        else:
            actual_replacement = new_text

    before = full_text[:start]
    after = full_text[end:]
    updated = _dedupe_horizontal_whitespace_at_edit_boundary(before, actual_replacement, after)
    return collapse_adjacent_duplicate_paragraphs(updated)


# ---------------------------------------------------------------------------
# [chunk_eof] normalisation
# ---------------------------------------------------------------------------

_CHUNK_EOF_TOKEN = "[chunk_eof]"


def normalize_chunk_eof_text(text: str) -> str:
    """Normalise a sentence text that may contain ``[chunk_eof]`` markers.

    For each ``[chunk_eof]`` occurrence:

    * strip any immediately following space characters (`` ``),
    * capitalise the very first printable character that follows.

    Only plain space characters (U+0020) are stripped — intentional newlines
    or other whitespace inside a sentence are left intact.

    Example::

        "лет,[chunk_eof]Вы"          → unchanged (already correct)
        "лет,[chunk_eof] вы"         → "лет,[chunk_eof]Вы"
        "лет,[chunk_eof]  вы"        → "лет,[chunk_eof]Вы"
        "a[chunk_eof] b[chunk_eof] c" → "a[chunk_eof]B[chunk_eof]C"
    """
    if _CHUNK_EOF_TOKEN not in text:
        return text
    parts = text.split(_CHUNK_EOF_TOKEN)
    normalised = [parts[0]]
    for part in parts[1:]:
        stripped = part.lstrip(" ")
        if stripped:
            stripped = stripped[0].upper() + stripped[1:]
        normalised.append(stripped)
    return _CHUNK_EOF_TOKEN.join(normalised)

