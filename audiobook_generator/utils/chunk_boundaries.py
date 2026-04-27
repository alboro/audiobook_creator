# Copyright (c) 2026 alboro <alboro@users.noreply.github.com>
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See: https://polyformproject.org/licenses/noncommercial/1.0.0/

from __future__ import annotations

import re
from typing import List

CHUNK_EOF_TAG = "[chunk_eof]"
SENTENCE_END_CHARS = ".!?"
TRAILING_CLOSERS = "\"'»”’`)]}"

CHUNK_EOF_RE = re.compile(r"\s*\[chunk_eof\]\s*", re.IGNORECASE)
CHUNK_EOF_AT_END_RE = re.compile(r"\s*\[chunk_eof\]\s*$", re.IGNORECASE)
CHUNK_SEPARATOR_RE = re.compile(
    r"(\s*\[chunk_eof\]\s*|(?<=[.!?]) +)",
    re.IGNORECASE,
)


def strip_chunk_boundary_tags(text: str) -> str:
    """Remove non-spoken chunk boundary tags from text before TTS calls."""
    return CHUNK_EOF_RE.sub(" ", text).strip()


def split_text_on_explicit_chunk_boundaries(text: str, min_chars: int = 0) -> List[str]:
    """Split only on explicit chunk boundary tags, removing the tags."""
    chunks = [
        strip_chunk_boundary_tags(part)
        for part in CHUNK_EOF_RE.split(text)
        if len(strip_chunk_boundary_tags(part)) >= min_chars
    ]
    return chunks


def ends_with_chunk_boundary(text: str) -> bool:
    """Return True when text ends with sentence punctuation or ``[chunk_eof]``."""
    stripped = text.strip()
    if not stripped:
        return False
    if CHUNK_EOF_AT_END_RE.search(stripped):
        return True
    bare = stripped.rstrip(TRAILING_CLOSERS).rstrip()
    return bool(bare) and bare[-1] in SENTENCE_END_CHARS


def ensure_chunk_eof_boundary(text: str) -> str:
    """End a forced chunk with ``[chunk_eof]`` instead of an artificial period."""
    stripped = CHUNK_EOF_AT_END_RE.sub("", text.strip()).rstrip()
    if not stripped:
        return ""
    stripped = _strip_terminal_period(stripped)
    return f"{stripped}{CHUNK_EOF_TAG}"


def split_text_by_chunk_boundaries(
    text: str,
    language: str = "ru",
    *,
    min_chars: int = 0,
) -> List[str]:
    """Split text on regular sentence boundaries and explicit ``[chunk_eof]`` tags.

    The returned chunks never include ``[chunk_eof]`` because it is a local
    processing marker, not text meant for the TTS engine.
    """
    if not text:
        return []

    result: List[str] = []
    for explicit_part in CHUNK_EOF_RE.split(text):
        explicit_part = explicit_part.strip()
        if not explicit_part:
            continue
        result.extend(_segment_sentences(explicit_part, language))

    return [
        clean
        for item in result
        if (clean := strip_chunk_boundary_tags(item)) and len(clean) >= min_chars
    ]


def split_text_preserve_chunk_separators(text: str) -> tuple[list[str], list[str]]:
    """Split text and preserve separators, including ``[chunk_eof]`` markers."""
    tokens = CHUNK_SEPARATOR_RE.split(text)
    sentences: list[str] = []
    separators: list[str] = []

    for tok in tokens:
        if not tok:
            continue
        if CHUNK_SEPARATOR_RE.fullmatch(tok):
            if separators:
                separators[-1] = _normalize_separator(tok)
            continue
        if tok.strip():
            sentences.append(tok)
            separators.append("")
        elif separators:
            separators[-1] = separators[-1] + tok

    return sentences, separators


def merge_broken_backtick_sentences(sentences: list[str]) -> list[str]:
    """Re-attach a stray closing backtick that sentencex may put in the next chunk."""
    if len(sentences) < 2:
        return sentences
    result: list[str] = []
    i = 0
    while i < len(sentences):
        sent = sentences[i]
        next_sent = sentences[i + 1] if i + 1 < len(sentences) else None
        if (
            next_sent is not None
            and sent.count("`") % 2 == 1
            and next_sent.startswith("`")
            and (len(next_sent) == 1 or next_sent[1] in (" ", "\t", "\n"))
        ):
            result.append(sent + "`")
            rest = next_sent[1:].lstrip()
            if rest:
                result.append(rest)
            i += 2
        else:
            result.append(sent)
            i += 1
    return result


def _segment_sentences(text: str, language: str) -> list[str]:
    try:
        from sentencex import segment  # type: ignore

        lang = (language or "ru").split("-")[0]
        return merge_broken_backtick_sentences(
            [item.strip() for item in segment(lang, text) if item and item.strip()]
        )
    except Exception:
        sentences, _separators = split_text_preserve_chunk_separators(text)
        return [item.strip() for item in sentences if item and item.strip()]


def _normalize_separator(separator: str) -> str:
    if CHUNK_EOF_RE.search(separator):
        return f" {CHUNK_EOF_TAG} "
    return " "


def _strip_terminal_period(text: str) -> str:
    suffix = ""
    base = text.rstrip()
    while base and base[-1] in TRAILING_CLOSERS:
        suffix = base[-1] + suffix
        base = base[:-1].rstrip()
    if base.endswith("."):
        base = base[:-1].rstrip()
    return f"{base}{suffix}".rstrip()
