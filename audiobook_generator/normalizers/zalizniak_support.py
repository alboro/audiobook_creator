# Copyright (c) 2026 alboro <alboro@users.noreply.github.com>
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See: https://polyformproject.org/licenses/noncommercial/1.0.0/

"""
Support module for integrating gramdict/zalizniak-2010 data into the
pronunciation lexicon DB.

The Zaliznyak Grammatical Dictionary (6th ed., 2010) covers ~110k Russian
words with authoritative stress marks directly embedded in the lemma as
U+0301 COMBINING ACUTE ACCENT.

Source: https://github.com/gramdict/zalizniak-2010
License: CC BY-NC (non-commercial use allowed)

Usage:
    from audiobook_generator.normalizers.zalizniak_support import (
        iter_zalizniak_entries,
        ensure_zalizniak_data_cached,
    )
"""
from __future__ import annotations

import logging
import re
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Iterator

from audiobook_generator.normalizers.pronunciation_lexicon_db import (
    PronunciationLexiconEntry,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ZALIZNIAK_SOURCE = "zalizniak"
COMBINING_ACUTE = "\u0301"
COMBINING_GRAVE = "\u0300"

_GITHUB_BASE = (
    "https://raw.githubusercontent.com/gramdict/zalizniak-2010/master/dictionary/"
)

# Text files to download: (url_suffix, is_proper_name)
_ZALIZNIAK_FILES: list[tuple[str, bool]] = [
    # Нарицательные (common nouns, adjectives, verbs, etc.)
    ("\u041d\u0430\u0440\u0438\u0446\u0430\u0442\u0435\u043b\u044c\u043d\u044b\u0435/\u0410.txt", False),
    ("\u041d\u0430\u0440\u0438\u0446\u0430\u0442\u0435\u043b\u044c\u043d\u044b\u0435/\u0411.txt", False),
    ("\u041d\u0430\u0440\u0438\u0446\u0430\u0442\u0435\u043b\u044c\u043d\u044b\u0435/\u0412.txt", False),
    ("\u041d\u0430\u0440\u0438\u0446\u0430\u0442\u0435\u043b\u044c\u043d\u044b\u0435/\u0413.txt", False),
    ("\u041d\u0430\u0440\u0438\u0446\u0430\u0442\u0435\u043b\u044c\u043d\u044b\u0435/\u0414.txt", False),
    ("\u041d\u0430\u0440\u0438\u0446\u0430\u0442\u0435\u043b\u044c\u043d\u044b\u0435/\u0415.txt", False),
    ("\u041d\u0430\u0440\u0438\u0446\u0430\u0442\u0435\u043b\u044c\u043d\u044b\u0435/\u0416.txt", False),
    ("\u041d\u0430\u0440\u0438\u0446\u0430\u0442\u0435\u043b\u044c\u043d\u044b\u0435/\u0417.txt", False),
    ("\u041d\u0430\u0440\u0438\u0446\u0430\u0442\u0435\u043b\u044c\u043d\u044b\u0435/\u0418.txt", False),
    ("\u041d\u0430\u0440\u0438\u0446\u0430\u0442\u0435\u043b\u044c\u043d\u044b\u0435/\u0419.txt", False),
    ("\u041d\u0430\u0440\u0438\u0446\u0430\u0442\u0435\u043b\u044c\u043d\u044b\u0435/\u041a.txt", False),
    ("\u041d\u0430\u0440\u0438\u0446\u0430\u0442\u0435\u043b\u044c\u043d\u044b\u0435/\u041b.txt", False),
    ("\u041d\u0430\u0440\u0438\u0446\u0430\u0442\u0435\u043b\u044c\u043d\u044b\u0435/\u041c.txt", False),
    ("\u041d\u0430\u0440\u0438\u0446\u0430\u0442\u0435\u043b\u044c\u043d\u044b\u0435/\u041d.txt", False),
    ("\u041d\u0430\u0440\u0438\u0446\u0430\u0442\u0435\u043b\u044c\u043d\u044b\u0435/\u041e.txt", False),
    ("\u041d\u0430\u0440\u0438\u0446\u0430\u0442\u0435\u043b\u044c\u043d\u044b\u0435/\u041f.txt", False),
    ("\u041d\u0430\u0440\u0438\u0446\u0430\u0442\u0435\u043b\u044c\u043d\u044b\u0435/\u0420.txt", False),
    ("\u041d\u0430\u0440\u0438\u0446\u0430\u0442\u0435\u043b\u044c\u043d\u044b\u0435/\u0421.txt", False),
    ("\u041d\u0430\u0440\u0438\u0446\u0430\u0442\u0435\u043b\u044c\u043d\u044b\u0435/\u0422.txt", False),
    ("\u041d\u0430\u0440\u0438\u0446\u0430\u0442\u0435\u043b\u044c\u043d\u044b\u0435/\u0423.txt", False),
    ("\u041d\u0430\u0440\u0438\u0446\u0430\u0442\u0435\u043b\u044c\u043d\u044b\u0435/\u0424.txt", False),
    ("\u041d\u0430\u0440\u0438\u0446\u0430\u0442\u0435\u043b\u044c\u043d\u044b\u0435/\u0425.txt", False),
    ("\u041d\u0430\u0440\u0438\u0446\u0430\u0442\u0435\u043b\u044c\u043d\u044b\u0435/\u0426.txt", False),
    ("\u041d\u0430\u0440\u0438\u0446\u0430\u0442\u0435\u043b\u044c\u043d\u044b\u0435/\u0427.txt", False),
    ("\u041d\u0430\u0440\u0438\u0446\u0430\u0442\u0435\u043b\u044c\u043d\u044b\u0435/\u0428.txt", False),
    ("\u041d\u0430\u0440\u0438\u0446\u0430\u0442\u0435\u043b\u044c\u043d\u044b\u0435/\u0429.txt", False),
    ("\u041d\u0430\u0440\u0438\u0446\u0430\u0442\u0435\u043b\u044c\u043d\u044b\u0435/\u042b.txt", False),
    ("\u041d\u0430\u0440\u0438\u0446\u0430\u0442\u0435\u043b\u044c\u043d\u044b\u0435/\u042c.txt", False),
    ("\u041d\u0430\u0440\u0438\u0446\u0430\u0442\u0435\u043b\u044c\u043d\u044b\u0435/\u042d.txt", False),
    ("\u041d\u0430\u0440\u0438\u0446\u0430\u0442\u0435\u043b\u044c\u043d\u044b\u0435/\u042e.txt", False),
    ("\u041d\u0430\u0440\u0438\u0446\u0430\u0442\u0435\u043b\u044c\u043d\u044b\u0435/\u042f.txt", False),
    # Глаголы (verbs)
    ("\u0413\u043b\u0430\u0433\u043e\u043b\u044b/\u0418.txt", False),
    ("\u0413\u043b\u0430\u0433\u043e\u043b\u044b/\u0419.txt", False),
    ("\u0413\u043b\u0430\u0433\u043e\u043b\u044b/\u0422.txt", False),
    ("\u0413\u043b\u0430\u0433\u043e\u043b\u044b/\u042c.txt", False),
    ("\u0413\u043b\u0430\u0433\u043e\u043b\u044b/\u042f.txt", False),
    # Собственные (proper names)
    ("\u0421\u043e\u0431\u0441\u0442\u0432\u0435\u043d\u043d\u044b\u0435.txt", True),
]

# Zaliznyak POS tag → canonical POS (order matters: most specific first)
_POS_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bмо-жо\b"), "NOUN"),
    (re.compile(r"\bжо\b"), "NOUN"),
    (re.compile(r"\bмо\b"), "NOUN"),
    (re.compile(r"\bсо\b"), "NOUN"),
    (re.compile(r"\bж\b"), "NOUN"),
    (re.compile(r"\bм\b"), "NOUN"),
    (re.compile(r"\bс\b"), "NOUN"),
    (re.compile(r"\bмс-п\b"), "PRON"),
    (re.compile(r"\bмс\b"), "PRON"),
    (re.compile(r"\bп\b"), "ADJ"),
    (re.compile(r"\bнар\b"), "ADV"),
    (re.compile(r"\bсв-нсв\b"), "VERB"),
    (re.compile(r"\bсв\b"), "VERB"),
    (re.compile(r"\bнсв\b"), "VERB"),
    (re.compile(r"\bмежд\b"), "INTJ"),
    (re.compile(r"\bсоюз\b"), "CONJ"),
    (re.compile(r"\bпредл\b"), "PREP"),
    (re.compile(r"\bчастица\b"), "PART"),
    (re.compile(r"\bчаст\b"), "PART"),
    (re.compile(r"\bчисл\b"), "NUM"),
    (re.compile(r"\bвводн\b"), "INTRO"),
    (re.compile(r"\bпредик\b"), "PRED"),
]

# Stress class code regex: optional leading digit, optional *, letter a-f (any case),
# optional prime(s), optional variant after //, optional em-dash
_CLASS_RE = re.compile(
    r"\b(\d+[*]?[a-fA-F](?:'|'')?(?://\d+[*]?[a-fA-F](?:'|'')?)?[—]?)\b"
)

# Homonym number prefix: "1/" or "2-3/"
_NUM_PREFIX_RE = re.compile(r"^\d+(?:-\d+)?/")

# Chars that are valid in Russian words (Cyrillic letters + hyphen)
_CYRILLIC_RE = re.compile(r"[\u0400-\u04FF]")


# ---------------------------------------------------------------------------
# Line parser
# ---------------------------------------------------------------------------

def parse_zalizniak_line(
    line: str,
    *,
    is_proper_name: bool = False,
) -> PronunciationLexiconEntry | None:
    """
    Parse one line from a Zaliznyak text file and return a
    PronunciationLexiconEntry (or None if the line should be skipped).

    Zaliznyak file format (simplified):
        [N/|N-M/]LEMMA POS STRESS_CLASS [notes]

    The lemma already carries primary stress as U+0301 (COMBINING ACUTE ACCENT).
    Secondary stress (U+0300 COMBINING GRAVE) is discarded.
    """
    line = line.strip()
    if not line:
        return None

    # Remove homonym prefix ("1/", "2-3/", …)
    line = _NUM_PREFIX_RE.sub("", line)

    # Split into lemma and the rest
    parts = line.split(None, 1)
    lemma_raw = parts[0]  # lemma with embedded stress marks
    rest = parts[1] if len(parts) > 1 else ""

    # Clean up the rest: remove [brackets] and (parentheses)
    rest_clean = re.sub(r"\[.*?\]", " ", rest)
    rest_clean = re.sub(r"\(.*?\)", " ", rest_clean)

    # --- spoken form (primary stress only) ---
    spoken_form = lemma_raw.replace(COMBINING_GRAVE, "")

    # --- surface form (no stress marks) ---
    surface_form = spoken_form.replace(COMBINING_ACUTE, "").lower()

    # Skip entries without any Cyrillic characters (punctuation-only, etc.)
    if not _CYRILLIC_RE.search(surface_form):
        return None

    # Skip entries with no stress mark at all (function words, etc.)
    # We still include unstressed monosyllabic words marked explicitly.
    # But genuinely unstressed entries (e.g., "_без удар._" in notes) are still
    # stored — they simply have spoken_form == surface_form.

    # --- POS ---
    pos: str | None = None
    for pattern, pos_tag in _POS_PATTERNS:
        if pattern.search(rest_clean):
            pos = pos_tag
            break

    # --- stress class (grammemes) ---
    grammemes: str | None = None
    m = _CLASS_RE.search(rest_clean)
    if m:
        grammemes = m.group(1)

    return PronunciationLexiconEntry(
        surface_form=surface_form,
        spoken_form=spoken_form,
        lemma=surface_form,       # lemma == surface form (Zaliznyak gives lemmas)
        pos=pos,
        grammemes=grammemes,
        is_proper_name=is_proper_name,
        source=ZALIZNIAK_SOURCE,
        confidence=0.95,
    )


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def get_zalizniak_cache_dir() -> Path:
    """Return the local cache directory for Zaliznyak data files."""
    project_root = Path(__file__).resolve().parents[2]
    return project_root / ".cache" / "zalizniak"


def _cached_file_path(url_suffix: str) -> Path:
    """Map a URL suffix to a local cache file path."""
    # Flatten the path into a single filename by replacing / with __
    flat_name = url_suffix.replace("/", "__")
    return get_zalizniak_cache_dir() / flat_name


def ensure_zalizniak_data_cached(*, force_refresh: bool = False) -> list[Path]:
    """
    Download Zaliznyak text files from GitHub if not already cached.

    Returns a list of (local_path, is_proper_name) tuples.
    """
    cache_dir = get_zalizniak_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)

    local_paths: list[Path] = []
    for url_suffix, _is_proper in _ZALIZNIAK_FILES:
        dest = _cached_file_path(url_suffix)
        if force_refresh or not dest.exists():
            # Percent-encode the Cyrillic path segments so urllib can handle them
            encoded_suffix = urllib.parse.quote(url_suffix, safe="/")
            url = _GITHUB_BASE + encoded_suffix
            log.info("Downloading Zaliznyak data: %s", url)
            try:
                with urllib.request.urlopen(url, timeout=30) as resp:
                    dest.write_bytes(resp.read())
                log.info("Cached: %s", dest.name)
            except Exception as exc:
                log.warning("Failed to download %s: %s", url, exc)
                continue
        local_paths.append(dest)

    return local_paths


# ---------------------------------------------------------------------------
# Entry iterator
# ---------------------------------------------------------------------------

def iter_zalizniak_entries(
    *,
    force_refresh: bool = False,
) -> Iterator[PronunciationLexiconEntry]:
    """
    Yield PronunciationLexiconEntry objects for every lemma in the Zaliznyak
    dictionary.

    Files are downloaded lazily and cached in ``.cache/zalizniak/``.
    """
    ensure_zalizniak_data_cached(force_refresh=force_refresh)

    seen: set[tuple[str, str | None]] = set()
    total = 0

    for url_suffix, is_proper_name in _ZALIZNIAK_FILES:
        local = _cached_file_path(url_suffix)
        if not local.exists():
            log.warning("Zaliznyak file not found (skipped): %s", local)
            continue

        text = local.read_text(encoding="utf-8")
        for line in text.splitlines():
            entry = parse_zalizniak_line(line, is_proper_name=is_proper_name)
            if entry is None:
                continue
            # De-duplicate identical (surface_form, spoken_form) pairs
            key = (entry.surface_form, entry.spoken_form)
            if key in seen:
                continue
            seen.add(key)
            total += 1
            yield entry

    log.info("Zaliznyak: yielded %d unique entries", total)

