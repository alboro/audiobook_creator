# Copyright (c) 2026 alboro <alboro@users.noreply.github.com>
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See: https://polyformproject.org/licenses/noncommercial/1.0.0/

from __future__ import annotations

import hashlib
import logging
import re
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

# espeak-ng main Russian word list (MIT / Apache 2.0).
# The module uses "ru_list" (the actual file that exists in the repo);
# function names retain the "listx" suffix so callers don't need changes.
ESPEAK_RU_LIST_URL = (
    "https://raw.githubusercontent.com/espeak-ng/espeak-ng/master/dictsource/ru_list"
)

# ---------------------------------------------------------------------------
# Espeak phoneme analysis
# ---------------------------------------------------------------------------

# A hard consonant in espeak's Russian notation is a consonant letter that is
# NOT immediately followed by `;` (the palatalization marker).
# The vowel "э" (IPA /ɛ/) is written as `E` (optionally preceded by stress
# marker `'` and/or secondary-stress marker `#`).
# Pattern: [hard_cons_letter](?!;) ['#]* E
_HARD_CONS_BEFORE_E = re.compile(
    r"[bvgdzkl mnprstfxjZS](?!;)['\#]*E",
    re.ASCII,
)

# ---------------------------------------------------------------------------
# Cyrillic substitution
# ---------------------------------------------------------------------------

# For a word identified as having hard-consonant е→э pronunciations,
# replace every occurrence of [Cyrillic consonant] + е with [consonant] + э.
_CYRILLIC_CONS_BEFORE_E = re.compile(
    r"([бвгдзклмнпрстфхцшжщ])е",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fetch_ru_listx(
    cache_dir: str | Path | None = None,
    force_refresh: bool = False,
) -> Path:
    """Download and cache the espeak-ng Russian word list.

    Despite the ``listx`` suffix in the name (kept for backward compatibility
    with callers), this fetches ``ru_list`` — the main Russian exception list
    in the espeak-ng repo.  The cached copy lives at
    ``.cache/espeak_ru_list.txt`` in the project root.

    Parameters
    ----------
    cache_dir:
        Override directory for the cached file.  Defaults to
        ``<project_root>/.cache/``.
    force_refresh:
        Re-download even if a cached copy already exists.
    """
    cache_path = _resolve_cache_dir(cache_dir) / "espeak_ru_list.txt"
    if not force_refresh and cache_path.exists():
        logger.debug("espeak ru_list cache hit: %s", cache_path)
        return cache_path
    logger.info("Downloading espeak-ng ru_list from %s …", ESPEAK_RU_LIST_URL)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(ESPEAK_RU_LIST_URL, timeout=30) as response:
        data = response.read()
    cache_path.write_bytes(data)
    logger.debug("Cached espeak ru_list → %s (%d bytes)", cache_path, len(data))
    return cache_path


def parse_ru_listx(cache_file: Path) -> dict[str, str]:
    """Parse the espeak ru_list file and return hard-consonant е→э overrides.

    For every single-Cyrillic-word entry whose espeak phoneme string contains
    a hard consonant (not palatalized) before the ``E`` phoneme (the Russian
    "э" sound), this function produces a mapping ``{word: tts_form}`` where
    all ``[consonant]е`` sequences in the orthographic form are replaced by
    ``[consonant]э``.

    Parameters
    ----------
    cache_file:
        Path to the cached ``ru_list`` file.

    Returns
    -------
    dict[str, str]
        Keys and values are lowercase Cyrillic strings.
    """
    pairs: dict[str, str] = {}
    content = cache_file.read_text(encoding="utf-8")
    for raw_line in content.splitlines():
        line = raw_line.strip()
        # Skip comments, flags, phrase entries, and blank lines.
        if not line or line.startswith("//") or line.startswith("?") or line.startswith("("):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        word, phoneme = parts[0], parts[1]
        # Only process pure Cyrillic single words.
        if not re.fullmatch(r"[а-яёА-ЯЁ]+", word):
            continue
        word_lower = word.lower()
        if "е" not in word_lower:
            continue
        # Accept only entries where the phoneme shows at least one hard
        # consonant directly before the E phoneme.
        if not _HARD_CONS_BEFORE_E.search(phoneme):
            continue
        tts_form = _CYRILLIC_CONS_BEFORE_E.sub(r"\1э", word_lower)
        if tts_form != word_lower:
            pairs[word_lower] = tts_form
    return pairs


def sha256_file(path: Path) -> str:
    """Return the hex-encoded SHA-256 digest of the file at *path*."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_cache_dir(cache_dir: str | Path | None) -> Path:
    if cache_dir:
        return Path(cache_dir)
    project_root = Path(__file__).resolve().parents[2]
    return project_root / ".cache"
