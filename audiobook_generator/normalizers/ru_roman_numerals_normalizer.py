# Copyright (c) 2026 alboro <alboro@users.noreply.github.com>
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See: https://polyformproject.org/licenses/noncommercial/1.0.0/

"""Standalone Roman numeral → Russian ordinal normalizer.

Converts standalone Roman numerals (not already consumed by ``ru_numbers``)
to Russian ordinal nominative masculine words.  Should be placed **after**
``ru_numbers`` in the pipeline, because ``ru_numbers`` already handles the
common "ROMAN + noun" pattern (e.g. "XVII глава").

Handled patterns
----------------
* **Multi-char** (≥2 Roman symbols) not adjacent to Cyrillic/Latin/digits:
  ``XIV`` → ``четырнадцатый``, ``Людовик XIV`` → ``Людовик четырнадцатый``
* **Single "I"** at the start of a line followed by punctuation
  (section headers like ``I. Общие положения``) → ``первый. Общие положения``

Known false-positive
--------------------
``CD`` is a valid Roman numeral (400) and will be converted to
``четырёхсотый``.  Add an upstream blocklist step if this is a concern.
"""

from __future__ import annotations

import logging
import re

from audiobook_generator.config.general_config import GeneralConfig
from audiobook_generator.normalizers.base_normalizer import BaseNormalizer
from audiobook_generator.normalizers.ru_text_utils import is_russian_language

logger = logging.getLogger(__name__)

try:
    from num2words import num2words as _num2words
except ImportError:  # pragma: no cover - dependency validation handles this at runtime
    _num2words = None

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Multi-character Roman numerals (≥2 symbols) not adjacent to Cyrillic/Latin/digits.
# Uppercase-only to reduce false positives with common lowercase abbreviations.
# TODO: consider a blocklist for common uppercase abbreviations (e.g. "TV") if needed.
_ROMAN_MULTI = re.compile(
    r"(?<![А-Яа-яЁёA-Za-z\d])([IVXLCDM]{2,})(?![А-Яа-яЁёA-Za-z\d])"
)

# Single "I" at the very start of a line, followed by sentence punctuation.
# Handles section/chapter headers: "I. Введение" → "первый. Введение".
_ROMAN_I_HEADING = re.compile(r"(?m)^(I)(?=[.,—–\-])")

# Strict Roman numeral validator — rejects invalid sequences like "VV", "LCD".
_VALID_ROMAN = re.compile(
    r"^(?=[IVXLCDM])M{0,3}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})$",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _roman_to_int(value: str) -> int | None:
    """Convert a Roman numeral string to an integer, or ``None`` if invalid."""
    if not _VALID_ROMAN.match(value):
        return None
    numerals = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    total = 0
    previous = 0
    for char in reversed(value.upper()):
        current = numerals[char]
        if current < previous:
            total -= current
        else:
            total += current
            previous = current
    return total if total > 0 else None


def _to_ordinal_ru(n: int) -> str:
    """Return Russian ordinal nominative masculine for *n* (e.g. 14 → 'четырнадцатый')."""
    spoken = _num2words(n, lang="ru", to="ordinal")
    return re.sub(r"\s+", " ", str(spoken)).strip()


# ---------------------------------------------------------------------------
# Normalizer class
# ---------------------------------------------------------------------------

class RomanNumeralsRuNormalizer(BaseNormalizer):
    STEP_NAME = "ru_roman_numerals"
    STEP_VERSION = 1

    def validate_config(self):
        if _num2words is None:
            raise ImportError(
                "ru_roman_numerals requires the 'num2words' package. "
                "Install it with: pip install num2words"
            )

    def normalize(self, text: str, chapter_title: str = "") -> str:
        if not is_russian_language(self.config.language):
            logger.info(
                "ru_roman_numerals skipped for chapter '%s' because language is '%s'",
                chapter_title,
                self.config.language,
            )
            return text

        normalized = text
        replacements = 0

        # Multi-character Roman numerals first (handles "XIV", "Людовик XIV", etc.)
        normalized, count = _ROMAN_MULTI.subn(self._replace_roman_multi, normalized)
        replacements += count

        # Single "I" at line start followed by punctuation ("I. Введение")
        normalized, count = _ROMAN_I_HEADING.subn(self._replace_roman_i_heading, normalized)
        replacements += count

        logger.info(
            "ru_roman_numerals normalizer applied to chapter '%s': %s replacements",
            chapter_title,
            replacements,
        )
        return normalized

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    def _replace_roman_multi(self, match: re.Match[str]) -> str:
        token = match.group(1)
        n = _roman_to_int(token)
        if n is None:
            return match.group(0)
        return _to_ordinal_ru(n)

    def _replace_roman_i_heading(self, match: re.Match[str]) -> str:  # noqa: ARG002
        # Lookahead keeps the following punctuation intact; only "I" is replaced.
        return "первый"
