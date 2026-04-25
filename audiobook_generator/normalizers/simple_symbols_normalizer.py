# Copyright (c) 2026 alboro <alboro@users.noreply.github.com>
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See: https://polyformproject.org/licenses/noncommercial/1.0.0/

from __future__ import annotations

import logging
import re
import unicodedata

from audiobook_generator.config.general_config import GeneralConfig
from audiobook_generator.normalizers.base_normalizer import BaseNormalizer
from audiobook_generator.normalizers.ru_text_utils import COMBINING_ACUTE

logger = logging.getLogger(__name__)

CHAR_REPLACEMENTS = {
    "\u00ab": '`',   # «
    "\u00bb": '`',   # »
    "\u201c": '`',   # "
    "\u201d": '`',   # "
    "\u201e": '`',   # „
    "\u201f": '`',   # ‟
    "\u2033": '`',   # ″
    "\u2036": '`',   # ‶
    "\u2018": "'",
    "\u2019": "'",
    "\u201a": "'",
    "\u201b": "'",
    "\u2032": "'",
    "\u2035": "'",
    "\u2010": "-",
    "\u2011": "-",
    "\u2012": "-",
    "\u2013": "-",
    "\u2014": "-",
    "\u2015": "-",
    "\u2212": "-",
    "\u2043": "-",
    "\u2026": "...",
    "\u00a0": " ",
    "\u2000": " ",
    "\u2001": " ",
    "\u2002": " ",
    "\u2003": " ",
    "\u2004": " ",
    "\u2005": " ",
    "\u2006": " ",
    "\u2007": " ",
    "\u2008": " ",
    "\u2009": " ",
    "\u200a": " ",
    "\u202f": " ",
    "\u205f": " ",
    "\u3000": " ",
}

DROP_CHARS = {
    "\ufeff",  # BOM
    "\u00ad",  # soft hyphen
    "\u200b",  # zero-width space
    "\u200c",  # zero-width non-joiner
    "\u200d",  # zero-width joiner
    "\u2060",  # word joiner
}

SPACE_RE = re.compile(r"[ \t\f\v]+")
SPACE_AROUND_NEWLINE_RE = re.compile(r" *\n *")


class SimpleSymbolsNormalizer(BaseNormalizer):
    STEP_NAME = "simple_symbols"

    def __init__(self, config: GeneralConfig):
        super().__init__(config)

    def validate_config(self):
        return None

    def normalize(self, text: str, chapter_title: str = "") -> str:
        normalized_chars: list[str] = []
        replacement_count = 0

        for char in text.replace("\r\n", "\n").replace("\r", "\n"):
            replacement, changed = self._normalize_char(char)
            normalized_chars.append(replacement)
            if changed:
                replacement_count += 1

        normalized = "".join(normalized_chars)
        normalized = SPACE_RE.sub(" ", normalized)
        normalized = SPACE_AROUND_NEWLINE_RE.sub("\n", normalized)

        logger.info(
            "Simple symbols normalizer applied to chapter '%s': %s replacements",
            chapter_title,
            replacement_count,
        )
        return normalized

    def _normalize_char(self, char: str) -> tuple[str, bool]:
        if char in CHAR_REPLACEMENTS:
            return CHAR_REPLACEMENTS[char], True

        if char in DROP_CHARS:
            return "", True

        if char == "\n":
            return "\n", False

        if char == COMBINING_ACUTE:
            return char, False

        if self._is_ascii_safe(char):
            return char, False

        category = unicodedata.category(char)
        if category.startswith("L") or category.startswith("N"):
            return char, False

        if category.startswith("Z"):
            return " ", True

        if category in {"Cc", "Cf", "Cs", "Co", "Cn"}:
            return "", True

        return " ", True

    @staticmethod
    def _is_ascii_safe(char: str) -> bool:
        codepoint = ord(char)
        if char in {"\t", " "}:
            return True
        return 0x21 <= codepoint <= 0x7E
