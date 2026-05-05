# Copyright (c) 2026 alboro <alboro@users.noreply.github.com>
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See: https://polyformproject.org/licenses/noncommercial/1.0.0/

"""Colloquial Russian time expression normalizer.

Converts digital clock notation (H:MM) to spoken Russian *before* the
``ru_numbers`` step, so that ``ru_numbers`` never sees converted tokens.

Converted patterns
------------------
* ``0:00``  → ``полночь``
* ``12:00`` → ``полдень``
* ``H:15``  → ``четверть <ordinal_gen(H+1)>``   (e.g. ``1:15`` → ``четверть второго``)
* ``H:30``  → ``половина <ordinal_gen(H+1)>``   (e.g. ``3:30`` → ``половина четвёртого``)
* ``H:45``  → ``без четверти <cardinal(H+1)>``  (e.g. ``5:45`` → ``без четверти шесть``)

23:15 / 23:30 / 23:45 are **not** converted (next_hour = 0; simple digital
format is produced by ``ru_numbers`` instead).  All other H:MM patterns are
also left unchanged for ``ru_numbers`` to handle.

Placement
---------
Must run **before** ``ru_numbers`` in the normalizer chain.
"""

from __future__ import annotations

import logging
import re

from audiobook_generator.config.general_config import GeneralConfig
from audiobook_generator.normalizers.base_normalizer import BaseNormalizer
from audiobook_generator.normalizers.ru_text_utils import is_russian_language

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lookup tables
# ---------------------------------------------------------------------------

# Genitive ordinal masculine singular for "четверть/половина X-ого".
# Key = next_hour (H + 1), range 1–23.
_HOUR_ORD_GEN: dict[int, str] = {
    1:  "первого",
    2:  "второго",
    3:  "третьего",
    4:  "четвёртого",
    5:  "пятого",
    6:  "шестого",
    7:  "седьмого",
    8:  "восьмого",
    9:  "девятого",
    10: "десятого",
    11: "одиннадцатого",
    12: "двенадцатого",
    13: "тринадцатого",
    14: "четырнадцатого",
    15: "пятнадцатого",
    16: "шестнадцатого",
    17: "семнадцатого",
    18: "восемнадцатого",
    19: "девятнадцатого",
    20: "двадцатого",
    21: "двадцать первого",
    22: "двадцать второго",
    23: "двадцать третьего",
}

# Cardinal nominative for "без четверти X".
# Key = next_hour (H + 1), range 1–23.
# Hour 1 is "час" (not "один") as per colloquial Russian.
_BEZ_CHETVERTI_HOUR: dict[int, str] = {
    1:  "час",
    2:  "два",
    3:  "три",
    4:  "четыре",
    5:  "пять",
    6:  "шесть",
    7:  "семь",
    8:  "восемь",
    9:  "девять",
    10: "десять",
    11: "одиннадцать",
    12: "двенадцать",
    13: "тринадцать",
    14: "четырнадцать",
    15: "пятнадцать",
    16: "шестнадцать",
    17: "семнадцать",
    18: "восемнадцать",
    19: "девятнадцать",
    20: "двадцать",
    21: "двадцать один",
    22: "двадцать два",
    23: "двадцать три",
}

# ---------------------------------------------------------------------------
# Pattern — identical to ru_numbers.TIME_PATTERN so no conflicts at seam
# ---------------------------------------------------------------------------

_TIME_PATTERN = re.compile(r"(?<!\d)([01]?\d|2[0-3]):([0-5]\d)(?!\d)")


# ---------------------------------------------------------------------------
# Normalizer class
# ---------------------------------------------------------------------------

class TimeRuNormalizer(BaseNormalizer):
    STEP_NAME = "ru_time"
    STEP_VERSION = 1

    def validate_config(self):
        pass  # no external dependencies

    def normalize(self, text: str, chapter_title: str = "") -> str:
        if not is_russian_language(self.config.language):
            logger.info(
                "ru_time skipped for chapter '%s' because language is '%s'",
                chapter_title,
                self.config.language,
            )
            return text

        normalized, count = _TIME_PATTERN.subn(self._replace_time, text)
        if count:
            logger.info(
                "ru_time normalizer applied to chapter '%s': %s replacements",
                chapter_title,
                count,
            )
        return normalized

    # ------------------------------------------------------------------
    # Handler
    # ------------------------------------------------------------------

    def _replace_time(self, match: re.Match[str]) -> str:
        hours = int(match.group(1))
        minutes = int(match.group(2))

        # Special named times
        if hours == 0 and minutes == 0:
            return "полночь"
        if hours == 12 and minutes == 0:
            return "полдень"

        # Colloquial quarter/half expressions (:15, :30, :45)
        # next_hour = 0 means 23:xx — use simple digital format (ru_numbers handles it)
        next_hour = hours + 1
        if next_hour > 23:
            return match.group(0)

        if minutes == 15:
            gen = _HOUR_ORD_GEN.get(next_hour)
            if gen:
                return f"четверть {gen}"

        elif minutes == 30:
            gen = _HOUR_ORD_GEN.get(next_hour)
            if gen:
                return f"половина {gen}"

        elif minutes == 45:
            card = _BEZ_CHETVERTI_HOUR.get(next_hour)
            if card:
                return f"без четверти {card}"

        # All other times — leave unchanged for ru_numbers
        return match.group(0)
