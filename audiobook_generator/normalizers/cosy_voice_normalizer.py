# Copyright (c) 2026 alboro <alboro@users.noreply.github.com>
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See: https://polyformproject.org/licenses/noncommercial/1.0.0/

"""CosyVoiceNormalizer — lightweight fixes for CosyVoice TTS quirks.

Rules applied in order:

1. **Capitalize first letter** — if the chunk starts with a lowercase letter,
   make it uppercase.  CosyVoice sometimes mis-stresses or swallows the first
   syllable of lower-cased text.

2. **Single-letter leading word** — if the first word of the chunk is a single
   alphabetic character (e.g. «А», «В», «И»), prepend «- » so that CosyVoice
   reads it naturally instead of clipping it.

   Example: «А было дело так» → «- А было дело так»
"""

from __future__ import annotations

import logging

from audiobook_generator.config.general_config import GeneralConfig
from audiobook_generator.normalizers.base_normalizer import BaseNormalizer

logger = logging.getLogger(__name__)


class CosyVoiceNormalizer(BaseNormalizer):
    STEP_NAME = "cosy_voice"
    STEP_VERSION = 1

    def __init__(self, config: GeneralConfig):
        super().__init__(config)

    def validate_config(self):
        return None

    def normalize(self, text: str, chapter_title: str = "") -> str:
        if not text:
            return text

        # Rule 1: capitalize the first letter of the chunk.
        if text[0].islower():
            text = text[0].upper() + text[1:]

        # Rule 2: prepend "- " when the first word is a single letter.
        # Scan from the start of the text (skipping leading spaces, if any)
        # to measure the length of the first word.
        stripped = text.lstrip()
        if stripped and stripped[0].isalpha():
            i = 1
            while i < len(stripped) and not stripped[i].isspace():
                i += 1
            # i == 1 means the first word is exactly one character
            if i == 1:
                text = "- " + text
                logger.debug(
                    "cosy_voice: prepended '- ' to single-letter leading word in %r",
                    chapter_title,
                )

        return text
