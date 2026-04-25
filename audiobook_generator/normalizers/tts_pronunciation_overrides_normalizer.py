# Copyright (c) 2026 alboro <alboro@users.noreply.github.com>
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See: https://polyformproject.org/licenses/noncommercial/1.0.0/

from __future__ import annotations

import logging
import re

from audiobook_generator.config.general_config import GeneralConfig
from audiobook_generator.normalizers.base_normalizer import BaseNormalizer
from audiobook_generator.normalizers.ru_text_utils import (
    is_russian_language,
    load_mapping_file,
    preserve_case,
)

logger = logging.getLogger(__name__)

# Default overrides kept as reference — override via normalize_tts_pronunciation_overrides_words in config.
# Format: "word=replacement,word2=replacement2" (comma-separated).
BUILTIN_TTS_PRONUNCIATION_OVERRIDES: dict[str, str] = {
    "отель": "отэль",
    "отеля": "отэля",
    "отелю": "отэлю",
    "отелем": "отэлем",
    "отеле": "отэле",
    "отели": "отэли",
    "отелей": "отэлей",
    "отелям": "отэлям",
    "отелями": "отэлями",
    "отелях": "отэлях",
}


def _parse_inline_overrides(raw: str | None) -> dict[str, str]:
    """Parse 'word=replacement,word2=replacement2' config string into a dict."""
    if not raw:
        return {}
    result = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if "=" in pair:
            src, _, rep = pair.partition("=")
            src, rep = src.strip().lower(), rep.strip()
            if src and rep:
                result[src] = rep
    return result


class TTSPronunciationOverridesNormalizer(BaseNormalizer):
    STEP_NAME = "tts_pronunciation_overrides"

    def __init__(self, config: GeneralConfig):
        inline_raw = getattr(config, "normalize_tts_pronunciation_overrides_words", None)
        inline_overrides = _parse_inline_overrides(inline_raw)
        if inline_overrides:
            # If inline config is provided, use ONLY those (user takes full control)
            self.replacements = inline_overrides
        else:
            # Fall back to builtin defaults
            self.replacements = BUILTIN_TTS_PRONUNCIATION_OVERRIDES.copy()
        # Always merge with file-based overrides on top
        self.replacements.update(
            {
                source.lower(): replacement
                for source, replacement in load_mapping_file(
                    config.normalize_tts_pronunciation_overrides_file
                ).items()
            }
        )
        super().__init__(config)

    def validate_config(self):
        return None

    def normalize(self, text: str, chapter_title: str = "") -> str:
        if not is_russian_language(self.config.language):
            logger.info(
                "tts_pronunciation_overrides skipped for chapter '%s' because language is '%s'",
                chapter_title,
                self.config.language,
            )
            return text

        normalized = text
        replacements = 0
        for source, replacement in sorted(
            self.replacements.items(),
            key=lambda item: len(item[0]),
            reverse=True,
        ):
            pattern = self._build_pattern(source)
            normalized, count = pattern.subn(
                lambda match: preserve_case(match.group(0), replacement),
                normalized,
            )
            replacements += count

        logger.info(
            "tts_pronunciation_overrides normalizer applied to chapter '%s': %s replacements",
            chapter_title,
            replacements,
        )
        return normalized

    def _build_pattern(self, source: str) -> re.Pattern[str]:
        escaped = re.escape(source)
        if re.fullmatch(r"[а-яё-]+", source, re.IGNORECASE):
            return re.compile(rf"(?<!\w){escaped}(?!\w)", re.IGNORECASE)
        return re.compile(escaped, re.IGNORECASE)
