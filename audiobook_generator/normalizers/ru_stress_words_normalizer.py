from __future__ import annotations

import logging
import re

from audiobook_generator.config.general_config import GeneralConfig
from audiobook_generator.normalizers.base_normalizer import BaseNormalizer
from audiobook_generator.normalizers.ru_text_utils import (
    COMBINING_ACUTE,
    load_mapping_file,
    normalize_stress_marks,
    preserve_case,
    strip_combining_acute,
    is_russian_language,
)

logger = logging.getLogger(__name__)

STRESSABLE_WORD_PATTERN = re.compile(rf"[А-Яа-яЁё{COMBINING_ACUTE}-]+")

BUILTIN_STRESS_OVERRIDES = {
    "чудес": "чуде́с",
    "чудеса": "чудеса́",
    "каштановые": "каштАновые",
    "пенились": "пЕнились",
    "бордюром": "бордЮром",
    "крылом": "крылОм",
}

""" deprecated """
class StressWordsRuNormalizer(BaseNormalizer):
    STEP_NAME = "ru_stress_words"

    def __init__(self, config: GeneralConfig):
        self.replacements = BUILTIN_STRESS_OVERRIDES.copy()
        self.replacements.update(
            {
                strip_combining_acute(source).lower(): replacement
                for source, replacement in load_mapping_file(
                    config.normalize_stress_exceptions_file
                ).items()
            }
        )
        super().__init__(config)

    def validate_config(self):
        return None

    def normalize(self, text: str, chapter_title: str = "") -> str:
        if not is_russian_language(self.config.language):
            logger.info(
                "stress_words_ru skipped for chapter '%s' because language is '%s'",
                chapter_title,
                self.config.language,
            )
            return text

        replacements = 0

        def replace_word(match: re.Match[str]) -> str:
            nonlocal replacements
            source = match.group(0)
            key = strip_combining_acute(source).lower()
            replacement = self.replacements.get(key)
            if not replacement:
                return source
            replacements += 1
            return normalize_stress_marks(
                preserve_case(strip_combining_acute(source), replacement)
            )

        normalized = STRESSABLE_WORD_PATTERN.sub(replace_word, text)
        logger.info(
            "stress_words_ru normalizer applied to chapter '%s': %s replacements",
            chapter_title,
            replacements,
        )
        return normalized
