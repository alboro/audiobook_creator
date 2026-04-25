from __future__ import annotations

import logging
import re

from audiobook_generator.config.general_config import GeneralConfig
from audiobook_generator.normalizers.base_normalizer import BaseNormalizer
from audiobook_generator.normalizers.ru_text_utils import is_russian_language

logger = logging.getLogger(__name__)

INITIALS_WITH_SURNAME_PATTERN = re.compile(
    r"(?<!\w)(?P<initials>(?:[А-ЯЁ]\s*\.\s*){1,3})(?P<surname>[А-ЯЁ][а-яё]+(?:-[А-ЯЁ][а-яё]+)*)"
)

RUSSIAN_LETTER_NAMES = {
    "А": "А",
    "Б": "Бэ",
    "В": "Вэ",
    "Г": "Гэ",
    "Д": "Дэ",
    "Е": "Е",
    "Ё": "Ё",
    "Ж": "Жэ",
    "З": "Зэ",
    "И": "И",
    "Й": "Й",
    "К": "Ка",
    "Л": "Эл",
    "М": "Эм",
    "Н": "Эн",
    "О": "О",
    "П": "Пэ",
    "Р": "Эр",
    "С": "Эс",
    "Т": "Тэ",
    "У": "У",
    "Ф": "Эф",
    "Х": "Ха",
    "Ц": "Цэ",
    "Ч": "Че",
    "Ш": "Ша",
    "Щ": "Ща",
    "Ъ": "Твердый-знак",
    "Ы": "Ы",
    "Ь": "Мягкий-знак",
    "Э": "Э",
    "Ю": "Ю",
    "Я": "Я",
}


class InitialsRuNormalizer(BaseNormalizer):
    STEP_NAME = "ru_initials"

    def __init__(self, config: GeneralConfig):
        super().__init__(config)

    def validate_config(self):
        return None

    def normalize(self, text: str, chapter_title: str = "") -> str:
        if not is_russian_language(self.config.language):
            logger.info(
                "initials_ru skipped for chapter '%s' because language is '%s'",
                chapter_title,
                self.config.language,
            )
            return text

        normalized, replacements = INITIALS_WITH_SURNAME_PATTERN.subn(self._replace_match, text)
        logger.info(
            "initials_ru normalizer applied to chapter '%s': %s replacements",
            chapter_title,
            replacements,
        )
        return normalized

    def _replace_match(self, match: re.Match[str]) -> str:
        initials = re.findall(r"[А-ЯЁ]", match.group("initials"))
        spoken_initials = " ".join(RUSSIAN_LETTER_NAMES.get(letter, letter) for letter in initials)
        return f"{spoken_initials} {match.group('surname')}"
