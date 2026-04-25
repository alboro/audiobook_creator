from __future__ import annotations

import logging
import re

from audiobook_generator.config.general_config import GeneralConfig
from audiobook_generator.normalizers.base_normalizer import BaseNormalizer
from audiobook_generator.normalizers.ru_text_utils import (
    COMBINING_ACUTE,
    is_russian_language,
)
from audiobook_generator.normalizers.tsnorm_support import create_tsnorm_backend, load_tsnorm_backend

logger = logging.getLogger(__name__)

# Include combining diacritics (U+0300-U+036F, e.g. COMBINING ACUTE ACCENT U+0301 for stress marks)
# so that "Права́ми" is matched as a whole word and the existing stress is detected.
PROPER_NOUN_WORD_PATTERN = re.compile(
    r"\b[А-ЯЁ][А-ЯЁа-яё\u0300-\u036f]*(?:-[А-ЯЁ][А-ЯЁа-яё\u0300-\u036f]*)*\b"
)
NON_NAME_WORDS = {
    "а",
    "без",
    "более",
    "бы",
    "в",
    "во",
    "вот",
    "все",
    "всё",
    "вы",
    "да",
    "для",
    "до",
    "его",
    "ее",
    "её",
    "если",
    "есть",
    "еще",
    "ещё",
    "же",
    "за",
    "здесь",
    "и",
    "из",
    "или",
    "им",
    "их",
    "к",
    "как",
    "когда",
    "кто",
    "ли",
    "меня",
    "мне",
    "мы",
    "на",
    "над",
    "не",
    "нет",
    "но",
    "о",
    "об",
    "однако",
    "он",
    "она",
    "оно",
    "они",
    "от",
    "по",
    "под",
    "при",
    "с",
    "со",
    "так",
    "там",
    "то",
    "тут",
    "ты",
    "у",
    "уже",
    "что",
    "чтобы",
    "это",
    "эта",
    "этот",
    "эти",
    "я",
}
SENTENCE_BOUNDARY_CHARS = ".!?…"
OPENING_PUNCTUATION = "\"'«“„([{-–—"
INITIALS_BEFORE_PATTERN = re.compile(r"(?:\b[А-ЯЁ]\.\s*){1,3}$")


class ProperNounsRuNormalizer(BaseNormalizer):
    STEP_NAME = "ru_proper_names"
    STEP_VERSION = 2  # bumped: paradox guard applied after tsnorm accentuation

    def __init__(self, config: GeneralConfig):
        self.backend = None
        super().__init__(config)
        self.backend = create_tsnorm_backend(
            stress_mark=COMBINING_ACUTE,
            stress_mark_pos="after",
            stress_yo=True,
            stress_monosyllabic=False,
            min_word_len=config.normalize_tsnorm_min_word_length or 2,
        )

    def validate_config(self):
        try:
            load_tsnorm_backend()
        except ImportError as exc:
            raise ImportError(
                "proper_nouns_ru requires the 'tsnorm' package. Install dependencies in a Python 3.10-3.12 environment."
            ) from exc

    def get_resume_signature(self) -> dict:
        return {
            **super().get_resume_signature(),
            "paradox_words": getattr(self.config, "normalize_stress_paradox_words", None) or "",
            "min_word_len": self.config.normalize_tsnorm_min_word_length or 2,
        }

    def normalize(self, text: str, chapter_title: str = "") -> str:
        if not is_russian_language(self.config.language):
            logger.info(
                "proper_nouns_ru skipped for chapter '%s' because language is '%s'",
                chapter_title,
                self.config.language,
            )
            return text

        replacements = 0

        def replace_match(match: re.Match[str]) -> str:
            nonlocal replacements
            word = match.group(0)
            if not self._should_accent_candidate(text, match.start(), word):
                return word

            accented = self._accentuate_candidate(word)
            if accented == word:
                return word

            replacements += 1
            return accented

        normalized = PROPER_NOUN_WORD_PATTERN.sub(replace_match, text)
        logger.info(
            "proper_nouns_ru normalizer applied to chapter '%s': %s replacements",
            chapter_title,
            replacements,
        )
        # Apply paradox guard to fix any stress errors introduced by tsnorm
        from audiobook_generator.normalizers.ru_tts_stress_paradox_guard import get_paradox_guard
        guard = get_paradox_guard(self.config)
        return guard.apply_paradox_overrides(normalized)

    def _should_accent_candidate(self, text: str, start_index: int, word: str) -> bool:
        import unicodedata as _ud
        if COMBINING_ACUTE in word:
            return False
        # The regex may stop before a trailing combining accent (e.g. "Царя́" → matches
        # "Царя", leaving "́" at match_end).  Check the char right after the match.
        match_end = start_index + len(word)
        if match_end < len(text) and _ud.combining(text[match_end]):
            return False
        if len(word) < 2:
            return False
        if word.isupper():
            return False
        if word.lower() in NON_NAME_WORDS:
            return False
        if self._has_initials_before(text, start_index):
            return True
        if self._is_sentence_start(text, start_index):
            return False
        return True

    def _is_sentence_start(self, text: str, start_index: int) -> bool:
        idx = start_index - 1
        while idx >= 0 and text[idx].isspace():
            idx -= 1
        while idx >= 0 and text[idx] in OPENING_PUNCTUATION:
            idx -= 1
            while idx >= 0 and text[idx].isspace():
                idx -= 1
        if idx < 0:
            return True
        return text[idx] in SENTENCE_BOUNDARY_CHARS

    def _has_initials_before(self, text: str, start_index: int) -> bool:
        window_start = max(0, start_index - 24)
        prefix = text[window_start:start_index]
        return bool(INITIALS_BEFORE_PATTERN.search(prefix))

    def _accentuate_candidate(self, word: str) -> str:
        if callable(self.backend):
            return self.backend(word)
        if hasattr(self.backend, "normalize"):
            return self.backend.normalize(word)
        return word
