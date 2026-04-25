from __future__ import annotations

import logging

from audiobook_generator.config.general_config import GeneralConfig
from audiobook_generator.normalizers.base_normalizer import BaseNormalizer
from audiobook_generator.normalizers.ru_text_utils import (
    COMBINING_ACUTE,
    is_russian_language,
    normalize_stress_marks,
)
from audiobook_generator.normalizers.tsnorm_support import create_tsnorm_backend, load_tsnorm_backend

logger = logging.getLogger(__name__)


class TSNormRuNormalizer(BaseNormalizer):
    STEP_NAME = "ru_tsnorm"

    def __init__(self, config: GeneralConfig):
        self.backend = None
        super().__init__(config)

        self.backend = create_tsnorm_backend(
            stress_mark=COMBINING_ACUTE,
            stress_mark_pos="after",
            stress_yo=bool(config.normalize_tsnorm_stress_yo),
            stress_monosyllabic=bool(config.normalize_tsnorm_stress_monosyllabic),
            min_word_len=config.normalize_tsnorm_min_word_length or 2,
        )

    def validate_config(self):
        try:
            load_tsnorm_backend()
        except ImportError as exc:
            raise ImportError(
                "tsnorm_ru requires the 'tsnorm' package. Install dependencies in a Python 3.10-3.12 environment."
            ) from exc

    def normalize(self, text: str, chapter_title: str = "") -> str:
        if not is_russian_language(self.config.language):
            logger.info(
                "tsnorm_ru skipped for chapter '%s' because language is '%s'",
                chapter_title,
                self.config.language,
            )
            return text

        if callable(self.backend):
            normalized = self.backend(text)
        elif hasattr(self.backend, "normalize"):
            normalized = self.backend.normalize(text)
        else:  # pragma: no cover - defensive compatibility path
            raise TypeError("tsnorm backend does not provide a callable or normalize() method")
        normalized = normalize_stress_marks(normalized)
        logger.info(
            "tsnorm_ru normalizer applied to chapter '%s': yo=%s, monosyllabic=%s, min_word_len=%s",
            chapter_title,
            bool(self.config.normalize_tsnorm_stress_yo),
            bool(self.config.normalize_tsnorm_stress_monosyllabic),
            self.config.normalize_tsnorm_min_word_length or 2,
        )
        return normalized
