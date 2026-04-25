# Copyright (c) 2026 alboro <alboro@users.noreply.github.com>
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See: https://polyformproject.org/licenses/noncommercial/1.0.0/

import logging

from audiobook_generator.config.general_config import GeneralConfig
from audiobook_generator.normalizers.base_normalizer import BaseNormalizer
from audiobook_generator.utils.utils import split_text

logger = logging.getLogger(__name__)


class OpenAINormalizer(BaseNormalizer):
    STEP_NAME = "openai"
    STEP_VERSION = 2

    def __init__(self, config: GeneralConfig):
        config.normalize_provider = config.normalize_provider or "openai"
        config.normalize_model = config.normalize_model or "gpt-4.1-mini"
        config.normalize_max_chars = config.normalize_max_chars or 4000

        super().__init__(config)

    def validate_config(self):
        llm = self.get_normalizer_llm()
        if self.config.normalize_max_chars == 0:
            raise ValueError("Normalizer max chars must be positive")
        if not llm.is_available:
            logger.warning(
                "openai normalizer: no LLM configured — step will be skipped. "
                "Set normalize_base_url / normalize_api_key to enable it."
            )
            return
        if not llm.settings.system_prompt:
            raise ValueError("Normalizer system prompt must not be empty")
        if "{text}" not in llm.settings.user_prompt_template:
            raise ValueError("Normalizer user prompt template must contain the {text} placeholder")

    def normalize(self, text: str, chapter_title: str = "") -> str:
        if not self.get_normalizer_llm().is_available:
            logger.info("openai normalizer skipped for chapter '%s': no LLM configured", chapter_title)
            return text
        chunks = self.plan_processing_units(text, chapter_title=chapter_title)
        normalized_chunks = []
        for idx, chunk in enumerate(chunks, start=1):
            normalized_chunks.append(
                self.process_unit(
                    chunk,
                    chapter_title=chapter_title,
                    unit_index=idx,
                    unit_count=len(chunks),
                ).strip()
            )
        return self.merge_processed_units(normalized_chunks, chapter_title=chapter_title)

    def supports_chunked_resume(self) -> bool:
        return True

    def get_resume_signature(self) -> dict:
        llm = self.get_normalizer_llm()
        return {
            **super().get_resume_signature(),
            "provider": llm.settings.provider,
            "model": llm.settings.model,
            "base_url": llm.settings.base_url,
            "max_chars": llm.settings.max_chars,
            "system_prompt": llm.settings.system_prompt,
            "user_prompt_template": llm.settings.user_prompt_template,
        }

    def plan_processing_units(self, text: str, chapter_title: str = "") -> list[str]:
        llm = self.get_normalizer_llm()
        max_chars = llm.settings.max_chars
        if not max_chars or max_chars < 0:
            return [text]
        chunks = split_text(text, max_chars, self.config.language)
        return self._merge_small_chunks(chunks, max_chars)

    def process_unit(
        self,
        unit: str,
        *,
        chapter_title: str = "",
        unit_index: int,
        unit_count: int,
    ) -> str:
        llm = self.get_normalizer_llm()
        logger.info(
            "Normalizing chapter '%s' chunk %s/%s, length=%s",
            chapter_title,
            unit_index,
            unit_count,
            len(unit),
        )
        user_prompt = llm.render_user_prompt(chapter_title=chapter_title, text=unit)
        return llm.complete(
            user_prompt=user_prompt,
            model=self.get_normalizer_model(),
            temperature=0,
        )

    def get_step_artifacts(self, text: str, chapter_title: str = "") -> dict[str, str]:
        llm = self.get_normalizer_llm()
        return {
            "00_system_prompt.txt": llm.settings.system_prompt,
            "01_user_prompt_template.txt": llm.settings.user_prompt_template,
            "02_llm_settings.json": self._render_settings_json(),
        }

    def get_unit_artifacts(
        self,
        unit: str,
        *,
        chapter_title: str = "",
        unit_index: int,
        unit_count: int,
    ) -> dict[str, str]:
        llm = self.get_normalizer_llm()
        return {
            "00_system_prompt.txt": llm.settings.system_prompt,
            "01_user_prompt.txt": llm.render_user_prompt(chapter_title=chapter_title, text=unit),
        }

    def _render_settings_json(self) -> str:
        llm = self.get_normalizer_llm()
        return (
            "{\n"
            f'  "provider": "{llm.settings.provider}",\n'
            f'  "model": "{llm.settings.model}",\n'
            f'  "base_url": "{llm.settings.base_url or ""}",\n'
            f'  "max_chars": {llm.settings.max_chars}\n'
            "}\n"
        )

    @staticmethod
    def _merge_small_chunks(chunks: list[str], preferred_max_chars: int) -> list[str]:
        if len(chunks) < 2:
            return chunks

        min_chunk_chars = max(1200, int(preferred_max_chars * 0.6))
        absolute_max_chars = max(preferred_max_chars, int(preferred_max_chars * 1.35))
        merged: list[str] = []

        for chunk in chunks:
            if (
                merged
                and len(chunk) < min_chunk_chars
                and len(merged[-1]) + 2 + len(chunk) <= absolute_max_chars
            ):
                merged[-1] = f"{merged[-1]}\n\n{chunk}".strip()
            else:
                merged.append(chunk)

        return merged
