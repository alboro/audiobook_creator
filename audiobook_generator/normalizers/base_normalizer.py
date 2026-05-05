# Copyright (c) 2026 alboro <alboro@users.noreply.github.com>
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See: https://polyformproject.org/licenses/noncommercial/1.0.0/

from typing import List

from audiobook_generator.config.general_config import GeneralConfig

# ---------------------------------------------------------------------------
# Normalizer registry
# Each entry: step_name → (module_path, class_name)
# The step_name MUST match the STEP_NAME constant defined on the class itself.
# Convention: step_name == <file_basename_without_normalizer_suffix>
# e.g. "ru_tsnorm" ↔ ru_tsnorm_normalizer.py ↔ TSNormRuNormalizer.STEP_NAME
# ---------------------------------------------------------------------------
NORMALIZER_REGISTRY: dict[str, tuple[str, str]] = {
    # key                                    module (relative)                                                   class
    "openai":                               ("audiobook_generator.normalizers.openai_normalizer",                              "OpenAINormalizer"),
    "simple_symbols":                       ("audiobook_generator.normalizers.simple_symbols_normalizer",                      "SimpleSymbolsNormalizer"),
    # tts_llm_safe_split: algorithmic split + optional LLM punctuation refinement
    "tts_llm_safe_split":                   ("audiobook_generator.normalizers.tts_safe_split_normalizer",                      "TTSSafeSplitNormalizer"),
    # tts_hard_consonants: де→дэ / те→тэ hard-consonant substitutions for TTS
    "tts_hard_consonants":                  ("audiobook_generator.normalizers.tts_hard_consonants_normalizer",                 "TTSHardConsonantsNormalizer"),
    "ru_initials":                          ("audiobook_generator.normalizers.ru_initials_normalizer",                         "InitialsRuNormalizer"),
    # ru_time: colloquial time phrases (полночь, четверть второго, …) — run BEFORE ru_numbers
    "ru_time":                              ("audiobook_generator.normalizers.ru_time_normalizer",                             "TimeRuNormalizer"),
    "ru_numbers":                           ("audiobook_generator.normalizers.ru_numbers_normalizer",                          "NumbersRuNormalizer"),
    # ru_roman_numerals: standalone Roman numerals not caught by ru_numbers — run AFTER ru_numbers
    "ru_roman_numerals":                    ("audiobook_generator.normalizers.ru_roman_numerals_normalizer",                   "RomanNumeralsRuNormalizer"),
    "ru_abbreviations":                     ("audiobook_generator.normalizers.ru_abbreviations_normalizer",                    "AbbreviationsRuNormalizer"),
    # ru_llm_stress_ambiguity: LLM-assisted homograph stress resolution
    "ru_llm_stress_ambiguity":              ("audiobook_generator.normalizers.ru_stress_ambiguity_normalizer",                 "StressAmbiguityLLMNormalizer"),
    # ru_proper_names: deterministic capitalised-word stress via tsnorm backend
    "ru_proper_names":                      ("audiobook_generator.normalizers.ru_proper_nouns_normalizer",                     "ProperNounsRuNormalizer"),
    # ru_llm_proper_nouns_pronunciation: LLM-assisted proper-name pronunciation selection
    "ru_llm_proper_nouns_pronunciation":    ("audiobook_generator.normalizers.ru_proper_nouns_pronunciation_normalizer",       "ProperNounsPronunciationRuNormalizer"),
    "ru_tsnorm":                            ("audiobook_generator.normalizers.ru_tsnorm_normalizer",                           "TSNormRuNormalizer"),
    "remove_endnotes":                      ("audiobook_generator.normalizers.remove_endnotes_normalizer",                     "RemoveEndnotesNormalizer"),
    "remove_reference_numbers":             ("audiobook_generator.normalizers.remove_reference_numbers_normalizer",            "RemoveReferenceNumbersNormalizer"),
    # cosy_voice: TTS-specific fixes for CosyVoice (capitalize + single-letter leading word)
    "cosy_voice":                           ("audiobook_generator.normalizers.cosy_voice_normalizer",                          "CosyVoiceNormalizer"),
    # ── Deprecated aliases (kept for backward compatibility, will be removed in a future version) ──
    "tts_safe_split":                       ("audiobook_generator.normalizers.tts_safe_split_normalizer",                      "TTSSafeSplitAlgorithmicNormalizer"),
    "ru_proper_nouns":                      ("audiobook_generator.normalizers.ru_proper_nouns_normalizer",                     "ProperNounsRuNormalizer"),
    "ru_proper_nouns_pronunciation":        ("audiobook_generator.normalizers.ru_proper_nouns_pronunciation_normalizer",       "ProperNounsPronunciationRuNormalizer"),
}


class BaseNormalizer:
    STEP_VERSION = 1

    def __init__(self, config: GeneralConfig):
        self.config = config
        self.validate_config()

    def validate_config(self):
        raise NotImplementedError

    def normalize(self, text: str, chapter_title: str = "") -> str:
        raise NotImplementedError

    def get_step_name(self) -> str:
        return getattr(self, "STEP_NAME", self.__class__.__name__.lower())

    def should_log_changes(self) -> bool:
        """Check if change logging is enabled for this normalizer step."""
        # Default to False for performance unless explicitly enabled
        return getattr(self.config, 'normalize_log_changes', False)

    def get_resume_signature(self) -> dict:
        return {
            "step_name": self.get_step_name(),
            "step_version": getattr(self, "STEP_VERSION", 1),
        }

    def supports_chunked_resume(self) -> bool:
        return False

    def plan_processing_units(self, text: str, chapter_title: str = "") -> list[str]:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support chunked processing units"
        )

    def process_unit(
        self,
        unit: str,
        *,
        chapter_title: str = "",
        unit_index: int,
        unit_count: int,
    ) -> str:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support chunked processing units"
        )

    def merge_processed_units(
        self,
        processed_units: list[str],
        *,
        chapter_title: str = "",
    ) -> str:
        return "\n\n".join(chunk for chunk in processed_units if chunk).strip()

    def get_step_artifacts(self, text: str, chapter_title: str = "") -> dict[str, str]:
        return {}

    def get_unit_artifacts(
        self,
        unit: str,
        *,
        chapter_title: str = "",
        unit_index: int,
        unit_count: int,
    ) -> dict[str, str]:
        return {}

    def get_post_step_artifacts(
        self,
        *,
        input_text: str,
        output_text: str,
        chapter_title: str = "",
    ) -> dict[str, str]:
        return {}

    def get_normalizer_llm(self):
        cached_runtime = getattr(self.config, "_normalizer_llm_runtime", None)
        if cached_runtime is None:
            from audiobook_generator.normalizers.llm_support import NormalizerLLM

            cached_runtime = NormalizerLLM(self.config)
            setattr(self.config, "_normalizer_llm_runtime", cached_runtime)
        return cached_runtime

    def has_normalizer_llm(self) -> bool:
        return self.get_normalizer_llm().is_available

    def get_normalizer_model(self) -> str | None:
        """Return the model for this normalizer.

        Checks ``normalize_{step_name}_model`` first (per-normalizer override),
        then falls back to the global ``normalize_model``.
        """
        step = self.get_step_name().replace("-", "_")
        per_step = getattr(self.config, f"normalize_{step}_model", None)
        return per_step or getattr(self.config, "normalize_model", None)

    def normalize_with_trace(self, text: str, chapter_title: str = "") -> tuple[str, list[tuple[str, str]]]:
        normalized = self.normalize(text, chapter_title=chapter_title)
        return normalized, [(self.get_step_name(), normalized)]


def get_supported_normalizers() -> List[str]:
    return list(NORMALIZER_REGISTRY.keys())


def get_normalizer(config: GeneralConfig) -> "BaseNormalizer":
    if not config.normalize:
        raise ValueError("Text normalization is disabled")
    steps = _resolve_normalizer_steps(config)
    normalizers = [_create_normalizer(step, config) for step in steps]

    if len(normalizers) == 1:
        return normalizers[0]

    return ChainNormalizer(config=config, normalizers=normalizers, steps=steps)


class ChainNormalizer(BaseNormalizer):
    def __init__(self, config: GeneralConfig, normalizers: List[BaseNormalizer], steps: List[str]):
        self.normalizers = normalizers
        self.steps = steps
        super().__init__(config)

    def validate_config(self):
        if not self.normalizers:
            raise ValueError("At least one normalizer step must be configured")

    def normalize(self, text: str, chapter_title: str = "") -> str:
        normalized = text
        for step_name, normalizer in zip(self.steps, self.normalizers):
            normalized = normalizer.normalize(normalized, chapter_title=chapter_title)
        return normalized

    def iter_steps(self):
        return list(zip(self.steps, self.normalizers))

    def normalize_with_trace(self, text: str, chapter_title: str = "") -> tuple[str, list[tuple[str, str]]]:
        normalized = text
        trace = []
        for step_name, normalizer in zip(self.steps, self.normalizers):
            if hasattr(normalizer, "normalize_with_trace"):
                normalized, step_trace = normalizer.normalize_with_trace(
                    normalized,
                    chapter_title=chapter_title,
                )
            else:
                normalized = normalizer.normalize(normalized, chapter_title=chapter_title)
                step_trace = [(step_name, normalized)]

            if not step_trace:
                step_trace = [(step_name, normalized)]

            for traced_name, traced_text in step_trace:
                trace.append((traced_name or step_name, traced_text))

        return normalized, trace


def _resolve_normalizer_steps(config: GeneralConfig) -> List[str]:
    if config.normalize_steps:
        steps = [normalize_step_name(step) for step in config.normalize_steps.split(",") if step.strip()]
        if not steps:
            raise ValueError("normalize_steps must contain at least one normalizer step")
        return steps

    return [normalize_step_name(config.normalize_provider or "openai")]


_DEPRECATED_STEP_ALIASES = {
    "ru_stress_ambiguity": "ru_llm_stress_ambiguity",
    "ru_proper_nouns": "ru_proper_names",
    "ru_proper_nouns_pronunciation": "ru_llm_proper_nouns_pronunciation",
    # tts_pronunciation_overrides is superseded by tts_hard_consonants
    "tts_pronunciation_overrides": "tts_hard_consonants",
}

import logging as _logging
_reg_logger = _logging.getLogger(__name__)


def normalize_step_name(step: str) -> str:
    """Resolve a step name to its canonical form.

    The canonical name is the key in NORMALIZER_REGISTRY, which equals the
    STEP_NAME constant defined on the corresponding normalizer class.
    Deprecated aliases are accepted with a warning.
    Raises ValueError for unknown names.
    """
    key = step.strip().lower()
    if key in _DEPRECATED_STEP_ALIASES:
        new_key = _DEPRECATED_STEP_ALIASES[key]
        _reg_logger.warning(
            "Normalizer step '%s' is deprecated; use '%s' instead.", key, new_key
        )
        return new_key
    if key not in NORMALIZER_REGISTRY:
        raise ValueError(
            f"Unknown normalizer step: '{step}'. "
            f"Valid steps: {', '.join(sorted(k for k in NORMALIZER_REGISTRY if k not in _DEPRECATED_STEP_ALIASES))}"
        )
    return key


def _create_normalizer(step: str, config: GeneralConfig) -> "BaseNormalizer":
    entry = NORMALIZER_REGISTRY.get(step)
    if entry is None:
        raise ValueError(f"Unknown normalizer step: '{step}'")
    module_path, class_name = entry
    import importlib
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(config)
