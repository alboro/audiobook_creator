# Copyright (c) 2026 alboro <alboro@users.noreply.github.com>
# Licensed under the PolyForm Noncommercial License 1.0.0.

"""WhisperSimilarityChecker — SequenceMatcher-based text similarity check."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from audiobook_generator.core.audio_checkers.base_audio_chunk_checker import (
    BaseAudioChunkChecker,
    CheckResult,
    normalize_for_compare,
    normalize_for_phonetic_compare,
    similarity,
)

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLD = 0.70

# Deterministic normalizer steps applied to both texts before similarity scoring
# to expand spoken forms (abbreviations, numbers, symbols …).
_PRE_COMPARE_STEPS = "simple_symbols,ru_initials,ru_abbreviations,ru_numbers,ru_proper_names"


def _build_pre_compare_normalizer(language: str = "ru"):
    """Build a lightweight normalizer chain used only for pre-compare expansion.

    Returns a callable ``normalize(text) -> str`` or None if construction fails.
    """
    try:
        import types
        from audiobook_generator.normalizers.base_normalizer import (
            NORMALIZER_REGISTRY,
            ChainNormalizer,
        )

        cfg = types.SimpleNamespace(
            language=language,
            normalize=True,
            normalize_steps=_PRE_COMPARE_STEPS,
            normalize_log_changes=False,
            normalize_tts_safe_max_chars=180,
            normalize_tts_safe_comma_as_period=False,
            normalize_tts_pronunciation_overrides_words=None,
            normalize_stress_paradox_words=None,
            normalize_tsnorm_min_word_length=2,
            normalize_tsnorm_stress_yo=False,
            normalize_tsnorm_stress_monosyllabic=False,
            normalize_model=None,
            normalize_provider=None,
            normalize_api_key=None,
            normalize_base_url=None,
            normalize_max_chars=4000,
            normalize_system_prompt=None,
            normalize_system_prompt_file=None,
            normalize_prompt_file=None,
            normalize_user_prompt_file=None,
            output_folder=None,
            prepared_text_folder=None,
            _normalizer_llm_runtime=None,
        )

        steps = [s.strip() for s in _PRE_COMPARE_STEPS.split(",") if s.strip()]
        normalizers = []
        for step in steps:
            entry = NORMALIZER_REGISTRY.get(step)
            if not entry:
                continue
            import importlib
            mod = importlib.import_module(entry[0])
            cls = getattr(mod, entry[1])
            normalizers.append(cls(cfg))

        if not normalizers:
            return None

        chain = ChainNormalizer(config=cfg, normalizers=normalizers, steps=steps)

        def _normalize(text: str) -> str:
            try:
                return chain.normalize(text)
            except Exception:
                return text

        return _normalize
    except Exception as exc:
        logger.warning("Could not build pre-compare normalizer: %s", exc)
        return None


class WhisperSimilarityChecker(BaseAudioChunkChecker):
    """Compare Whisper transcription with original text using SequenceMatcher.

    Marks a chunk as disputed when the character-level similarity ratio falls
    below ``audio_check_threshold``.  Applies the same deterministic normaliser
    chain to both sides before comparison so spoken-form expansions (numbers,
    abbreviations …) don't cause false positives.

    Config keys:
        audio_check_threshold  – similarity threshold (default 0.70).
    """

    name = "whisper_similarity"

    def __init__(self, config):
        super().__init__(config)
        _t = getattr(config, "audio_check_threshold", None)
        try:
            self.threshold = float(_t) if _t is not None else DEFAULT_THRESHOLD
        except (TypeError, ValueError):
            self.threshold = DEFAULT_THRESHOLD

        _lang = (getattr(config, "language", None) or "ru").split("-")[0]
        self._pre_compare = _build_pre_compare_normalizer(_lang)

    def check(
        self,
        audio_file: Path,
        original_text: str,
        transcription: str,
        chunk_cache_row: Optional[dict],
    ) -> CheckResult:
        orig_prepared = original_text
        trans_prepared = transcription

        if self._pre_compare is not None:
            orig_prepared = self._pre_compare(original_text)
            trans_prepared = self._pre_compare(transcription)

        orig_norm = normalize_for_compare(orig_prepared)
        trans_norm = normalize_for_compare(trans_prepared)
        sim = similarity(orig_norm, trans_norm)

        # Keep the regular similarity score, but allow a second pass with the
        # narrow phonetic key so short chunks are not falsely disputed by
        # harmless Whisper variants like "век" -> "вег".
        orig_phon = normalize_for_phonetic_compare(orig_prepared)
        trans_phon = normalize_for_phonetic_compare(trans_prepared)
        sim = max(sim, similarity(orig_phon, trans_phon))
        return CheckResult(disputed=sim < self.threshold, similarity=sim)
