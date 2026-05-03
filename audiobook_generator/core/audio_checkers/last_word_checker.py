# Copyright (c) 2026 alboro <alboro@users.noreply.github.com>
# Licensed under the PolyForm Noncommercial License 1.0.0.

"""LastWordChecker — disputed if transcription ends with a different word."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from audiobook_generator.core.audio_checkers.base_audio_chunk_checker import (
    BaseAudioChunkChecker,
    CheckResult,
    normalize_for_compare,
    ends_with_boundary_word,
)

logger = logging.getLogger(__name__)


class LastWordChecker(BaseAudioChunkChecker):
    """Mark a chunk as disputed when the transcription ends with a different word
    than the original text.

    This catches synthesis artefacts where the TTS engine adds an unrelated
    trailing utterance after the sentence — e.g., a repeated phrase, a filler
    word, or a garbled outro that Whisper transcribes as a final token.

    The same pre-compare normaliser used by WhisperSimilarityChecker is applied
    before word comparison so that number/abbreviation differences don't cause
    false positives.

    No config keys — always active when included in ``audio_check_checkers``.
    """

    name = "last_word"

    def __init__(self, config):
        super().__init__(config)
        try:
            from audiobook_generator.core.audio_checkers.whisper_similarity_checker import (
                _build_pre_compare_normalizer,
            )
            _lang = (getattr(config, "language", None) or "ru").split("-")[0]
            self._pre_compare = _build_pre_compare_normalizer(_lang)
        except Exception as exc:
            logger.debug("LastWordChecker: could not build pre-compare: %s", exc)
            self._pre_compare = None

    def check(
        self,
        audio_file: Path,
        original_text: str,
        transcription: str,
        chunk_cache_row: Optional[dict],
    ) -> CheckResult:
        if not transcription or not transcription.strip():
            # Empty transcription is already handled by whisper_similarity
            return CheckResult(disputed=False)

        orig = original_text
        trans = transcription
        if self._pre_compare is not None:
            try:
                orig = self._pre_compare(original_text)
                trans = self._pre_compare(transcription)
            except Exception:
                pass

        orig_words = normalize_for_compare(orig).split()

        if not orig_words:
            return CheckResult(disputed=False)

        disputed = not ends_with_boundary_word(orig_words[-1], trans)
        if disputed:
            logger.debug(
                "last_word mismatch: orig=%r  trans=%r",
                orig_words[-1],
                normalize_for_compare(trans)[-40:],
            )
        return CheckResult(disputed=disputed)
