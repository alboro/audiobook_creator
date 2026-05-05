# Copyright (c) 2026 alboro <alboro@users.noreply.github.com>
# Licensed under the PolyForm Noncommercial License 1.0.0.

"""FirstWordChecker — disputed if transcription starts with a different word."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from audiobook_generator.core.audio_checkers.base_audio_chunk_checker import (
    BaseAudioChunkChecker,
    CheckResult,
    normalize_for_compare,
    starts_with_boundary_word,
)

logger = logging.getLogger(__name__)


class FirstWordChecker(BaseAudioChunkChecker):
    """Mark a chunk as disputed when the transcription starts with a different word
    than the original text.

    This catches synthesis artefacts where the TTS engine emits an unrelated
    preamble utterance ("очка", "ота", etc.) that Whisper picks up as a word
    before the actual sentence starts.

    The same pre-compare normaliser used by WhisperSimilarityChecker is applied
    before word comparison so that number/abbreviation differences (e.g.
    "1917" vs "тысяча девятьсот семнадцатый") don't cause false positives.

    No config keys — always active when included in ``audio_check_checkers``.
    """

    name = "first_word"

    def __init__(self, config):
        super().__init__(config)
        # Reuse the same pre-compare builder as WhisperSimilarityChecker so
        # digit/abbreviation expansions apply here too.
        try:
            from audiobook_generator.core.audio_checkers.whisper_similarity_checker import (
                _build_pre_compare_normalizer,
            )
            _lang = (getattr(config, "language", None) or "ru").split("-")[0]
            self._pre_compare = _build_pre_compare_normalizer(_lang)
        except Exception as exc:
            logger.debug("FirstWordChecker: could not build pre-compare: %s", exc)
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

        disputed = not starts_with_boundary_word(orig_words[0], trans)
        if disputed:
            # Whisper sometimes inserts a spurious space inside a word
            # (e.g. "под мрачным" for "подмрачным").  Retry on spaceless
            # transcription as a last-resort tolerance.
            disputed = not starts_with_boundary_word(orig_words[0], trans.replace(" ", ""))
        if disputed:
            logger.debug(
                "first_word mismatch: orig=%r  trans=%r",
                orig_words[0],
                normalize_for_compare(trans)[:40],
            )
        return CheckResult(disputed=disputed)
