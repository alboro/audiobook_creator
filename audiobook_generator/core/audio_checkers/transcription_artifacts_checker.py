# Copyright (c) 2026 alboro <alboro@users.noreply.github.com>
# Licensed under the PolyForm Noncommercial License 1.0.0.

"""TranscriptionArtifactsChecker — disputed if transcription contains forbidden substrings.

Catches TTS artefacts where the engine vocalises punctuation or emits spurious
words that Whisper picks up.  A typical CosyVoice example: a trailing period
"." in the source text causes the model to say "точка" aloud, which Whisper
transcribes as "очка" or "точка".

Config key
----------
audio_checker_transcription_artifacts
    Comma-separated list of substrings (case-insensitive) to look for anywhere
    in the Whisper transcription.  If any substring is found the chunk is marked
    as disputed for manual review.

    Example::

        audio_checker_transcription_artifacts = точка,очка

    Leave empty (or omit) to disable the check.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from audiobook_generator.core.audio_checkers.base_audio_chunk_checker import (
    BaseAudioChunkChecker,
    CheckResult,
)

logger = logging.getLogger(__name__)


def _parse_artifacts(raw: str | None) -> list[str]:
    """Parse ``'substr1,substr2'`` config string into a list of lowercase substrings."""
    if not raw:
        return []
    return [s.strip().lower() for s in raw.split(",") if s.strip()]


class TranscriptionArtifactsChecker(BaseAudioChunkChecker):
    """Mark a chunk as disputed when the Whisper transcription contains any of
    the configured artefact substrings.

    The check is intentionally broad (substring, case-insensitive) so that
    partial matches like ``очка`` inside ``точка`` are also caught by a single
    entry.  False positives are expected and harmless — they show up in the
    review UI for human confirmation.

    Config keys
    -----------
    audio_checker_transcription_artifacts
        Comma-separated substrings to search for in the transcription.
        Default: empty (checker is a no-op when no patterns are configured).
    """

    name = "transcription_artifacts"

    # Result stored in generic checker_transcription_artifacts_passed column.
    uses_fallback_passed_column = True

    def __init__(self, config):
        super().__init__(config)
        raw = getattr(config, "audio_checker_transcription_artifacts", None)
        self._patterns: list[str] = _parse_artifacts(raw)
        if self._patterns:
            logger.debug(
                "TranscriptionArtifactsChecker: watching for %d pattern(s): %s",
                len(self._patterns),
                self._patterns,
            )
        else:
            logger.debug(
                "TranscriptionArtifactsChecker: no patterns configured — "
                "checker will always pass."
            )

    def check(
        self,
        audio_file: Path,
        original_text: str,
        transcription: str,
        chunk_cache_row: Optional[dict],
    ) -> CheckResult:
        if not self._patterns or not transcription:
            return CheckResult(disputed=False)

        trans_lower = transcription.lower()
        for pattern in self._patterns:
            if pattern in trans_lower:
                logger.debug(
                    "transcription_artifacts: pattern %r found in transcription %r (chunk %s)",
                    pattern,
                    transcription[:80],
                    audio_file.name if audio_file else "?",
                )
                return CheckResult(disputed=True)

        return CheckResult(disputed=False)
