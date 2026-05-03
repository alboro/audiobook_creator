# Copyright (c) 2026 alboro <alboro@users.noreply.github.com>
# Licensed under the PolyForm Noncommercial License 1.0.0.

"""Pluggable audio chunk checker framework.

Usage::

    from audiobook_generator.core.audio_checkers import build_checkers

    checkers = build_checkers(config)  # honours config.audio_check_checkers
    for checker in checkers:
        result = checker.check(audio_file, original_text, transcription, cache_row)
"""

from audiobook_generator.core.audio_checkers.base_audio_chunk_checker import (
    AUDIO_CHECKER_REGISTRY,
    BaseAudioChunkChecker,
    CheckResult,
    build_checkers,
    normalize_for_compare,
)

__all__ = [
    "AUDIO_CHECKER_REGISTRY",
    "BaseAudioChunkChecker",
    "CheckResult",
    "build_checkers",
    "normalize_for_compare",
]

