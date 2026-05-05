# Copyright (c) 2026 alboro <alboro@users.noreply.github.com>
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See: https://polyformproject.org/licenses/noncommercial/1.0.0/

"""Deprecated shim — use ``tts_hard_consonants`` instead.

This module previously provided ``TTSPronunciationOverridesNormalizer``
(step name ``tts_pronunciation_overrides``).  It has been superseded by
``TTSHardConsonantsNormalizer`` (step name ``tts_hard_consonants``), which
covers all the same cases plus a full set of algorithmic hard-consonant
substitution rules.

The class ``TTSPronunciationOverridesNormalizer`` is kept as a re-export so
that any code that imports it by name continues to work.  The step name
``tts_pronunciation_overrides`` is registered as a deprecated alias in
``base_normalizer._DEPRECATED_STEP_ALIASES``.

Builtin overrides (the ``отель`` forms) have moved to
``tts_hard_consonants_normalizer.BUILTIN_TTS_HARD_CONSONANTS_OVERRIDES``.
"""

from __future__ import annotations

import logging

# Re-export the new class under the old name so downstream imports keep working.
from audiobook_generator.normalizers.tts_hard_consonants_normalizer import (  # noqa: F401
    TTSHardConsonantsNormalizer as TTSPronunciationOverridesNormalizer,
)

# Also re-export the helpers that external code may reference.
from audiobook_generator.normalizers.tts_hard_consonants_normalizer import (  # noqa: F401
    BUILTIN_TTS_HARD_CONSONANTS_OVERRIDES as BUILTIN_TTS_PRONUNCIATION_OVERRIDES,
    _parse_inline_overrides,
)

logger = logging.getLogger(__name__)
logger.debug(
    "tts_pronunciation_overrides_normalizer is deprecated; "
    "use tts_hard_consonants_normalizer instead."
)
