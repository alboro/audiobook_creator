# Copyright (c) 2026 alboro <alboro@users.noreply.github.com>
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See: https://polyformproject.org/licenses/noncommercial/1.0.0/

"""Global pymorphy3 MorphAnalyzer cache.

The pymorphy3 MorphAnalyzer loads large dictionary files on initialization,
which takes several seconds. Since multiple normalizers use it, we cache
a single instance here to avoid repeated expensive initialization.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

_morph_analyzer_cache: Optional[object] = None


def get_morph_analyzer():
    """Get a cached pymorphy3.MorphAnalyzer instance.

    Returns None if pymorphy3 is not available.
    Creates the analyzer on first call and caches it for subsequent calls.
    """
    global _morph_analyzer_cache

    if _morph_analyzer_cache is not None:
        return _morph_analyzer_cache

    try:
        import pymorphy3  # type: ignore
        logger.info("Initializing pymorphy3.MorphAnalyzer (this may take a few seconds)...")
        _morph_analyzer_cache = pymorphy3.MorphAnalyzer()
        logger.info("pymorphy3.MorphAnalyzer initialized and cached")
        return _morph_analyzer_cache
    except ImportError:
        logger.warning("pymorphy3 not available")
        _morph_analyzer_cache = None  # Cache the None result too
        return None
