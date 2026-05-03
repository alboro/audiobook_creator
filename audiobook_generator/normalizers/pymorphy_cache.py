# Copyright (c) 2026 alboro <alboro@users.noreply.github.com>
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See: https://polyformproject.org/licenses/noncommercial/1.0.0/

"""Global pymorphy3 MorphAnalyzer cache.

The pymorphy3 MorphAnalyzer loads large dictionary files on initialization,
which takes several seconds. Since multiple normalizers use it, we cache
a single instance here to avoid repeated expensive initialization.
"""

import logging

logger = logging.getLogger(__name__)

# Sentinel — distinguishes "never tried" (None) from "tried and unavailable"
_UNAVAILABLE = object()

_morph_analyzer_cache = None  # None = not yet initialised


def get_morph_analyzer():
    """Get a cached pymorphy3.MorphAnalyzer instance.

    Returns None if pymorphy3 is not available or could not be initialised.
    Creates the analyzer on the first successful call and caches it for all
    subsequent calls.  Unavailability is also cached so repeated import
    attempts are never made.
    """
    global _morph_analyzer_cache

    if _morph_analyzer_cache is _UNAVAILABLE:
        return None

    if _morph_analyzer_cache is not None:
        return _morph_analyzer_cache

    try:
        import pymorphy3  # type: ignore
        logger.info("Initializing pymorphy3.MorphAnalyzer (this may take a few seconds)...")
        _morph_analyzer_cache = pymorphy3.MorphAnalyzer()
        logger.info("pymorphy3.MorphAnalyzer initialized and cached")
        return _morph_analyzer_cache
    except Exception as exc:
        logger.warning("pymorphy3 not available: %s", exc)
        _morph_analyzer_cache = _UNAVAILABLE
        return None
