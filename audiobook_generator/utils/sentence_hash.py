# Copyright (c) 2026 alboro <alboro@users.noreply.github.com>
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See: https://polyformproject.org/licenses/noncommercial/1.0.0/

"""Shared sentence content-hash utility.

The hash is keyed on text only (no voice/model), so the same sentence always
produces the same hash regardless of TTS settings.  Used by:
- ChunkedAudioGenerator (to name chunk audio files)
- AudioChunkStore (sentence_text_versions table)
- Review UI (to locate chunk files on disk)
- ExistingChaptersLoader (to check which chunks have been synthesised)
"""
from __future__ import annotations

import hashlib


def sentence_hash(text: str) -> str:
    """Return a 16-hex-char SHA-256 hash of *text* (stripped)."""
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()[:16]

