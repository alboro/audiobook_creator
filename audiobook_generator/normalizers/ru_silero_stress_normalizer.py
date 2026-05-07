# Copyright (c) 2026 alboro <alboro@users.noreply.github.com>
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See: https://polyformproject.org/licenses/noncommercial/1.0.0/

"""
Silero-based Russian homograph stress normalizer
=================================================
Places stress marks (combining acute U+0301) on **known Russian homographs**
using the silero-stress model for context-aware disambiguation.

Key design:
- Only words that appear in silero's built-in homograph dictionary (``homodict``,
  ~1 900 entries) receive stress marks.  All other words are left untouched.
- The BERT-based silero accentor disambiguates within each sentence, so the
  correct form is chosen based on context (e.g. "ча́сти дела" vs "части́ замка").
- Words that already carry a combining-acute stress mark are skipped.
- Words listed in ``normalize_stress_paradox_words`` are skipped (paradox guard).
- The silero model is loaded lazily and cached at the module level.

Registration:
    STEP_NAME = "ru_silero_stress"

Configuration:
    No dedicated INI keys required.  The normalizer honours the general
    ``normalize_stress_paradox_words`` key (comma-separated list of words
    that must not receive stress marks).

Requirements:
    pip install silero-stress>=1.4
"""

from __future__ import annotations

import logging
import re

from audiobook_generator.config.general_config import GeneralConfig
from audiobook_generator.normalizers.base_normalizer import BaseNormalizer
from audiobook_generator.normalizers.ru_text_utils import (
    COMBINING_ACUTE,
    normalize_stress_marks,
    plus_stress_to_combining_acute,
    preserve_case,
    strip_combining_acute,
    is_russian_language,
)
from audiobook_generator.utils.chunk_boundaries import split_text_preserve_chunk_separators

logger = logging.getLogger(__name__)

# Matches a run of Cyrillic letters (no hyphens — same as silero's word units).
_RU_WORD_RE = re.compile(r"[А-Яа-яЁё]+")

# Same pattern but also allows the leading/embedded '+' that silero inserts as a
# stress marker:  "ч+асти", "+один", "т+ому".
_RU_PLUS_WORD_RE = re.compile(r"[+А-Яа-яЁё]+")

# ---------------------------------------------------------------------------
# Module-level lazy singleton for the silero accentor
# ---------------------------------------------------------------------------

_accentor_instance = None   # the loaded silero accentor, or False if unavailable
_accentor_loaded = False    # True once we have attempted to load


def _load_silero_accentor():
    """Return the silero Russian accentor, loading it on first call.

    Returns ``None`` if silero-stress is not installed or fails to load.
    The result is cached at module level so the ~50 MB model is loaded once.
    """
    global _accentor_instance, _accentor_loaded
    if _accentor_loaded:
        return _accentor_instance

    _accentor_loaded = True
    try:
        from silero_stress.accentor import load_accentor  # type: ignore[import]
        _accentor_instance = load_accentor("ru")
        logger.info("ru_silero_stress: silero accentor loaded (ru)")
    except ImportError:
        logger.warning(
            "ru_silero_stress: silero-stress not installed — step will be skipped. "
            "Install it with: pip install silero-stress"
        )
        _accentor_instance = None
    except Exception as exc:
        logger.warning("ru_silero_stress: failed to load silero accentor: %s", exc)
        _accentor_instance = None

    return _accentor_instance


def _build_homodict(accentor) -> dict[str, list[str]]:
    """Extract the combined homograph dictionary from a loaded accentor.

    Merges ``homodict`` (stress homographs, ~1 900 entries) and
    ``yohomodict`` (ё/е alternation homographs, ~160 entries).
    Both are plain ``{word_lowercase: ['+form1', '+form2']}`` dicts.
    """
    hs = accentor.homosolver
    combined: dict[str, list[str]] = {}
    # yohomodict first so that homodict (stress-only) takes precedence on overlap
    combined.update(hs.yohomodict)
    combined.update(hs.homodict)
    return combined


# ---------------------------------------------------------------------------
# Core per-segment processing
# ---------------------------------------------------------------------------

def _stress_homographs_in_segment(
    segment: str,
    accentor,
    homodict: dict[str, list[str]],
    paradox_guard,
) -> str:
    """Apply context-aware stress to homograph words within *segment*.

    Only words present in *homodict* are modified; all other words are
    returned verbatim.  Words already carrying a stress mark (COMBINING_ACUTE)
    or listed in *paradox_guard* are always skipped.

    Args:
        segment:       Plain text segment (no ``[chunk_eof]`` markers).
        accentor:      Loaded silero accentor (callable).
        homodict:      Combined homograph dictionary from silero.
        paradox_guard: :class:`TTSStressParadoxGuard` instance.

    Returns:
        The segment with stress marks inserted on homograph tokens.
    """
    word_matches = list(_RU_WORD_RE.finditer(segment))
    if not word_matches:
        return segment

    # Quick-exit: does this segment contain any unstressed homographs?
    has_candidate = any(
        COMBINING_ACUTE not in m.group(0)
        and strip_combining_acute(m.group(0)).lower() in homodict
        and not paradox_guard.is_paradox_word(strip_combining_acute(m.group(0)).lower())
        for m in word_matches
    )
    if not has_candidate:
        return segment

    # Run silero for context-aware stress placement on the whole segment.
    try:
        accented_segment = accentor(segment)
    except Exception as exc:
        logger.warning(
            "ru_silero_stress: accentor raised an error — skipping segment: %s", exc
        )
        return segment

    # Extract accented-word tokens from silero output.
    # silero preserves word order/count; it only inserts '+' within tokens.
    acc_matches = list(_RU_PLUS_WORD_RE.finditer(accented_segment))

    # Alignment sanity check: token counts must agree.
    if len(acc_matches) != len(word_matches):
        logger.debug(
            "ru_silero_stress: token count mismatch (orig=%d, accented=%d) — "
            "skipping segment",
            len(word_matches),
            len(acc_matches),
        )
        return segment

    # Rebuild the segment, replacing only homograph tokens.
    parts: list[str] = []
    prev_end = 0
    for orig_m, acc_m in zip(word_matches, acc_matches):
        # Copy non-word gap verbatim.
        parts.append(segment[prev_end : orig_m.start()])
        prev_end = orig_m.end()

        orig_word = orig_m.group(0)
        acc_word = acc_m.group(0)   # e.g. "ч+асти", "+один", "тому"
        clean = strip_combining_acute(orig_word).lower()

        if (
            clean in homodict
            and COMBINING_ACUTE not in orig_word
            and not paradox_guard.is_paradox_word(clean)
        ):
            # Convert silero's '+' notation → combining acute, preserve case.
            stressed = plus_stress_to_combining_acute(acc_word)
            stressed = preserve_case(orig_word, stressed)
            stressed = normalize_stress_marks(stressed)
            parts.append(stressed)
        else:
            parts.append(orig_word)

    # Trailing non-word tail.
    parts.append(segment[prev_end:])
    return "".join(parts)


# ---------------------------------------------------------------------------
# Normalizer class
# ---------------------------------------------------------------------------

class SileroStressNormalizer(BaseNormalizer):
    """Context-aware Russian homograph stress normalizer backed by silero-stress.

    Applies stress marks only to words found in silero's homograph dictionary
    (~2 000 entries).  Uses silero's BERT-based accentor for context-aware
    disambiguation — the same word can receive different stress in different
    sentences.

    Example::

        "части дела"  →  "ча́сти де́ла"   (genitive — first syllable stressed)
        "по частѝ"    unchanged            (already stressed)
    """

    STEP_NAME = "ru_silero_stress"
    STEP_VERSION = 1

    def __init__(self, config: GeneralConfig):
        # Lazy — populated on first normalize() call.
        self._accentor = None
        self._homodict: dict[str, list[str]] | None = None
        super().__init__(config)

    # ------------------------------------------------------------------
    # BaseNormalizer interface
    # ------------------------------------------------------------------

    def validate_config(self):
        # Nothing to validate; silero absence is handled gracefully at runtime.
        pass

    def normalize(self, text: str, chapter_title: str = "") -> str:
        if not is_russian_language(self.config.language):
            logger.info(
                "ru_silero_stress skipped for chapter '%s': language is '%s'",
                chapter_title,
                self.config.language,
            )
            return text

        accentor = self._ensure_accentor()
        if accentor is None:
            logger.info(
                "ru_silero_stress skipped for chapter '%s': silero not available",
                chapter_title,
            )
            return text

        homodict = self._ensure_homodict(accentor)
        if not homodict:
            return text

        from audiobook_generator.normalizers.ru_tts_stress_paradox_guard import (
            get_paradox_guard,
        )
        paradox_guard = get_paradox_guard(self.config)

        # Split on both [chunk_eof] markers and sentence-ending punctuation (.!?)
        # so that silero receives a clean, self-contained sentence as context per
        # segment.  Separators are preserved for exact reconstruction.
        sentences, separators = split_text_preserve_chunk_separators(text)
        if not sentences:
            return text

        processed_sentences = [
            _stress_homographs_in_segment(sent, accentor, homodict, paradox_guard)
            for sent in sentences
        ]
        result = "".join(s + sep for s, sep in zip(processed_sentences, separators))

        if result != text:
            changes = sum(
                1
                for a, b in zip(
                    _RU_WORD_RE.findall(text),
                    _RU_WORD_RE.findall(result),
                )
                if a != b
            )
            logger.info(
                "ru_silero_stress applied to chapter '%s': %d word(s) stressed",
                chapter_title,
                changes,
            )

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_accentor(self):
        if self._accentor is None:
            self._accentor = _load_silero_accentor()
        return self._accentor

    def _ensure_homodict(self, accentor) -> dict[str, list[str]]:
        if self._homodict is None:
            self._homodict = _build_homodict(accentor)
        return self._homodict
