# Copyright (c) 2026 alboro <alboro@users.noreply.github.com>
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See: https://polyformproject.org/licenses/noncommercial/1.0.0/

"""
TTS Stress Paradox Guard
========================
Some words have a "stress paradox": adding a stress mark makes the TTS engine
read the word *worse* than without any mark.  For example, placing an accent on
"Томас" results in the TTS reading "ТомА́с" — an ugly artefact.

Words are configured via the INI key ``normalize_stress_paradox_words`` as a
comma-separated list.  Each entry may carry a stress mark (``+`` notation or
combining acute U+0301) to define the **canonical pronunciation** for that
exact surface form.  All other inflected forms of the same lexeme are handled
by stripping any stress marks placed by upstream normalizers.

Logic
-----
1. Parse each entry: convert ``+`` notation → combining acute; strip stress to
   get the *base* (look-up) form.
2. Expand to all inflected forms via ``pymorphy3`` (falls back gracefully if the
   analyser returns nothing useful).
3. Build a mapping  ``{stripped_lowercase_form → canonical_or_None}``:
   - For the exact base form that matches the config entry: canonical = entry
     (with stress, if specified).
   - For every other inflected form: canonical = ``None``  (→ strip stress).
4. ``apply_paradox_overrides(text)`` walks every Russian word token in *text*
   and, if the token belongs to the paradox set, replaces it with the canonical
   form (or the bare token with stress stripped), preserving original case.
5. ``filter_candidates(candidates)`` removes paradox words from the LLM
   stress-ambiguity candidate dict so the LLM is never asked to choose stress
   for words that must stay as-is.

Configuration example (config.local.ini or per-book .ini):

    [normalize]
    normalize_stress_paradox_words = Т+омас, Пейн, Кутон

Public API
----------
``TTSStressParadoxGuard(word_entries)``         — construct from a list of strings
``TTSStressParadoxGuard.from_config(raw)``      — factory from INI raw string
``guard.is_paradox_word(word)``                 — bool, ignores stress / case
``guard.apply_paradox_overrides(text)``         — str → str, post-processing pass
``guard.filter_candidates(candidates)``         — remove paradox words from LLM dict
``get_paradox_guard(config)``                   — module-level singleton factory
"""
from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Iterable

logger = logging.getLogger(__name__)

# Combining acute accent (U+0301) used project-wide for stress marks
COMBINING_ACUTE = "\u0301"

# Matches a Russian word token (possibly already carrying a stress mark)
_RU_WORD_RE = re.compile(
    rf"[А-Яа-яЁё]+(?:{re.escape(COMBINING_ACUTE)}[А-Яа-яЁё]*)*",
    re.UNICODE,
)

# Converts «+Г» notation to «Г\u0301» (stress mark AFTER the stressed letter)
_PLUS_RE = re.compile(r"\+([А-Яа-яЁё])")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _plus_to_acute(text: str) -> str:
    """Convert «То+мас» → «То́мас» (combining acute after stressed letter)."""
    return _PLUS_RE.sub(lambda m: m.group(1) + COMBINING_ACUTE, text)


def _strip_acute(text: str) -> str:
    return text.replace(COMBINING_ACUTE, "")


def _preserve_case(source: str, replacement: str) -> str:
    """Apply capitalisation of *source* to *replacement*."""
    if not source or not replacement:
        return replacement
    if source.isupper():
        return replacement.upper()
    if source[0].isupper():
        return replacement[0].upper() + replacement[1:]
    return replacement.lower()


@lru_cache(maxsize=512)
def _expand_forms_pymorphy(base_lower: str) -> frozenset[str]:
    """Return all inflected forms for *base_lower* using pymorphy3."""
    from .pymorphy_cache import get_morph_analyzer

    morph = get_morph_analyzer()
    if morph is None:
        logger.warning("pymorphy3 not available; paradox guard will only match exact forms")
        return frozenset([base_lower])

    forms: set[str] = {base_lower}
    for parse in morph.parse(base_lower):
        for form in parse.lexeme:
            forms.add(form.word.lower())
    return frozenset(forms)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class TTSStressParadoxGuard:
    """Tracks words that must not receive stress marks before TTS (or must
    keep only the explicitly configured stress)."""

    def __init__(self, word_entries: Iterable[str]):
        # Map: stripped_lowercase_surface_form → canonical_form (str with
        # optional stress mark) or None (meaning "strip any stress").
        self._map: dict[str, str | None] = {}
        for entry in word_entries:
            entry = entry.strip()
            if not entry:
                continue
            self._register(entry)

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, raw: str | None) -> "TTSStressParadoxGuard":
        """Build from a comma-separated INI value (or None/empty → empty guard)."""
        if not raw:
            return cls([])
        entries = [e.strip() for e in raw.split(",") if e.strip()]
        return cls(entries)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_paradox_word(self, word: str) -> bool:
        """Return True if *word* (any form, any stress, any case) is in the guard."""
        key = _strip_acute(word).lower()
        return key in self._map

    def apply_paradox_overrides(self, text: str) -> str:
        """Replace every paradox word occurrence with its canonical form.

        Canonical form is:
        - The configured stressed form, if the config entry had a stress mark
          AND this form is the configured base form.
        - The bare word (no stress), otherwise.
        """
        if not self._map:
            return text

        def _replace(m: re.Match) -> str:
            token = m.group(0)
            key = _strip_acute(token).lower()
            if key not in self._map:
                return token
            canonical = self._map[key]
            if canonical is None:
                # Strip any stress marks and preserve original casing
                return _preserve_case(token, key)
            else:
                # Enforce canonical (may include stress mark), preserve case
                return _preserve_case(token, canonical)

        return _RU_WORD_RE.sub(_replace, text)

    def filter_candidates(
        self, candidates: dict[str, list[str]]
    ) -> dict[str, list[str]]:
        """Remove paradox words from an {word: [variant1, variant2]} LLM candidate dict."""
        result = {}
        for word, variants in candidates.items():
            key = _strip_acute(word).lower()
            if key in self._map:
                logger.debug("Stress paradox guard: skipping LLM for '%s'", word)
            else:
                result[word] = variants
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _register(self, entry: str) -> None:
        """Parse one config entry and populate self._map for all its forms.

        Multi-word entries (e.g. ``Льво́м Толсты́м``) are split on whitespace and
        each word is registered independently.
        """
        words = entry.strip().split()
        if len(words) > 1:
            for word in words:
                self._register(word)
            return

        # Single-word path
        # Normalize stress notation to combining acute
        canonical_with_stress = _plus_to_acute(entry)
        base_lower = _strip_acute(canonical_with_stress).lower()
        has_stress = COMBINING_ACUTE in canonical_with_stress

        # Expand to all morphological forms
        all_forms = _expand_forms_pymorphy(base_lower)

        for form in all_forms:
            if form == base_lower and has_stress:
                # For the exact configured form: enforce canonical pronunciation
                self._map[form] = canonical_with_stress.lower()
            else:
                # For all other forms: strip stress
                # Only set if not already set (first entry wins)
                if form not in self._map:
                    self._map[form] = None

        # Ensure the base form itself is always present
        if base_lower not in self._map:
            self._map[base_lower] = canonical_with_stress.lower() if has_stress else None


# ---------------------------------------------------------------------------
# Module-level singleton (keyed on the raw config value)
# ---------------------------------------------------------------------------

_guard_cache: dict[str | None, TTSStressParadoxGuard] = {}


def get_paradox_guard(config=None) -> TTSStressParadoxGuard:
    """Return a cached TTSStressParadoxGuard for the given config.

    Pass a GeneralConfig (or any object with
    ``normalize_stress_paradox_words`` attribute), or None for the
    no-words guard.
    """
    raw: str | None = None
    if config is not None:
        raw = getattr(config, "normalize_stress_paradox_words", None)

    if raw not in _guard_cache:
        _guard_cache[raw] = TTSStressParadoxGuard.from_config(raw)
    return _guard_cache[raw]
