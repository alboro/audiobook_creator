# Copyright (c) 2026 alboro <alboro@users.noreply.github.com>
# Licensed under the PolyForm Noncommercial License 1.0.0.

"""Base class, registry, and shared text utilities for audio chunk checkers."""

from __future__ import annotations

import importlib
import logging
import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared text normalisation (same logic as AudioChecker._normalize_for_compare)
# ---------------------------------------------------------------------------

def normalize_for_compare(text: str) -> str:
    """Keep only Cyrillic / Latin letters and spaces, lowercase.

    Stress marks (combining acute accents U+0301) and all other diacritics
    are stripped so Whisper's plain output can match stress-annotated TTS text.
    """
    nfd = unicodedata.normalize("NFD", text.replace("+", ""))
    stripped = "".join(c for c in nfd if unicodedata.category(c) != "Mn")
    only_letters = re.sub(r"[^\w ]", " ", stripped.lower())
    only_letters = re.sub(r"[0-9_]", " ", only_letters)
    return re.sub(r"\s+", " ", only_letters).strip()


def similarity(a: str, b: str) -> float:
    """Character-level similarity ratio in [0, 1]."""
    if not a and not b:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


_CYRILLIC_RE = re.compile(r"[а-яё]", re.IGNORECASE)
_BOUNDARY_WORD_RE = re.compile(r"\w+", re.UNICODE)
_BOUNDARY_SOFTENER_VOWELS_RE = re.compile(r"[ьъ](?=[аеёиоуыэюя])")
_BOUNDARY_IOTATED_E_RE = re.compile(r"(^|[аеёиоуыэюя])и(?=е)")
_BOUNDARY_CONSONANTS = frozenset("бвгджзйклмнпрстфхцчшщ")
_BOUNDARY_VOICING_CANONICAL = {
    "б": "п",
    "п": "п",
    "в": "ф",
    "ф": "ф",
    "г": "к",
    "к": "к",
    "д": "т",
    "т": "т",
    "ж": "ш",
    "ш": "ш",
    "з": "с",
    "с": "с",
}
_BOUNDARY_INITIAL_CANONICAL = {
    "б": "п",
    "п": "п",
}


def _get_morph_analyzer():
    """Return the shared pymorphy3 analyzer from the module-level cache.

    Delegates to :func:`audiobook_generator.normalizers.pymorphy_cache.get_morph_analyzer`
    so that all callers across the project share a single ``MorphAnalyzer`` instance
    rather than each loading the heavy dictionary files independently.
    """
    try:
        from audiobook_generator.normalizers.pymorphy_cache import get_morph_analyzer
        return get_morph_analyzer()
    except Exception as exc:
        logger.debug("pymorphy3 unavailable for boundary checks: %s", exc)
        return None


@lru_cache(maxsize=8192)
def _word_lemma_variants(word: str) -> frozenset[str]:
    """Return surface form + candidate lemmas for a single normalized word."""
    normalized = normalize_for_compare(word)
    if not normalized:
        return frozenset()

    forms = {normalized}
    if " " in normalized or not _CYRILLIC_RE.search(normalized):
        return frozenset(forms)

    morph = _get_morph_analyzer()
    if morph is None:
        return frozenset(forms)

    try:
        for parse in morph.parse(normalized)[:4]:
            lemma = normalize_for_compare(parse.normal_form or "")
            if lemma:
                forms.add(lemma)
    except Exception as exc:
        logger.debug("pymorphy3 parse failed for %r: %s", normalized, exc)

    return frozenset(forms)


def words_match_morphologically(left: str, right: str) -> bool:
    """Return True when two boundary words are equal or share a Russian lemma."""
    left_norm = normalize_for_compare(left)
    right_norm = normalize_for_compare(right)
    if not left_norm or not right_norm:
        return left_norm == right_norm
    if left_norm == right_norm:
        return True
    return bool(_word_lemma_variants(left_norm) & _word_lemma_variants(right_norm))


@lru_cache(maxsize=8192)
def _word_boundary_sound_key(word: str) -> str:
    """Return a conservative pronunciation key for Russian boundary words.

    This is intentionally narrow and only smooths over common Whisper/TTS
    surface variations that do not change how the word sounds at a chunk edge:

    - ``е`` ~= ``йе`` ~= ``йэ``
    - ``те`` ~= ``тэ`` ~= ``тьэ`` (and same idea for other consonants)

    The goal is to suppress false ``disputed`` statuses for near-identical
    speech recognitions, not to implement a full phonetic algorithm.
    """
    normalized = normalize_for_compare(word)
    if not normalized or " " in normalized:
        return normalized

    key = normalized.replace("э", "е")
    key = _BOUNDARY_SOFTENER_VOWELS_RE.sub("", key)
    # normalize_for_compare() decomposes "й" into "и" + combining breve and then
    # strips the combining mark, so "йе"/"йэ" arrive here as "ие"/"иэ".
    key = _BOUNDARY_IOTATED_E_RE.sub(r"\1", key)
    if key:
        key = _BOUNDARY_INITIAL_CANONICAL.get(key[0], key[0]) + key[1:]

    chars: list[str] = []
    for idx, char in enumerate(key):
        next_char = key[idx + 1] if idx + 1 < len(key) else ""
        if char in _BOUNDARY_VOICING_CANONICAL and (
            not next_char or next_char in _BOUNDARY_CONSONANTS
        ):
            chars.append(_BOUNDARY_VOICING_CANONICAL[char])
        else:
            chars.append(char)
    return "".join(chars)


@lru_cache(maxsize=8192)
def _word_boundary_sound_variants(word: str) -> frozenset[str]:
    """Return sound-normalized candidates for a boundary word."""
    variants = set(_word_lemma_variants(word))
    if not variants:
        normalized = normalize_for_compare(word)
        if normalized:
            variants.add(normalized)
    keys = {_word_boundary_sound_key(variant) for variant in variants if variant}
    return frozenset(key for key in keys if key)


def words_match_for_boundary(left: str, right: str) -> bool:
    """Return True when two boundary words are effectively the same for UI review.

    Comparison order:
    1. exact normalized equality
    2. shared Russian lemma via pymorphy3
    3. narrow sound-equivalence rules for Whisper boundary drift
    """
    left_norm = normalize_for_compare(left)
    right_norm = normalize_for_compare(right)
    if not left_norm or not right_norm:
        return left_norm == right_norm
    if left_norm == right_norm:
        return True
    if words_match_morphologically(left_norm, right_norm):
        return True
    return bool(_word_boundary_sound_variants(left_norm) & _word_boundary_sound_variants(right_norm))


def normalize_for_phonetic_compare(text: str) -> str:
    """Return a comparison string tolerant to narrow Russian Whisper sound drift.

    This keeps the ordinary text normalization and then applies the same
    per-word sound key we use for boundary matching.  It is intentionally
    conservative: useful for cases like ``век``/``вег`` or ``бог``/``бок``,
    but not a generic fuzzy matcher for arbitrary words.
    """
    normalized = normalize_for_compare(text)
    if not normalized:
        return normalized
    return " ".join(_word_boundary_sound_key(word) for word in normalized.split())


def starts_with_boundary_word(original_word: str, transcription_text: str) -> bool:
    """Return True when transcription begins with the original boundary word.

    Primary check uses the whole normalized transcription string, so missing
    spaces like ``подмрачным`` still match original ``под``.  A token fallback
    is kept for inflection-only cases like ``вступление`` / ``вступлении``.
    """
    original_norm = normalize_for_compare(original_word)
    transcription_norm = normalize_for_compare(transcription_text)
    if not original_norm or not transcription_norm:
        return original_norm == transcription_norm
    if transcription_norm.startswith(original_norm):
        return True

    original_phon = _word_boundary_sound_key(original_norm)
    transcription_phon = normalize_for_phonetic_compare(transcription_norm)
    if original_phon and transcription_phon.startswith(original_phon):
        return True

    match = _BOUNDARY_WORD_RE.match(transcription_norm)
    if not match:
        return False
    return words_match_for_boundary(original_norm, match.group(0))


def ends_with_boundary_word(original_word: str, transcription_text: str) -> bool:
    """Return True when transcription ends with the original boundary word.

    Like :func:`starts_with_boundary_word`, this first checks the full string
    suffix to tolerate merged words, then falls back to the last token for
    inflection-only variants.
    """
    original_norm = normalize_for_compare(original_word)
    transcription_norm = normalize_for_compare(transcription_text)
    if not original_norm or not transcription_norm:
        return original_norm == transcription_norm
    if transcription_norm.endswith(original_norm):
        return True

    original_phon = _word_boundary_sound_key(original_norm)
    transcription_phon = normalize_for_phonetic_compare(transcription_norm)
    if original_phon and transcription_phon.endswith(original_phon):
        return True

    matches = _BOUNDARY_WORD_RE.findall(transcription_norm)
    if not matches:
        return False
    return words_match_for_boundary(original_norm, matches[-1])


# ---------------------------------------------------------------------------
# CheckResult
# ---------------------------------------------------------------------------

@dataclass
class CheckResult:
    """Outcome of a single BaseAudioChunkChecker.check() call.

    ``disputed`` is the primary output.  All other fields are optional extras
    stored alongside similarity data in AudioChunkStore.  A checker only
    populates fields it owns; the orchestrator aggregates across all checkers.
    """

    disputed: bool

    # --- populated by WhisperSimilarityChecker ---
    similarity: Optional[float] = None

    # --- populated by ReferenceChecker ---
    reference_check_score: Optional[float] = None
    reference_check_threshold: Optional[float] = None
    reference_check_status: Optional[str] = None
    reference_check_payload: Optional[dict] = None



# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseAudioChunkChecker:
    """Abstract base for all audio chunk quality checkers.

    Subclasses must:
      - Set a unique class-level ``name`` string (used in INI and logs).
      - Implement ``check()``.

    Constructor always receives the application ``config`` object so every
    checker can read its own settings without extra plumbing.

    Two class-level hooks let the Review UI server derive pass/fail without
    re-running the checker:

    ``uses_fallback_passed_column``
        When ``True`` (default) the checker has no dedicated DB column and
        ``audio_checker.py`` stores its result in a generic
        ``checker_<name>_passed`` fallback column after each run.
        Set to ``False`` for checkers whose result is already encoded in
        their own existing ``chunk_cache`` columns (e.g. *whisper_similarity*
        uses the ``similarity`` column).

    ``evaluate_from_row(row, config)``
        Called by the Review UI to get pass/fail from a cached DB row without
        re-running the checker.  Returns ``None`` when the row doesn't contain
        enough data.

    ``score_from_row(row, config)``
        Returns the numeric metric for display (e.g. similarity score).
        Returns ``None`` for binary-only checkers.
    """

    name: str = ""

    #: When True the checker relies on a generic ``checker_<name>_passed``
    #: fallback column written by AudioChecker after every run.
    #: Checkers that store results in their own dedicated columns set this
    #: to False so the fallback column is never written or read.
    uses_fallback_passed_column: bool = True

    def __init__(self, config):
        self.config = config

    # ------------------------------------------------------------------
    # Per-row evaluation (called by Review UI without re-running the checker)
    # ------------------------------------------------------------------

    @classmethod
    def evaluate_from_row(cls, row: dict, config) -> Optional[bool]:
        """Derive pass/fail for this checker from an existing ``chunk_cache`` row.

        Called by the Review UI server to determine the checker's verdict
        without re-running it.  Returns ``None`` when the row does not contain
        enough data to make a determination.

        Default: reads the generic ``checker_<name>_passed`` fallback column
        (``1`` = passed, ``0`` = failed, absent / NULL = unknown).
        Subclasses with their own dedicated column(s) override this.
        """
        col = f"checker_{cls.name}_passed"
        val = row.get(col)
        if val is None:
            return None
        return val == 1

    @classmethod
    def score_from_row(cls, row: dict, config) -> Optional[float]:
        """Return the stored numeric score for display, or ``None``.

        Returns ``None`` for checkers that produce only a binary verdict.
        Subclasses with a numeric metric (e.g. similarity ratio) override this.
        """
        return None

    # ------------------------------------------------------------------

    def check(
        self,
        audio_file: Path,
        original_text: str,
        transcription: str,
        chunk_cache_row: Optional[dict],
    ) -> CheckResult:
        """Return a CheckResult for a single audio chunk.

        Args:
            audio_file:       Path to the synthesised audio file.
            original_text:    The original sentence text (TTS source).
            transcription:    Whisper transcription (already computed centrally).
            chunk_cache_row:  Row from AudioChunkStore cache, or None.

        Returns:
            CheckResult with at least ``disputed`` set.
        """
        raise NotImplementedError(f"{self.__class__.__name__}.check() not implemented")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

#: Maps checker name (as used in INI ``audio_check_checkers``) to
#: (module_path, class_name).  Third-party checkers can extend this dict
#: before AudioChecker is instantiated.
AUDIO_CHECKER_REGISTRY: dict[str, tuple[str, str]] = {
    "whisper_similarity": (
        "audiobook_generator.core.audio_checkers.whisper_similarity_checker",
        "WhisperSimilarityChecker",
    ),
    "first_word": (
        "audiobook_generator.core.audio_checkers.first_word_checker",
        "FirstWordChecker",
    ),
    "last_word": (
        "audiobook_generator.core.audio_checkers.last_word_checker",
        "LastWordChecker",
    ),
    "reference": (
        "audiobook_generator.core.audio_checkers.reference_checker",
        "ReferenceChecker",
    ),
    "transcription_artifacts": (
        "audiobook_generator.core.audio_checkers.transcription_artifacts_checker",
        "TranscriptionArtifactsChecker",
    ),
}

#: Default checker pipeline when ``audio_check_checkers`` is not set in config.
DEFAULT_CHECKERS = "whisper_similarity,first_word,last_word"


def build_checkers(config) -> list[BaseAudioChunkChecker]:
    """Instantiate checkers listed in ``config.audio_check_checkers``.

    Falls back to :data:`DEFAULT_CHECKERS` when the config key is absent/empty.
    Unknown names are logged as warnings and skipped.
    """
    spec = (getattr(config, "audio_check_checkers", None) or DEFAULT_CHECKERS).strip()
    names = [n.strip() for n in spec.split(",") if n.strip()]
    result: list[BaseAudioChunkChecker] = []
    for name in names:
        entry = AUDIO_CHECKER_REGISTRY.get(name)
        if not entry:
            logger.warning("Unknown audio checker %r — skipping.", name)
            continue
        mod_path, cls_name = entry
        try:
            mod = importlib.import_module(mod_path)
            cls = getattr(mod, cls_name)
            result.append(cls(config))
            logger.debug("Registered audio checker: %s", name)
        except Exception as exc:
            logger.error("Failed to load audio checker %r: %s", name, exc)
    return result
