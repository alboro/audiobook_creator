"""Russian abbreviation expansion normalizer.

Evaluation summary (TODO 3):

``runorm`` (pip install runorm, v1.1)
  - ``RuleNormalizer``: deterministic, handles ALL-CAPS Russian acronyms
    (США → эс-ша-а) and basic currency symbol normalization.  Useful for
    letter-by-letter expansion of acronyms.  Integrated as an optional backend
    below when ``runorm`` is installed.
  - ``RUNorm`` (neural): downloads BERT+T5 models from HuggingFace (~200 MB+),
    requires ~1 GB RAM for inference on CPU.  Too heavy for the default
    pipeline; its number/date normalization is less capable than our own
    ``numbers_ru`` step.  Not integrated.

``saarus72/text_normalization`` (GitHub only, HuggingFace model ``saarus72/russian_text_normalizer``)
  - T5/FRED-T5-large based neural normalization (~1.2 GB).  NOT a pip package.
  - Handles numbers, dates, abbreviations and Latin transliteration well.
  - However, it requires significant compute, has no pip distribution, and its
    normalization scope is already covered by our deterministic pipeline
    (``numbers_ru``) + the existing ``openai`` LLM step.  Not integrated.

This normalizer therefore:
  1. Always applies a deterministic table of common Russian text abbreviations
     (т.д., т.е., и пр., и др., напр., см., ср., тыс., млн., млрд., …).
  2. Optionally uses ``runorm.RuleNormalizer`` for ALL-CAPS Russian acronyms
     when the ``runorm`` package is installed (gracefully skipped otherwise).
"""
from __future__ import annotations

import logging
import re

from audiobook_generator.config.general_config import GeneralConfig
from audiobook_generator.normalizers.base_normalizer import BaseNormalizer
from audiobook_generator.normalizers.ru_text_utils import is_russian_language

logger = logging.getLogger(__name__)

try:
    from runorm.runorm import RuleNormalizer as _RunormRuleNormalizer
    _HAS_RUNORM = True
except ImportError:  # pragma: no cover - runorm is optional
    _RunormRuleNormalizer = None
    _HAS_RUNORM = False

# ---------------------------------------------------------------------------
# Russian letter names (алфавитные названия) used for acronym expansion.
# When runorm is not available we use this table directly.
# ---------------------------------------------------------------------------
_RU_LETTER_NAMES: dict[str, str] = {
    "А": "а",   "Б": "бэ",  "В": "вэ",  "Г": "гэ",  "Д": "дэ",
    "Е": "е",   "Ё": "ё",   "Ж": "жэ",  "З": "зэ",  "И": "и",
    "Й": "ий",  "К": "ка",  "Л": "эл",  "М": "эм",  "Н": "эн",
    "О": "о",   "П": "пэ",  "Р": "эр",  "С": "эс",  "Т": "тэ",
    "У": "у",   "Ф": "эф",  "Х": "ха",  "Ц": "цэ",  "Ч": "чэ",
    "Ш": "ша",  "Щ": "ща",  "Ъ": "твёрдый знак",
    "Ы": "ы",   "Ь": "мягкий знак",
    "Э": "э",   "Ю": "ю",   "Я": "я",
}

# Pattern: 2+ consecutive Cyrillic uppercase letters not flanked by Cyrillic letters.
# NOTE: We intentionally do NOT use a very large upper bound here — real Russian
# acronyms (США, НАТО, ООН, ВЛКСМ, ДОСААФ…) are typically ≤ 7 characters.
# Longer ALL-CAPS sequences are usually regular words in uppercase (headings, emphasis)
# and must NOT be letter-expanded; they are filtered in _expand_acronym_match below.
_ACRONYM_PATTERN = re.compile(r"\b([А-ЯЁ]{2,})\b")

# Russian uppercase vowels — used to distinguish acronyms from regular words written in caps.
_RU_VOWELS_UPPER = frozenset("АЕЁИОУЫЭЮЯ")

# Heuristic: if an ALL-CAPS word has ≥ 3 vowels, it is almost certainly a regular word
# (e.g. ЗАВЕТА = З-А-В-Е-Т-А → 3 vowels; АМЕРИКАНСКОЙ → 5 vowels), not an acronym.
_MAX_ACRONYM_VOWELS = 2
# Hard length cap for acronym expansion; longer words are always treated as regular words.
_MAX_ACRONYM_LEN = 7

# ---------------------------------------------------------------------------
# Deterministic abbreviation expansion table.
# Compound / longer entries must precede shorter ones.
# ---------------------------------------------------------------------------
_ABBREV_TABLE: list[tuple[re.Pattern, str]] = [
    # Compound sequences
    (re.compile(r"\bи\s+т\.\s*д\.", re.IGNORECASE), "и так далее"),
    (re.compile(r"\bи\s+т\.\s*п\.", re.IGNORECASE), "и тому подобное"),
    (re.compile(r"\bи\s+пр\.", re.IGNORECASE), "и прочее"),
    (re.compile(r"\bи\s+др\.", re.IGNORECASE), "и другие"),
    # Single-token
    (re.compile(r"\bт\.\s*д\.", re.IGNORECASE), "так далее"),
    (re.compile(r"\bт\.\s*е\.", re.IGNORECASE), "то есть"),
    (re.compile(r"\bт\.\s*п\.", re.IGNORECASE), "тому подобному"),
    (re.compile(r"\bнапр\.", re.IGNORECASE), "например"),
    (re.compile(r"\bсм\.", re.IGNORECASE), "смотрите"),
    (re.compile(r"\bср\.", re.IGNORECASE), "сравните"),
    # Units of quantity (note: no case agreement — left to TTS)
    (re.compile(r"\bтыс\.", re.IGNORECASE), "тысяч"),
    (re.compile(r"\bмлн\.", re.IGNORECASE), "миллионов"),
    (re.compile(r"\bмлрд\.", re.IGNORECASE), "миллиардов"),
    # Publication-related
    (re.compile(r"\bизд\.", re.IGNORECASE), "издание"),
    # "л." (лист) is ONLY expanded after a digit to avoid false positives like "Самуил." → "Самуилист"
    (re.compile(r"(?<=\d)\s*л\.", re.IGNORECASE), " листов"),
]


def _expand_acronym(acronym: str) -> str:
    """Convert an ALL-CAPS Cyrillic acronym to hyphen-joined letter names.

    Example: "США" → "эс-ша-а"
    """
    parts = [_RU_LETTER_NAMES.get(ch.upper(), ch) for ch in acronym]
    return "-".join(parts)


class AbbreviationsRuNormalizer(BaseNormalizer):
    """Expand common Russian text abbreviations and optionally ALL-CAPS acronyms.

    Pipeline step name: ``abbreviations_ru``

    Two-pass strategy
    -----------------
    1. Deterministic table substitution (always active): т.д., т.е., и пр., …
    2. ALL-CAPS acronym expansion (active when ``runorm`` is installed *or*
       always via built-in letter table — configurable).

    The ``runorm`` backend is tried first; if unavailable the built-in
    ``_expand_acronym()`` function is used instead.
    """

    STEP_NAME = "ru_abbreviations"

    def __init__(self, config: GeneralConfig):
        # Pre-initialise the optional runorm backend once (lazy-loaded on
        # first normalize() call to avoid heavy imports during __init__).
        self._runorm: object | None = None
        self._runorm_checked = False
        super().__init__(config)

    def validate_config(self):
        pass  # runorm is optional; nothing required

    def normalize(self, text: str, chapter_title: str = "") -> str:
        if not is_russian_language(self.config.language):
            logger.info(
                "abbreviations_ru skipped for chapter '%s' because language is '%s'",
                chapter_title,
                self.config.language,
            )
            return text

        normalized = text
        changed = 0

        # Pass 1: deterministic table
        for pattern, replacement in _ABBREV_TABLE:
            new, n = pattern.subn(replacement, normalized)
            if n:
                normalized = new
                changed += n

        # Pass 2: ALL-CAPS acronym expansion.
        # Strip combining diacritics first so that pre-stressed words like "ТО́МАС"
        # are matched as a single token ("ТОМАС") rather than split at the accent.
        import unicodedata as _ud
        stripped = "".join(ch for ch in normalized if not _ud.combining(ch))
        # Build a mapping from stripped positions back to original positions.
        orig_positions: list[int] = []
        for i, ch in enumerate(normalized):
            if not _ud.combining(ch):
                orig_positions.append(i)
        orig_positions.append(len(normalized))  # sentinel

        expanded_parts: list[str] = []
        prev_stripped = 0
        prev_orig = 0

        for m in _ACRONYM_PATTERN.finditer(stripped):
            s_start, s_end = m.start(1), m.end(1)
            o_start = orig_positions[s_start]
            o_end = orig_positions[s_end]

            replacement = self._expand_acronym_match_str(m.group(1), stripped, s_start, s_end)

            # Copy original text up to this match (preserving combining chars), then replacement
            expanded_parts.append(normalized[prev_orig:o_start])
            if replacement != m.group(1):
                expanded_parts.append(replacement)
                changed += 1
            else:
                expanded_parts.append(normalized[o_start:o_end])
            prev_stripped = s_end
            prev_orig = o_end

        expanded_parts.append(normalized[prev_orig:])
        normalized = "".join(expanded_parts)

        logger.info(
            "abbreviations_ru applied to chapter '%s': %d replacements (runorm=%s)",
            chapter_title,
            changed,
            "available" if _HAS_RUNORM else "unavailable",
        )
        return normalized

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _expand_acronym_match(self, match: re.Match[str]) -> str:
        """Callback for _ACRONYM_PATTERN.subn — kept for backwards compatibility."""
        return self._expand_acronym_match_str(match.group(1), match.string, match.start(1), match.end(1))

    def _expand_acronym_match_str(self, acronym: str, text: str, s_start: int, s_end: int) -> str:
        """Expand a single ALL-CAPS acronym, or return it unchanged if it looks like a name.

        Returns the original text unchanged if the matched word looks like a
        regular Russian word written in ALL-CAPS rather than a true acronym.
        """

        # Reject overly long words (regular words in ALL-CAPS, e.g. headings)
        if len(acronym) > _MAX_ACRONYM_LEN:
            return acronym

        # Reject words with too many vowels (real words, not acronyms)
        vowel_count = sum(1 for ch in acronym if ch in _RU_VOWELS_UPPER)
        if vowel_count > _MAX_ACRONYM_VOWELS:
            return acronym

        # Reject if there is another ALL-CAPS word adjacent (before or after),
        # optionally separated by punctuation and/or spaces.
        # This handles headings/titles like "РУКА, СОЗДАВШАЯ НАС" or "ТОМАС ПЭЙН".
        # Check word to the right: skip punctuation/spaces after the acronym
        right_idx = s_end
        while right_idx < len(text) and text[right_idx] in ' ,;:.!?-–—':
            right_idx += 1
        if right_idx < len(text) and re.match(r"[А-ЯЁ]{2,}", text[right_idx:]):
            return acronym
        # Check word to the left: skip punctuation/spaces before the acronym
        left_idx = s_start - 1
        while left_idx >= 0 and text[left_idx] in ' ,;:.!?-–—':
            left_idx -= 1
        if left_idx >= 0:
            prefix = text[:left_idx + 1]
            if re.search(r"[А-ЯЁ]{2,}$", prefix):
                return acronym

        if _HAS_RUNORM:
            return self._expand_via_runorm(acronym)

        return _expand_acronym(acronym)

    def _expand_via_runorm(self, acronym: str) -> str:
        """Use runorm to get letter-by-letter pronunciation, then join with hyphens."""
        runorm_obj = self._get_runorm()
        if runorm_obj is None:
            return _expand_acronym(acronym)
        try:
            # expand_abbreviations works on a full text fragment; wrap the acronym
            expanded = runorm_obj.expand_abbreviations(acronym)
            # runorm outputs space-separated letter names; convert to hyphen-joined
            return "-".join(expanded.split())
        except Exception:
            return _expand_acronym(acronym)

    def _get_runorm(self):
        """Lazily initialise the runorm RuleNormalizer (once per instance)."""
        if self._runorm_checked:
            return self._runorm
        self._runorm_checked = True
        if _RunormRuleNormalizer is not None:
            try:
                self._runorm = _RunormRuleNormalizer()
            except Exception:
                self._runorm = None
        return self._runorm

