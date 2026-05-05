# Copyright (c) 2026 alboro <alboro@users.noreply.github.com>
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See: https://polyformproject.org/licenses/noncommercial/1.0.0/

"""TTS hard-consonant normalizer (де→дэ / те→тэ / etc.).

Applies phonetic substitutions so that TTS engines pronounce loanwords with
the correct hard-consonant articulation before "е" sounds.  Three layers are
combined (applied in order, lower layers take precedence):

1. **Regex patterns** — algorithmic rules covering the most productive
   word-internal patterns (де, те, не, ле, ме …).  Applied first,
   left-to-right, longest composite patterns first.
2. **Word-level overrides** — explicit word→tts_form mappings merged from:
   - builtin defaults (e.g. все формы слова «отель»)
   - DB overrides from the espeak-ng ``ru_list`` (if DB is available)
   - a file specified in config (``normalize_tts_hard_consonants_file``)
   - inline pairs from config (``normalize_tts_hard_consonants_words``)
   Later sources override earlier ones.

Patterns are applied to the original text BEFORE stress-marking normalizers.
The step name is ``tts_hard_consonants``.  The old step name
``tts_pronunciation_overrides`` is a deprecated alias handled in
``base_normalizer.py``.
"""

from __future__ import annotations

import logging
import re
from typing import Callable

from audiobook_generator.config.general_config import GeneralConfig
from audiobook_generator.normalizers.base_normalizer import BaseNormalizer
from audiobook_generator.normalizers.ru_text_utils import (
    is_russian_language,
    load_mapping_file,
    preserve_case,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cyrillic word-left boundary (not preceded by any Cyrillic letter incl. Ё)
# ---------------------------------------------------------------------------
_WB = r"(?<![А-Яа-яЁё])"

# Cyrillic consonant character class (used in lookahead assertions)
_CONS = r"[бвгджзйклмнпрстфхцчшщ]"

# ---------------------------------------------------------------------------
# Regex patterns (applied in order — composite/longest first)
#
# Each entry: (compiled_pattern, replacement_string)
# The replacement string is lowercased; case is restored by preserve_case().
# ---------------------------------------------------------------------------
_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # ── Special composite patterns (handle multi-substitution words) ───────
    # "интернет" and all its forms: both те→тэ and не→нэ simultaneously.
    (re.compile(_WB + r"интернет", re.IGNORECASE), "интэрнэт"),
    # "менедж" prefix: covers менеджер, менеджмент, etc. — both ме→мэ and не→нэ.
    (re.compile(_WB + r"менедж", re.IGNORECASE), "мэнэдж"),

    # ── де → дэ ────────────────────────────────────────────────────────────
    (re.compile(r"(?<=мо)де(?=м)", re.IGNORECASE), "дэ"),        # модем
    (re.compile(_WB + r"моде(?=л)", re.IGNORECASE), "модэ"),     # модель, моделирование
    (re.compile(r"(?<=н)де(?=кс)", re.IGNORECASE), "дэ"),        # индекс
    (re.compile(_WB + r"шедевр", re.IGNORECASE), "шэдэвр"),       # шедевр (composite: ш+е→шэ AND д+е→дэ)
    (re.compile(r"(?<=тен)де(?=нц)", re.IGNORECASE), "дэ"),      # тенденция

    # ── те → тэ ────────────────────────────────────────────────────────────
    (re.compile(r"(?<=компью)те(?=р)", re.IGNORECASE), "тэ"),    # компьютер
    (re.compile(r"(?<=прин)те(?=р)", re.IGNORECASE), "тэ"),      # принтер
    (re.compile(_WB + r"конте(?=н)", re.IGNORECASE), "контэ"),   # контент
    (re.compile(r"(?<=бак)те(?=р)", re.IGNORECASE), "тэ"),       # бактерия
    (re.compile(r"(?<=аль)те(?=р)", re.IGNORECASE), "тэ"),       # альтернатива
    # интер+[consonant]: интернет caught above; handles интерфейс etc.
    (re.compile(r"(?<=ин)те(?=р" + _CONS + r")", re.IGNORECASE), "тэ"),
    (re.compile(_WB + r"анте(?=нн)", re.IGNORECASE), "антэ"),    # антенна
    (re.compile(r"(?<=роу)те(?=р)", re.IGNORECASE), "тэ"),       # роутер
    (re.compile(r"(?<=бар)те(?=р)", re.IGNORECASE), "тэ"),       # бартер
    (re.compile(r"(?<=экс)те(?=нс)", re.IGNORECASE), "тэ"),      # экстенсивный
    (re.compile(r"(?<=ин)те(?=нс)", re.IGNORECASE), "тэ"),       # интенсивный
    (re.compile(r"(?<=про)те(?=кт|з)", re.IGNORECASE), "тэ"),    # протект, протез
    (re.compile(r"(?<=эс)те(?=т)", re.IGNORECASE), "тэ"),        # эстет
    (re.compile(r"(?<=син)те(?=з|т)", re.IGNORECASE), "тэ"),     # синтез, синтетика

    # ── не → нэ ────────────────────────────────────────────────────────────
    (re.compile(r"(?<=биз)не(?=с)", re.IGNORECASE), "нэ"),       # бизнес
    (re.compile(_WB + r"кларне(?=т)", re.IGNORECASE), "кларнэ"), # кларнет
    (re.compile(r"(?<=э)не(?=рг)", re.IGNORECASE), "нэ"),        # энергия

    # ── ле → лэ ────────────────────────────────────────────────────────────
    (re.compile(r"(?<=п)ле(?=ер)", re.IGNORECASE), "лэ"),        # плеер
]

# ---------------------------------------------------------------------------
# Builtin word-level overrides (explicit forms, no regex guessing needed)
# ---------------------------------------------------------------------------
BUILTIN_TTS_HARD_CONSONANTS_OVERRIDES: dict[str, str] = {
    # Все падежные формы слова «отель»
    "отель":    "отэль",
    "отеля":    "отэля",
    "отелю":    "отэлю",
    "отелем":   "отэлем",
    "отеле":    "отэле",
    "отели":    "отэли",
    "отелей":   "отэлей",
    "отелям":   "отэлям",
    "отелями":  "отэлями",
    "отелях":   "отэлях",
}


# ---------------------------------------------------------------------------
# Config-string parser (same format as old tts_pronunciation_overrides)
# ---------------------------------------------------------------------------

def _parse_inline_overrides(raw: str | None) -> dict[str, str]:
    """Parse ``'word=replacement,word2=replacement2'`` config string."""
    if not raw:
        return {}
    result: dict[str, str] = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if "=" in pair:
            src, _, rep = pair.partition("=")
            src, rep = src.strip().lower(), rep.strip()
            if src and rep:
                result[src] = rep
    return result


# ---------------------------------------------------------------------------
# Normalizer class
# ---------------------------------------------------------------------------

class TTSHardConsonantsNormalizer(BaseNormalizer):
    """Apply hard-consonant (де→дэ / те→тэ / etc.) TTS substitutions.

    Step name: ``tts_hard_consonants``.
    """

    STEP_NAME = "tts_hard_consonants"
    STEP_VERSION = 1

    def __init__(self, config: GeneralConfig):
        # --- Build word-level overrides (builtin < db < file < manual) ---
        overrides: dict[str, str] = BUILTIN_TTS_HARD_CONSONANTS_OVERRIDES.copy()

        # DB overrides: use if the DB already exists (no auto-build/download).
        try:
            db_path = _resolve_db_path(config)
            if db_path is not None:
                from audiobook_generator.normalizers.pronunciation_lexicon_db import (
                    ESPEAK_HARD_CONSONANTS_SOURCE,
                    PronunciationLexiconDB,
                )
                db = PronunciationLexiconDB(db_path)
                overrides.update(db.get_tts_overrides(ESPEAK_HARD_CONSONANTS_SOURCE))
        except Exception as exc:  # pragma: no cover
            logger.warning(
                "tts_hard_consonants: could not load DB overrides: %s", exc
            )

        # File overrides: new key, with fallback to old key.
        file_path = (
            getattr(config, "normalize_tts_hard_consonants_file", None)
            or getattr(config, "normalize_tts_pronunciation_overrides_file", None)
        )
        overrides.update(
            {src.lower(): rep for src, rep in load_mapping_file(file_path).items()}
        )

        # Inline/manual overrides.
        inline_raw = getattr(config, "normalize_tts_hard_consonants_words", None)
        overrides.update(_parse_inline_overrides(inline_raw))

        self._overrides = overrides
        # Pre-compile word-boundary patterns for each override key.
        self._override_patterns: list[tuple[re.Pattern[str], str]] = [
            (_build_word_pattern(src), rep)
            for src, rep in sorted(
                overrides.items(), key=lambda kv: len(kv[0]), reverse=True
            )
        ]
        super().__init__(config)

    # ------------------------------------------------------------------

    def validate_config(self) -> None:
        return None

    def normalize(self, text: str, chapter_title: str = "") -> str:
        if not is_russian_language(self.config.language):
            logger.info(
                "tts_hard_consonants skipped for chapter '%s': language is '%s'",
                chapter_title,
                self.config.language,
            )
            return text

        result = text
        regex_hits = 0

        # 1. Apply regex patterns.
        for pattern, replacement in _PATTERNS:
            result, n = pattern.subn(
                _make_case_replacer(replacement),
                result,
            )
            regex_hits += n

        # 2. Apply word-level overrides (longest first).
        word_hits = 0
        for pattern, replacement in self._override_patterns:
            result, n = pattern.subn(
                lambda m, rep=replacement: preserve_case(m.group(0), rep),
                result,
            )
            word_hits += n

        logger.info(
            "tts_hard_consonants applied to chapter '%s': "
            "%d regex hits, %d word-override hits",
            chapter_title,
            regex_hits,
            word_hits,
        )
        return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_case_replacer(replacement: str) -> Callable[[re.Match[str]], str]:
    def _replace(match: re.Match[str]) -> str:
        return preserve_case(match.group(0), replacement)
    return _replace


def _build_word_pattern(source: str) -> re.Pattern[str]:
    escaped = re.escape(source)
    if re.fullmatch(r"[а-яё-]+", source, re.IGNORECASE):
        return re.compile(rf"(?<!\w){escaped}(?!\w)", re.IGNORECASE)
    return re.compile(escaped, re.IGNORECASE)


def _resolve_db_path(config: GeneralConfig) -> str | None:
    """Return the DB path from config, or the default path if the file exists."""
    explicit = getattr(config, "normalize_pronunciation_lexicon_db", None)
    if explicit:
        return explicit
    # Use default path only if the file already exists (no auto-creation here).
    try:
        from audiobook_generator.normalizers.pronunciation_lexicon_db import (
            get_default_pronunciation_lexicon_db_path,
        )
        default = get_default_pronunciation_lexicon_db_path()
        if default.exists():
            return str(default)
    except Exception:  # pragma: no cover
        pass
    return None
