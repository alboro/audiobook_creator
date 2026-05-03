from __future__ import annotations

import logging
import re

from audiobook_generator.config.general_config import GeneralConfig
from audiobook_generator.normalizers.base_normalizer import BaseNormalizer
from audiobook_generator.normalizers.ru_text_utils import is_russian_language

logger = logging.getLogger(__name__)

try:
    from num2words import num2words
except ImportError:  # pragma: no cover - dependency validation handles this at runtime
    num2words = None

try:
    from pymorphy3 import MorphAnalyzer
except ImportError:  # pragma: no cover - dependency validation handles this at runtime
    MorphAnalyzer = None

# ---------------------------------------------------------------------------
# Lookup tables defined first — patterns below reference them
# ---------------------------------------------------------------------------

# Russian month names in genitive case (used in Russian dates)
MONTH_GENITIVE_RU: dict[str, int] = {
    "января": 1, "февраля": 2, "марта": 3, "апреля": 4,
    "мая": 5, "июня": 6, "июля": 7, "августа": 8,
    "сентября": 9, "октября": 10, "ноября": 11, "декабря": 12,
}
_MONTH_NUM_TO_GEN: dict[int, str] = {v: k for k, v in MONTH_GENITIVE_RU.items()}

# "год" in various grammatical cases used after a year number
YEAR_NOUN_FORMS: dict[str, tuple[str, str]] = {
    "год":   ("m", "n"),  # 1917 год   → тысяча девятьсот семнадцатый год
    "года":  ("m", "g"),  # до 1917 года
    "году":  ("m", "p"),  # в 1917 году  (prepositional; dative "к 1917 году" has same surface "году")
    "годом": ("m", "i"),  # 1917 годом
    "годе":  ("m", "p"),  # rare prepositional
}

# Currency symbol → (nominative singular, genitive singular, genitive plural)
# Used to pick the right Russian noun form after the amount.
CURRENCY_SYMBOL_FORMS: dict[str, tuple[str, str, str]] = {
    "$": ("доллар",  "доллара",  "долларов"),
    "€": ("евро",    "евро",     "евро"),
    "£": ("фунт",    "фунта",    "фунтов"),
    "₽": ("рубль",   "рубля",    "рублей"),
}

# Common fractions: (numerator, denominator) → spoken Russian form
COMMON_FRACTIONS: dict[tuple[int, int], str] = {
    (1, 2):   "половина",
    (1, 3):   "треть",
    (2, 3):   "две трети",
    (1, 4):   "четверть",
    (3, 4):   "три четверти",
    (1, 5):   "одна пятая",
    (2, 5):   "две пятых",
    (3, 5):   "три пятых",
    (4, 5):   "четыре пятых",
    (1, 10):  "одна десятая",
    (3, 10):  "три десятых",
    (7, 10):  "семь десятых",
    (9, 10):  "девять десятых",
    (1, 100): "один процент",
}

# Ordinal genitive plural forms for denominator in generic fractions
_FRAC_DENOM_FORMS: dict[int, str] = {
    2: "вторых", 3: "третьих", 4: "четвёртых",
    5: "пятых", 6: "шестых", 7: "седьмых", 8: "восьмых",
    9: "девятых", 10: "десятых", 11: "одиннадцатых",
    12: "двенадцатых", 100: "сотых", 1000: "тысячных",
}

# ---------------------------------------------------------------------------
# Regex patterns (applied in order listed in normalize())
# ---------------------------------------------------------------------------

LIST_ITEM_PATTERN = re.compile(r"(^|\n)(\s*)(\d+)\.(?=\s)", re.MULTILINE)
NUMBER_SIGN_PATTERN = re.compile(r"№\s*(\d+)")

# Numeric date DD.MM.YYYY — must come BEFORE DECIMAL_PATTERN (to prevent "14.04" eating).
# Restrict to months 01-12 in the regex so invalid months fall through to DECIMAL.
NUMERIC_DATE_PATTERN = re.compile(
    r"(?<![/\d])(\d{1,2})\.(0?[1-9]|1[0-2])\.([12]\d{3})(?![/\d])"
)

# Full date with text month + year noun: "14 апреля 2026 года"
# Must run before PARTIAL_DATE_PATTERN so the year is consumed together.
_MONTHS_GEN_RE = "|".join(re.escape(m) for m in sorted(MONTH_GENITIVE_RU, key=len, reverse=True))
FULL_DATE_PATTERN = re.compile(
    r"(?<!\w)(\d{1,2})\s+(" + _MONTHS_GEN_RE + r")\s+(\d{4})\s+(год[аеу]?|годом)\b",
    re.IGNORECASE,
)

# Partial date "14 апреля" — after FULL_DATE_PATTERN
PARTIAL_DATE_PATTERN = re.compile(
    r"(?<!\w)(\d{1,2})\s+(" + _MONTHS_GEN_RE + r")\b",
    re.IGNORECASE,
)

# Year + "год" form: "1917 год", "в 1917 году" — before ORDINAL_NOUN_PATTERN
YEAR_RANGE_PATTERN = re.compile(
    r"(?<!\w)([12]\d{3})\s*([–—-])\s*([12]\d{3})\s+(годы|год(?:а|у|ом|е)?)\b",
    re.IGNORECASE,
)
YEAR_PATTERN = re.compile(
    r"(?<!\w)([12]\d{3})\s+(год[аеу]?|годом)\b",
    re.IGNORECASE,
)

# Time H:MM — restricted to valid clock values (0-23 h, 0-59 min) to avoid
# matching things like "100:200" that should be left for CARDINAL.
TIME_PATTERN = re.compile(r"(?<!\d)([01]?\d|2[0-3]):([0-5]\d)(?!\d)")

# Fraction N/M — before DECIMAL and CARDINAL
FRACTION_PATTERN = re.compile(r"(?<![.\w/])(\d+)/(\d+)(?![.\w/])")

# Currency symbol before number: $100, €50, ₽1000, £200
CURRENCY_BEFORE_PATTERN = re.compile(r"([$€£₽])(\d+(?:[.,]\d+)?)\b")

DECIMAL_PATTERN = re.compile(r"(?<![\w/])(-?\d+[.,]\d+)(?![\w/])")
RANGE_PATTERN = re.compile(r"(?<![\w/])(\d+)\s*([–—-])\s*(\d+)(?![\w/])")

# ---------------------------------------------------------------------------
# Ordinal-noun agreement tables
# ---------------------------------------------------------------------------

ORDINAL_NOUN_FORMS: dict[str, tuple[str, str]] = {
    # century
    "век": ("m", "n"), "века": ("m", "g"), "веку": ("m", "d"),
    "веком": ("m", "i"), "веке": ("m", "p"),
    # столетие
    "столетие": ("n", "n"), "столетия": ("n", "g"), "столетию": ("n", "d"),
    "столетием": ("n", "i"), "столетии": ("n", "p"),
    # глава
    "глава": ("f", "n"), "главы": ("f", "g"), "главе": ("f", "d"),
    "главу": ("f", "a"), "главой": ("f", "i"), "главою": ("f", "i"),
    # том
    "том": ("m", "n"), "тома": ("m", "g"), "тому": ("m", "d"),
    "томом": ("m", "i"), "томе": ("m", "p"),
    # раздел
    "раздел": ("m", "n"), "раздела": ("m", "g"), "разделу": ("m", "d"),
    "разделом": ("m", "i"), "разделе": ("m", "p"),
    # пункт
    "пункт": ("m", "n"), "пункта": ("m", "g"), "пункту": ("m", "d"),
    "пунктом": ("m", "i"), "пункте": ("m", "p"),
    # страница
    "страница": ("f", "n"), "страницы": ("f", "g"), "странице": ("f", "d"),
    "страницу": ("f", "a"), "страницей": ("f", "i"), "страницею": ("f", "i"),
    # книга
    "книга": ("f", "n"), "книги": ("f", "g"), "книге": ("f", "d"),
    "книгу": ("f", "a"), "книгой": ("f", "i"), "книгою": ("f", "i"),
    # номер
    "номер": ("m", "n"), "номера": ("m", "g"), "номеру": ("m", "d"),
    "номером": ("m", "i"), "номере": ("m", "p"),
    # часть
    "часть": ("f", "n"), "части": ("f", "g"), "частью": ("f", "i"),
    # выпуск
    "выпуск": ("m", "n"), "выпуска": ("m", "g"), "выпуску": ("m", "d"),
    "выпуском": ("m", "i"), "выпуске": ("m", "p"),
    # серия
    "серия": ("f", "n"), "серии": ("f", "g"), "серие": ("f", "d"),
    "серию": ("f", "a"), "серией": ("f", "i"),
    # сезон
    "сезон": ("m", "n"), "сезона": ("m", "g"), "сезону": ("m", "d"),
    "сезоном": ("m", "i"), "сезоне": ("m", "p"),
    # эпизод
    "эпизод": ("m", "n"), "эпизода": ("m", "g"), "эпизоду": ("m", "d"),
    "эпизодом": ("m", "i"), "эпизоде": ("m", "p"),
    # параграф
    "параграф": ("m", "n"), "параграфа": ("m", "g"), "параграфу": ("m", "d"),
    "параграфом": ("m", "i"), "параграфе": ("m", "p"),
    # статья
    "статья": ("f", "n"), "статьи": ("f", "g"), "статье": ("f", "d"),
    "статью": ("f", "a"), "статьёй": ("f", "i"), "статьей": ("f", "i"),
    # урок
    "урок": ("m", "n"), "урока": ("m", "g"), "уроку": ("m", "d"),
    "уроком": ("m", "i"), "уроке": ("m", "p"),
    # вопрос
    "вопрос": ("m", "n"), "вопроса": ("m", "g"), "вопросу": ("m", "d"),
    "вопросом": ("m", "i"), "вопросе": ("m", "p"),
    # задание
    "задание": ("n", "n"), "задания": ("n", "g"), "заданию": ("n", "d"),
    "заданием": ("n", "i"), "задании": ("n", "p"),
    # приложение
    "приложение": ("n", "n"), "приложения": ("n", "g"), "приложению": ("n", "d"),
    "приложением": ("n", "i"), "приложении": ("n", "p"),
    # подраздел
    "подраздел": ("m", "n"), "подраздела": ("m", "g"), "подразделу": ("m", "d"),
    "подразделом": ("m", "i"), "подразделе": ("m", "p"),
    # акт (законодательный)
    "акт": ("m", "n"), "акта": ("m", "g"), "акту": ("m", "d"),
    "актом": ("m", "i"), "акте": ("m", "p"),
}

ORDINAL_NOUN_PATTERN = re.compile(
    r"\b([IVXLCDM]+|\d+)\s+("
    + "|".join(re.escape(item) for item in sorted(ORDINAL_NOUN_FORMS, key=len, reverse=True))
    + r")\b",
    re.IGNORECASE,
)

ALLOWED_CARDINAL_NOUN_LEMMAS: set[str] = {
    # original
    "глава", "том", "раздел", "пункт", "страница", "книга", "часть", "номер",
    # new
    "выпуск", "серия", "сезон", "эпизод", "параграф", "статья",
    "урок", "вопрос", "задание", "приложение", "подраздел", "акт",
}

ORDINAL_SUFFIX_CONFIG: dict[str, dict] = {
    "й":   {"to": "ordinal", "case": "n", "gender": "m"},
    "ый":  {"to": "ordinal", "case": "n", "gender": "m"},
    "ий":  {"to": "ordinal", "case": "n", "gender": "m"},
    "ой":  {"to": "ordinal", "case": "n", "gender": "m"},
    "ая":  {"to": "ordinal", "case": "n", "gender": "f"},
    "я":   {"to": "ordinal", "case": "n", "gender": "f"},
    "ое":  {"to": "ordinal", "case": "n", "gender": "n"},
    "ее":  {"to": "ordinal", "case": "n", "gender": "n"},
    "е":   {"to": "ordinal", "case": "n", "gender": "n"},
    "ого": {"to": "ordinal", "case": "g", "gender": "m"},
    "го":  {"to": "ordinal", "case": "g", "gender": "m"},
    "ому": {"to": "ordinal", "case": "d", "gender": "m"},
    "му":  {"to": "ordinal", "case": "d", "gender": "m"},
    "ым":  {"to": "ordinal", "case": "i", "gender": "m"},
    "им":  {"to": "ordinal", "case": "i", "gender": "m"},
    "ом":  {"to": "ordinal", "case": "p", "gender": "m"},
    "ем":  {"to": "ordinal", "case": "p", "gender": "m"},
    "ую":  {"to": "ordinal", "case": "a", "gender": "f"},
    "ю":   {"to": "ordinal", "case": "a", "gender": "f"},
    "ые":  {"to": "ordinal", "case": "n", "plural": True},
    "ие":  {"to": "ordinal", "case": "n", "plural": True},
    "ых":  {"to": "ordinal", "case": "g", "plural": True},
    "их":  {"to": "ordinal", "case": "g", "plural": True},
    "ыми": {"to": "ordinal", "case": "i", "plural": True},
    "ими": {"to": "ordinal", "case": "i", "plural": True},
    # Short prepositional masculine: "в 2017-м году" → "две тысячи семнадцатом году"
    "м":   {"to": "ordinal", "case": "p", "gender": "m"},
}
SAFE_NO_HYPHEN_SUFFIXES = {"й", "я", "е", "ая", "ое", "ее", "го", "му", "ю"}

ORDINAL_PATTERN = re.compile(
    r"(?<![\w/])(?P<number>-?\d+)(?P<sep>-?)(?P<suffix>ый|ий|ой|й|ая|я|ое|ее|е|ого|го|ому|му|ыми|ими|ых|их|ую|ю|ом|ем|ым|им|ые|ие|м)(?![\w/])",
    re.IGNORECASE,
)

PYMORPHY_GENDER_MAP = {"masc": "m", "femn": "f", "neut": "n"}
PYMORPHY_CASE_MAP = {
    "nomn": "n", "gent": "g", "datv": "d",
    "accs": "a", "ablt": "i", "loct": "p", "loc2": "p",
}
NUMBER_WITH_NOUN_FORMS: dict[int, dict[tuple[str, str], str]] = {
    1: {
        ("n", "m"): "один", ("n", "f"): "одна", ("n", "n"): "одно",
        ("g", "m"): "одного", ("g", "f"): "одной", ("g", "n"): "одного",
        ("d", "m"): "одному", ("d", "f"): "одной", ("d", "n"): "одному",
        ("a", "m"): "один", ("a", "f"): "одну", ("a", "n"): "одно",
        ("i", "m"): "одним", ("i", "f"): "одной", ("i", "n"): "одним",
        ("p", "m"): "одном", ("p", "f"): "одной", ("p", "n"): "одном",
    },
    2: {
        ("n", "m"): "два", ("n", "f"): "две", ("n", "n"): "два",
        ("g", "m"): "двух", ("g", "f"): "двух", ("g", "n"): "двух",
        ("d", "m"): "двум", ("d", "f"): "двум", ("d", "n"): "двум",
        ("a", "m"): "два", ("a", "f"): "две", ("a", "n"): "два",
        ("i", "m"): "двумя", ("i", "f"): "двумя", ("i", "n"): "двумя",
        ("p", "m"): "двух", ("p", "f"): "двух", ("p", "n"): "двух",
    },
}

# Cardinal with pymorphy3 noun agreement — extended from [12] to any \d+
# (TODO 2: use pymorphy3 for harder numeric agreement with a broader noun set)
CARDINAL_WITH_NOUN_PATTERN = re.compile(
    r"(?<![\w/])(?P<number>\d+)\s+(?P<noun>[А-Яа-яЁё-]+)"
)
CARDINAL_PATTERN = re.compile(r"(?<![\w/])(-?\d+)(?![\w/])")


# ---------------------------------------------------------------------------
# Normalizer class
# ---------------------------------------------------------------------------

class NumbersRuNormalizer(BaseNormalizer):
    STEP_NAME = "ru_numbers"

    def __init__(self, config: GeneralConfig):
        from .pymorphy_cache import get_morph_analyzer
        self.morph = get_morph_analyzer()
        super().__init__(config)

    def validate_config(self):
        if num2words is None:
            raise ImportError(
                "numbers_ru requires the 'num2words' package. Install dependencies again after updating requirements.txt."
            )

    def normalize(self, text: str, chapter_title: str = "") -> str:
        if not is_russian_language(self.config.language):
            logger.info(
                "numbers_ru skipped for chapter '%s' because language is '%s'",
                chapter_title,
                self.config.language,
            )
            return text

        normalized = text
        replacements = 0

        # Structural markers
        normalized, count = LIST_ITEM_PATTERN.subn(self._replace_list_item, normalized)
        replacements += count
        normalized, count = NUMBER_SIGN_PATTERN.subn(self._replace_number_sign, normalized)
        replacements += count

        # Dates (NUMERIC before DECIMAL to prevent "14.04" being consumed as decimal)
        normalized, count = NUMERIC_DATE_PATTERN.subn(self._replace_numeric_date, normalized)
        replacements += count
        normalized, count = FULL_DATE_PATTERN.subn(self._replace_full_date, normalized)
        replacements += count
        normalized, count = PARTIAL_DATE_PATTERN.subn(self._replace_partial_date, normalized)
        replacements += count

        # Years with "год" form
        normalized, count = YEAR_RANGE_PATTERN.subn(self._replace_year_range, normalized)
        replacements += count
        normalized, count = YEAR_PATTERN.subn(self._replace_year, normalized)
        replacements += count

        # Time "H:MM"
        normalized, count = TIME_PATTERN.subn(self._replace_time, normalized)
        replacements += count

        # Ordinal noun constructs ("5 серия", "XVII параграф", "к 3 сезону")
        normalized, count = ORDINAL_NOUN_PATTERN.subn(self._replace_ordinal_noun, normalized)
        replacements += count

        # Ordinal suffix ("17-й", "2017-м")
        normalized, count = ORDINAL_PATTERN.subn(self._replace_ordinal, normalized)
        replacements += count

        # Fractions ("1/2") — before DECIMAL and CARDINAL
        normalized, count = FRACTION_PATTERN.subn(self._replace_fraction, normalized)
        replacements += count

        # Currency symbols ("$100", "₽1000")
        normalized, count = CURRENCY_BEFORE_PATTERN.subn(self._replace_currency_before, normalized)
        replacements += count

        # Decimal numbers
        normalized, count = DECIMAL_PATTERN.subn(self._replace_decimal, normalized)
        replacements += count

        # Numeric ranges ("5-10")
        normalized, count = RANGE_PATTERN.subn(self._replace_range, normalized)
        replacements += count

        # Cardinal with grammatical noun agreement via pymorphy3 (TODO 2)
        if self.morph is not None:
            normalized, count = CARDINAL_WITH_NOUN_PATTERN.subn(
                self._replace_cardinal_with_noun, normalized
            )
            replacements += count

        # Catch-all cardinal
        normalized, count = CARDINAL_PATTERN.subn(self._replace_cardinal, normalized)
        replacements += count

        logger.info(
            "numbers_ru normalizer applied to chapter '%s': %s replacements",
            chapter_title,
            replacements,
        )
        return normalized

    # ------------------------------------------------------------------
    # Handlers — structural
    # ------------------------------------------------------------------

    def _replace_list_item(self, match: re.Match[str]) -> str:
        prefix, indent, raw_number = match.groups()
        spoken = self._to_words(int(raw_number), to="ordinal", gender="n")
        return f"{prefix}{indent}{spoken.capitalize()}."

    def _replace_number_sign(self, match: re.Match[str]) -> str:
        return f"номер {self._to_words(int(match.group(1)))}"

    # ------------------------------------------------------------------
    # Handlers — dates
    # ------------------------------------------------------------------

    def _replace_numeric_date(self, match: re.Match[str]) -> str:
        day, month, year = int(match.group(1)), int(match.group(2)), int(match.group(3))
        if not (1 <= month <= 12 and 1 <= day <= 31):
            return match.group(0)
        month_name = _MONTH_NUM_TO_GEN.get(month, str(month))
        try:
            day_spoken = self._to_words(day, to="ordinal", gender="m", case="g")
            year_spoken = self._to_words(year, to="ordinal", gender="m", case="g")
        except Exception:
            day_spoken = self._to_words(day)
            year_spoken = self._to_words(year)
        return f"{day_spoken} {month_name} {year_spoken} года"

    def _replace_full_date(self, match: re.Match[str]) -> str:
        day = int(match.group(1))
        month_name = match.group(2).lower()
        year = int(match.group(3))
        year_noun = match.group(4).lower()
        try:
            day_spoken = self._to_words(day, to="ordinal", gender="m", case="g")
            year_spoken = self._to_words(year, to="ordinal", gender="m", case="g")
        except Exception:
            day_spoken = self._to_words(day)
            year_spoken = self._to_words(year)
        return f"{day_spoken} {month_name} {year_spoken} {year_noun}"

    def _replace_partial_date(self, match: re.Match[str]) -> str:
        day = int(match.group(1))
        month_name = match.group(2).lower()
        try:
            day_spoken = self._to_words(day, to="ordinal", gender="m", case="g")
        except Exception:
            day_spoken = self._to_words(day)
        return f"{day_spoken} {month_name}"

    # ------------------------------------------------------------------
    # Handlers — years
    # ------------------------------------------------------------------

    def _replace_year_range(self, match: re.Match[str]) -> str:
        left_year = int(match.group(1))
        dash = match.group(2)
        right_year = int(match.group(3))
        year_noun = match.group(4)
        noun_lower = year_noun.lower()

        # Whisper often collapses "1794-1796 годы" to "... год". For compare/TTS
        # we normalize year ranges to the natural plural surface.
        output_noun = "годы" if noun_lower == "год" else year_noun
        if output_noun.lower() == "годы":
            gender, case = ("m", "n")
        else:
            noun_config = YEAR_NOUN_FORMS.get(noun_lower)
            if noun_config is None:
                return match.group(0)
            gender, case = noun_config

        try:
            left_spoken = self._to_words(left_year, to="ordinal", gender=gender, case=case)
            right_spoken = self._to_words(right_year, to="ordinal", gender=gender, case=case)
        except Exception:
            left_spoken = self._to_words(left_year)
            right_spoken = self._to_words(right_year)
        return f"{left_spoken} {dash} {right_spoken} {output_noun}"

    def _replace_year(self, match: re.Match[str]) -> str:
        year = int(match.group(1))
        year_noun = match.group(2)
        noun_config = YEAR_NOUN_FORMS.get(year_noun.lower())
        if noun_config is None:
            return match.group(0)
        gender, case = noun_config
        try:
            spoken = self._to_words(year, to="ordinal", gender=gender, case=case)
        except Exception:
            spoken = self._to_words(year)
        return f"{spoken} {year_noun}"

    # ------------------------------------------------------------------
    # Handlers — time
    # ------------------------------------------------------------------

    def _replace_time(self, match: re.Match[str]) -> str:
        hours = int(match.group(1))
        minutes = int(match.group(2))
        # Pattern already validates 0-23 h / 0-59 min, so no range check needed.
        hours_spoken = self._to_words(hours)
        if minutes == 0:
            minutes_spoken = "ноль-ноль"
        elif minutes < 10:
            minutes_spoken = f"ноль {self._to_words(minutes)}"
        else:
            minutes_spoken = self._to_words(minutes)
        return f"{hours_spoken} {minutes_spoken}"

    # ------------------------------------------------------------------
    # Handlers — fractions
    # ------------------------------------------------------------------

    def _replace_fraction(self, match: re.Match[str]) -> str:
        try:
            numerator = int(match.group(1))
            denominator = int(match.group(2))
        except ValueError:
            return match.group(0)
        if denominator == 0:
            return match.group(0)
        common = COMMON_FRACTIONS.get((numerator, denominator))
        if common:
            return common
        return self._fraction_to_words(numerator, denominator)

    def _fraction_to_words(self, numerator: int, denominator: int) -> str:
        denom_form = _FRAC_DENOM_FORMS.get(denominator)
        if denom_form is None:
            return f"{self._to_words(numerator)} / {self._to_words(denominator)}"
        try:
            numer_spoken = self._to_words(numerator, gender="f")
        except TypeError:
            numer_spoken = self._to_words(numerator)
        return f"{numer_spoken} {denom_form}"

    # ------------------------------------------------------------------
    # Handlers — currency
    # ------------------------------------------------------------------

    def _replace_currency_before(self, match: re.Match[str]) -> str:
        symbol = match.group(1)
        amount_str = match.group(2).replace(",", ".")
        forms = CURRENCY_SYMBOL_FORMS.get(symbol)
        if forms is None:
            return match.group(0)
        try:
            amount_float = float(amount_str)
        except ValueError:
            return match.group(0)
        noun_form = self._currency_noun_form(int(amount_float), forms)
        if "." in amount_str and not amount_str.endswith(".0"):
            amount_spoken = self._to_words(amount_str)
        else:
            amount_spoken = self._to_words(int(amount_float))
        return f"{amount_spoken} {noun_form}"

    @staticmethod
    def _currency_noun_form(n: int, forms: tuple[str, str, str]) -> str:
        """Pick the correct Russian noun form for a currency amount.

        forms[0] — nominative singular  (1 доллар)
        forms[1] — genitive singular    (2–4 доллара)
        forms[2] — genitive plural      (5+ / 11–19 долларов)
        """
        last2 = n % 100
        last1 = n % 10
        if 11 <= last2 <= 14:
            return forms[2]
        if last1 == 1:
            return forms[0]
        if 2 <= last1 <= 4:
            return forms[1]
        return forms[2]

    # ------------------------------------------------------------------
    # Handlers — ordinals, decimals, ranges, cardinals (original logic)
    # ------------------------------------------------------------------

    def _replace_ordinal_noun(self, match: re.Match[str]) -> str:
        raw_number, noun = match.groups()
        if raw_number in {"1", "2"} and self._best_noun_parse(noun) is not None:
            return match.group(0)

        noun_config = ORDINAL_NOUN_FORMS.get(noun.lower())
        if noun_config is None:
            return match.group(0)

        number = self._parse_number_token(raw_number)
        if number is None:
            return match.group(0)

        gender, case = noun_config
        spoken = self._to_words(number, to="ordinal", gender=gender, case=case)
        return f"{spoken} {noun}"

    def _replace_ordinal(self, match: re.Match[str]) -> str:
        raw_number = match.group("number")
        separator = match.group("sep")
        suffix = match.group("suffix").lower()
        if not separator and suffix not in SAFE_NO_HYPHEN_SUFFIXES:
            return match.group(0)

        config = ORDINAL_SUFFIX_CONFIG.get(suffix)
        if config is None:
            return match.group(0)

        return self._to_words(int(raw_number), **config)

    def _replace_decimal(self, match: re.Match[str]) -> str:
        raw_number = match.group(1).replace(",", ".")
        return self._to_words(raw_number)

    def _replace_range(self, match: re.Match[str]) -> str:
        left, dash, right = match.groups()
        return f"{self._to_words(int(left))} {dash} {self._to_words(int(right))}"

    def _replace_cardinal_with_noun(self, match: re.Match[str]) -> str:
        raw_number = match.group("number")
        noun = match.group("noun")
        if self.morph is None:
            return match.group(0)

        parse = self._best_noun_parse(noun)
        if parse is None:
            return match.group(0)

        spoken = self._to_agreed_cardinal(int(raw_number), parse)
        return f"{spoken} {noun}"

    def _replace_cardinal(self, match: re.Match[str]) -> str:
        return self._to_words(int(match.group(1)))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _best_noun_parse(self, noun: str):
        if self.morph is None:
            return None
        for parse in self.morph.parse(noun):
            if "NOUN" not in parse.tag:
                continue
            if parse.normal_form not in ALLOWED_CARDINAL_NOUN_LEMMAS:
                continue
            return parse
        return None

    def _to_agreed_cardinal(self, number: int, noun_parse) -> str:
        gender = PYMORPHY_GENDER_MAP.get(noun_parse.tag.gender)
        if gender is None and "ms-f" in str(noun_parse.tag):
            gender = "f"
        case = PYMORPHY_CASE_MAP.get(noun_parse.tag.case)
        direct_form = NUMBER_WITH_NOUN_FORMS.get(number, {}).get((case or "n", gender or "m"))
        if direct_form:
            return direct_form
        kwargs: dict = {}
        if gender:
            kwargs["gender"] = gender
        if case:
            kwargs["case"] = case
        try:
            return self._to_words(number, **kwargs)
        except TypeError:
            kwargs.pop("case", None)
            return self._to_words(number, **kwargs)

    def _parse_number_token(self, raw_number: str) -> int | None:
        if raw_number.isdigit():
            return int(raw_number)
        if re.fullmatch(r"[IVXLCDM]+", raw_number, re.IGNORECASE):
            return self._roman_to_int(raw_number)
        return None

    def _roman_to_int(self, value: str) -> int:
        numerals = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
        total = 0
        previous = 0
        for char in reversed(value.upper()):
            current = numerals[char]
            if current < previous:
                total -= current
            else:
                total += current
                previous = current
        return total

    def _to_words(self, value, **kwargs) -> str:
        spoken = num2words(value, lang="ru", **kwargs)
        spoken = re.sub(r"\s+", " ", str(spoken)).strip()
        if kwargs.get("to") != "ordinal":
            spoken = re.sub(r"^одна\s+тысяча\b", "тысяча", spoken, flags=re.IGNORECASE)
        return spoken
