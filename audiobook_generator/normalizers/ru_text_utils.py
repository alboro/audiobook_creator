# Copyright (c) 2026 alboro <alboro@users.noreply.github.com>
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See: https://polyformproject.org/licenses/noncommercial/1.0.0/

from __future__ import annotations

import re
from pathlib import Path

COMBINING_ACUTE = "\u0301"
CYRILLIC_WORD_PATTERN = re.compile(r"[А-Яа-яЁё-]+")
CYRILLIC_STRESSED_WORD_PATTERN = re.compile(
    rf"[А-Яа-яЁё-]+(?:{COMBINING_ACUTE}[А-Яа-яЁё-]*)*"
)
PLUS_STRESS_PATTERN = re.compile(r"(?P<mark>\+)(?P<letter>[А-Яа-яЁё])")


def is_russian_language(language: str | None) -> bool:
    return (language or "ru").split("-")[0].lower() == "ru"


def strip_combining_acute(text: str) -> str:
    return text.replace(COMBINING_ACUTE, "")


def collapse_extra_word_stress(word: str) -> str:
    accents = [index for index, char in enumerate(word) if char == COMBINING_ACUTE]
    if len(accents) <= 1:
        return word

    keep_index = accents[-1]
    return "".join(
        char
        for index, char in enumerate(word)
        if char != COMBINING_ACUTE or index == keep_index
    )


def normalize_stress_marks(text: str) -> str:
    return CYRILLIC_STRESSED_WORD_PATTERN.sub(
        lambda match: collapse_extra_word_stress(match.group(0)),
        text,
    )


def preserve_case(source: str, replacement: str) -> str:
    if not source:
        return replacement
    if source.isupper():
        return replacement.upper()
    if source[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def load_mapping_file(path: str | None) -> dict[str, str]:
    if not path:
        return {}

    mapping: dict[str, str] = {}
    for line_number, raw_line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "==" not in line:
            raise ValueError(
                f"Invalid mapping line {line_number} in {path!r}. "
                "Expected the format 'source==replacement'."
            )
        source, replacement = line.split("==", 1)
        source = source.strip()
        replacement = replacement.strip()
        if not source:
            raise ValueError(
                f"Invalid mapping line {line_number} in {path!r}. Source must not be empty."
            )
        mapping[source] = replacement
    return mapping


def plus_stress_to_combining_acute(text: str) -> str:
    converted = PLUS_STRESS_PATTERN.sub(
        lambda match: f"{match.group('letter')}{COMBINING_ACUTE}",
        text,
    )
    return normalize_stress_marks(converted.replace("+", ""))


def load_choice_mapping_file(path: str | None) -> dict[str, tuple[str, ...]]:
    if not path:
        return {}

    mapping: dict[str, tuple[str, ...]] = {}
    for line_number, raw_line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "==" not in line:
            raise ValueError(
                f"Invalid choice-mapping line {line_number} in {path!r}. "
                "Expected the format 'source==variant1|variant2'."
            )
        source, variants = line.split("==", 1)
        source = source.strip()
        option_list = tuple(
            normalize_stress_marks(plus_stress_to_combining_acute(option.strip()))
            for option in variants.split("|")
            if option.strip()
        )
        if not source:
            raise ValueError(
                f"Invalid choice-mapping line {line_number} in {path!r}. Source must not be empty."
            )
        if len(option_list) < 2:
            raise ValueError(
                f"Invalid choice-mapping line {line_number} in {path!r}. "
                "At least two variants are required."
            )
        mapping[source] = option_list
    return mapping
