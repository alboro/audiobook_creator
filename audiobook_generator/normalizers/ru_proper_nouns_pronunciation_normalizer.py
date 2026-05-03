from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from audiobook_generator.config.general_config import GeneralConfig
from audiobook_generator.normalizers.base_normalizer import BaseNormalizer
from audiobook_generator.normalizers.llm_support import (
    DEFAULT_CHOICE_SYSTEM_PROMPT,
    NormalizerLLMChoiceItem,
    NormalizerLLMChoiceOption,
    NormalizerLLMChoiceSelection,
    NormalizerLLMChoiceService,
)
from audiobook_generator.normalizers.ru_text_utils import (
    COMBINING_ACUTE,
    CYRILLIC_WORD_PATTERN,
    is_russian_language,
    load_mapping_file,
    normalize_stress_marks,
    preserve_case,
    strip_combining_acute,
)
from audiobook_generator.normalizers.tsnorm_support import (
    create_tsnorm_backend,
    load_tsnorm_backend,
)

logger = logging.getLogger(__name__)

NAME_TOKEN = rf"[А-ЯЁ][А-ЯЁа-яё{COMBINING_ACUTE}]*(?:-[А-ЯЁ][А-ЯЁа-яё{COMBINING_ACUTE}]*)*"
MULTIWORD_NAME_PATTERN = re.compile(
    rf"(?<!\w)(?P<value>{NAME_TOKEN}(?:\s+{NAME_TOKEN}){{1,3}})(?!\w)"
)
QUOTED_NAME_PATTERN = re.compile(
    rf"[\"«](?P<value>{NAME_TOKEN}(?:\s+{NAME_TOKEN}){{0,2}})[\"»]"
)
INITIALS_WITH_SURNAME_PATTERN = re.compile(
    rf"(?<!\w)(?P<value>[А-ЯЁ](?:-[А-ЯЁа-яё{COMBINING_ACUTE}]+){{2,}})(?!\w)"
)
LEADING_SENTENCE_WORDS = {
    "А",
    "Без",
    "Было",
    "В",
    "Ведь",
    "Вместо",
    "Во",
    "Вот",
    "Да",
    "Даже",
    "Если",
    "И",
    "Или",
    "Именно",
    "Итак",
    "Как",
    "Когда",
    "Конечно",
    "Которых",
    "Но",
    "Однако",
    "Они",
    "Он",
    "Она",
    "После",
    "Потом",
    "При",
    "С",
    "Так",
    "Также",
    "Там",
    "Тогда",
    "Уже",
    "Хотя",
    "Это",
}

PROPER_NOUN_CHOICE_SYSTEM_PROMPT = (
    DEFAULT_CHOICE_SYSTEM_PROMPT
    + """

Additional rules for this task:
- The target language is Russian text-to-speech.
- Focus on proper names, surnames, places, titles, and named entities.
- Prefer the option that will sound most natural when read aloud by a Russian TTS model.
- If stress marks help pronunciation, prefer the correctly stressed option.
- If a foreign name is better pronounced with a Russian-friendly spelling such as "Пэйн" instead of "Пейн", prefer that option.
- Do not rewrite surrounding context. Only choose among the provided options for the highlighted source text."""
)

BUILTIN_PROPER_NOUN_PRONUNCIATION_HINTS = {
    "пейн": "пэйн",
    "пейна": "пэйна",
    "пейну": "пэйну",
    "пейном": "пэйном",
    "пейне": "пэйне",
    "пейны": "пэйны",
    "пейнов": "пэйнов",
    "пейнам": "пэйнам",
    "пейнами": "пэйнами",
    "пейнах": "пэйнах",
}
BUILTIN_PROPER_NOUN_STRESS_HINTS = {
    "толстой": f"толсто{COMBINING_ACUTE}й",
    "толстого": f"толсто{COMBINING_ACUTE}го",
    "толстому": f"толсто{COMBINING_ACUTE}му",
    "толстым": f"толсты{COMBINING_ACUTE}м",
    "толстом": f"толсто{COMBINING_ACUTE}м",
    "толстая": f"толста{COMBINING_ACUTE}я",
    "лев": "лев",
    "лева": "льва",
    "леву": "льву",
    "левом": "львом",
    "леве": "льве",
    "томас": f"то{COMBINING_ACUTE}мас",
    "томаса": f"то{COMBINING_ACUTE}маса",
    "томасу": f"то{COMBINING_ACUTE}масу",
    "томасом": f"то{COMBINING_ACUTE}масом",
    "томасе": f"то{COMBINING_ACUTE}масе",
}


@dataclass(frozen=True)
class ProperNounCandidate:
    item_id: str
    start: int
    end: int
    source_text: str
    context: str
    options: tuple[NormalizerLLMChoiceOption, ...]

    def to_choice_item(self) -> NormalizerLLMChoiceItem:
        return NormalizerLLMChoiceItem(
            item_id=self.item_id,
            source_text=self.source_text,
            context=self.context,
            options=self.options,
            note="Choose the most natural Russian TTS pronunciation for this named entity.",
        )


class ProperNounsPronunciationRuNormalizer(BaseNormalizer):
    STEP_NAME = "ru_llm_proper_nouns_pronunciation"
    STEP_VERSION = 3  # bumped: builtin hints now applied without LLM + paradox guard integrated

    def __init__(self, config: GeneralConfig):
        self.backend = None
        self.pronunciation_overrides = BUILTIN_PROPER_NOUN_PRONUNCIATION_HINTS.copy()
        self.stress_overrides = BUILTIN_PROPER_NOUN_STRESS_HINTS.copy()
        self.pronunciation_overrides.update(
            {
                strip_combining_acute(source).lower(): replacement
                for source, replacement in load_mapping_file(
                    config.normalize_pronunciation_exceptions_file
                ).items()
            }
        )
        self.stress_overrides.update(
            {
                strip_combining_acute(source).lower(): replacement
                for source, replacement in load_mapping_file(
                    getattr(config, "normalize_stress_exceptions_file", None)
                ).items()
            }
        )
        self._planned_text = ""
        self._planned_candidates: dict[str, ProperNounCandidate] = {}
        self._planned_order: list[str] = []
        self._last_selections: dict[str, NormalizerLLMChoiceSelection] = {}
        super().__init__(config)
        self.backend = create_tsnorm_backend(
            stress_mark=COMBINING_ACUTE,
            stress_mark_pos="after",
            stress_yo=True,
            stress_monosyllabic=False,
            min_word_len=config.normalize_tsnorm_min_word_length or 2,
        )
        self.choice_service = NormalizerLLMChoiceService(self.get_normalizer_llm())

    def validate_config(self):
        if not self.has_normalizer_llm():
            logger.warning(
                "ru_proper_nouns_pronunciation: no LLM configured — step will be skipped. "
                "Set normalize_base_url / normalize_api_key to enable it."
            )
            return
        try:
            load_tsnorm_backend()
        except ImportError as exc:
            raise ImportError(
                "proper_nouns_pronunciation_ru requires the 'tsnorm' package. "
                "Install dependencies in a Python 3.10-3.12 environment."
            ) from exc

    def supports_chunked_resume(self) -> bool:
        return True

    def get_resume_signature(self) -> dict:
        llm = self.get_normalizer_llm()
        return {
            **super().get_resume_signature(),
            "provider": llm.settings.provider,
            "model": llm.settings.model,
            "base_url": llm.settings.base_url,
            "max_chars": llm.settings.max_chars,
            "choice_system_prompt": PROPER_NOUN_CHOICE_SYSTEM_PROMPT,
            "pronunciation_overrides": sorted(self.pronunciation_overrides.items()),
            "stress_overrides": sorted(self.stress_overrides.items()),
            "min_word_len": self.config.normalize_tsnorm_min_word_length or 2,
            "paradox_words": getattr(self.config, "normalize_stress_paradox_words", None) or "",
        }

    def _apply_builtin_hints(self, text: str) -> str:
        """Apply pronunciation and stress overrides deterministically (no LLM required).

        Words that are present in the builtin stress/pronunciation hint tables are
        corrected regardless of any stress marks that upstream normalizers may have
        placed on them.  All other words are left untouched (including their accents).

        Also applies ``normalize_stress_paradox_words`` overrides from config so that
        user-configured paradox words are enforced in the output even without LLM.
        """
        # Pattern that includes combining diacritics so we can match e.g. "То́лстым" whole.
        word_re = re.compile(
            rf"[А-ЯЁа-яё][А-ЯЁа-яё{re.escape(COMBINING_ACUTE)}]*",
            re.UNICODE,
        )

        def replace(match: re.Match[str]) -> str:
            word = match.group(0)
            bare = strip_combining_acute(word)
            key = bare.lower()
            # Check pronunciation override first (replaces the whole word)
            if key in self.pronunciation_overrides:
                replacement = self.pronunciation_overrides[key]
                return preserve_case(bare, replacement)
            # Then stress override
            if key in self.stress_overrides:
                replacement = self.stress_overrides[key]
                return preserve_case(bare, normalize_stress_marks(replacement))
            return word  # not in any hint table — keep as-is

        result = word_re.sub(replace, text)

        # Apply user-configured paradox overrides (normalize_stress_paradox_words).
        from audiobook_generator.normalizers.ru_tts_stress_paradox_guard import get_paradox_guard
        guard = get_paradox_guard(self.config)
        return guard.apply_paradox_overrides(result)

    def normalize(self, text: str, chapter_title: str = "") -> str:
        if not is_russian_language(self.config.language):
            logger.info(
                "proper_nouns_pronunciation_ru skipped for chapter '%s' because language is '%s'",
                chapter_title,
                self.config.language,
            )
            return text

        if not self.has_normalizer_llm():
            logger.info(
                "ru_proper_nouns_pronunciation: builtin hints applied to chapter '%s', "
                "LLM step skipped (no LLM configured)",
                chapter_title,
            )
            # Apply deterministic hints even without LLM.
            return self._apply_builtin_hints(text)

        units = self.plan_processing_units(text, chapter_title=chapter_title)
        processed_units = [
            self.process_unit(
                unit,
                chapter_title=chapter_title,
                unit_index=index,
                unit_count=len(units),
            )
            for index, unit in enumerate(units, start=1)
        ]
        result = self.merge_processed_units(processed_units, chapter_title=chapter_title)
        # Always enforce builtin hints as a final override over LLM choices.
        return self._apply_builtin_hints(result)

    def plan_processing_units(self, text: str, chapter_title: str = "") -> list[str]:
        if not self.has_normalizer_llm():
            self._planned_text = text
            self._planned_candidates = {}
            self._planned_order = []
            return []
        if not is_russian_language(self.config.language):
            self._planned_text = text
            self._planned_candidates = {}
            self._planned_order = []
            return []

        candidates = self._collect_candidates(text)
        self._planned_text = text
        self._planned_candidates = {candidate.item_id: candidate for candidate in candidates}
        self._planned_order = [candidate.item_id for candidate in candidates]
        self._last_selections = {}
        batches = self.choice_service.plan_batches(
            [candidate.to_choice_item() for candidate in candidates],
            system_prompt=PROPER_NOUN_CHOICE_SYSTEM_PROMPT,
        )
        return [self._serialize_batch(batch) for batch in batches]

    def process_unit(
        self,
        unit: str,
        *,
        chapter_title: str = "",
        unit_index: int,
        unit_count: int,
    ) -> str:
        batch = self._deserialize_batch(unit)
        logger.info(
            "Choosing proper-name pronunciations for chapter '%s' batch %s/%s, items=%s",
            chapter_title,
            unit_index,
            unit_count,
            len(batch),
        )
        selections = self.choice_service.choose_batch(
            batch,
            target_language=self.config.language,
            system_prompt=PROPER_NOUN_CHOICE_SYSTEM_PROMPT,
            model=self.get_normalizer_model(),
            temperature=0,
        )
        normalized_selections = {
            item_id: self._coerce_selection(item_id, selection)
            for item_id, selection in selections.items()
        }
        return json.dumps(
            {
                "selections": [
                    {
                        "id": item_id,
                        "option_id": selection.option_id,
                        "custom_text": selection.custom_text or "",
                        "cacheable": bool(selection.cacheable),
                        "reason": selection.reason or "",
                        "source": selection.source,
                    }
                    for item_id, selection in normalized_selections.items()
                ]
            },
            ensure_ascii=False,
            indent=2,
        )

    def merge_processed_units(
        self,
        processed_units: list[str],
        *,
        chapter_title: str = "",
    ) -> str:
        if not self._planned_candidates:
            return self._planned_text

        selections: dict[str, NormalizerLLMChoiceSelection] = {}
        for processed in processed_units:
            if not processed.strip():
                continue
            selections.update(self.choice_service.parse_choice_response_objects(processed))

        normalized = self._planned_text
        replacements = 0
        self._last_selections = selections
        for candidate in sorted(
            self._planned_candidates.values(),
            key=lambda item: item.start,
            reverse=True,
        ):
            selection = selections.get(
                candidate.item_id,
                NormalizerLLMChoiceSelection(item_id=candidate.item_id, option_id="original"),
            )
            replacement_text = normalize_stress_marks(
                self._resolve_selected_text(candidate, selection)
            )
            if replacement_text == candidate.source_text:
                continue
            normalized = (
                normalized[: candidate.start]
                + replacement_text
                + normalized[candidate.end :]
            )
            replacements += 1

        logger.info(
            "proper_nouns_pronunciation_ru applied to chapter '%s': %s replacements",
            chapter_title,
            replacements,
        )
        return normalized

    def get_step_artifacts(self, text: str, chapter_title: str = "") -> dict[str, str]:
        candidates = self._collect_candidates(text)
        manifest = [
            {
                "id": candidate.item_id,
                "source_text": candidate.source_text,
                "context": candidate.context,
                "options": [
                    {"id": option.option_id, "text": option.text}
                    for option in candidate.options
                ],
            }
            for candidate in candidates
        ]
        return {
            "00_choice_system_prompt.txt": PROPER_NOUN_CHOICE_SYSTEM_PROMPT,
            "01_choice_settings.json": self.choice_service.render_settings_json(
                system_prompt=PROPER_NOUN_CHOICE_SYSTEM_PROMPT,
                model=self.get_normalizer_model(),
            ),
            "02_candidates.json": json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        }

    def get_post_step_artifacts(
        self,
        *,
        input_text: str,
        output_text: str,
        chapter_title: str = "",
    ) -> dict[str, str]:
        if not self._planned_candidates:
            return {}

        case_lines: list[str] = []
        stats = {
            "chapter_title": chapter_title,
            "total_candidates": len(self._planned_candidates),
            "changed_candidates": 0,
            "selection_counts": {},
            "selection_source_counts": {},
        }

        option_counts: dict[str, int] = {}
        source_counts: dict[str, int] = {}
        for candidate_id in self._planned_order:
            candidate = self._planned_candidates[candidate_id]
            selection = self._last_selections.get(
                candidate_id,
                NormalizerLLMChoiceSelection(item_id=candidate_id, option_id="original"),
            )
            resolved = self._resolve_selected_text(candidate, selection)
            option_id = selection.option_id or ("custom" if selection.has_custom_text else "original")
            option_counts[option_id] = option_counts.get(option_id, 0) + 1
            source_counts[selection.source] = source_counts.get(selection.source, 0) + 1

            changed = resolved != candidate.source_text
            if changed:
                stats["changed_candidates"] += 1

            case_lines.extend(
                [
                    f"id: {candidate.item_id}",
                    f"changed: {'yes' if changed else 'no'}",
                    f"source_text: {candidate.source_text}",
                    f"selected_option: {option_id}",
                    f"selected_source: {selection.source}",
                    f"selected_text: {resolved}",
                    f"cacheable: {selection.cacheable}",
                    f"reason: {selection.reason or ''}",
                    f"context: {candidate.context}",
                    "options:",
                ]
            )
            for option in candidate.options:
                case_lines.append(f"  - {option.option_id}: {option.text}")
            if selection.has_custom_text:
                case_lines.append(f"  - custom: {selection.custom_text}")
            case_lines.append("")

        stats["selection_counts"] = option_counts
        stats["selection_source_counts"] = source_counts

        report_lines = [
            "# proper_nouns_pronunciation_ru selection report",
            "",
            f"- chapter_title: {chapter_title}",
            f"- total_candidates: {stats['total_candidates']}",
            f"- changed_candidates: {stats['changed_candidates']}",
            f"- selection_counts: {json.dumps(stats['selection_counts'], ensure_ascii=False, sort_keys=True)}",
            f"- selection_source_counts: {json.dumps(stats['selection_source_counts'], ensure_ascii=False, sort_keys=True)}",
            "",
            "## Cases",
            "",
        ]
        if case_lines:
            report_lines.extend(case_lines)
        else:
            report_lines.append("No cases.")
            report_lines.append("")

        return {
            "92_selection_report.txt": "\n".join(report_lines),
            "93_selection_stats.json": json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
        }

    def get_unit_artifacts(
        self,
        unit: str,
        *,
        chapter_title: str = "",
        unit_index: int,
        unit_count: int,
    ) -> dict[str, str]:
        batch = self._deserialize_batch(unit)
        return {
            "00_choice_system_prompt.txt": PROPER_NOUN_CHOICE_SYSTEM_PROMPT,
            "01_choice_user_prompt.txt": self.choice_service.render_user_prompt(
                batch,
                target_language=self.config.language,
            ),
        }

    def _collect_candidates(self, text: str) -> list[ProperNounCandidate]:
        candidates: list[ProperNounCandidate] = []
        occupied: list[tuple[int, int]] = []
        item_index = 1

        for pattern in (MULTIWORD_NAME_PATTERN, QUOTED_NAME_PATTERN, INITIALS_WITH_SURNAME_PATTERN):
            for match in pattern.finditer(text):
                start, end = match.span("value")
                source_text, start, end = self._normalize_candidate_span(
                    text,
                    source_text=match.group("value"),
                    start=start,
                    end=end,
                )
                if not source_text:
                    continue
                if self._overlaps_existing(start, end, occupied):
                    continue
                options = self._build_options(source_text)
                if len(options) < 2:
                    continue
                context = self._extract_context(text, start, end)
                item_id = f"proper_name_{item_index:04d}"
                item_index += 1
                candidates.append(
                    ProperNounCandidate(
                        item_id=item_id,
                        start=start,
                        end=end,
                        source_text=source_text,
                        context=context,
                        options=options,
                    )
                )
                occupied.append((start, end))

        return candidates

    def _build_options(self, source_text: str) -> tuple[NormalizerLLMChoiceOption, ...]:
        candidates = [
            ("original", source_text),
            ("accented", self._accent_phrase(source_text)),
            ("phonetic", self._apply_pronunciation_overrides(source_text)),
        ]
        phonetic = candidates[-1][1]
        candidates.append(("phonetic_accented", self._accent_phrase(phonetic)))
        candidates.append(("guided", self._apply_guided_name_hints(source_text)))
        guided = candidates[-1][1]
        candidates.append(("guided_accented", self._accent_phrase(guided)))

        options: list[NormalizerLLMChoiceOption] = []
        seen_texts: set[str] = set()
        for option_id, option_text in candidates:
            normalized_option_text = normalize_stress_marks(option_text.strip())
            if not normalized_option_text or normalized_option_text in seen_texts:
                continue
            options.append(
                NormalizerLLMChoiceOption(
                    option_id=option_id,
                    text=normalized_option_text,
                )
            )
            seen_texts.add(normalized_option_text)

        return tuple(options)

    def _accent_phrase(self, phrase: str) -> str:
        def replace_word(match: re.Match[str]) -> str:
            word = match.group(0)
            if COMBINING_ACUTE in word or word.isupper():
                return word
            accented = self._accentuate_word(word)
            return accented or word

        return normalize_stress_marks(CYRILLIC_WORD_PATTERN.sub(replace_word, phrase))

    def _apply_pronunciation_overrides(self, phrase: str) -> str:
        def replace_word(match: re.Match[str]) -> str:
            word = match.group(0)
            key = strip_combining_acute(word).lower()
            replacement = self.pronunciation_overrides.get(key)
            if not replacement:
                return word
            return preserve_case(strip_combining_acute(word), replacement)

        return normalize_stress_marks(CYRILLIC_WORD_PATTERN.sub(replace_word, phrase))

    def _apply_guided_name_hints(self, phrase: str) -> str:
        pronounced = self._apply_pronunciation_overrides(phrase)

        def replace_word(match: re.Match[str]) -> str:
            word = match.group(0)
            key = strip_combining_acute(word).lower()
            replacement = self.stress_overrides.get(key)
            if not replacement:
                return word
            return preserve_case(strip_combining_acute(word), replacement)

        return normalize_stress_marks(CYRILLIC_WORD_PATTERN.sub(replace_word, pronounced))

    def _accentuate_word(self, word: str) -> str:
        if callable(self.backend):
            return self.backend(word)
        if hasattr(self.backend, "normalize"):
            return self.backend.normalize(word)
        return word

    def _extract_context(self, text: str, start: int, end: int) -> str:
        left = start
        while left > 0 and text[left - 1] not in ".!?\n":
            left -= 1
        right = end
        while right < len(text) and text[right] not in ".!?\n":
            right += 1
        return text[left:right].strip()

    def _normalize_candidate_span(
        self,
        text: str,
        *,
        source_text: str,
        start: int,
        end: int,
    ) -> tuple[str, int, int]:
        if not source_text or " " not in source_text:
            return source_text, start, end

        sentence_start = self._is_sentence_start(text, start)
        if not sentence_start:
            return source_text, start, end

        words = source_text.split()
        if len(words) < 2:
            return source_text, start, end

        leading_word = strip_combining_acute(words[0]).strip()
        if leading_word not in LEADING_SENTENCE_WORDS:
            return source_text, start, end

        trimmed_text = " ".join(words[1:]).strip()
        if not trimmed_text:
            return "", start, end
        shift = source_text.find(trimmed_text)
        if shift < 0:
            return source_text, start, end
        return trimmed_text, start + shift, end

    @staticmethod
    def _is_sentence_start(text: str, start: int) -> bool:
        idx = start - 1
        while idx >= 0 and text[idx].isspace():
            idx -= 1
        if idx < 0:
            return True
        return text[idx] in ".!?\n"

    @staticmethod
    def _resolve_selected_text(
        candidate: ProperNounCandidate,
        selection: NormalizerLLMChoiceSelection,
    ) -> str:
        if selection.has_custom_text:
            return selection.custom_text or candidate.source_text
        for option in candidate.options:
            if option.option_id == selection.resolved_option_id():
                return option.text
        return candidate.source_text

    @staticmethod
    def _overlaps_existing(start: int, end: int, occupied: list[tuple[int, int]]) -> bool:
        for existing_start, existing_end in occupied:
            if start < existing_end and end > existing_start:
                return True
        return False

    @staticmethod
    def _serialize_batch(batch: list[NormalizerLLMChoiceItem]) -> str:
        return json.dumps(
            {
                "items": [
                    {
                        "id": item.item_id,
                        "source_text": item.source_text,
                        "context": item.context,
                        "note": item.note or "",
                        "options": [
                            {"id": option.option_id, "text": option.text}
                            for option in item.options
                        ],
                    }
                    for item in batch
                ]
            },
            ensure_ascii=False,
            indent=2,
        )

    @staticmethod
    def _deserialize_batch(serialized_batch: str) -> list[NormalizerLLMChoiceItem]:
        payload = json.loads(serialized_batch)
        items: list[NormalizerLLMChoiceItem] = []
        for raw_item in payload.get("items", []):
            items.append(
                NormalizerLLMChoiceItem(
                    item_id=raw_item["id"],
                    source_text=raw_item["source_text"],
                    context=raw_item["context"],
                    note=raw_item.get("note") or None,
                    options=tuple(
                        NormalizerLLMChoiceOption(
                            option_id=raw_option["id"],
                            text=raw_option["text"],
                        )
                        for raw_option in raw_item.get("options", [])
                    ),
                )
            )
        return items

    @staticmethod
    def _coerce_selection(
        item_id: str,
        selection: NormalizerLLMChoiceSelection | str,
    ) -> NormalizerLLMChoiceSelection:
        if isinstance(selection, NormalizerLLMChoiceSelection):
            return selection
        return NormalizerLLMChoiceSelection(item_id=item_id, option_id=str(selection))
