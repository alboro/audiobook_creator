from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from audiobook_generator.config.general_config import GeneralConfig
from audiobook_generator.normalizers.base_normalizer import BaseNormalizer
from audiobook_generator.normalizers.llm_support import (
    NormalizerLLMChoiceItem,
    NormalizerLLMChoiceOption,
    NormalizerLLMChoiceSelection,
    NormalizerLLMChoiceService,
)
from audiobook_generator.normalizers.pronunciation_lexicon_db import (
    PronunciationLexiconDB,
    PronunciationLexiconEntry,
    ensure_pronunciation_lexicon_db,
)
from audiobook_generator.normalizers.ru_text_utils import (
    COMBINING_ACUTE,
    is_russian_language,
    normalize_stress_marks,
    preserve_case,
    strip_combining_acute,
)

logger = logging.getLogger(__name__)

AMBIGUOUS_WORD_PATTERN = re.compile(rf"[А-Яа-яЁё{COMBINING_ACUTE}-]+")
_RUSSIAN_VOWELS = frozenset("аеёиоуыэюяАЕЁИОУЫЭЮЯ")


def _count_vowels(word: str) -> int:
    """Return the number of vowel characters (i.e. syllables) in the word."""
    return sum(1 for ch in word if ch in _RUSSIAN_VOWELS)

STRESS_AMBIGUITY_CHOICE_SYSTEM_PROMPT = """Расставь ударения в тексте для TTS.

В каждой строке — контекст предложения, где неоднозначное слово заменено вариантами:
  (N.1.форма:грамматика|N.2.форма:грамматика)
N — номер слова в пакете. Подсказка после «:» необязательна.

Ответ — ТОЛЬКО список выборов, по одному на строку, формат N.K:
  1.2
  2.1
Если контекст не позволяет однозначно выбрать — пиши N.0.
Никаких пояснений."""


def _get_stress_ambiguity_prompt(config) -> str:
    """Return the system prompt for stress ambiguity selection, from config or default."""
    custom = getattr(config, "normalize_stress_ambiguity_system_prompt", None)
    if custom and isinstance(custom, str):
        return custom
    return STRESS_AMBIGUITY_CHOICE_SYSTEM_PROMPT


@dataclass(frozen=True)
class StressAmbiguityCandidate:
    item_id: str
    start: int
    end: int
    source_text: str
    context: str
    options: tuple[NormalizerLLMChoiceOption, ...]
    lexicon_entries: tuple[PronunciationLexiconEntry, ...]

    def to_choice_item(self) -> NormalizerLLMChoiceItem:
        return NormalizerLLMChoiceItem(
            item_id=self.item_id,
            source_text=self.source_text,
            context=self.context,
            options=self.options,
            note="Выбери вариант ударения, наиболее соответствующий роли слова в предложении.",
        )


class StressAmbiguityLLMNormalizer(BaseNormalizer):
    STEP_NAME = "ru_llm_stress_ambiguity"
    STEP_VERSION = 5

    def __init__(self, config: GeneralConfig):
        if is_russian_language(config.language):
            db_path = getattr(config, "normalize_pronunciation_lexicon_db", None) or None
            self.lexicon_db = ensure_pronunciation_lexicon_db(db_path)
        else:
            self.lexicon_db = None
        self._lexicon_entry_cache: dict[str, tuple[PronunciationLexiconEntry, ...]] = {}
        self._planned_text = ""
        self._planned_candidates: dict[str, StressAmbiguityCandidate] = {}
        self._planned_order: list[str] = []
        self._planned_indices: list[list[dict]] = []   # per-batch N-index, populated by plan_processing_units
        self._last_selections: dict[str, NormalizerLLMChoiceSelection] = {}
        super().__init__(config)
        self.choice_service = NormalizerLLMChoiceService(self.get_normalizer_llm())

    def validate_config(self):
        if not self.has_normalizer_llm():
            logger.warning(
                "ru_llm_stress_ambiguity: no LLM configured — step will be skipped. "
                "Set normalize_base_url / normalize_api_key to enable it."
            )

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
            "choice_system_prompt": _get_stress_ambiguity_prompt(self.config),
            "pronunciation_lexicon_db": str(self.lexicon_db.path) if self.lexicon_db else None,
            "pronunciation_lexicon_sources": (
                self._load_built_sources(self.lexicon_db) if self.lexicon_db else []
            ),
            "pronunciation_lexicon_stats": (
                self.lexicon_db.get_stats() if self.lexicon_db else None
            ),
        }

    def normalize(self, text: str, chapter_title: str = "") -> str:
        if not self.has_normalizer_llm():
            logger.info(
                "ru_llm_stress_ambiguity skipped for chapter '%s': no LLM configured",
                chapter_title,
            )
            return text
        if not is_russian_language(self.config.language):
            logger.info(
                "ru_llm_stress_ambiguity skipped for chapter '%s' because language is '%s'",
                chapter_title,
                self.config.language,
            )
            return text

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
        return self.merge_processed_units(processed_units, chapter_title=chapter_title)

    def plan_processing_units(self, text: str, chapter_title: str = "") -> list[str]:
        if not self.has_normalizer_llm():
            self._planned_text = text
            self._planned_candidates = {}
            self._planned_order = []
            self._planned_indices = []
            return []
        if not is_russian_language(self.config.language):
            self._planned_text = text
            self._planned_candidates = {}
            self._planned_order = []
            self._planned_indices = []
            return []

        candidates = self._collect_candidates(text)
        self._planned_text = text
        self._planned_candidates = {candidate.item_id: candidate for candidate in candidates}
        self._planned_order = [candidate.item_id for candidate in candidates]
        self._last_selections = {}
        batches = self.choice_service.plan_batches(
            [candidate.to_choice_item() for candidate in candidates],
            system_prompt=_get_stress_ambiguity_prompt(self.config),
        )
        units: list[str] = []
        self._planned_indices = []
        for batch in batches:
            prompt_text, index = self._render_compact_prompt(batch)
            self._planned_indices.append(index)
            units.append(json.dumps({"prompt": prompt_text, "index": index}, ensure_ascii=False))
        return units

    def process_unit(
        self,
        unit: str,
        *,
        chapter_title: str = "",
        unit_index: int,
        unit_count: int,
    ) -> str:
        data = json.loads(unit)
        prompt_text = data["prompt"]
        logger.info(
            "Choosing stress ambiguities for chapter '%s' batch %s/%s",
            chapter_title,
            unit_index,
            unit_count,
        )
        return self.choice_service.llm.complete(
            user_prompt=prompt_text,
            system_prompt=_get_stress_ambiguity_prompt(self.config),
            model=self.get_normalizer_model(),
            temperature=0,
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
        for i, raw_response in enumerate(processed_units):
            if not raw_response.strip():
                continue
            index = self._planned_indices[i] if i < len(self._planned_indices) else []
            selections.update(self._parse_compact_response(raw_response, index))

        # Fill fallbacks for items the LLM skipped
        for item_id in self._planned_order:
            if item_id not in selections:
                selections[item_id] = NormalizerLLMChoiceSelection(
                    item_id=item_id,
                    option_id="original",
                    source="fallback",
                )

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
            "stress_ambiguity_llm applied to chapter '%s': %s replacements",
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
                "lexicon_entries": [
                    self._entry_to_payload(entry)
                    for entry in candidate.lexicon_entries
                ],
            }
            for candidate in candidates
        ]
        return {
            "00_choice_system_prompt.txt": _get_stress_ambiguity_prompt(self.config),
            "01_choice_settings.json": self.choice_service.render_settings_json(
                system_prompt=_get_stress_ambiguity_prompt(self.config),
                model=self.get_normalizer_model(),
            ),
            "02_candidates.json": json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            "03_pronunciation_lexicon.json": json.dumps(
                {
                    "db_path": str(self.lexicon_db.path) if self.lexicon_db else None,
                    "built_sources": self._load_built_sources(self.lexicon_db)
                    if self.lexicon_db
                    else [],
                    "stats": self.lexicon_db.get_stats() if self.lexicon_db else {},
                    "legacy_stress_ambiguity_file_ignored": bool(
                        getattr(self.config, "normalize_stress_ambiguity_file", None)
                    ),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
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
            case_lines.append("lexicon_entries:")
            for entry in candidate.lexicon_entries:
                case_lines.append(
                    "  - spoken_form: {spoken}, lemma: {lemma}, pos: {pos}, grammemes: {grammemes}, "
                    "is_proper_name: {is_proper_name}, source: {source}, confidence: {confidence}".format(
                        spoken=entry.spoken_form or "",
                        lemma=entry.lemma or "",
                        pos=entry.pos or "",
                        grammemes=entry.grammemes or "",
                        is_proper_name=str(entry.is_proper_name).lower(),
                        source=entry.source,
                        confidence="" if entry.confidence is None else entry.confidence,
                    )
                )
            case_lines.append("")

        stats["selection_counts"] = option_counts
        stats["selection_source_counts"] = source_counts

        report_lines = [
            "# stress_ambiguity_llm selection report",
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
        data = json.loads(unit)
        return {
            "00_choice_system_prompt.txt": _get_stress_ambiguity_prompt(self.config),
            "input.txt": data["prompt"],
        }

    def _collect_candidates(self, text: str) -> list[StressAmbiguityCandidate]:
        if not self.lexicon_db:
            return []

        from audiobook_generator.normalizers.ru_tts_stress_paradox_guard import get_paradox_guard
        paradox_guard = get_paradox_guard(self.config)

        candidates: list[StressAmbiguityCandidate] = []
        item_index = 1
        for match in AMBIGUOUS_WORD_PATTERN.finditer(text):
            source_text = match.group(0)
            if COMBINING_ACUTE in source_text:
                continue

            key = strip_combining_acute(source_text).lower()

            # Skip monosyllabic words — stress position is unambiguous (only one vowel)
            if _count_vowels(key) <= 1:
                logger.debug("Skipping monosyllabic word '%s'", source_text)
                continue

            # Skip words that are known to be mispronounced when stressed
            if paradox_guard.is_paradox_word(key):
                logger.debug("Stress paradox guard: skipping '%s'", source_text)
                continue

            lexicon_entries = self._lookup_ambiguous_entries(key)
            if not lexicon_entries:
                continue

            options = self._build_options(source_text, lexicon_entries)
            if len(options) < 2:
                continue

            item_id = f"stress_ambiguity_{item_index:04d}"
            item_index += 1
            candidates.append(
                StressAmbiguityCandidate(
                    item_id=item_id,
                    start=match.start(),
                    end=match.end(),
                    source_text=source_text,
                    context=self._extract_context(text, match.start(), match.end()),
                    options=options,
                    lexicon_entries=lexicon_entries,
                )
            )
        return candidates

    def _lookup_ambiguous_entries(
        self,
        key: str,
    ) -> tuple[PronunciationLexiconEntry, ...]:
        if not self.lexicon_db:
            return ()

        cached = self._lexicon_entry_cache.get(key)
        if cached is not None:
            return cached

        entries = self.lexicon_db.lookup_ambiguous_entries(key)
        self._lexicon_entry_cache[key] = entries
        return entries

    def _build_options(
        self,
        source_text: str,
        lexicon_entries: tuple[PronunciationLexiconEntry, ...],
    ) -> tuple[NormalizerLLMChoiceOption, ...]:
        options: list[NormalizerLLMChoiceOption] = [
            NormalizerLLMChoiceOption("original", source_text)
        ]
        seen_texts = {source_text}

        # Build spoken_form → set of grammeme labels for LLM hints
        form_to_grammemes: dict[str, list[str]] = {}
        for entry in lexicon_entries:
            if entry.spoken_form and entry.grammemes and entry.grammemes != "canonical":
                form_to_grammemes.setdefault(entry.spoken_form, []).append(entry.grammemes)

        unique_spoken_forms = sorted(
            {
                entry.spoken_form
                for entry in lexicon_entries
                if entry.spoken_form
            }
        )
        for index, spoken_form in enumerate(unique_spoken_forms, start=1):
            preserved = normalize_stress_marks(
                preserve_case(strip_combining_acute(source_text), spoken_form)
            )
            if not preserved or preserved in seen_texts:
                continue
            # Discard variants that are capitalised when the source starts lowercase —
            # these come from proper-name entries (e.g. "Ве́ры" for lowercase "веры").
            if source_text and source_text[0].islower() and preserved[0].isupper():
                continue

            # Compose a compact grammatical hint for the LLM
            grammeme_labels = form_to_grammemes.get(spoken_form, [])
            unique_labels = list(dict.fromkeys(grammeme_labels))  # deduplicated, order-preserved
            hint: str | None = "; ".join(unique_labels) if unique_labels else None

            options.append(
                NormalizerLLMChoiceOption(
                    option_id=f"variant_{index}",
                    text=preserved,
                    hint=hint,
                )
            )
            seen_texts.add(preserved)
        return tuple(options)

    @staticmethod
    def _render_compact_prompt(
        items: list[NormalizerLLMChoiceItem],
    ) -> tuple[str, list[dict]]:
        """Render compact inline prompt and index for response parsing.

        Each item's source_text is replaced inline in its context sentence:
          Слово (1.1.форма:грамматика|1.2.форма).

        Multiple items from the same sentence appear on the same line.

        Returns:
            prompt_text: the text to send to LLM
            index: list of {"num": N, "item_id": ..., "options": {"1": id, ...}}
        """
        # Group items by context sentence, preserving insertion order
        ctx_order: list[str] = []
        ctx_items: dict[str, list[tuple[int, NormalizerLLMChoiceItem]]] = {}
        for i, item in enumerate(items, start=1):
            ctx = item.context
            if ctx not in ctx_items:
                ctx_order.append(ctx)
                ctx_items[ctx] = []
            ctx_items[ctx].append((i, item))

        lines: list[str] = []
        index: list[dict] = []

        for ctx in ctx_order:
            line = ctx
            for num, item in ctx_items[ctx]:
                opt_map: dict[str, str] = {"0": "original"}
                parts: list[str] = []
                k = 1
                for option in item.options:
                    if option.option_id == "original":
                        continue
                    opt_map[str(k)] = option.option_id
                    part = f"{num}.{k}.{option.text}"
                    if option.hint:
                        part += f":{option.hint}"
                    parts.append(part)
                    k += 1
                inline = "(" + "|".join(parts) + ")"
                if item.source_text in line:
                    line = line.replace(item.source_text, inline, 1)
                else:
                    line += f" [{item.source_text}→{inline}]"
                index.append({"num": num, "item_id": item.item_id, "options": opt_map})
            lines.append(line)

        return "\n".join(lines), index

    @staticmethod
    def _parse_compact_response(
        response_text: str,
        index: list[dict],
    ) -> dict[str, NormalizerLLMChoiceSelection]:
        """Parse compact ``N.K`` response lines into item_id → selection."""
        num_to_entry = {entry["num"]: entry for entry in index}
        result: dict[str, NormalizerLLMChoiceSelection] = {}
        for line in response_text.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            m = re.search(r'\b(\d+)\.(\d+)\b', line)
            if not m:
                continue
            item_num = int(m.group(1))
            opt_key = m.group(2)
            entry = num_to_entry.get(item_num)
            if not entry:
                logger.warning("Compact response: unknown item number %d", item_num)
                continue
            option_id = entry["options"].get(opt_key)
            if option_id is None:
                logger.warning(
                    "Compact response: unknown option %s for item %d, using original",
                    opt_key, item_num,
                )
                option_id = "original"
            result[entry["item_id"]] = NormalizerLLMChoiceSelection(
                item_id=entry["item_id"],
                option_id=option_id,
                source="llm",
            )
        return result

    def _extract_context(self, text: str, start: int, end: int) -> str:
        left = start
        while left > 0 and text[left - 1] not in ".!?\n":
            left -= 1
        right = end
        while right < len(text) and text[right] not in ".!?\n":
            right += 1
        return text[left:right].strip()

    @staticmethod
    def _resolve_selected_text(
        candidate: StressAmbiguityCandidate,
        selection: NormalizerLLMChoiceSelection,
    ) -> str:
        if selection.has_custom_text:
            return selection.custom_text or candidate.source_text
        for option in candidate.options:
            if option.option_id == selection.resolved_option_id():
                return option.text
        return candidate.source_text

    @staticmethod
    def _load_built_sources(lexicon_db: PronunciationLexiconDB | None) -> list[str]:
        if not lexicon_db:
            return []
        return json.loads(lexicon_db.get_metadata("built_sources") or "[]")

    @staticmethod
    def _entry_to_payload(entry: PronunciationLexiconEntry) -> dict[str, object]:
        return {
            "surface_form": entry.surface_form,
            "spoken_form": entry.spoken_form,
            "lemma": entry.lemma,
            "pos": entry.pos,
            "grammemes": entry.grammemes,
            "is_proper_name": entry.is_proper_name,
            "source": entry.source,
            "confidence": entry.confidence,
        }
