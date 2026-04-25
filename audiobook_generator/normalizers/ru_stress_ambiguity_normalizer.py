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

_DEFAULT_STRESS_AMBIGUITY_EXTRA = """

Additional rules for this task:
- The target language is Russian text-to-speech.
- Focus only on ambiguous Russian word stress or pronunciation inside the highlighted source_text.
- The provided options come from a pronunciation lexicon and represent valid spoken variants for the same written word form.
- Choose the option that best matches the local sentence context.
- Prefer adding a stress mark only when it genuinely helps the Russian TTS model avoid the wrong reading.
- Leave the original option if the context is insufficient or the pronunciation is not clearly determined.
- Set cacheable to true only if the best choice is stable for the same source_text regardless of wider context.
- Do not rewrite surrounding context. Only choose among the provided options for the highlighted source_text unless a clearly better custom_text is necessary."""

STRESS_AMBIGUITY_CHOICE_SYSTEM_PROMPT = DEFAULT_CHOICE_SYSTEM_PROMPT + _DEFAULT_STRESS_AMBIGUITY_EXTRA


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
            note="Choose the stress or pronunciation variant that best fits this sentence.",
        )


class StressAmbiguityLLMNormalizer(BaseNormalizer):
    STEP_NAME = "ru_llm_stress_ambiguity"
    STEP_VERSION = 3

    def __init__(self, config: GeneralConfig):
        self.lexicon_db = (
            ensure_pronunciation_lexicon_db(config.normalize_pronunciation_lexicon_db)
            if is_russian_language(config.language) and config.normalize_pronunciation_lexicon_db
            else None
        )
        self._lexicon_entry_cache: dict[str, tuple[PronunciationLexiconEntry, ...]] = {}
        self._planned_text = ""
        self._planned_candidates: dict[str, StressAmbiguityCandidate] = {}
        self._planned_order: list[str] = []
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
            system_prompt=_get_stress_ambiguity_prompt(self.config),
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
            "Choosing stress ambiguities for chapter '%s' batch %s/%s, items=%s",
            chapter_title,
            unit_index,
            unit_count,
            len(batch),
        )
        selections = self.choice_service.choose_batch(
            batch,
            target_language=self.config.language,
            system_prompt=_get_stress_ambiguity_prompt(self.config),
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
        batch = self._deserialize_batch(unit)
        return {
            "00_choice_system_prompt.txt": _get_stress_ambiguity_prompt(self.config),
            "01_choice_user_prompt.txt": self.choice_service.render_user_prompt(
                batch,
                target_language=self.config.language,
            ),
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
            options.append(
                NormalizerLLMChoiceOption(
                    option_id=f"variant_{index}",
                    text=preserved,
                )
            )
            seen_texts.add(preserved)
        return tuple(options)

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
