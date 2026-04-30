# Copyright (c) 2026 alboro <alboro@users.noreply.github.com>
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See: https://polyformproject.org/licenses/noncommercial/1.0.0/

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import openai
from openai import OpenAI

from audiobook_generator.config.general_config import GeneralConfig

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = """You normalize ebook text for text-to-speech.

Rules:
- Preserve meaning and structure.
- Do not summarize, censor, or add new facts.
- Keep paragraph breaks where they help narration.
- Fix obvious formatting artifacts and awkward punctuation.
- Expand only symbols or shorthand that would sound bad when read aloud.
- Never output labels, headings, field names, or wrapper text such as "Chapter", "Text", or "Title".
- Return only the normalized text."""
DEFAULT_USER_PROMPT_TEMPLATE = "{text}"
DEFAULT_CHOICE_SYSTEM_PROMPT = """You choose the best pronunciation-friendly variant for named entities in text prepared for text-to-speech.

Rules:
- Select exactly one option for each item unless you know a clearly better custom variant.
- Prefer the smallest change that improves pronunciation.
- Preserve meaning, grammar, case, and number.
- Prefer Russian-friendly pronunciation when the target language is Russian.
- Keep the original option if none of the alternatives is clearly better.
- You may return custom_text only if none of the provided options is satisfactory and you are confident in a better pronunciation-safe variant.
- Set cacheable to true only when the best choice does not depend on wider context and can be safely reused for the same source_text and options.
- Return JSON only, with this exact shape:
{"selections":[{"id":"item-1","option_id":"option-id","custom_text":"","cacheable":false,"reason":""}]}"""
CHOICE_PROMPT_MARGIN_CHARS = 400
CHOICE_CACHE_VERSION = 1
RATE_LIMIT_RETRY_INTERVAL_S = 180  # 3 minutes between rate-limit retries


@dataclass(frozen=True)
class NormalizerLLMChoiceOption:
    option_id: str
    text: str
    hint: str | None = None  # grammatical hint shown to LLM, e.g. "genitive singular"


@dataclass(frozen=True)
class NormalizerLLMChoiceItem:
    item_id: str
    source_text: str
    context: str
    options: tuple[NormalizerLLMChoiceOption, ...]
    note: str | None = None


@dataclass(frozen=True)
class NormalizerLLMChoiceSelection:
    item_id: str
    option_id: str | None = None
    custom_text: str | None = None
    cacheable: bool = False
    reason: str | None = None
    source: str = "llm"

    @property
    def has_custom_text(self) -> bool:
        return bool((self.custom_text or "").strip())

    def resolved_option_id(self) -> str:
        return self.option_id or "original"


@dataclass(frozen=True)
class NormalizerLLMSettings:
    provider: str
    model: str
    base_url: str | None
    api_key: str | None
    max_chars: int
    system_prompt: str
    user_prompt_template: str
    choice_cache_path: str
    reasoning_effort: str | None = None  # "low" | "medium" | "high"


class NormalizerLLMChoiceCache:
    def __init__(self, cache_path: str | Path):
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache = self._load()

    def get(self, cache_key: str) -> NormalizerLLMChoiceSelection | None:
        payload = self._cache.get(cache_key)
        if not isinstance(payload, dict):
            return None
        return self._selection_from_payload(payload, source="cache")

    def put(self, cache_key: str, selection: NormalizerLLMChoiceSelection):
        self._cache[cache_key] = {
            "item_id": selection.item_id,
            "option_id": selection.option_id,
            "custom_text": selection.custom_text,
            "cacheable": bool(selection.cacheable),
            "reason": selection.reason or "",
        }
        self._write()

    def _load(self) -> dict[str, dict]:
        if not self.cache_path.is_file():
            return {}
        try:
            payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        if isinstance(payload, dict):
            return payload
        return {}

    def _write(self):
        self.cache_path.write_text(
            json.dumps(self._cache, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )

    @staticmethod
    def _selection_from_payload(payload: dict, *, source: str) -> NormalizerLLMChoiceSelection:
        return NormalizerLLMChoiceSelection(
            item_id=str(payload.get("item_id") or "").strip(),
            option_id=(str(payload.get("option_id") or "").strip() or None),
            custom_text=(str(payload.get("custom_text") or "").strip() or None),
            cacheable=bool(payload.get("cacheable")),
            reason=(str(payload.get("reason") or "").strip() or None),
            source=source,
        )


class NormalizerLLM:
    def __init__(self, config: GeneralConfig):
        self.config = config
        self.settings = resolve_normalizer_llm_settings(config)
        self.client = self._build_client()

    @property
    def is_available(self) -> bool:
        return self.client is not None

    def ensure_available(self):
        if self.is_available:
            return
        raise ValueError(
            "LLM-backed normalizer features are not available. "
            "Provide normalize_api_key / normalize_base_url or the matching environment variables."
        )

    def render_user_prompt(self, *, chapter_title: str, text: str) -> str:
        try:
            return self.settings.user_prompt_template.format(
                chapter_title=chapter_title,
                text=text,
            )
        except KeyError as exc:
            raise ValueError(
                "Unknown placeholder in normalize_user_prompt_file. "
                "Supported placeholders: {chapter_title}, {text}."
            ) from exc

    def complete(
        self,
        *,
        user_prompt: str,
        system_prompt: str | None = None,
        model: str | None = None,
        temperature: float = 0,
    ) -> str:
        self.ensure_available()
        effective_model = model or self.settings.model
        extra_kwargs: dict = {}
        if self.settings.reasoning_effort:
            extra_kwargs["reasoning_effort"] = self.settings.reasoning_effort

        attempt = 0
        while True:
            attempt += 1
            try:
                response = self.client.chat.completions.create(
                    model=effective_model,
                    temperature=temperature,
                    messages=[
                        {"role": "system", "content": system_prompt or self.settings.system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    **extra_kwargs,
                )
                return (response.choices[0].message.content or "").strip()
            except openai.RateLimitError as exc:
                logger.warning(
                    "LLM rate limit reached (attempt %d): %s — sleeping %d s before retry.",
                    attempt,
                    exc,
                    RATE_LIMIT_RETRY_INTERVAL_S,
                )
                time.sleep(RATE_LIMIT_RETRY_INTERVAL_S)

    def _build_client(self):
        if self.settings.provider != "openai":
            return None

        api_key = self.settings.api_key
        if not api_key and not self.settings.base_url:
            return None
        if not api_key and self.settings.base_url:
            api_key = "dummy"

        return OpenAI(
            api_key=api_key,
            base_url=self.settings.base_url,
            max_retries=4,
        )


class NormalizerLLMChoiceService:
    def __init__(self, llm: NormalizerLLM):
        self.llm = llm
        self.cache = NormalizerLLMChoiceCache(llm.settings.choice_cache_path)

    def plan_batches(
        self,
        items: Sequence[NormalizerLLMChoiceItem],
        *,
        system_prompt: str = DEFAULT_CHOICE_SYSTEM_PROMPT,
        max_chars: int | None = None,
    ) -> list[list[NormalizerLLMChoiceItem]]:
        if not items:
            return []

        preferred_limit = max_chars or self.llm.settings.max_chars or 4000
        budget = max(1500, preferred_limit - CHOICE_PROMPT_MARGIN_CHARS)
        batches: list[list[NormalizerLLMChoiceItem]] = []
        current_batch: list[NormalizerLLMChoiceItem] = []

        for item in items:
            candidate_batch = current_batch + [item]
            candidate_size = self._estimate_batch_size(
                candidate_batch,
                system_prompt=system_prompt,
            )
            if current_batch and candidate_size > budget:
                batches.append(current_batch)
                current_batch = [item]
                continue

            current_batch = candidate_batch

        if current_batch:
            batches.append(current_batch)

        return batches

    def render_user_prompt(
        self,
        items: Sequence[NormalizerLLMChoiceItem],
        *,
        target_language: str | None = None,
    ) -> str:
        payload = {
            "target_language": target_language or "",
            "items": [
                {
                    "id": item.item_id,
                    "source_text": item.source_text,
                    "context": item.context,
                    "note": item.note or "",
                    "options": [
                        {
                            "id": option.option_id,
                            "text": option.text,
                            **({} if not option.hint else {"hint": option.hint}),
                        }
                        for option in item.options
                    ],
                }
                for item in items
            ],
        }
        return (
            "Choose the best option for each item and return JSON only.\n\n"
            + json.dumps(payload, ensure_ascii=False, indent=2)
        )

    def choose_batch(
        self,
        items: Sequence[NormalizerLLMChoiceItem],
        *,
        target_language: str | None = None,
        system_prompt: str = DEFAULT_CHOICE_SYSTEM_PROMPT,
        model: str | None = None,
        temperature: float = 0,
        use_cache: bool = True,
    ) -> dict[str, NormalizerLLMChoiceSelection]:
        if not items:
            return {}

        if use_cache:
            cached, unresolved = self._resolve_cached(items, target_language=target_language, system_prompt=system_prompt)
        else:
            cached, unresolved = {}, list(items)
        if not unresolved:
            return cached

        user_prompt = self.render_user_prompt(unresolved, target_language=target_language)
        response_text = self.llm.complete(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            model=model,
            temperature=temperature,
        )
        llm_selections = self.parse_choice_response_objects(response_text)

        resolved = dict(cached)
        for item in unresolved:
            selection = llm_selections.get(item.item_id)
            if selection is None:
                selection = NormalizerLLMChoiceSelection(
                    item_id=item.item_id,
                    option_id="original",
                    source="fallback",
                )
            resolved[item.item_id] = selection
            if use_cache and selection.cacheable:
                cache_key = self._make_cache_key(
                    item,
                    target_language=target_language,
                    system_prompt=system_prompt,
                )
                self.cache.put(cache_key, selection)

        return resolved

    def render_settings_json(
        self,
        *,
        system_prompt: str = DEFAULT_CHOICE_SYSTEM_PROMPT,
    ) -> str:
        settings = {
            "provider": self.llm.settings.provider,
            "model": self.llm.settings.model,
            "base_url": self.llm.settings.base_url or "",
            "max_chars": self.llm.settings.max_chars,
            "choice_system_prompt": system_prompt,
            "choice_cache_path": self.llm.settings.choice_cache_path,
            "choice_cache_version": CHOICE_CACHE_VERSION,
        }
        return json.dumps(settings, ensure_ascii=False, indent=2) + "\n"

    @staticmethod
    def parse_choice_response(response_text: str) -> dict[str, str]:
        return {
            item_id: selection.resolved_option_id()
            for item_id, selection in NormalizerLLMChoiceService.parse_choice_response_objects(
                response_text
            ).items()
        }

    @staticmethod
    def parse_choice_response_objects(response_text: str) -> dict[str, NormalizerLLMChoiceSelection]:
        data = _parse_json_response(response_text)
        selections = data.get("selections") if isinstance(data, dict) else data
        if not isinstance(selections, list):
            raise ValueError("LLM choice response must contain a 'selections' array.")

        result: dict[str, NormalizerLLMChoiceSelection] = {}
        for selection in selections:
            if not isinstance(selection, dict):
                raise ValueError("Each LLM choice selection must be an object.")
            item_id = str(selection.get("id") or "").strip()
            option_id = str(selection.get("option_id") or "").strip() or None
            custom_text = str(selection.get("custom_text") or "").strip() or None
            cacheable = _coerce_bool(selection.get("cacheable"))
            reason = str(selection.get("reason") or "").strip() or None
            if not item_id:
                raise ValueError("Each LLM choice selection must contain 'id'.")
            if not option_id and not custom_text:
                raise ValueError(
                    "Each LLM choice selection must contain 'option_id' or 'custom_text'."
                )
            result[item_id] = NormalizerLLMChoiceSelection(
                item_id=item_id,
                option_id=option_id,
                custom_text=custom_text,
                cacheable=cacheable,
                reason=reason,
                source="llm",
            )

        return result

    def _resolve_cached(
        self,
        items: Sequence[NormalizerLLMChoiceItem],
        *,
        target_language: str | None,
        system_prompt: str,
    ) -> tuple[dict[str, NormalizerLLMChoiceSelection], list[NormalizerLLMChoiceItem]]:
        cached: dict[str, NormalizerLLMChoiceSelection] = {}
        unresolved: list[NormalizerLLMChoiceItem] = []
        for item in items:
            cache_key = self._make_cache_key(
                item,
                target_language=target_language,
                system_prompt=system_prompt,
            )
            selection = self.cache.get(cache_key)
            if selection is None:
                unresolved.append(item)
                continue
            cached[item.item_id] = NormalizerLLMChoiceSelection(
                item_id=item.item_id,
                option_id=selection.option_id,
                custom_text=selection.custom_text,
                cacheable=selection.cacheable,
                reason=selection.reason,
                source="cache",
            )
        return cached, unresolved

    def _make_cache_key(
        self,
        item: NormalizerLLMChoiceItem,
        *,
        target_language: str | None,
        system_prompt: str,
    ) -> str:
        payload = {
            "version": CHOICE_CACHE_VERSION,
            "target_language": target_language or "",
            "source_text": item.source_text,
            "options": [{"id": option.option_id, "text": option.text} for option in item.options],
            "system_prompt": system_prompt,
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _estimate_batch_size(
        self,
        items: Iterable[NormalizerLLMChoiceItem],
        *,
        system_prompt: str,
    ) -> int:
        prompt = self.render_user_prompt(list(items))
        return len(system_prompt) + len(prompt)


def resolve_normalizer_llm_settings(config: GeneralConfig) -> NormalizerLLMSettings:
    provider = config.normalize_provider or "openai"
    model = config.normalize_model or "gpt-4.1-mini"
    max_chars = config.normalize_max_chars or 4000

    system_prompt_path = config.normalize_system_prompt_file or config.normalize_prompt_file
    system_prompt = DEFAULT_SYSTEM_PROMPT
    if getattr(config, "normalize_system_prompt", None):
        system_prompt = config.normalize_system_prompt.strip()
    elif system_prompt_path:
        system_prompt = Path(system_prompt_path).read_text(encoding="utf-8").strip()

    user_prompt_template = DEFAULT_USER_PROMPT_TEMPLATE
    if config.normalize_user_prompt_file:
        user_prompt_template = Path(config.normalize_user_prompt_file).read_text(
            encoding="utf-8"
        ).strip()

    base_url = (
        config.normalize_base_url
        or os.getenv("NORMALIZER_OPENAI_BASE_URL")
        or os.getenv("OPENAI_BASE_URL")
    )
    api_key = (
        config.normalize_api_key
        or os.getenv("NORMALIZER_OPENAI_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )

    return NormalizerLLMSettings(
        provider=provider,
        model=model,
        base_url=base_url,
        api_key=api_key,
        max_chars=max_chars,
        system_prompt=system_prompt,
        user_prompt_template=user_prompt_template,
        choice_cache_path=str(_resolve_choice_cache_path(config)),
        reasoning_effort=getattr(config, "normalize_reasoning_effort", None) or None,
    )


def _resolve_choice_cache_path(config: GeneralConfig) -> Path:
    if getattr(config, "prepared_text_folder", None):
        base_dir = Path(config.prepared_text_folder)
    elif getattr(config, "output_folder", None):
        base_dir = Path(config.output_folder)
    else:
        base_dir = Path.cwd()
    return base_dir / "_state" / "normalizer_llm_choice_cache.json"


def _parse_json_response(response_text: str):
    if not response_text:
        raise ValueError("LLM response is empty.")

    raw = response_text.strip()
    fenced_match = None
    if raw.startswith("```"):
        lines = raw.splitlines()
        if len(lines) >= 3:
            fenced_match = "\n".join(lines[1:-1]).strip()
    if fenced_match:
        raw = fenced_match

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(raw[start : end + 1])
        start = raw.find("[")
        end = raw.rfind("]")
        if start != -1 and end != -1 and end > start:
            return json.loads(raw[start : end + 1])
        raise


def _coerce_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)
