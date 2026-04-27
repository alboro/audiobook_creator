# Copyright (c) 2026 alboro <alboro@users.noreply.github.com>
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See: https://polyformproject.org/licenses/noncommercial/1.0.0/

from __future__ import annotations

import json as _json
import logging
import re

from audiobook_generator.config.general_config import GeneralConfig
from audiobook_generator.normalizers.base_normalizer import BaseNormalizer
from audiobook_generator.utils.chunk_boundaries import (
    CHUNK_EOF_TAG,
    ends_with_chunk_boundary,
    ensure_chunk_eof_boundary,
    merge_broken_backtick_sentences,
    strip_chunk_boundary_tags,
    split_text_by_chunk_boundaries,
    split_text_preserve_chunk_separators,
)

logger = logging.getLogger(__name__)

DEFAULT_SAFE_MAX_CHARS = 180
MIN_SPLIT_FRACTION = 0.45
MIN_SPLIT_FRAGMENT_CHARS = 24
MIN_SPLIT_FRAGMENT_WORDS = 2
# Sentences shorter than this will be merged with the next sentence to avoid TTS instability
MIN_TTS_SAFE_CHARS = 12
LEFT_TRIM_CHARS = " \t\r\n,;:-–—"
RIGHT_TRIM_CHARS = " \t\r\n,;:-–—"

PRIORITY_PATTERNS = (
    # 0. Existing sentence boundary: period/!/? followed by space and capital letter.
    #    This is always the best split point — avoids breaking clauses mid-sentence.
    re.compile(r"(?<=[.!?])\s+(?=[А-ЯЁA-Z«\"])", re.UNICODE),
    re.compile(r"[;:](?=\s|$)"),
    # Conjunctions after punctuation only — prevents splitting "Ветхого и Нового" (no comma).
    # Exclude "и" followed by adverbial particles ("и более того", "и даже", "и при этом" etc.)
    re.compile(
        r"(?<=[,;])\s+(?=(?:а|но|однако)\b)"
        r"|(?<=[,;])\s+(?=и\s+(?!(?:более|даже|тем|при|всё|всего|ещё|только|притом|при этом)\b))",
        re.IGNORECASE,
    ),
    re.compile(r",\s+(?=(?:а также|однако|но|зато|поэтому|причем|притом|при этом|затем|потом)\b)", re.IGNORECASE),
    re.compile(
        r",\s+(?=(?:котор(?:ый|ая|ое|ые|ого|ому|ым|ых|ую|ой|ою)|обосновывающ\w*|существующ\w*|позволяющ\w*|делающ\w*|создающ\w*)\b)",
        re.IGNORECASE,
    ),
    re.compile(r",(?=\s|$)"),
    re.compile(r"\s-\s"),
)

DEFAULT_SAFE_SPLIT_SYSTEM_PROMPT = (
    "Ты — препроцессор текста для синтеза речи (TTS). "
    "Получаешь JSON-список объектов. Каждый объект содержит:\n"
    "  - `id`: идентификатор\n"
    "  - `sentence`: ПРЕДЛОЖЕНИЕ, которое нужно разбить\n"
    "  - `context_before`, `context_after`: контекст ДЛЯ ПОНИМАНИЯ ТОЛЬКО — НЕ включать в ответ\n\n"
    "Задача: разбить с минимальным вмешательством поле `sentence` на более короткие части не больше %max_chars% символов в каждой части.\n"
    "Правила:\n"
    "- Разбивай только по содержимому поля `sentence`, не включай в ответ контекст.\n"
    "- Каждая часть должна быть законченной фразой, естественно звучащей вслух.\n"
    "- Пауза между частями разделяемого предложения не должна вызывать слушательский дискомфорт.\n"
    "- Лёгкое перефразирование может помогать разбиению.\n"
    "- Не удаляй и не добавляй фактический смысл.\n"
    "- В получившемся предложении: должны начинаться с заглувной буквы, а знаки препинания должны быть синтаксически верными.\n"
    "- Все не касающиеся разбиения формы слов, даже если они странные, оставляй, как есть.\n"
    "- Возвращай JSON-массив: "
    '[{"id":"<id>","parts":["часть1","часть2",...]}]\n'
    "- Если `sentence` уже укладывается в лимит — верни его одной частью."
)


def _get_safe_split_prompt(config, max_chars: int) -> str:
    custom = getattr(config, "normalize_safe_split_system_prompt", None)
    template = custom.strip() if custom and isinstance(custom, str) else DEFAULT_SAFE_SPLIT_SYSTEM_PROMPT
    return template.replace("%max_chars%", str(max_chars))


class TTSSafeSplitNormalizer(BaseNormalizer):
    STEP_NAME = "tts_llm_safe_split"

    def __init__(self, config: GeneralConfig):
        self.max_chars = config.normalize_tts_safe_max_chars or DEFAULT_SAFE_MAX_CHARS
        self.comma_as_period: bool = bool(getattr(config, "normalize_tts_safe_comma_as_period", False))
        super().__init__(config)

    def validate_config(self):
        if self.max_chars < 40:
            raise ValueError("normalize_tts_safe_max_chars must be at least 40")

    def normalize(self, text: str, chapter_title: str = "") -> str:
        """LLM-only pass: send sentences exceeding max_chars to the LLM for splitting."""
        return self._llm_split_long_sentences(text, chapter_title=chapter_title)

    def _preprocess_punctuation(self, text: str) -> str:
        """Replace punctuation that is safe/configured to treat as sentence boundaries."""
        text = re.sub(r";\s*", ". ", text)
        if self.comma_as_period:
            text = re.sub(r",\s*", ". ", text)
        return text

    # ------------------------------------------------------------------
    # LLM pass

    def _llm_split_long_sentences(self, text: str, *, chapter_title: str = "") -> str:
        """Send all over-limit sentences (with context) to the LLM in bulk batches.

        Paragraphs are processed independently so that paragraph breaks are preserved.
        """
        if not self.has_normalizer_llm():
            return text

        # Split into (paragraph, separator) pairs so we can reconstruct faithfully.
        para_tokens = re.split(r"(\n{2,})", text)
        paragraphs: list[str] = []
        para_seps: list[str] = []
        for tok in para_tokens:
            if re.fullmatch(r"\n{2,}", tok):
                if para_seps:
                    para_seps[-1] = tok
            else:
                paragraphs.append(tok)
                para_seps.append("")

        # For each paragraph, split into sentences and gather over-limit candidates.
        # Each candidate is tagged with (para_idx, sent_idx) for stitching back.
        all_items: list[dict] = []          # LLM payload items
        para_data: list[tuple] = []         # (sentences, separators) per paragraph

        for pi, para in enumerate(paragraphs):
            sentences, separators = _split_text_preserve_separators(para)
            para_data.append((sentences, separators))
            for si, sent in enumerate(sentences):
                if len(sent) > self.max_chars:
                    ctx_before = " ".join(sentences[max(0, si - 2): si])
                    ctx_after = " ".join(sentences[si + 1: si + 3])
                    all_items.append({
                        "para": pi,
                        "sent": si,
                        "id": f"p{pi}s{si}",
                        "sentence": sent,
                        "context_before": ctx_before,
                        "context_after": ctx_after,
                    })

        if not all_items:
            return text

        logger.info(
            "TTS safe split LLM: chapter '%s' — %s sentence(s) over %s chars",
            chapter_title, len(all_items), self.max_chars,
        )

        try:
            replacements = self._call_llm_bulk(all_items, chapter_title=chapter_title)
        except Exception as exc:
            logger.warning(
                "TTS safe split LLM skipped for chapter '%s': %s", chapter_title, exc,
            )
            return text

        # Apply replacements back per paragraph, then rejoin paragraphs.
        result_paras: list[str] = []
        for pi, (sentences, separators) in enumerate(para_data):
            new_sentences: list[str] = []
            new_separators: list[str] = []
            for si, sent in enumerate(sentences):
                item_id = f"p{pi}s{si}"
                parts = replacements.get(item_id)
                if parts and isinstance(parts, list):
                    clean_parts = [part.strip() for part in parts if str(part).strip()]
                    for j, part in enumerate(clean_parts):
                        new_sentences.append(
                            self._finalize_sentence(
                                part,
                                artificial_boundary=j < len(clean_parts) - 1,
                            )
                        )
                        new_separators.append(" " if j < len(clean_parts) - 1 else
                                              (separators[si] if si < len(separators) else ""))
                else:
                    new_sentences.append(sent)
                    new_separators.append(separators[si] if si < len(separators) else "")
            result_paras.append(_rejoin_sentences(new_sentences, new_separators))

        # Rejoin paragraphs with their original separators.
        out = []
        for i, para in enumerate(result_paras):
            out.append(para)
            if para_seps[i]:
                out.append(para_seps[i])
        result = "".join(out).strip()

        logger.info(
            "TTS safe split LLM done for chapter '%s': %s item(s) processed",
            chapter_title, len(all_items),
        )
        return result

    def _call_llm_bulk(self, items: list[dict], *, chapter_title: str) -> dict[str, list[str]]:
        """Send bulk request; return {id: [part, ...]} mapping."""
        llm = self.get_normalizer_llm()
        system_prompt = _get_safe_split_prompt(self.config, self.max_chars)
        max_chars = self.max_chars

        # Build batches to stay within normalize_max_chars limit
        llm_max = (self.config.normalize_max_chars or 4000)
        batches = _make_batches(items, system_prompt=system_prompt, budget=llm_max)

        result: dict[str, list[str]] = {}
        for batch_num, batch in enumerate(batches, 1):
            payload = [
                {
                    "id": item["id"],
                    "sentence": item["sentence"],
                    "context_before": item["context_before"],
                    "context_after": item["context_after"],
                }
                for item in batch
            ]
            user_prompt = (
                f"Разбей поле `sentence` каждого объекта на части не длиннее {max_chars} символов.\n"
                f"Поля `context_before` и `context_after` — только для понимания контекста, НЕ включай их в ответ.\n\n"
                + _json.dumps(payload, ensure_ascii=False, indent=2)
            )
            logger.debug(
                "TTS safe split LLM [ch='%s' batch=%d/%d] REQUEST:\n%s",
                chapter_title, batch_num, len(batches), user_prompt,
            )
            response = llm.complete(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                model=self.get_normalizer_model(),
                temperature=0,
            )
            logger.debug(
                "TTS safe split LLM [ch='%s' batch=%d/%d] RESPONSE:\n%s",
                chapter_title, batch_num, len(batches), response,
            )
            parsed = _parse_split_response(response)
            result.update(parsed)

        return result

    # ------------------------------------------------------------------
    # Algorithmic splitting helpers (unchanged)
    # ------------------------------------------------------------------

    def _normalize_paragraph(self, paragraph: str) -> tuple[str, int, int]:
        compact = re.sub(r"\s+", " ", paragraph).strip()
        if not compact:
            return "", 0, 0

        language = (self.config.language or "ru").split("-")[0].lower()
        sentences = split_text_by_chunk_boundaries(compact, language)
        if not sentences:
            sentences = [compact]

        safe_sentences = []
        inserted_splits = 0
        for sentence in sentences:
            split_parts = self._split_long_sentence(sentence)
            safe_sentences.extend(split_parts)
            inserted_splits += max(0, len(split_parts) - 1)

        # Merge very short sentences with neighbors to avoid TTS instability.
        merged_sentences = []
        i = 0
        while i < len(safe_sentences):
            current = safe_sentences[i]
            if len(current) < MIN_TTS_SAFE_CHARS:
                if (
                    merged_sentences
                    and len(merged_sentences[-1]) + 1 + len(current) <= self.max_chars
                ):
                    prev = merged_sentences.pop()
                    if current and current[-1] in "!?":
                        merged_sentences.append(f"{prev} {current}")
                    else:
                        base_curr = strip_chunk_boundary_tags(current).rstrip(".").rstrip()
                        merged_sentences.append(f"{prev} {base_curr}.")
                    i += 1
                elif i + 1 < len(safe_sentences):
                    next_sent = safe_sentences[i + 1]
                    if current and current[-1] in "!?":
                        merged_sentences.append(f"{current} {next_sent}")
                    else:
                        base = strip_chunk_boundary_tags(current).rstrip(".").rstrip()
                        merged_sentences.append(f"{base} {next_sent}")
                    i += 2
                else:
                    merged_sentences.append(current)
                    i += 1
            else:
                merged_sentences.append(current)
                i += 1

        return " ".join(merged_sentences).strip(), len(merged_sentences), inserted_splits

    def _split_long_sentence(self, sentence: str) -> list[str]:
        pending = [sentence.strip()]
        result = []

        while pending:
            current = pending.pop(0).strip()
            if not current:
                continue
            if len(current) <= self.max_chars:
                result.append(self._finalize_sentence(current))
                continue

            split_index = self._find_split_index(current)
            if split_index is None:
                result.append(self._finalize_sentence(current))
                continue

            left = current[:split_index].rstrip(LEFT_TRIM_CHARS)
            right = current[split_index:].lstrip(RIGHT_TRIM_CHARS)
            if not left or not right or not self._is_acceptable_split(left, right):
                result.append(self._finalize_sentence(current))
                continue

            result.append(self._finalize_sentence(left, artificial_boundary=True))
            pending.insert(0, self._normalize_sentence_start(right))

        return result

    def _find_split_index(self, sentence: str) -> int | None:
        if len(sentence) <= self.max_chars:
            return None

        min_index = max(20, int(self.max_chars * MIN_SPLIT_FRACTION))
        window = sentence[: self.max_chars + 1]

        for pattern in PRIORITY_PATTERNS:
            candidate_indexes = [
                match.end()
                for match in pattern.finditer(window)
                if match.end() >= min_index
            ]
            split_index = self._select_best_candidate(sentence, candidate_indexes)
            if split_index is not None:
                return split_index

        space_indexes = [
            index + 1
            for index, char in enumerate(window)
            if char == " " and index + 1 >= min_index
        ]
        split_index = self._select_best_candidate(sentence, space_indexes)
        if split_index is not None:
            return split_index

        return self.max_chars

    def _select_best_candidate(self, sentence: str, candidate_indexes: list[int]) -> int | None:
        for split_index in sorted(candidate_indexes, reverse=True):
            left = sentence[:split_index].rstrip(LEFT_TRIM_CHARS)
            right = sentence[split_index:].lstrip(RIGHT_TRIM_CHARS)
            if left and right and self._is_acceptable_split(left, right):
                return split_index
        return None

    def _is_acceptable_split(self, left: str, right: str) -> bool:
        if not left or not right:
            return False

        left_words = [word for word in left.split() if word]
        right_words = [word for word in right.split() if word]
        if len(left) < MIN_SPLIT_FRAGMENT_CHARS and len(left_words) < MIN_SPLIT_FRAGMENT_WORDS:
            return False
        if len(right) < MIN_SPLIT_FRAGMENT_CHARS and len(right_words) < MIN_SPLIT_FRAGMENT_WORDS:
            return False

        if left_words and left and not ends_with_chunk_boundary(left):
            last_word = left_words[-1].strip('.,!?;:\'"»«`')
            if len(last_word) <= 3:
                return False

        first_alpha = next((ch for ch in right if ch.isalpha()), None)
        if first_alpha is not None and first_alpha.islower() and '\u0400' <= first_alpha <= '\u04ff':
            return False

        return True

    def _normalize_sentence_start(self, sentence: str) -> str:
        sentence = sentence.lstrip()
        if not sentence:
            return ""

        chars = list(sentence)
        for index, char in enumerate(chars):
            if char.isalpha():
                chars[index] = char.upper()
                break
            if char.isdigit():
                break
        return "".join(chars)

    def _finalize_sentence(self, sentence: str, *, artificial_boundary: bool = False) -> str:
        sentence = sentence.strip()
        if not sentence:
            return ""
        if artificial_boundary:
            return ensure_chunk_eof_boundary(sentence)
        if ends_with_chunk_boundary(sentence):
            return sentence
        return f"{sentence}."


class TTSSafeSplitAlgorithmicNormalizer(TTSSafeSplitNormalizer):
    """Deterministic (algorithmic) safe-split — no LLM, use as ``tts_safe_split`` step."""

    STEP_NAME = "tts_safe_split"

    def normalize(self, text: str, chapter_title: str = "") -> str:
        """Algorithmic-only pass: preprocess punctuation + deterministic sentence splitting."""
        text = self._preprocess_punctuation(text)
        parts = re.split(r"(\n\s*\n+)", text)
        normalized_parts = []
        sentence_count = 0
        inserted_splits = 0

        for part in parts:
            if not part:
                continue
            if re.fullmatch(r"\n\s*\n+", part):
                normalized_parts.append(part)
                continue
            normalized_paragraph, paragraph_sentences, paragraph_splits = self._normalize_paragraph(part)
            normalized_parts.append(normalized_paragraph)
            sentence_count += paragraph_sentences
            inserted_splits += paragraph_splits

        logger.info(
            "TTS safe split (algorithmic) applied to '%s': %s sentences, %s splits, max_chars=%s",
            chapter_title, sentence_count, inserted_splits, self.max_chars,
        )
        return "".join(normalized_parts).strip()


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _split_text_preserve_separators(text: str) -> tuple[list[str], list[str]]:
    """Split text on sentence boundaries. Return (sentences, separators).

    separators[i] is the whitespace/text between sentences[i] and sentences[i+1].
    len(separators) == len(sentences) (last entry is always "").
    """
    return split_text_preserve_chunk_separators(text)


def _rejoin_sentences(sentences: list[str], separators: list[str]) -> str:
    if not sentences:
        return ""
    parts = []
    for i, sent in enumerate(sentences):
        parts.append(sent)
        sep = separators[i] if i < len(separators) else ""
        if sep and i < len(sentences) - 1:
            parts.append(sep)
    return "".join(parts)


def _make_batches(
    items: list[dict],
    *,
    system_prompt: str,
    budget: int,
) -> list[list[dict]]:
    """Split items into batches so that each batch user-prompt stays within budget."""
    batches: list[list[dict]] = []
    current: list[dict] = []
    overhead = len(system_prompt) + 200  # constant overhead per call

    for item in items:
        item_size = len(item["sentence"]) + len(item["context_before"]) + len(item["context_after"]) + 80
        current_size = overhead + sum(
            len(it["sentence"]) + len(it["context_before"]) + len(it["context_after"]) + 80
            for it in current
        )
        if current and current_size + item_size > budget:
            batches.append(current)
            current = [item]
        else:
            current.append(item)

    if current:
        batches.append(current)

    return batches


def _merge_broken_backtick_sentences(sentences: list[str]) -> list[str]:
    """Re-attach a stray closing backtick that sentencex left at the start of the next sentence.

    sentencex sometimes splits a backtick-quoted span like ``word!`` into
    two pieces: [``word!``, `` rest``] (the closing backtick ends up at
    the beginning of the next fragment).
    This function detects such pairs and fixes them to [``word!```, ``rest``].
    Detection: current sentence has an odd number of backtick chars (unmatched opening quote)
    AND the next sentence starts with a lone backtick (`` ` `` at pos 0, then space or end).
    """
    return merge_broken_backtick_sentences(sentences)


def _parse_split_response(response_text: str) -> dict[str, list[str]]:
    """Parse LLM response into {id: [part, ...]}."""
    if not response_text:
        return {}

    raw = response_text.strip()
    # Strip markdown code fence if present
    if raw.startswith("```"):
        lines = raw.splitlines()
        if len(lines) >= 3:
            raw = "\n".join(lines[1:-1]).strip()

    try:
        data = _json.loads(raw)
    except _json.JSONDecodeError:
        # Try to find JSON array in the response
        start = raw.find("[")
        end = raw.rfind("]")
        if start != -1 and end != -1 and end > start:
            data = _json.loads(raw[start: end + 1])
        else:
            raise ValueError(f"Cannot parse LLM response as JSON: {raw[:200]!r}")

    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array, got: {type(data)}")

    result: dict[str, list[str]] = {}
    for entry in data:
        if not isinstance(entry, dict):
            continue
        item_id = str(entry.get("id") or "").strip()
        parts = entry.get("parts")
        if not item_id or not isinstance(parts, list):
            continue
        result[item_id] = [str(p) for p in parts if str(p).strip()]

    return result
