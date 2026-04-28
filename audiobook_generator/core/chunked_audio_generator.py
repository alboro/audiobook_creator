# Copyright (c) 2026 alboro <alboro@users.noreply.github.com>
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See: https://polyformproject.org/licenses/noncommercial/1.0.0/

"""Chunked TTS audio generation with sentence-level resume.

Usage
-----
Instead of sending an entire chapter text as one TTS call, this module:

1. Splits the chapter text into sentences.
2. Detects quoted blocks and further splits them, optionally using a secondary
   voice (``voice_name2``) for character speech.
3. Computes a content-hash for each sentence (voice suffix added when voice2 used).
4. Skips sentences whose audio file already exists on disk (file = truth).
5. Calls the TTS provider for each missing sentence.
6. Trims trailing silence from each synthesised chunk (keeps a 200ms tail).
7. Concatenates all chunk audio files into the chapter output file.

Enable with ``--chunked_audio`` CLI flag.
Secondary voice: set ``voice_name2`` in config/ini for quoted character speech.
"""
from __future__ import annotations

import logging
import os
import re
from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional, Tuple

from audiobook_generator.core.audio_chunk_store import AudioChunkStore
from audiobook_generator.core.audio_tags import AudioTags
from audiobook_generator.utils.chunk_boundaries import split_text_by_chunk_boundaries
from audiobook_generator.utils.sentence_hash import sentence_hash as _sentence_hash

logger = logging.getLogger(__name__)

# Minimum sentence length to synthesise (skip whitespace-only fragments).
MIN_SENTENCE_CHARS = 3

# Silence trimming: keep this many ms of audio after the last non-silent segment.
SILENCE_TAIL_MS = 200
# Silence threshold in dBFS below which a segment is considered silent.
SILENCE_THRESH_DBFS = -45

# Opening/closing quote pairs to detect character speech blocks.
# Each tuple: (open_char, close_char)
_QUOTE_PAIRS: List[Tuple[str, str]] = [
    ('\u00ab', '\u00bb'),   # «»  (Russian guillemets)
    ('\u201c', '\u201d'),   # ""  (English/typographic double quotes)
    ('\u2018', '\u2019'),   # ''  (single typographic)
    ('"', '"'),             # straight double quotes
]
# Opening chars for fast lookup
_OPEN_QUOTES = {p[0] for p in _QUOTE_PAIRS}
# Mapping open → close
_CLOSE_FOR_OPEN = {p[0]: p[1] for p in _QUOTE_PAIRS}

# Regex that matches a *close-quote artifact* sentence: a very short fragment that
# consists only of a closing quote character optionally followed by sentence-ending
# punctuation.  These are produced when sentencex treats "?" or "!" inside a quoted
# block as an EOS, leaving the closing quote character in its own tiny fragment.
# Example: from «…королевства?». sentencex yields «…королевства?» + «».»
# The second fragment must be re-attached to the previous sentence so that
# _find_quoted_span can still locate the end of the quoted span.
_CLOSE_QUOTE_ARTIFACT_RE = re.compile(
    r'^[\u00bb\u201d\u2019"][\s.!?,;:…\-]*$'
)


def _merge_close_quote_artifacts(sentences: List[str]) -> List[str]:
    """Re-attach orphaned close-quote artifact sentences to the preceding sentence.

    When sentencex splits on ``?`` or ``!`` inside a quoted block, the closing
    quote character (e.g. ``\u201d.`` or ``".``) ends up as a tiny fragment that
    would otherwise be filtered out by the ``MIN_SENTENCE_CHARS`` threshold,
    making ``_find_quoted_span`` unable to locate the end of the span.

    This function finds such fragments and merges them back onto the previous
    sentence so that the close-quote character remains visible to span detection.
    """
    if len(sentences) < 2:
        return sentences
    result: List[str] = []
    for s in sentences:
        stripped = s.strip()
        if result and stripped and _CLOSE_QUOTE_ARTIFACT_RE.match(stripped):
            result[-1] = result[-1].rstrip() + stripped
        else:
            result.append(s)
    return result


def split_into_sentences(text: str, language: str = "ru") -> List[str]:
    """Split *text* into sentences using sentencex.

    Falls back to splitting on double-newlines if sentencex is not available.

    Close-quote artifact fragments (e.g. ``\u201d.`` produced when sentencex
    treats ``?`` or ``!`` inside a quoted block as an EOS) are silently
    re-attached to the preceding sentence before the length filter is applied,
    so that ``_find_quoted_span`` can still detect the quoted-block boundary.
    """
    raw = split_text_by_chunk_boundaries(text, language)
    raw = _merge_close_quote_artifacts(raw)
    return [s for s in raw if len(s.strip()) >= MIN_SENTENCE_CHARS]


def _is_fully_quoted(text: str) -> Optional[Tuple[str, str]]:
    """If the entire *text* is wrapped in a matching quote pair, return (open, close).

    Allows optional punctuation after the closing quote (e.g. «...». ).
    Returns None if not a quoted block.

    Also returns None when the opening quote character is immediately followed by
    punctuation or whitespace (e.g. ``". Narrator text...``) — that pattern indicates
    a *closing* quote artifact left over from a sentence split, not a real opening.
    """
    t = text.strip()
    if not t:
        return None
    open_char = t[0]
    if open_char not in _OPEN_QUOTES:
        return None
    # Reject closing-quote artifacts: a real opening quote is followed by word content,
    # not by punctuation or whitespace (e.g. '". Narrator …' is a close-quote leftover).
    if len(t) < 2 or t[1] in '.!?,;: \t':
        return None
    close_char = _CLOSE_FOR_OPEN[open_char]
    # Find last occurrence of closing quote (may be followed by punctuation/space)
    idx = t.rfind(close_char)
    if idx <= 0:
        return None
    # Everything after the close quote should only be punctuation/whitespace
    after = t[idx + 1:].strip()
    if after and not re.fullmatch(r'[\s.,!?;:\-…]+', after):
        return None
    # The opening quote must come before the closing quote
    return (open_char, close_char)


def _find_quoted_span(
    sentences: List[str], start: int
) -> Optional[Tuple[int, Optional[str], Optional[str]]]:
    """If ``sentences[start]`` opens a quoted block closed in a later sentence, return
    ``(span_end, voiced_last, unvoiced_rest)``; otherwise return ``None``.

    * **span_end** – exclusive index of the span (sentences[start:span_end] are in-quote).
    * **voiced_last** / **unvoiced_rest** – when the sentence that contains the closing
      quote also has narrator text *after* it (e.g. ``…церкви". Плут, переписывавший…``),
      the sentence is split here:  ``voiced_last`` = portion up to and including the
      closing quote + any immediately trailing punctuation; ``unvoiced_rest`` = the
      narrator text that follows.  Both are ``None`` when the sentence ends cleanly
      after the closing quote.

    Uses the **first** occurrence of the closing quote char (not the last) so that a
    close-quote mid-sentence is not confused with a later quote-within-quote.
    """
    t = sentences[start].strip()
    if not t or t[0] not in _OPEN_QUOTES:
        return None
    # Same closing-quote-artifact guard as _is_fully_quoted.
    if len(t) < 2 or t[1] in '.!?,;: \t':
        return None
    open_char = t[0]
    close_char = _CLOSE_FOR_OPEN[open_char]
    # If the closing quote is already present, _is_fully_quoted handles this sentence.
    if close_char in t[1:]:
        return None
    for end in range(start + 1, len(sentences)):
        sent = sentences[end].strip()
        if close_char not in sent:
            continue
        # Use the FIRST occurrence so we don't skip past a mid-sentence close.
        idx = sent.find(close_char)
        tail = sent[idx + 1:]
        tail_stripped = tail.strip()
        if not tail_stripped or re.fullmatch(r'[\s.,!?;:\-…]+', tail_stripped):
            # Clean close — nothing substantive follows the closing quote.
            return (end + 1, None, None)
        # The closing quote is somewhere mid-sentence; split at that point.
        # Absorb any punctuation immediately after the close quote into the voiced part.
        punct_match = re.match(r'^[\s.,!?;:\-…]*', tail)
        punct_suffix = punct_match.group(0).rstrip()   # e.g. "." (no trailing space)
        voiced_last = sent[:idx + 1] + punct_suffix
        # Remainder starts after the consumed punctuation + whitespace.
        remainder_start = idx + 1 + len(punct_match.group(0))
        unvoiced_rest = sent[remainder_start:].strip()
        if not unvoiced_rest:
            return (end + 1, None, None)
        return (end + 1, voiced_last, unvoiced_rest)
    return None


def split_sentences_with_voices(
    text: str,
    language: str = "ru",
    voice2: Optional[str] = None,
) -> List[Tuple[str, Optional[str]]]:
    """Split *text* into (sentence_text, voice_override) pairs.

    When *voice2* is set, sentences that form a fully-quoted block are
    further split internally and tagged with *voice2*.  If *voice2* is None,
    all sentences use the default voice (None).

    Also handles quoted blocks that were split across a ``[chunk_eof]`` boundary:
    if ``sentences[i]`` starts with an opening quote but has no matching closing
    quote, the function looks ahead to find the sentence that closes the block and
    assigns *voice2* to all sentences in the span.
    """
    sentences = split_into_sentences(text, language)
    result: List[Tuple[str, Optional[str]]] = []

    i = 0
    while i < len(sentences):
        sentence = sentences[i]
        quote_pair = _is_fully_quoted(sentence) if voice2 else None
        if quote_pair:
            open_char, close_char = quote_pair
            # Extract inner text (strip outer quotes, preserve trailing punctuation)
            inner = sentence.strip()
            inner = inner[len(open_char):]  # remove opening quote
            close_idx = inner.rfind(close_char)
            after_close = inner[close_idx + len(close_char):].strip()
            inner = inner[:close_idx].strip()  # remove closing quote
            # If the outer sentence had sentence-ending punctuation after the closing quote,
            # append it to the inner text so the last sub-sentence isn't cut off.
            if after_close and re.match(r'^[.!?…]', after_close):
                inner = inner + after_close[0]
            inner_sentences = split_into_sentences(inner, language)
            if len(inner_sentences) > 1:
                logger.debug(
                    "Quoted block split into %d sub-sentences (voice2=%s): %s…",
                    len(inner_sentences), voice2, sentence[:60],
                )
                for sub in inner_sentences:
                    result.append((sub, voice2))
                i += 1
                continue
            # Only one inner sentence — keep original sentence text with voice2
            result.append((sentence, voice2))
        elif voice2:
            # Detect a quoted block split by [chunk_eof]: the opening and closing quotes
            # appear in different sentence items after boundary splitting.
            span_result = _find_quoted_span(sentences, i)
            if span_result is not None:
                span_end, voiced_last, unvoiced_rest = span_result
                logger.debug(
                    "Cross-boundary quoted span sentences %d..%d assigned voice2=%s: %s…",
                    i, span_end - 1, voice2, sentence[:60],
                )
                # All sentences in the span except the last get voice2 unchanged.
                for k in range(i, span_end - 1):
                    result.append((sentences[k], voice2))
                # Last sentence in span: may need to be split.
                if voiced_last is not None:
                    result.append((voiced_last, voice2))
                    result.append((unvoiced_rest, None))
                else:
                    result.append((sentences[span_end - 1], voice2))
                i = span_end
                continue
            result.append((sentence, None))
        else:
            result.append((sentence, None))
        i += 1

    return result


def _apply_boundary_fades(
    data: bytes,
    sampwidth: int,
    nchannels: int,
    fade_samples: int,
    fade_in: bool,
    fade_out: bool,
) -> bytes:
    """Apply linear fade-in / fade-out to raw PCM bytes at chunk boundaries.

    Uses NumPy when available (fast), otherwise a pure-Python fallback.
    Supports 8-, 16-, and 32-bit signed PCM (the most common WAV formats).
    Returns the original *data* unchanged when the format is unsupported or
    *fade_samples* ≤ 0.
    """
    if not (fade_in or fade_out) or fade_samples <= 0:
        return data

    try:
        import numpy as np
        dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}
        dtype = dtype_map.get(sampwidth)
        if dtype is None:
            return data
        samples = np.frombuffer(data, dtype=dtype).copy()
        n_frames = len(samples) // nchannels
        actual = min(fade_samples, n_frames)
        if actual <= 0:
            return data
        # Reshape to (frames, channels) for broadcasting
        s2d = samples.reshape(n_frames, nchannels).astype(np.float32)
        if fade_in:
            ramp = np.linspace(0.0, 1.0, actual, dtype=np.float32)
            s2d[:actual] *= ramp[:, np.newaxis]
        if fade_out:
            ramp = np.linspace(1.0, 0.0, actual, dtype=np.float32)
            s2d[-actual:] *= ramp[:, np.newaxis]
        info = np.iinfo(dtype)
        return np.clip(s2d, info.min, info.max).astype(dtype).tobytes()
    except ImportError:
        pass

    # ── Pure-Python fallback ──────────────────────────────────────────────────
    import array as _arr
    fmt_map = {1: 'b', 2: 'h', 4: 'i'}
    fmt = fmt_map.get(sampwidth)
    if fmt is None:
        return data
    samples = _arr.array(fmt, data)
    n_frames = len(samples) // nchannels
    actual = min(fade_samples, n_frames)
    if actual <= 0:
        return data
    if fade_in:
        for fi in range(actual):
            factor = fi / actual
            for ch in range(nchannels):
                idx = fi * nchannels + ch
                samples[idx] = int(samples[idx] * factor)
    if fade_out:
        for fi in range(actual):
            factor = (actual - fi - 1) / actual
            for ch in range(nchannels):
                idx = (n_frames - actual + fi) * nchannels + ch
                samples[idx] = int(samples[idx] * factor)
    return samples.tobytes()


def _merge_wav_files(chunk_paths: List[str], output_path: str, smooth_join_ms: int = 0) -> None:
    """Fast WAV concatenation using Python's stdlib ``wave`` module (O(n)).

    Optionally applies a short linear fade-out at the tail of each chunk and
    a fade-in at the head of the next chunk (*smooth_join_ms* > 0) to
    eliminate audible clicks / crackling at boundaries.
    """
    import wave

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with wave.open(chunk_paths[0], 'rb') as first_w:
        params = first_w.getparams()

    fade_samples = int(params.framerate * smooth_join_ms / 1000) if smooth_join_ms > 0 else 0
    n = len(chunk_paths)

    with wave.open(output_path, 'wb') as out_w:
        out_w.setnchannels(params.nchannels)
        out_w.setsampwidth(params.sampwidth)
        out_w.setframerate(params.framerate)

        for i, path in enumerate(chunk_paths):
            try:
                with wave.open(path, 'rb') as w:
                    data = w.readframes(w.getnframes())
            except Exception as exc:
                logger.warning("Skipping unreadable chunk %s: %s", path, exc)
                continue

            if fade_samples > 0:
                data = _apply_boundary_fades(
                    data,
                    params.sampwidth,
                    params.nchannels,
                    fade_samples,
                    fade_in=(i > 0),
                    fade_out=(i < n - 1),
                )
            out_w.writeframes(data)

    logger.debug(
        "Merged %d WAV chunks into %s (smooth_join_ms=%d)",
        n, output_path, smooth_join_ms,
    )


def _merge_via_pydub(chunk_paths: List[str], output_path: str, fmt: str, smooth_join_ms: int = 0) -> None:
    """Merge non-WAV audio files using pydub with optional crossfade."""
    try:
        from pydub import AudioSegment  # type: ignore
    except ImportError:
        raise RuntimeError(
            "pydub is required for non-WAV audio merging. "
            "Install it with: pip install pydub"
        )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    combined = None
    for path in chunk_paths:
        seg = AudioSegment.from_file(path)
        if combined is None:
            combined = seg
        elif smooth_join_ms > 0:
            combined = combined.append(seg, crossfade=smooth_join_ms)
        else:
            combined += seg

    if combined is not None:
        combined.export(str(output_path), format=fmt)
    logger.debug("Merged %d chunks via pydub into %s", len(chunk_paths), output_path)


def _trim_trailing_silence(audio_path: str, tail_ms: int = SILENCE_TAIL_MS) -> None:
    """Trim trailing silence from *audio_path* in-place, keeping *tail_ms* ms after the last
    non-silent segment.

    Algorithm:
    1. Find the last non-silent chunk using pydub.silence.detect_nonsilent.
    2. Cut everything after it plus tail_ms.
    3. Overwrite the file.
    """
    try:
        from pydub import AudioSegment  # type: ignore
        from pydub.silence import detect_nonsilent  # type: ignore

        seg = AudioSegment.from_file(audio_path)
        nonsilent_ranges = detect_nonsilent(
            seg,
            min_silence_len=100,
            silence_thresh=SILENCE_THRESH_DBFS,
            seek_step=10,
        )
        if not nonsilent_ranges:
            return  # entire segment is silent — leave as is

        last_end_ms = nonsilent_ranges[-1][1]
        trim_end_ms = min(last_end_ms + tail_ms, len(seg))
        if trim_end_ms >= len(seg) - 10:
            return  # nothing significant to trim

        trimmed = seg[:trim_end_ms]
        ext = Path(audio_path).suffix.lstrip(".") or "wav"
        trimmed.export(audio_path, format=ext)
        logger.debug("Trimmed trailing silence: kept %d ms (was %d ms) in %s",
                     trim_end_ms, len(seg), audio_path)
    except ImportError:
        logger.debug("pydub not available — skipping silence trimming for %s", audio_path)
    except Exception as exc:
        logger.debug("Silence trimming failed for %s: %s", audio_path, exc)


def _merge_audio_files(chunk_paths: List[str], output_path: str, smooth_join_ms: int = 0) -> None:
    """Concatenate audio chunk files into *output_path*.

    For WAV files uses Python's stdlib ``wave`` module (fast, O(n)).
    For other formats falls back to pydub.

    If *smooth_join_ms* > 0, a short linear fade-out / fade-in is applied at
    each chunk boundary to eliminate audible clicks (no pydub needed for WAV).
    """
    if not chunk_paths:
        return
    fmt = Path(output_path).suffix.lstrip(".").lower() or "wav"
    if fmt == "wav":
        _merge_wav_files(chunk_paths, output_path, smooth_join_ms)
    else:
        _merge_via_pydub(chunk_paths, output_path, fmt, smooth_join_ms)


class ChunkedAudioGenerator:
    """Per-chapter chunked TTS synthesiser with filesystem-based resume."""

    def __init__(
        self,
        *,
        config,
        chunk_store: AudioChunkStore,
        tts_provider,
        chunks_base_dir: str,
        run_id: str = "",
    ):
        self.config = config
        self.store = chunk_store
        self.tts_provider = tts_provider
        self.chunks_base_dir = Path(chunks_base_dir)

    def _chunk_dir(self, chapter_key: str) -> Path:
        """Return (and create) the directory for this chapter's chunk files."""
        d = self.chunks_base_dir / chapter_key
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _chunk_path(self, chapter_key: str, s_hash: str, voice_override: Optional[str] = None) -> str:
        ext = self.tts_provider.get_output_file_extension()
        # Single naming scheme regardless of voice: <hash>.<ext>
        # Voice assignment is determined by context (quoted block detection), not the filename.
        return str(self._chunk_dir(chapter_key) / f"{s_hash}.{ext}")

    @contextmanager
    def _voice_override(self, voice: Optional[str]):
        """Temporarily swap config.voice_name for one TTS call."""
        if voice is None or voice == self.config.voice_name:
            yield
            return
        original = self.config.voice_name
        self.config.voice_name = voice
        try:
            yield
        finally:
            self.config.voice_name = original

    def _chapter_wav_is_uptodate(self, output_file: str, chunk_paths: List[str]) -> bool:
        """Return True if *output_file* exists and is newer than every chunk in *chunk_paths*.

        Used to skip redundant re-merges when the chapter WAV already reflects all
        current chunks (e.g. after a previous successful run).
        """
        if not os.path.exists(output_file):
            return False
        wav_mtime = os.path.getmtime(output_file)
        for cp in chunk_paths:
            if not os.path.exists(cp):
                return False
            if os.path.getmtime(cp) > wav_mtime:
                return False
        return True

    def _prepare_tts_text(self, text: str) -> str:
        prepare = getattr(self.tts_provider, "prepare_tts_text", None)
        if callable(prepare):
            return prepare(text)
        return text

    def process_chapter(
        self,
        *,
        chapter_idx: int,
        chapter_key: str,
        text_for_tts: str,
        output_file: str,
        audio_tags: AudioTags,
        synthesize_only: bool = False,
    ) -> bool:
        """Synthesise all sentences for *chapter_key*, then merge into *output_file*.

        Resume logic: if the chunk file already exists on disk, the sentence
        is skipped (no TTS call).  Returns True on success.

        If ``config.voice_name2`` is set, quoted character-speech blocks are
        further split and synthesised with the secondary voice.

        synthesize_only=True
            Produce chunk files but skip the final merge step (useful for
            ``--mode audio_chunks`` where you want to synthesise first and
            merge later).
        """
        voice2 = getattr(self.config, 'voice_name2', None) or None
        sentence_voice_pairs = split_sentences_with_voices(
            text_for_tts, self.config.language or "ru", voice2=voice2
        )
        if not sentence_voice_pairs:
            logger.warning("Chapter %d '%s' produced no sentences; skipping.", chapter_idx, chapter_key)
            return False

        logger.info(
            "Chapter %d: %d sentences to synthesise (chunked mode, voice2=%s).",
            chapter_idx, len(sentence_voice_pairs), voice2,
        )

        # Record sentence texts in version history (INSERT OR IGNORE — no-op if already present).
        registered_new = 0
        registered_existing = 0
        for sentence, _voice in sentence_voice_pairs:
            s_hash = _sentence_hash(sentence)
            if self.store.save_sentence_version(s_hash, sentence):
                registered_new += 1
            else:
                registered_existing += 1
        logger.info(
            "Chapter %d sentence text history: %d new, %d already known.",
            chapter_idx, registered_new, registered_existing,
        )

        # Synthesise missing chunks (file existence = already done).
        synthesised = 0
        skipped = 0
        errors = 0
        for pos, (sentence, voice) in enumerate(sentence_voice_pairs):
            s_hash = _sentence_hash(sentence)
            chunk_path = self._chunk_path(chapter_key, s_hash, voice)
            if os.path.exists(chunk_path):
                skipped += 1
                continue
            try:
                with self._voice_override(voice):
                    self.tts_provider.text_to_speech(
                        self._prepare_tts_text(sentence), chunk_path, audio_tags
                    )
                if getattr(self.config, "tts_trim_silence", True):
                    _trim_trailing_silence(chunk_path)
                synthesised += 1
            except Exception as exc:
                logger.error(
                    "Chapter %d sentence %d synthesis failed: %s", chapter_idx, pos, exc
                )
                errors += 1

        logger.info(
            "Chapter %d synthesis done: %d new, %d skipped, %d errors.",
            chapter_idx, synthesised, skipped, errors,
        )

        if errors > 0:
            logger.warning(
                "Chapter %d has %d synthesis errors; output may be incomplete.", chapter_idx, errors
            )

        if synthesize_only:
            logger.info(
                "Chapter %d chunk synthesis complete (synthesize_only=True, merge skipped).",
                chapter_idx,
            )
            return True

        # Collect synthesised chunk paths in sentence order (FS-based).
        chunk_paths = [
            self._chunk_path(chapter_key, _sentence_hash(s), v)
            for s, v in sentence_voice_pairs
            if os.path.exists(self._chunk_path(chapter_key, _sentence_hash(s), v))
        ]
        if not chunk_paths:
            logger.error("Chapter %d: no audio chunks available for merging.", chapter_idx)
            return False

        # Skip merge if the chapter WAV already reflects all current chunks.
        if self._chapter_wav_is_uptodate(output_file, chunk_paths):
            logger.info(
                "Chapter %d WAV is up-to-date, skipping re-merge: %s", chapter_idx, output_file
            )
            return True

        # Smooth join setting from config (default: 30 ms).
        smooth_join_ms = 0
        if getattr(self.config, "tts_chunk_smooth_join", True):
            smooth_join_ms = int(getattr(self.config, "tts_chunk_smooth_join_ms", 30) or 30)

        # Merge chunks into the chapter output file.
        try:
            _merge_audio_files(chunk_paths, output_file, smooth_join_ms)
            logger.info("Chapter %d merged into %s", chapter_idx, output_file)
            return True
        except Exception as exc:
            logger.error("Chapter %d merge failed: %s", chapter_idx, exc)
            return False
