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
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
START_CLICK_PEAK_THRESHOLD = 0.03

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


def _pcm32_to_int16(data: bytes) -> bytes:
    """Convert 32-bit signed PCM bytes (little-endian) to 16-bit signed PCM.

    PCM-32 from some TTS servers stores full-range int32 values — right-shift
    by 16 gives the correct int16 equivalent.
    """
    import struct as _struct
    import array as _array
    n = len(data) // 4
    int32_vals = _struct.unpack_from(f'<{n}i', data)
    return _array.array('h', (v >> 16 for v in int32_vals)).tobytes()


def _read_wav_frames(path: str) -> Tuple[int, int, int, bytes]:
    """Return ``(nchannels, sampwidth, framerate, frames_bytes)``.

    Normalises **all** WAV formats to 16-bit signed PCM so that
    ``_merge_wav_files`` always writes a uniform int16 output regardless of
    whether the TTS server produced PCM-16, PCM-32, or IEEE float-32:

    * PCM-16  (format 1, sampwidth=2) → returned as-is.
    * PCM-32  (format 1, sampwidth=4) → right-shifted to int16.
    * Float-32 (format 3)             → converted to int16 via float→int clamp.

    Raises ``wave.Error`` for truly unreadable or malformed files.
    """
    import wave as _wave
    import struct as _struct
    import array as _array

    # ── Fast path: standard PCM WAV ──────────────────────────────────────────
    try:
        with _wave.open(path, 'rb') as w:
            p = w.getparams()
            frames = w.readframes(p.nframes)
            if p.sampwidth == 2:
                # Already int16 — return as-is.
                return p.nchannels, 2, p.framerate, frames
            if p.sampwidth == 4:
                # PCM-32: right-shift to int16.
                return p.nchannels, 2, p.framerate, _pcm32_to_int16(frames)
            # Other widths (8-bit, 24-bit): return raw and let caller deal with it.
            return p.nchannels, p.sampwidth, p.framerate, frames
    except _wave.Error as exc:
        if 'unknown format: 3' not in str(exc):
            raise

    # ── Slow path: IEEE float-32 WAV — parse RIFF manually ───────────────────
    with open(path, 'rb') as f:
        raw = f.read()

    if raw[:4] != b'RIFF' or raw[8:12] != b'WAVE':
        raise _wave.Error(f"not a valid WAV file: {path}")

    nchannels: Optional[int] = None
    sample_rate: Optional[int] = None
    audio_data: bytes = b''

    pos = 12
    while pos + 8 <= len(raw):
        cid = raw[pos:pos + 4]
        csz = _struct.unpack_from('<I', raw, pos + 4)[0]
        if cid == b'fmt ':
            nchannels = _struct.unpack_from('<H', raw, pos + 8 + 2)[0]
            sample_rate = _struct.unpack_from('<I', raw, pos + 8 + 4)[0]
        elif cid == b'data':
            audio_data = raw[pos + 8: pos + 8 + csz]
        pos += 8 + csz + (csz & 1)  # RIFF pads odd-sized chunks

    if not nchannels or not sample_rate:
        raise _wave.Error(f"malformed IEEE float WAV (missing fmt/data): {path}")

    # Convert float32 LE samples to int16 PCM
    float_count = len(audio_data) // 4
    floats = _struct.unpack_from(f'<{float_count}f', audio_data)
    pcm = _array.array('h', (
        max(-32768, min(32767, int(f * 32767.0))) for f in floats
    ))
    return nchannels, 2, sample_rate, pcm.tobytes()


def _remove_dc_offset(data: bytes, sampwidth: int, nchannels: int) -> bytes:
    """Subtract the mean (DC bias) from each channel of raw PCM data.

    A non-zero DC offset causes a step-change click at each chunk boundary
    even if the fade/crossfade brings the amplitude to zero, because the
    *baseline* is different between chunks.  Removing it beforehand makes
    every chunk oscillate symmetrically around zero.

    No-ops when |DC| < 1 LSB or when numpy is unavailable.
    """
    try:
        import numpy as np
        dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}
        dtype = dtype_map.get(sampwidth)
        if dtype is None:
            return data
        samples = np.frombuffer(data, dtype=dtype).astype(np.float32)
        dc = float(samples.mean())
        if abs(dc) < 1.0:
            return data
        info = np.iinfo(dtype)
        corrected = np.clip(samples - dc, info.min, info.max).astype(dtype)
        return corrected.tobytes()
    except ImportError:
        return data


def _write_merged_wav_numpy(
    chunks: "List[bytes]",
    output_path: str,
    nchannels: int,
    sampwidth: int,
    framerate: int,
    crossfade_samples: int,
    gap_bytes: bytes,
) -> None:
    """Merge *chunks* into *output_path* using a streaming O(N) approach.

    Only the crossfade tail of the previous chunk (≤30 ms) is kept in memory
    between iterations; the body is written to disk immediately.  This avoids
    the O(N²) buffer-copy loop of the original iterative ``_crossfade_pcm``
    approach, which caused multi-hour runtimes for chapters with 1000+ chunks.

    Requires NumPy; raises ``ImportError`` if not available (caller falls back
    to the legacy path).
    """
    import wave
    import numpy as np

    dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}
    dt = dtype_map.get(sampwidth)
    if dt is None:
        raise TypeError(f"Unsupported sampwidth: {sampwidth}")
    info = np.iinfo(dt)
    cf = crossfade_samples

    # Pre-compute cosine ramps once; reused at every boundary.
    ramp_down: "np.ndarray | None" = None
    ramp_up:   "np.ndarray | None" = None
    if cf > 0:
        t = np.linspace(0.0, 1.0, cf, dtype=np.float64)
        ramp_down = (0.5 * (1.0 + np.cos(np.pi * t)))[:, np.newaxis]  # 1→0
        ramp_up   = (0.5 * (1.0 - np.cos(np.pi * t)))[:, np.newaxis]  # 0→1

    def _to_arr(data: bytes) -> "np.ndarray":
        return np.frombuffer(data, dtype=dt).reshape(-1, nchannels).astype(np.float64)

    def _write(w: "wave.Wave_write", arr: "np.ndarray") -> None:
        if len(arr):
            w.writeframes(np.clip(arr, info.min, info.max).astype(dt).tobytes())

    # Interleave silent gap chunks when gap_ms > 0.
    all_chunks: List[bytes] = list(chunks)
    if gap_bytes and len(all_chunks) > 1:
        with_gaps: List[bytes] = []
        for i, c in enumerate(all_chunks):
            with_gaps.append(c)
            if i < len(all_chunks) - 1:
                with_gaps.append(gap_bytes)
        all_chunks = with_gaps

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with wave.open(output_path, "wb") as w:
        w.setnchannels(nchannels)
        w.setsampwidth(sampwidth)
        w.setframerate(framerate)

        # ``tail`` = float64 array of the last ``cf`` frames from the previous
        # chunk, held in memory for blending with the next chunk's head.
        tail: "np.ndarray | None" = None

        for i, data in enumerate(all_chunks):
            arr = _to_arr(data)
            is_last = (i == len(all_chunks) - 1)

            # ── Blend previous tail with our head ────────────────────────────
            if tail is not None:
                if cf > 0 and len(tail) > 0 and len(arr) > 0:
                    actual_cf = min(cf, len(tail), len(arr))
                    region = (
                        tail[:actual_cf] * ramp_down[:actual_cf]
                        + arr[:actual_cf] * ramp_up[:actual_cf]
                    )
                    _write(w, region)
                    arr = arr[actual_cf:]   # consume blended head
                else:
                    _write(w, tail)
                tail = None

            # ── Write body; save tail for next boundary ───────────────────────
            if not is_last and cf > 0 and len(arr) > cf:
                _write(w, arr[:-cf])       # body (all except last cf frames)
                tail = arr[-cf:].copy()    # tail saved for next iteration
            else:
                _write(w, arr)             # last chunk — write everything

        # Flush any leftover tail (shouldn't happen, but be safe).
        if tail is not None:
            _write(w, tail)


def _crossfade_pcm(
    data_a: bytes,
    data_b: bytes,
    sampwidth: int,
    nchannels: int,
    crossfade_samples: int,
) -> bytes:
    """True crossfade: overlap the end of *data_a* with the start of *data_b*.

    Uses complementary cosine ramps (equal-power) so the boundary region
    always sums to full energy — no volume dip, no click.

    The returned buffer has length:
        len(data_a) + len(data_b) - crossfade_samples * frame_size

    Falls back to a pure-Python cosine crossfade when NumPy is unavailable
    (slower but correct).  Falls back to plain concatenation when *crossfade_samples*
    is 0 or either operand is shorter than the crossfade window.
    """
    frame_size = sampwidth * nchannels
    n_a = len(data_a) // frame_size
    n_b = len(data_b) // frame_size
    cf = min(crossfade_samples, n_a, n_b)
    if cf <= 0:
        return data_a + data_b

    try:
        import numpy as np
        dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}
        dtype = dtype_map.get(sampwidth)
        if dtype is None:
            return data_a + data_b

        a = np.frombuffer(data_a, dtype=dtype).reshape(n_a, nchannels).astype(np.float64)
        b = np.frombuffer(data_b, dtype=dtype).reshape(n_b, nchannels).astype(np.float64)

        t = np.linspace(0.0, 1.0, cf, dtype=np.float64)
        ramp_down = 0.5 * (1.0 + np.cos(np.pi * t))   # 1 → 0  (cosine)
        ramp_up   = 0.5 * (1.0 - np.cos(np.pi * t))   # 0 → 1  (cosine)

        region = (a[-cf:] * ramp_down[:, np.newaxis] +
                  b[:cf]  * ramp_up  [:, np.newaxis])

        info = np.iinfo(dtype)
        result = np.concatenate([a[:-cf], np.clip(region, info.min, info.max), b[cf:]])
        return result.astype(dtype).tobytes()

    except ImportError:
        pass

    # ── Pure-Python fallback (no numpy) ──────────────────────────────────────
    import array as _arr
    import math
    fmt_map = {1: 'b', 2: 'h', 4: 'i'}
    fmt = fmt_map.get(sampwidth)
    if fmt is None:
        return data_a + data_b

    arr_a = _arr.array(fmt, data_a)
    arr_b = _arr.array(fmt, data_b)
    n_a_f = len(arr_a) // nchannels
    n_b_f = len(arr_b) // nchannels
    cf2 = min(crossfade_samples, n_a_f, n_b_f)

    result = _arr.array(fmt, arr_a[:(n_a_f - cf2) * nchannels])
    for fi in range(cf2):
        t = fi / cf2
        down = 0.5 * (1.0 + math.cos(math.pi * t))
        up   = 0.5 * (1.0 - math.cos(math.pi * t))
        for ch in range(nchannels):
            i_a = (n_a_f - cf2 + fi) * nchannels + ch
            i_b = fi * nchannels + ch
            mixed = int(arr_a[i_a] * down + arr_b[i_b] * up)
            limit = (1 << (sampwidth * 8 - 1)) - 1
            result.append(max(-limit - 1, min(limit, mixed)))
    result.extend(arr_b[cf2 * nchannels:])
    return result.tobytes()


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


# LF-preamble detection thresholds
_LF_PREAMBLE_ZCR_ARTIFACT_MAX = 0.040  # avg ZCR over first 60ms must be below this
_LF_PREAMBLE_LFR_ARTIFACT_MIN = 0.60   # peak LF ratio (LPF-proxy) over first 60ms
_LF_PREAMBLE_RMS_MIN = 0.012           # avg RMS of artifact zone (rel. to local peak)
_LF_PREAMBLE_SPEECH_ZCR_MIN = 0.030   # ZCR above this → speech has started
_LF_PREAMBLE_CHECK_MS = 350           # only look within the first N ms
_LF_PREAMBLE_ARTIFACT_ZONE_MS = 60    # initial "artifact zone" to characterise
_LF_PREAMBLE_WINDOW_MS = 20           # analysis window per step
_LF_PREAMBLE_HOP_MS = 10             # hop between windows
_LF_PREAMBLE_LPF_MS = 2.0            # moving-average window → ~500 Hz LPF proxy
# Hard cap on how many ms the function may trim.  The genuine CosyVoice LF
# pre-phonation burst never exceeds ~50 ms; anything longer is almost certainly
# real speech (e.g. a word starting with a nasal like "М" that has low ZCR and
# high low-frequency energy throughout its duration).
_LF_PREAMBLE_MAX_TRIM_MS = 80


def _lf_preamble_trim_frames(
    samples,       # array.array('h') — int16 PCM, little-endian
    nchannels: int,
    framerate: int,
) -> int:
    """Return the number of frames to remove from the start to skip a low-frequency
    preamble artifact (the "ock" TTS pre-phonation burst).

    Detection strategy:
      1. Compute average ZCR and peak LF-ratio over the first
         _LF_PREAMBLE_ARTIFACT_ZONE_MS (60 ms) — the "artifact zone".
      2. If avg ZCR < _LF_PREAMBLE_ZCR_ARTIFACT_MAX  AND
            peak LFR > _LF_PREAMBLE_LFR_ARTIFACT_MIN  AND
            avg RMS  > _LF_PREAMBLE_RMS_MIN            → artifact detected.
      3. Walk forward from the artifact zone's end to find the first window
         where ZCR ≥ _LF_PREAMBLE_SPEECH_ZCR_MIN (speech onset).
      4. Return the speech onset frame index as the trim point.

    Returns 0 if no preamble is detected.
    """
    n_frames = len(samples) // nchannels
    max_check = min(n_frames, int(framerate * _LF_PREAMBLE_CHECK_MS / 1000))
    artifact_zone = min(max_check, int(framerate * _LF_PREAMBLE_ARTIFACT_ZONE_MS / 1000))
    win_fr = max(2, int(framerate * _LF_PREAMBLE_WINDOW_MS / 1000))
    hop_fr = max(1, int(framerate * _LF_PREAMBLE_HOP_MS / 1000))
    lpf_fr = max(1, int(framerate * _LF_PREAMBLE_LPF_MS / 1000))

    if max_check < win_fr * 2:
        return 0

    # Build normalised mono view of the first max_check frames
    total_look = min(max_check + win_fr, n_frames)
    if nchannels == 1:
        raw = [samples[i] for i in range(total_look)]
    else:
        raw = [
            sum(samples[i * nchannels + ch] for ch in range(nchannels)) / nchannels
            for i in range(total_look)
        ]

    abs_peak = max(abs(v) for v in raw) if raw else 0
    if abs_peak < 32767 * 0.01:      # practically silent file — nothing to trim
        return 0
    norm = [v / abs_peak for v in raw]

    def _analyze(start: int):
        """Return (zcr, lfr, rms) for a window starting at *start* (frame index)."""
        end = min(len(norm), start + win_fr)
        w = norm[start:end]
        n = len(w)
        if n < 2:
            return 0.0, 0.0, 0.0
        # RMS
        rms = (sum(v * v for v in w) / n) ** 0.5
        # ZCR
        zcr = sum(1 for i in range(1, n) if (w[i] >= 0) != (w[i - 1] >= 0)) / (2 * n)
        # LF-ratio via moving-average LPF proxy (rectangular, cutoff ~1/lpf_fr Hz per sample)
        acc = 0.0
        lpf = []
        for i, v in enumerate(w):
            acc += v
            if i >= lpf_fr:
                acc -= w[i - lpf_fr]
            lpf.append(acc / min(i + 1, lpf_fr))
        lpf_rms = (sum(v * v for v in lpf) / n) ** 0.5
        lfr = lpf_rms / (rms + 1e-9)
        return zcr, lfr, rms

    # ── Characterise the artifact zone (first 60 ms) ─────────────────────────
    artifact_zcrs = []
    artifact_rms_vals = []
    artifact_max_lfr = 0.0

    for start in range(0, artifact_zone, hop_fr):
        zcr, lfr, rms = _analyze(start)
        artifact_zcrs.append(zcr)
        artifact_rms_vals.append(rms)
        artifact_max_lfr = max(artifact_max_lfr, lfr)

    if not artifact_zcrs:
        return 0

    avg_zcr = sum(artifact_zcrs) / len(artifact_zcrs)
    avg_rms = sum(artifact_rms_vals) / len(artifact_rms_vals)

    if avg_rms < _LF_PREAMBLE_RMS_MIN:
        return 0  # starts with silence — nothing to trim
    if avg_zcr >= _LF_PREAMBLE_ZCR_ARTIFACT_MAX:
        return 0  # ZCR too high — looks like normal speech or noise
    if artifact_max_lfr < _LF_PREAMBLE_LFR_ARTIFACT_MIN:
        return 0  # signal not dominated by low frequencies

    # ── Walk forward to find where speech begins ────────────────────────────
    speech_onset_frame = 0
    max_trim_frames = int(framerate * _LF_PREAMBLE_MAX_TRIM_MS / 1000)
    search_start = max(hop_fr, artifact_zone - win_fr)
    for start in range(search_start, max_check, hop_fr):
        zcr, _lfr, _rms = _analyze(start)
        if zcr >= _LF_PREAMBLE_SPEECH_ZCR_MIN:
            speech_onset_frame = start
            break

    if speech_onset_frame <= 0:
        return 0  # no speech found after artifact — don't trim

    if speech_onset_frame > max_trim_frames:
        # Speech onset is too far in — almost certainly real speech (e.g. a word
        # starting with a low-ZCR nasal consonant).  Genuine CosyVoice preamble
        # bursts always resolve within _LF_PREAMBLE_MAX_TRIM_MS ms.
        logger.debug(
            "LF preamble: speech onset at %.0f ms exceeds max trim %.0f ms — skipping trim "
            "(avg_ZCR=%.3f max_LFR=%.3f avg_RMS=%.3f)",
            speech_onset_frame * 1000.0 / framerate,
            _LF_PREAMBLE_MAX_TRIM_MS,
            avg_zcr, artifact_max_lfr, avg_rms,
        )
        return 0

    logger.debug(
        "LF preamble detected: trim=%.0f ms "
        "(avg_ZCR=%.3f max_LFR=%.3f avg_RMS=%.3f), speech onset at %.0f ms",
        speech_onset_frame * 1000.0 / framerate,
        avg_zcr, artifact_max_lfr, avg_rms,
        speech_onset_frame * 1000.0 / framerate,
    )
    return speech_onset_frame


def _remove_lf_preamble_from_pcm(
    data: bytes,
    sampwidth: int,
    nchannels: int,
    framerate: int,
    fade_ms: int,
) -> bytes:
    """Detect and remove a low-frequency pre-speech preamble artifact from PCM data.

    Only operates on 16-bit PCM.  Returns *data* unchanged when no artifact is found.
    """
    if sampwidth != 2 or nchannels <= 0 or framerate <= 0:
        return data

    import array as _arr
    import sys as _sys

    samples = _arr.array('h')
    samples.frombytes(data)
    if _sys.byteorder != 'little':
        samples.byteswap()

    trim_frames = _lf_preamble_trim_frames(samples, nchannels, framerate)
    if trim_frames <= 0:
        return data

    del samples[:trim_frames * nchannels]

    fade_frames = min(len(samples) // nchannels, int(framerate * fade_ms / 1000))
    for frame in range(fade_frames):
        factor = (frame + 1) / fade_frames
        for ch in range(nchannels):
            idx = frame * nchannels + ch
            samples[idx] = int(samples[idx] * factor)

    if _sys.byteorder != 'little':
        samples.byteswap()
    return samples.tobytes()


# ---------------------------------------------------------------------------
# Gap-preamble detection: [active content] → [silence gap] → [speech]
# ---------------------------------------------------------------------------
# CosyVoice (and other TTS engines) occasionally emit a spurious vocalization
# at the very start of a chunk — before any actual speech.  The artifact is
# typically 200-600 ms long and is followed by a clear silence gap (≥ 200 ms)
# before the real first word.  The lf_preamble detector above handles a *short*
# (< 80 ms) low-frequency burst; this detector handles the longer "warm-up"
# artifact pattern.
#
# Detection criteria:
#   1. The chunk starts with active audio (not silence).
#   2. Within the first _GAP_PREAMBLE_MAX_MS, a silence gap appears.
#   3. The gap lasts at least _GAP_PREAMBLE_MIN_GAP_MS.
#   4. Active audio (real speech) follows the gap.
#   5. The inferred speech-start point is at most _GAP_PREAMBLE_MAX_TRIM_MS.
#
# Safety: only trim if the silence gap is long (≥ 300 ms by default) to avoid
# accidentally cutting a natural comma pause or the first word of the sentence.

_GAP_PREAMBLE_MAX_MS = 700        # search for silence gap starting within first N ms
_GAP_PREAMBLE_MIN_GAP_MS = 300    # silence gap must last at least this long
_GAP_PREAMBLE_SILENCE_THRESH = 0.04  # normalised RMS below this → silence (~−28 dBFS)
_GAP_PREAMBLE_WIN_MS = 10         # RMS analysis window
_GAP_PREAMBLE_MAX_TRIM_MS = 1200  # hard cap: inferred speech start must be before this


def _gap_preamble_trim_frames(
    samples,       # array.array('h') — int16 PCM, little-endian
    nchannels: int,
    framerate: int,
) -> int:
    """Return the number of frames to remove from the start to skip a
    gap-preamble artifact (spurious vocalization followed by silence).

    Returns 0 if the pattern is not detected or if trimming would be unsafe.
    """
    n_frames = len(samples) // nchannels
    win_fr = max(1, int(framerate * _GAP_PREAMBLE_WIN_MS / 1000))
    max_search_fr = min(n_frames, int(framerate * _GAP_PREAMBLE_MAX_MS / 1000))
    min_gap_fr = max(win_fr, int(framerate * _GAP_PREAMBLE_MIN_GAP_MS / 1000))
    max_trim_fr = int(framerate * _GAP_PREAMBLE_MAX_TRIM_MS / 1000)

    if n_frames < win_fr * 3:
        return 0

    # Build a normalised mono view up to max_trim_frames.
    total_look = min(n_frames, max_trim_fr + win_fr)
    if nchannels == 1:
        mono = [samples[i] for i in range(total_look)]
    else:
        mono = [
            sum(samples[i * nchannels + ch] for ch in range(nchannels)) / nchannels
            for i in range(total_look)
        ]

    abs_peak = max(abs(v) for v in mono) if mono else 0
    if abs_peak < 32767 * 0.005:
        return 0  # essentially silent — nothing to trim

    thresh = _GAP_PREAMBLE_SILENCE_THRESH  # normalised

    def _win_rms(start: int) -> float:
        end = min(len(mono), start + win_fr)
        w = mono[start:end]
        if not w:
            return 0.0
        return (sum(v * v for v in w) / len(w)) ** 0.5 / abs_peak

    # ── Condition 1: chunk starts with active content ────────────────────────
    if _win_rms(0) < thresh:
        return 0  # starts with silence — no preamble

    # ── Condition 2: find first silence gap within max_search_fr ────────────
    gap_start_fr: Optional[int] = None
    pos = win_fr  # start searching after the first active window
    while pos < max_search_fr:
        if _win_rms(pos) < thresh:
            gap_start_fr = pos
            break
        pos += win_fr

    if gap_start_fr is None:
        return 0  # no silence gap found — no preamble

    # ── Condition 3: measure gap length ──────────────────────────────────────
    gap_pos = gap_start_fr
    while gap_pos < total_look and _win_rms(gap_pos) < thresh:
        gap_pos += win_fr

    gap_fr = gap_pos - gap_start_fr
    if gap_fr < min_gap_fr:
        return 0  # gap too short (likely a natural intra-word pause)

    # ── Condition 4: active content follows the gap ───────────────────────────
    speech_start_fr = gap_pos
    if speech_start_fr >= len(mono) or _win_rms(speech_start_fr) < thresh:
        return 0  # nothing follows the gap — don't trim

    # ── Condition 5: hard cap on trim amount ─────────────────────────────────
    if speech_start_fr > max_trim_fr:
        logger.debug(
            "Gap preamble: speech onset at %.0f ms exceeds hard cap %.0f ms — skipping",
            speech_start_fr * 1000.0 / framerate,
            _GAP_PREAMBLE_MAX_TRIM_MS,
        )
        return 0

    logger.debug(
        "Gap preamble detected: preamble=0..%.0f ms, gap=%.0f..%.0f ms, "
        "trimming %.0f ms",
        gap_start_fr * 1000.0 / framerate,
        gap_start_fr * 1000.0 / framerate,
        speech_start_fr * 1000.0 / framerate,
        speech_start_fr * 1000.0 / framerate,
    )
    return speech_start_fr


def _remove_gap_preamble_from_pcm(
    data: bytes,
    sampwidth: int,
    nchannels: int,
    framerate: int,
    fade_ms: int,
) -> bytes:
    """Detect and remove a gap-preamble artifact from PCM data.

    Only operates on 16-bit PCM.  Returns *data* unchanged when no artifact
    is found.
    """
    if sampwidth != 2 or nchannels <= 0 or framerate <= 0:
        return data

    import array as _arr
    import sys as _sys

    samples = _arr.array('h')
    samples.frombytes(data)
    if _sys.byteorder != 'little':
        samples.byteswap()

    trim_frames = _gap_preamble_trim_frames(samples, nchannels, framerate)
    if trim_frames <= 0:
        return data

    del samples[:trim_frames * nchannels]

    fade_frames = min(len(samples) // nchannels, int(framerate * fade_ms / 1000))
    for frame in range(fade_frames):
        factor = (frame + 1) / fade_frames
        for ch in range(nchannels):
            idx = frame * nchannels + ch
            samples[idx] = int(samples[idx] * factor)

    if _sys.byteorder != 'little':
        samples.byteswap()
    return samples.tobytes()


def _remove_start_click_from_pcm(    data: bytes,
    sampwidth: int,
    nchannels: int,
    framerate: int,
    trim_ms: int,
    fade_ms: int,
    peak_threshold: float = START_CLICK_PEAK_THRESHOLD,
) -> bytes:
    """Remove a short synthetic click/burst at the beginning of a PCM chunk.

    CosyVoice can emit a loud 5-10 ms transient before the useful signal.  We
    only touch 16-bit PCM here because all float WAV chunks are converted to
    int16 by _read_wav_frames before merge.
    """
    if trim_ms <= 0 or sampwidth != 2 or nchannels <= 0 or framerate <= 0:
        return data

    import array as _arr
    import sys as _sys

    samples = _arr.array('h')
    samples.frombytes(data)
    if _sys.byteorder != 'little':
        samples.byteswap()

    n_frames = len(samples) // nchannels
    if n_frames <= 1:
        return data

    max_trim_frames = min(n_frames - 1, int(framerate * trim_ms / 1000))
    if max_trim_frames <= 0:
        return data

    detect_frames = min(max_trim_frames, max(1, int(framerate * 0.003)))
    detect_samples = detect_frames * nchannels
    peak = max(abs(v) for v in samples[:detect_samples])
    if peak < int(32767 * peak_threshold):
        return data

    quiet_peak = int(32767 * min(peak_threshold * 0.5, 0.015))
    quiet_window_frames = max(1, int(framerate * 0.0015))
    search_start = min(max_trim_frames, max(1, int(framerate * 0.003)))
    trim_frames = min(max_trim_frames, max(1, int(framerate * 0.005)))
    for frame in range(search_start, max_trim_frames - quiet_window_frames + 1):
        window_start = frame * nchannels
        window_end = (frame + quiet_window_frames) * nchannels
        if max(abs(v) for v in samples[window_start:window_end]) <= quiet_peak:
            trim_frames = frame
            break

    del samples[:trim_frames * nchannels]

    fade_frames = min(len(samples) // nchannels, int(framerate * fade_ms / 1000))
    if fade_frames > 0:
        for frame in range(fade_frames):
            factor = (frame + 1) / fade_frames
            for ch in range(nchannels):
                idx = frame * nchannels + ch
                samples[idx] = int(samples[idx] * factor)

    if _sys.byteorder != 'little':
        samples.byteswap()
    return samples.tobytes()


def _merge_wav_files(
    chunk_paths: List[str],
    output_path: str,
    smooth_join_ms: int = 0,
    dc_remove: bool = False,
    gap_ms: int = 0,
    start_declick_ms: int = 0,
    start_declick_fade_ms: int = 0,
    lf_preamble_fade_ms: int = 0,
    gap_preamble_fade_ms: int = 0,
) -> None:
    """Fast WAV concatenation with optional de-click, DC removal, and crossfade."""
    import wave

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # ── Read all chunks ───────────────────────────────────────────────────────
    out_nchannels = out_sampwidth = out_framerate = None
    chunks: List[bytes] = []

    for cp in chunk_paths:
        try:
            nc, sw, fr, data = _read_wav_frames(cp)
        except Exception as exc:
            logger.warning("Skipping unreadable chunk %s: %s", cp, exc)
            continue
        if out_nchannels is None:
            out_nchannels, out_sampwidth, out_framerate = nc, sw, fr
        if start_declick_ms > 0:
            data = _remove_start_click_from_pcm(
                data,
                sw,
                nc,
                fr,
                start_declick_ms,
                start_declick_fade_ms,
            )
        if lf_preamble_fade_ms > 0:
            data = _remove_lf_preamble_from_pcm(data, sw, nc, fr, lf_preamble_fade_ms)
        if gap_preamble_fade_ms > 0:
            data = _remove_gap_preamble_from_pcm(data, sw, nc, fr, gap_preamble_fade_ms)
        if dc_remove:
            data = _remove_dc_offset(data, sw, nc)
        chunks.append(data)

    if not chunks:
        raise wave.Error("No readable WAV chunks found among %d paths" % len(chunk_paths))

    # ── Build silence gap bytes (inserted between chunks when gap_ms > 0) ────
    gap_bytes = b''
    if gap_ms > 0:
        gap_frames = int(out_framerate * gap_ms / 1000)
        gap_bytes = bytes(gap_frames * out_sampwidth * out_nchannels)

    # ── Merge with crossfade ──────────────────────────────────────────────────
    crossfade_samples = int(out_framerate * smooth_join_ms / 1000) if smooth_join_ms > 0 else 0

    # Fast O(N) streaming path (requires numpy).  Streams the body of every
    # chunk directly to disk, holding only a crossfade-length tail (≤30 ms)
    # in memory between iterations.  Falls back to the legacy O(N²) in-memory
    # loop when numpy is absent or sampwidth is exotic.
    try:
        _write_merged_wav_numpy(
            chunks,
            output_path,
            nchannels=out_nchannels,
            sampwidth=out_sampwidth,
            framerate=out_framerate,
            crossfade_samples=crossfade_samples,
            gap_bytes=gap_bytes,
        )
    except (ImportError, TypeError):
        # ── Legacy O(N²) fallback ─────────────────────────────────────────────
        combined = chunks[0]
        for next_data in chunks[1:]:
            if gap_bytes:
                # Append silence, then crossfade silence-end → next chunk start
                combined = _crossfade_pcm(
                    combined + gap_bytes, next_data,
                    out_sampwidth, out_nchannels, crossfade_samples,
                )
            else:
                combined = _crossfade_pcm(
                    combined, next_data,
                    out_sampwidth, out_nchannels, crossfade_samples,
                )

        with wave.open(output_path, 'wb') as out_w:
            out_w.setnchannels(out_nchannels)
            out_w.setsampwidth(out_sampwidth)
            out_w.setframerate(out_framerate)
            out_w.writeframes(combined)

    logger.debug(
        "Merged %d WAV chunks into %s "
        "(crossfade=%dms, dc_remove=%s, gap=%dms, start_declick=%dms, lf_preamble=%s)",
        len(chunks),
        output_path,
        smooth_join_ms,
        dc_remove,
        gap_ms,
        start_declick_ms,
        f"fade={lf_preamble_fade_ms}ms" if lf_preamble_fade_ms else "off",
    )


def _merge_via_pydub(
    chunk_paths: List[str],
    output_path: str,
    fmt: str,
    smooth_join_ms: int = 0,
    gap_ms: int = 0,
    start_declick_ms: int = 0,
    start_declick_fade_ms: int = 0,
    lf_preamble_fade_ms: int = 0,
    gap_preamble_fade_ms: int = 0,
) -> None:
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
        if start_declick_ms > 0 and len(seg) > start_declick_ms:
            seg = seg[start_declick_ms:]
            if start_declick_fade_ms > 0:
                seg = seg.fade_in(start_declick_fade_ms)
        if lf_preamble_fade_ms > 0:
            import array as _arr
            _raw = seg.get_array_of_samples()
            _data = _arr.array('h', _raw)
            _nc = seg.channels
            _fr = seg.frame_rate
            _trim = _lf_preamble_trim_frames(_data, _nc, _fr)
            if _trim > 0:
                seg = seg[int(_trim * 1000.0 / _fr):]
                if lf_preamble_fade_ms > 0:
                    seg = seg.fade_in(lf_preamble_fade_ms)
        if gap_preamble_fade_ms > 0:
            import array as _arr2
            _raw2 = seg.get_array_of_samples()
            _data2 = _arr2.array('h', _raw2)
            _nc2 = seg.channels
            _fr2 = seg.frame_rate
            _trim2 = _gap_preamble_trim_frames(_data2, _nc2, _fr2)
            if _trim2 > 0:
                seg = seg[int(_trim2 * 1000.0 / _fr2):]
                seg = seg.fade_in(gap_preamble_fade_ms)
        if combined is None:
            combined = seg
            continue
        if gap_ms > 0:
            combined += AudioSegment.silent(duration=gap_ms, frame_rate=combined.frame_rate)
        if smooth_join_ms > 0:
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


# ---------------------------------------------------------------------------
# Per-voice audio tempo (time-stretch via ffmpeg atempo)
# ---------------------------------------------------------------------------

def _atempo_filter(tempo: float) -> str:
    """Return an ffmpeg audio-filter string that applies *tempo* time-stretch.

    The ``atempo`` filter accepts values in the range [0.5, 2.0].  For values
    outside this range we chain multiple ``atempo`` stages, e.g.::

        tempo=3.0  →  "atempo=2.0,atempo=1.5"
        tempo=0.25 →  "atempo=0.5,atempo=0.5"

    A tempo of 1.0 returns an empty string (no filter needed).
    """
    if abs(tempo - 1.0) < 1e-6:
        return ""
    filters: List[str] = []
    remaining = tempo
    while remaining > 2.0 + 1e-9:
        filters.append("atempo=2.0")
        remaining /= 2.0
    while remaining < 0.5 - 1e-9:
        filters.append("atempo=0.5")
        remaining /= 0.5
    filters.append(f"atempo={remaining:.6f}")
    return ",".join(filters)


def _apply_audio_tempo(
    src: str,
    dst: str,
    tempo: float,
    ffmpeg_path: str = "ffmpeg",
) -> None:
    """Time-stretch *src* WAV by *tempo* and write result to *dst*.

    Uses ffmpeg ``atempo`` filter (no pitch change).  Raises ``RuntimeError``
    on ffmpeg failure.
    """
    af = _atempo_filter(tempo)
    if not af:
        # tempo ≈ 1.0 — just copy
        import shutil
        shutil.copy2(src, dst)
        return
    cmd = [
        ffmpeg_path or "ffmpeg",
        "-y",
        "-i", src,
        "-af", af,
        dst,
    ]
    logger.debug("Applying audio_tempo=%.4f to %s → %s", tempo, src, dst)
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg atempo failed for {src!r} (tempo={tempo}): {result.stderr[-500:]}"
        )



def _prepare_chunks_with_tempo(
    chunk_paths: List[str],
    chunk_voice_map: Dict[str, Optional[str]],
    voices_config: Dict[str, dict],
    primary_voice: str,
    ffmpeg_path: str = "ffmpeg",
    tmp_dir: Optional[str] = None,
) -> List[str]:
    """Return a new list of chunk paths with per-voice ``audio_tempo`` applied.

    Each chunk whose effective voice has ``audio_tempo`` ≠ 1.0 in *voices_config*
    is time-stretched into *tmp_dir* (a new WAV file).  Chunks without a tempo
    override are returned unchanged (path identity).

    *chunk_voice_map* maps chunk path → voice name (None means primary voice).
    *primary_voice* is used to resolve None entries.

    If no voice has a non-trivial ``audio_tempo`` (or *tmp_dir* is None), the
    original list is returned as-is — no files are written.

    Caller is responsible for cleaning up *tmp_dir*.
    """
    needs_tempo: Dict[str, float] = {}
    for voice_key, vcfg in (voices_config or {}).items():
        t = vcfg.get("audio_tempo")
        if t is not None and abs(float(t) - 1.0) > 1e-6:
            needs_tempo[voice_key] = float(t)

    if not needs_tempo or not tmp_dir:
        return list(chunk_paths)

    result: List[str] = []
    counter = 0
    for cp in chunk_paths:
        voice = chunk_voice_map.get(cp) if chunk_voice_map else None
        effective_voice = voice if voice is not None else primary_voice
        tempo = needs_tempo.get(effective_voice)
        if tempo is not None:
            counter += 1
            dst = os.path.join(tmp_dir, f"tempo_{counter:05d}_{Path(cp).name}")
            _apply_audio_tempo(cp, dst, tempo, ffmpeg_path=ffmpeg_path)
            result.append(dst)
        else:
            result.append(cp)
    return result


def _voices_need_tempo(voices_config: Optional[Dict[str, dict]]) -> bool:
    """True iff any voice in *voices_config* has audio_tempo ≠ 1.0."""
    if not voices_config:
        return False
    for vcfg in voices_config.values():
        t = vcfg.get("audio_tempo")
        if t is not None and abs(float(t) - 1.0) > 1e-6:
            return True
    return False


def _merge_audio_files(
    chunk_paths: List[str],
    output_path: str,
    smooth_join_ms: int = 0,
    dc_remove: bool = False,
    gap_ms: int = 0,
    start_declick_ms: int = 0,
    start_declick_fade_ms: int = 0,
    lf_preamble_fade_ms: int = 0,
    gap_preamble_fade_ms: int = 0,
    *,
    voices_config: Optional[Dict[str, dict]] = None,
    chunk_voice_map: Optional[Dict[str, Optional[str]]] = None,
    primary_voice: Optional[str] = None,
    ffmpeg_path: str = "ffmpeg",
) -> None:
    """Concatenate audio chunk files into *output_path*.

    For WAV files uses Python's stdlib ``wave`` module with optional true
    cosine crossfade, DC offset removal, and silence gap insertion.
    For other formats falls back to pydub (crossfade only).

    Per-chunk audio_tempo
    ---------------------
    When *voices_config*, *chunk_voice_map*, and *primary_voice* are all
    provided AND at least one voice has ``audio_tempo`` ≠ 1.0, every chunk
    whose voice has a tempo override is first time-stretched (via ffmpeg
    ``atempo``) into a temporary directory before merging.  The temp dir is
    cleaned up after the merge completes (success or failure).
    """
    if not chunk_paths:
        return

    tempo_tmp_dir: Optional[str] = None
    merge_paths = chunk_paths
    try:
        if (
            voices_config
            and chunk_voice_map is not None
            and primary_voice is not None
            and _voices_need_tempo(voices_config)
        ):
            tempo_tmp_dir = tempfile.mkdtemp(prefix="epub2ab_tempo_")
            merge_paths = _prepare_chunks_with_tempo(
                chunk_paths,
                chunk_voice_map,
                voices_config,
                primary_voice,
                ffmpeg_path=ffmpeg_path,
                tmp_dir=tempo_tmp_dir,
            )
            n_stretched = sum(1 for a, b in zip(chunk_paths, merge_paths) if a != b)
            logger.info(
                "audio_tempo: prepared %d/%d chunk(s) (primary=%s) in %s",
                n_stretched, len(chunk_paths), primary_voice, tempo_tmp_dir,
            )

        fmt = Path(output_path).suffix.lstrip(".").lower() or "wav"
        if fmt == "wav":
            _merge_wav_files(
                merge_paths,
                output_path,
                smooth_join_ms,
                dc_remove,
                gap_ms,
                start_declick_ms,
                start_declick_fade_ms,
                lf_preamble_fade_ms,
                gap_preamble_fade_ms,
            )
        else:
            _merge_via_pydub(
                merge_paths,
                output_path,
                fmt,
                smooth_join_ms,
                gap_ms,
                start_declick_ms,
                start_declick_fade_ms,
                lf_preamble_fade_ms,
                gap_preamble_fade_ms,
            )
    finally:
        if tempo_tmp_dir:
            shutil.rmtree(tempo_tmp_dir, ignore_errors=True)


class ChunkedAudioGenerator:
    """Per-chapter chunked TTS synthesiser with filesystem-based resume."""

    def __init__(
        self,
        *,
        config,
        chunk_store: Optional["AudioChunkStore"],
        tts_provider,
        chunks_base_dir: str,
        run_id: str = "",
    ):
        self.config = config
        self.store = chunk_store  # may be None when DB tracking is disabled
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
        """Temporarily swap config.voice_name (and per-voice speed) for one TTS call.

        When ``voices_config`` contains a ``speed`` entry for *voice*, that value
        is also applied for the duration of the call and restored afterwards.
        """
        if voice is None or voice == self.config.voice_name:
            yield
            return
        original_voice = self.config.voice_name
        original_speed = getattr(self.config, 'speed', None)
        self.config.voice_name = voice
        # Apply per-voice speed if configured.
        _voices_cfg = getattr(self.config, 'voices_config', {})
        _voice_speed = _voices_cfg.get(voice, {}).get('speed')
        if _voice_speed is not None:
            self.config.speed = _voice_speed
        try:
            yield
        finally:
            self.config.voice_name = original_voice
            self.config.speed = original_speed

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
        # Skipped when store is None (chunked_audio_no_db=true).
        if self.store is not None:
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

        # Collect synthesised chunk paths in sentence order (FS-based) and build
        # parallel voice map for per-chunk audio_tempo handling.
        chunk_paths: List[str] = []
        chunk_voice_map: Dict[str, Optional[str]] = {}
        for s, v in sentence_voice_pairs:
            cp = self._chunk_path(chapter_key, _sentence_hash(s), v)
            if os.path.exists(cp):
                chunk_paths.append(cp)
                chunk_voice_map[cp] = v  # None → primary voice

        if not chunk_paths:
            logger.error("Chapter %d: no audio chunks available for merging.", chapter_idx)
            return False

        voices_config: Dict[str, dict] = getattr(self.config, "voices_config", {}) or {}
        ffmpeg_path: str = getattr(self.config, "ffmpeg_path", None) or "ffmpeg"
        primary_voice: str = self.config.voice_name or ""
        needs_audio_tempo = _voices_need_tempo(voices_config)

        # Skip merge if the chapter WAV already reflects all current chunks.
        start_declick_ms = 0
        start_declick_fade_ms = 0
        if getattr(self.config, "tts_chunk_declick_start", False):
            start_declick_ms = int(getattr(self.config, "tts_chunk_declick_start_ms", 10) or 10)
            start_declick_fade_ms = int(getattr(self.config, "tts_chunk_declick_fade_ms", 6) or 6)

        lf_preamble_fade_ms = 0
        if getattr(self.config, "tts_chunk_declick_lf_preamble", False):
            lf_preamble_fade_ms = int(
                getattr(self.config, "tts_chunk_declick_lf_preamble_fade_ms", 8) or 8
            )

        gap_preamble_fade_ms = 0
        if getattr(self.config, "tts_chunk_declick_gap_preamble", False):
            gap_preamble_fade_ms = int(
                getattr(self.config, "tts_chunk_declick_gap_preamble_fade_ms", 10) or 10
            )

        if (
            not start_declick_ms
            and not lf_preamble_fade_ms
            and not gap_preamble_fade_ms
            and not needs_audio_tempo
            and self._chapter_wav_is_uptodate(output_file, chunk_paths)
        ):
            logger.info(
                "Chapter %d WAV is up-to-date, skipping re-merge: %s", chapter_idx, output_file
            )
            return True

        # Smooth join / crossfade settings from config.
        smooth_join_ms = 0
        if getattr(self.config, "tts_chunk_smooth_join", True):
            smooth_join_ms = int(getattr(self.config, "tts_chunk_smooth_join_ms", 30) or 30)
        dc_remove = bool(getattr(self.config, "tts_chunk_dc_remove", True))
        gap_ms = int(getattr(self.config, "tts_chunk_merge_gap_ms", 0) or 0)

        # Merge — _merge_audio_files applies per-chunk audio_tempo internally
        # when voices_config / chunk_voice_map are provided.
        try:
            _merge_audio_files(
                chunk_paths,
                output_file,
                smooth_join_ms,
                dc_remove,
                gap_ms,
                start_declick_ms,
                start_declick_fade_ms,
                lf_preamble_fade_ms,
                gap_preamble_fade_ms,
                voices_config=voices_config,
                chunk_voice_map=chunk_voice_map,
                primary_voice=primary_voice,
                ffmpeg_path=ffmpeg_path,
            )
            logger.info("Chapter %d merged into %s", chapter_idx, output_file)
            return True
        except Exception as exc:
            logger.error("Chapter %d merge failed: %s", chapter_idx, exc)
            return False
