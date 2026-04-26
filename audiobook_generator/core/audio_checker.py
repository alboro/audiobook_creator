# Copyright (c) 2026 alboro <alboro@users.noreply.github.com>
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See: https://polyformproject.org/licenses/noncommercial/1.0.0/

"""Audio quality checker.

Transcribes every synthesised audio chunk with Whisper, normalises both
the original text and the transcription to "letters + spaces only", then
compares them.  Chunks whose similarity falls below *threshold* are marked
as disputed in the AudioChunkStore so they can be reviewed in the Review UI.

Approach: text-first.
  1. Scan ``text/<latest_run>/`` for chapter .txt files.
  2. Split each chapter text into sentences (same logic as ChunkedAudioGenerator).
  3. Compute sentence hash; check if ``wav/chunks/<chapter_key>/<hash>.*`` exists.
  4. Transcribe existing audio and compare with original sentence text.

NOTE: Standard Whisper (faster-whisper) does NOT output Russian stress marks
(ударения / combining acute accents).  Stress marks are therefore stripped from
the original text before comparison so that annotated TTS source text can match
plain Whisper output.  If per-word stress quality matters, review disputed chunks
manually in the Review UI and re-synthesise as needed.

Usage (CLI):
    .venv/bin/python -m audiobook_generator.core.audio_checker \
        --output_folder /path/to/book_output \
        [--model small] [--language ru] [--threshold 0.70]
"""
from __future__ import annotations

import logging
import re
import unicodedata
from datetime import UTC, datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Text normalisation for comparison
# ---------------------------------------------------------------------------

def _normalize_for_compare(text: str) -> str:
    """Keep only Cyrillic / Latin letters and spaces, lowercase.

    Stress marks (combining acute accents U+0301) and all other diacritics
    are stripped before comparison so Whisper's plain output can match the
    stress-annotated TTS source text.
    """
    # Strip Silero-style plus and combining diacritical marks
    nfd = unicodedata.normalize("NFD", text.replace("+", ""))
    stripped = "".join(c for c in nfd if unicodedata.category(c) != "Mn")
    # Keep only letters (any script) and spaces, lowercase
    only_letters = re.sub(r"[^\w ]", " ", stripped.lower())
    # Remove digits and underscores that \w matched
    only_letters = re.sub(r"[0-9_]", " ", only_letters)
    return re.sub(r"\s+", " ", only_letters).strip()


def _similarity(a: str, b: str) -> float:
    """Character-level similarity ratio in [0, 1]."""
    if not a and not b:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


# ---------------------------------------------------------------------------
# AudioChecker
# ---------------------------------------------------------------------------

DEFAULT_THRESHOLD = 0.70
DEFAULT_MODEL = "small"
DEFAULT_LANGUAGE = "ru"

_AUDIO_EXTENSIONS = ["wav", "mp3", "ogg", "m4a"]

# Deterministic normalizer steps applied to the original sentence text before
# similarity comparison.  These expand abbreviations / numbers / symbols to
# their spoken forms so Whisper's plain output can match the TTS source.
_PRE_COMPARE_STEPS = "simple_symbols,ru_initials,ru_abbreviations,ru_numbers,ru_proper_names"


def _build_pre_compare_normalizer(language: str = "ru"):
    """Build a lightweight normalizer chain used only for pre-compare expansion.

    Returns a callable ``normalize(text) -> str`` or None if construction fails.
    """
    try:
        import types
        from audiobook_generator.normalizers.base_normalizer import (
            NORMALIZER_REGISTRY, ChainNormalizer,
        )

        # Minimal config stub — only fields required by the deterministic normalizers.
        cfg = types.SimpleNamespace(
            language=language,
            normalize=True,
            normalize_steps=_PRE_COMPARE_STEPS,
            normalize_log_changes=False,
            normalize_tts_safe_max_chars=180,
            normalize_tts_safe_comma_as_period=False,
            normalize_tts_pronunciation_overrides_words=None,
            normalize_stress_paradox_words=None,
            normalize_tsnorm_min_word_length=2,
            normalize_tsnorm_stress_yo=False,
            normalize_tsnorm_stress_monosyllabic=False,
            normalize_model=None,
            normalize_provider=None,
            normalize_api_key=None,
            normalize_base_url=None,
            normalize_max_chars=4000,
            normalize_system_prompt=None,
            normalize_system_prompt_file=None,
            normalize_prompt_file=None,
            normalize_user_prompt_file=None,
            output_folder=None,
            prepared_text_folder=None,
            _normalizer_llm_runtime=None,
        )

        steps = [s.strip() for s in _PRE_COMPARE_STEPS.split(",") if s.strip()]
        normalizers = []
        for step in steps:
            entry = NORMALIZER_REGISTRY.get(step)
            if not entry:
                continue
            mod_path, cls_name = entry
            import importlib
            mod = importlib.import_module(mod_path)
            cls = getattr(mod, cls_name)
            normalizers.append(cls(cfg))

        if not normalizers:
            return None

        chain = ChainNormalizer(config=cfg, normalizers=normalizers, steps=steps)

        def _normalize(text: str) -> str:
            try:
                return chain.normalize(text)
            except Exception:
                return text

        return _normalize
    except Exception as exc:
        logger.warning("Could not build pre-compare normalizer: %s", exc)
        return None


class AudioChecker:
    """Walk text files to find synthesised chunks and quality-check them."""

    def __init__(
        self,
        output_folder: str | Path,
        model_size: str = DEFAULT_MODEL,
        language: str = DEFAULT_LANGUAGE,
        threshold: float = DEFAULT_THRESHOLD,
        device: str = "cpu",
        compute_type: str = "int8",
        config=None,
    ):
        self.output_folder = Path(output_folder)
        self.language = language
        self.threshold = threshold
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._model = None  # lazy-loaded
        # Pre-compare normalizer: expand abbreviations/numbers before similarity scoring.
        # Built from config language if available, otherwise use default "ru".
        _lang = getattr(config, "language", language) if config else language
        self._pre_compare = _build_pre_compare_normalizer(_lang)

    # ------------------------------------------------------------------

    def _get_model(self):
        if self._model is None:
            from faster_whisper import WhisperModel  # type: ignore[import]
            logger.info(
                "Loading Whisper model '%s' on %s …",
                self._model_size, self._device,
            )
            self._model = WhisperModel(
                self._model_size,
                device=self._device,
                compute_type=self._compute_type,
            )
            logger.info("Whisper model ready.")
        return self._model

    def _transcribe(self, wav_path: Path) -> str:
        model = self._get_model()
        segments, _info = model.transcribe(
            str(wav_path),
            language=self.language.split("-")[0],  # "ru-RU" → "ru"
            beam_size=5,
        )
        return " ".join(s.text for s in segments).strip()

    def _find_audio_file(self, chapter_key: str, sentence_hash: str) -> Optional[Path]:
        """Find the audio file for a given chapter_key and hash. Returns None if not found."""
        chunks_dir = self.output_folder / "wav" / "chunks" / chapter_key
        for ext in _AUDIO_EXTENSIONS:
            p = chunks_dir / f"{sentence_hash}.{ext}"
            if p.exists():
                return p
        return None

    def _is_cached_transcription_fresh(self, audio_file: Path, checked_at: str | None) -> bool:
        """Return True when cached transcription is at least as new as the audio file."""
        if not checked_at:
            return False
        try:
            checked_dt = datetime.fromisoformat(checked_at)
            if checked_dt.tzinfo is None:
                checked_dt = checked_dt.replace(tzinfo=UTC)
            audio_mtime = datetime.fromtimestamp(audio_file.stat().st_mtime, tz=UTC)
        except Exception:
            return False
        return checked_dt >= audio_mtime

    def _get_transcription(self, audio_file: Path, chapter_key: str, sentence_hash: str, store) -> str:
        """Reuse cached raw transcription when possible, otherwise run Whisper.

        Cache is considered valid when:
          - a non-manual raw_transcription (or transcription) is stored, AND
          - the cached entry is at least as recent as the audio file on disk.

        If the audio file was re-synthesised after the last check, the cache is
        stale and Whisper is re-run so the new audio is correctly evaluated.
        """
        cache_row = store.get_cached_transcription_entry(chapter_key, sentence_hash)
        if cache_row:
            cached_raw = cache_row["raw_transcription"] or None
            if not cached_raw:
                legacy_transcription = (cache_row["transcription"] or "").strip()
                if legacy_transcription and not legacy_transcription.startswith("[manual]"):
                    cached_raw = legacy_transcription
            if cached_raw and self._is_cached_transcription_fresh(audio_file, cache_row["checked_at"]):
                logger.debug(
                    "[%s] Hash %s – using cached transcription from chunk_cache",
                    chapter_key, sentence_hash[:10],
                )
                return cached_raw
            elif cached_raw:
                logger.debug(
                    "[%s] Hash %s – cache stale (audio newer than last check), re-transcribing",
                    chapter_key, sentence_hash[:10],
                )
        return self._transcribe(audio_file)

    # ------------------------------------------------------------------

    def run(self, store) -> dict[str, int]:
        """Check all synthesised audio chunks and mark disputed ones in *store*.

        Uses a text-first approach:
          1. Find latest text run folder (``text/<NNN>/``).
          2. For each chapter .txt file, derive chapter_key from filename stem.
          3. Split chapter text into sentences (same logic as ChunkedAudioGenerator).
          4. For each sentence: compute hash → check FS → transcribe if found → compare.

        Falls back to FS-only scan (wav/chunks/) when no text folder is found.

        Returns counters: {"checked": N, "disputed": M, "skipped": K}.
        """
        from audiobook_generator.core.audio_chunk_store import AudioChunkStore
        assert isinstance(store, AudioChunkStore)

        # Pre-flight: verify faster-whisper is importable before processing any files.
        try:
            import faster_whisper  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "faster-whisper is not installed or failed to import.\n"
                "Install it with:\n"
                "    pip install faster-whisper\n"
                "On Windows you may also need the Microsoft Visual C++ 2019 Redistributable:\n"
                "    https://aka.ms/vs/17/release/vc_redist.x64.exe\n"
                "If you use CUDA, make sure cuBLAS and cuDNN are available (see faster-whisper docs)."
            )

        counters = {"checked": 0, "disputed": 0, "skipped": 0}

        # --- Text-first approach -------------------------------------------
        from audiobook_generator.utils.existing_chapters_loader import (
            find_latest_run_folder,
            load_chapters_from_run_folder,
        )
        from audiobook_generator.utils.existing_chapters_loader import split_text_into_chunks

        run_folder = find_latest_run_folder(self.output_folder)
        if run_folder:
            chapters = load_chapters_from_run_folder(run_folder)
            logger.info(
                "Text-first mode: found %d chapters in %s", len(chapters), run_folder
            )
            for chapter in chapters:
                self._check_chapter_text_first(chapter, store, counters)
        else:
            # Fallback: walk wav/chunks/ FS directly (no text files available)
            logger.warning(
                "No text/ run folder found under %s — falling back to FS-only scan.",
                self.output_folder,
            )
            self._run_fs_fallback(store, counters)

        logger.info(
            "Audio check complete — checked=%d, disputed=%d, skipped=%d",
            counters["checked"], counters["disputed"], counters["skipped"],
        )
        return counters

    def _check_chapter_text_first(self, chapter, store, counters: dict) -> None:
        """Check all sentences for a chapter using its text file as the source of truth."""
        try:
            text = Path(chapter.text_path).read_text(encoding="utf-8")
        except Exception as exc:
            logger.error("Cannot read text file %s: %s", chapter.text_path, exc)
            return

        from audiobook_generator.utils.existing_chapters_loader import split_text_into_chunks
        from audiobook_generator.utils.sentence_hash import sentence_hash as _sentence_hash

        sentences = split_text_into_chunks(text, self.language.split("-")[0])
        chapter_key = chapter.chapter_key

        logger.info(
            "Chapter '%s': %d sentences to check.", chapter_key, len(sentences)
        )

        for sentence in sentences:
            s_hash = _sentence_hash(sentence)
            audio_file = self._find_audio_file(chapter_key, s_hash)
            if audio_file is None:
                # Not synthesised yet — skip
                counters["skipped"] += 1
                continue
            self._check_one_file(audio_file, chapter_key, s_hash, sentence, store, counters)

    def _run_fs_fallback(self, store, counters: dict) -> None:
        """Fallback: walk wav/chunks/<chapter_key>/*.* and look up text from DB."""
        wav_root = self.output_folder / "wav" / "chunks"
        if not wav_root.exists():
            logger.warning("No wav/chunks directory found at %s", wav_root)
            return

        chapter_dirs = sorted(d for d in wav_root.iterdir() if d.is_dir())
        logger.info("Found %d chapter chunk directories (FS fallback).", len(chapter_dirs))

        for chapter_dir in chapter_dirs:
            chapter_key = chapter_dir.name
            audio_files = sorted(
                f for f in chapter_dir.iterdir()
                if f.is_file() and f.suffix.lstrip(".") in _AUDIO_EXTENSIONS
            )
            logger.info("Chapter '%s': %d chunks (FS fallback).", chapter_key, len(audio_files))
            for audio_file in audio_files:
                s_hash = audio_file.stem
                original_text = store.get_latest_sentence_text(s_hash)
                if not original_text:
                    logger.warning(
                        "  [%s] Hash %s – no text in DB, skipping.",
                        chapter_key, s_hash[:10],
                    )
                    counters["skipped"] += 1
                    continue
                self._check_one_file(audio_file, chapter_key, s_hash, original_text, store, counters)

    def _check_one_file(
        self,
        audio_file: Path,
        chapter_key: str,
        sentence_hash: str,
        original_text: str,
        store,
        counters: dict,
    ) -> None:
        try:
            transcription = self._get_transcription(audio_file, chapter_key, sentence_hash, store)
        except Exception as exc:
            logger.error(
                "  [%s] Hash %s – transcription error: %s",
                chapter_key, sentence_hash[:10], exc,
            )
            counters["skipped"] += 1
            return

        orig_norm = _normalize_for_compare(original_text)
        trans_norm = _normalize_for_compare(transcription)

        # Apply pre-compare semantic normalisation (expand abbreviations, numbers, etc.)
        # to the original text so it matches the spoken form that Whisper transcribed.
        if self._pre_compare is not None:
            orig_norm = _normalize_for_compare(self._pre_compare(original_text))
            # Also normalize the transcription through simple_symbols to clean punctuation.
            trans_norm = _normalize_for_compare(transcription)

        sim = _similarity(orig_norm, trans_norm)

        counters["checked"] += 1

        disputed = sim < self.threshold
        status = "DISPUTED" if disputed else "ok"
        logger.info(
            "  [%s] %s  sim=%.2f  %s | orig: %s",
            chapter_key, sentence_hash[:10], sim, status, original_text[:60],
        )
        if disputed:
            store.save_disputed_chunk(
                chapter_key=chapter_key,
                sentence_hash=sentence_hash,
                original_text=original_text,
                transcription=transcription,
                similarity=sim,
                raw_transcription=transcription,
            )
            counters["disputed"] += 1
        else:
            store.save_checked_chunk(
                chapter_key=chapter_key,
                sentence_hash=sentence_hash,
                original_text=original_text,
                transcription=transcription,
                similarity=sim,
                raw_transcription=transcription,
            )


# ---------------------------------------------------------------------------
# Standalone CLI entry point
# ---------------------------------------------------------------------------

def _build_store(output_folder: Path):
    from audiobook_generator.core.audio_chunk_store import AudioChunkStore
    db_path = output_folder / "wav" / "_state" / "audio_chunks.sqlite3"
    if not db_path.exists():
        raise FileNotFoundError(f"DB not found: {db_path}")
    return AudioChunkStore(db_path)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Transcribe audio chunks and mark suspected synthesis errors as disputed."
    )
    parser.add_argument("output_folder", help="Book output folder (contains wav/chunks/)")
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Whisper model size (default: {DEFAULT_MODEL}). Options: tiny, base, small, medium, large-v3.",
    )
    parser.add_argument(
        "--language", default=DEFAULT_LANGUAGE,
        help=f"Language code for Whisper (default: {DEFAULT_LANGUAGE}).",
    )
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_THRESHOLD,
        help=f"Similarity threshold below which a chunk is marked disputed (default: {DEFAULT_THRESHOLD}).",
    )
    parser.add_argument(
        "--device", default="cpu",
        help="Inference device: cpu or cuda (default: cpu).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    output_folder = Path(args.output_folder).resolve()
    store = _build_store(output_folder)

    checker = AudioChecker(
        output_folder=output_folder,
        model_size=args.model,
        language=args.language,
        threshold=args.threshold,
        device=args.device,
    )
    checker.run(store)


if __name__ == "__main__":
    main()

