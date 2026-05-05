# Copyright (c) 2026 alboro <alboro@users.noreply.github.com>
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See: https://polyformproject.org/licenses/noncommercial/1.0.0/

"""Audio quality checker.

Transcribes every synthesised audio chunk with Whisper, then runs the
configured checker pipeline to decide whether the chunk is *disputed*.

Checker pipeline is controlled by ``audio_check_checkers`` in the INI file
(comma-separated names from AUDIO_CHECKER_REGISTRY).  Default:
``whisper_similarity,first_word,last_word``.

Approach: text-first.
  1. Scan ``text/<latest_run>/`` for chapter .txt files.
  2. Split each chapter text into sentences (same logic as ChunkedAudioGenerator).
  3. Compute sentence hash; check if ``wav/chunks/<chapter_key>/<hash>.*`` exists.
  4. Transcribe existing audio and run checker pipeline.

Usage (CLI):
    .venv/bin/python -m audiobook_generator.core.audio_checker \\
        --output_folder /path/to/book_output \\
        [--model small] [--language ru] [--threshold 0.70]
        [--checkers whisper_similarity,first_word,last_word,reference]
"""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Backward-compatible re-exports (imported by existing test code)
# ---------------------------------------------------------------------------
from audiobook_generator.core.audio_checkers.base_audio_chunk_checker import (  # noqa: E402
    normalize_for_compare as _normalize_for_compare,
)
from audiobook_generator.core.audio_checkers.whisper_similarity_checker import (  # noqa: E402
    _build_pre_compare_normalizer,
)
from audiobook_generator.core.audio_chunk_store import (  # noqa: E402
    STATUS_CHECKED, STATUS_DISPUTED, STATUS_RESOLVED,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_THRESHOLD = 0.70
DEFAULT_MODEL = "small"
DEFAULT_LANGUAGE = "ru"

_AUDIO_EXTENSIONS = ["wav", "mp3", "ogg", "m4a"]


# ---------------------------------------------------------------------------
# Windows CUDA helper (unchanged)
# ---------------------------------------------------------------------------

def _iter_windows_cuda_dll_dirs() -> list[Path]:
    """Return candidate DLL directories that may satisfy faster-whisper on Windows."""
    if os.name != "nt":
        return []

    candidates: list[Path] = []
    seen: set[str] = set()

    def _maybe_add(path: Path) -> None:
        key = str(path).lower()
        if key in seen or not path.is_dir():
            return
        marker_names = ("cublas64_12.dll", "cudart64_12.dll", "cudnn64_9.dll")
        if not any((path / marker).exists() for marker in marker_names):
            return
        seen.add(key)
        candidates.append(path)

    current_prefix = Path(sys.executable).resolve().parent.parent
    _maybe_add(current_prefix / "Lib" / "site-packages" / "torch" / "lib")
    _maybe_add(current_prefix / "Lib" / "site-packages" / "ctranslate2")

    repo_root = Path(__file__).resolve().parents[2]
    projects_root = repo_root.parent
    if projects_root.is_dir():
        for project_dir in projects_root.iterdir():
            if not project_dir.is_dir():
                continue
            _maybe_add(project_dir / ".venv" / "Lib" / "site-packages" / "torch" / "lib")
            _maybe_add(project_dir / ".venv" / "Lib" / "site-packages" / "ctranslate2")
            conda_envs = project_dir / ".conda" / "miniforge" / "envs"
            if conda_envs.is_dir():
                for env_dir in conda_envs.iterdir():
                    if not env_dir.is_dir():
                        continue
                    _maybe_add(env_dir / "Lib" / "site-packages" / "torch" / "lib")
                    _maybe_add(env_dir / "Lib" / "site-packages" / "ctranslate2")

    return candidates


# ---------------------------------------------------------------------------
# AudioChecker
# ---------------------------------------------------------------------------

class AudioChecker:
    """Walk text files to find synthesised chunks and quality-check them.

    Transcription is always performed centrally (with caching) and passed to
    every checker in the pipeline.  The pipeline is configured via
    ``audio_check_checkers`` in the INI file.
    """

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
        self.threshold = threshold   # kept for logging / boundary-mismatch override
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._model = None           # lazy-loaded
        self._dll_dir_handles = []
        self._cuda_runtime_prepared = False
        self.force = bool(getattr(config, "audio_check_force", False)) if config else False

        self._prepared_text_folder = (
            getattr(config, "prepared_text_folder", None) if config else None
        )

        # Build checker pipeline from config.audio_check_checkers
        from audiobook_generator.core.audio_checkers import build_checkers
        if config is not None:
            self._checkers = build_checkers(config)
        else:
            import types
            _stub = types.SimpleNamespace(
                audio_check_checkers=None,
                audio_check_threshold=threshold,
                audio_reference_check_command=None,
                audio_reference_check_threshold=None,
                audio_reference_check_timeout=None,
                audio_reference_check_cache_dir=None,
                audio_reference_check_stress=None,
                language=language,
                output_folder=str(output_folder),
                ffmpeg_path=None,
                prepared_text_folder=None,
            )
            self._checkers = build_checkers(_stub)

        logger.info(
            "Audio checker pipeline: %s",
            ", ".join(c.name for c in self._checkers) or "(empty)",
        )

    # ------------------------------------------------------------------
    # Backward-compatible _pre_compare proxy
    # (tests set checker._pre_compare = fn to override the similarity normalizer)
    # ------------------------------------------------------------------

    @property
    def _pre_compare(self):
        from audiobook_generator.core.audio_checkers.whisper_similarity_checker import (
            WhisperSimilarityChecker,
        )
        for c in self._checkers:
            if isinstance(c, WhisperSimilarityChecker):
                return c._pre_compare
        return None

    @_pre_compare.setter
    def _pre_compare(self, value):
        from audiobook_generator.core.audio_checkers.whisper_similarity_checker import (
            WhisperSimilarityChecker,
        )
        for c in self._checkers:
            if isinstance(c, WhisperSimilarityChecker):
                c._pre_compare = value
                return

    # ------------------------------------------------------------------

    def _select_text_run_folder(self) -> Optional[Path]:
        if self._prepared_text_folder:
            folder = Path(self._prepared_text_folder)
            if not folder.is_absolute():
                folder = self.output_folder / folder
            return folder if folder.exists() else None

        from audiobook_generator.utils.existing_chapters_loader import find_latest_run_folder
        return find_latest_run_folder(self.output_folder)

    def _prepare_windows_cuda_runtime(self) -> None:
        if self._cuda_runtime_prepared or os.name != "nt" or str(self._device).lower() != "cuda":
            return
        added_dirs: list[str] = []
        path_entries = os.environ.get("PATH", "").split(os.pathsep)
        path_entries_lower = {entry.lower() for entry in path_entries if entry}

        for dll_dir in _iter_windows_cuda_dll_dirs():
            dll_dir_str = str(dll_dir)
            if hasattr(os, "add_dll_directory"):
                try:
                    self._dll_dir_handles.append(os.add_dll_directory(dll_dir_str))
                except (FileNotFoundError, OSError):
                    pass
            if dll_dir_str.lower() not in path_entries_lower:
                os.environ["PATH"] = dll_dir_str + os.pathsep + os.environ.get("PATH", "")
                path_entries_lower.add(dll_dir_str.lower())
            added_dirs.append(dll_dir_str)

        if added_dirs:
            logger.info("Prepared CUDA DLL paths for audio_check: %s", "; ".join(added_dirs))
        else:
            logger.warning(
                "audio_check: device=cuda on Windows but no CUDA DLL directories were found "
                "in the venv. If Whisper fails to load, set device=cpu or ensure "
                "cublas64_12.dll / cudart64_12.dll / cudnn64_9.dll are on PATH."
            )
        self._cuda_runtime_prepared = True

    def _get_model(self):
        if self._model is not None:
            return self._model

        self._prepare_windows_cuda_runtime()
        from faster_whisper import WhisperModel  # type: ignore[import]

        # ctranslate2 requires device-appropriate compute types.
        # "int8" is CPU-optimal; on CUDA the correct types are float16 / int8_float16.
        # Pass "auto" to let ctranslate2 pick the best type for the device, unless the
        # caller explicitly overrode compute_type to something other than the default.
        compute_type = self._compute_type
        if compute_type == "int8" and str(self._device).lower() == "cuda":
            compute_type = "auto"
            logger.info(
                "audio_check: compute_type upgraded from 'int8' to 'auto' for device=cuda "
                "(int8 is CPU-only; ctranslate2 will choose float16 or int8_float16)."
            )

        logger.info(
            "Loading Whisper model '%s' on %s (compute_type=%s) …",
            self._model_size, self._device, compute_type,
        )
        try:
            self._model = WhisperModel(
                self._model_size,
                device=self._device,
                compute_type=compute_type,
            )
        except Exception as exc:
            logger.error(
                "audio_check: failed to load Whisper on %s (%s). "
                "Falling back to CPU. Original error: %s",
                self._device, compute_type, exc,
            )
            self._device = "cpu"
            self._model = WhisperModel(
                self._model_size,
                device="cpu",
                compute_type="int8",
            )

        logger.info("Whisper model ready (device=%s).", self._device)
        return self._model

    def _transcribe(self, wav_path: Path) -> str:
        model = self._get_model()
        segments, _info = model.transcribe(
            str(wav_path),
            language=self.language.split("-")[0],
            beam_size=5,
        )
        return " ".join(s.text for s in segments).strip()

    def _find_audio_file(self, chapter_key: str, sentence_hash: str) -> Optional[Path]:
        chunks_dir = self.output_folder / "wav" / "chunks" / chapter_key
        for ext in _AUDIO_EXTENSIONS:
            p = chunks_dir / f"{sentence_hash}.{ext}"
            if p.exists():
                return p
        return None

    def _is_cached_transcription_fresh(self, audio_file: Path, checked_at: str | None) -> bool:
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

    def _get_transcription(
        self, audio_file: Path, chapter_key: str, sentence_hash: str, store
    ) -> str:
        """Reuse cached transcription when possible, otherwise run Whisper."""
        cache_row = store.get_cached_transcription_entry(chapter_key, sentence_hash)
        if cache_row:
            cached_raw = cache_row["raw_transcription"] or None
            if not cached_raw:
                legacy = (cache_row["transcription"] or "").strip()
                if legacy and not legacy.startswith("[manual]"):
                    cached_raw = legacy
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

    def _normalize_transcription_for_storage(self, transcription: str) -> str:
        """Return the compare-ready transcription shown in review/UI.

        ``raw_transcription`` keeps the exact Whisper output.  This field stores
        the deterministic pre-compare form (numbers / abbreviations expanded)
        so users can see what the similarity checker actually compared.
        """
        pre_compare = self._pre_compare
        if pre_compare is None:
            return transcription
        try:
            normalized = pre_compare(transcription)
        except Exception:
            return transcription
        return normalized or transcription

    # ------------------------------------------------------------------

    def run(self, store) -> dict[str, int]:
        """Check all synthesised audio chunks and mark disputed ones in *store*.

        Returns counters: ``{"checked": N, "disputed": M, "skipped": K}``.
        """
        from audiobook_generator.core.audio_chunk_store import AudioChunkStore
        assert isinstance(store, AudioChunkStore)

        counters: dict[str, int] = {"checked": 0, "disputed": 0, "skipped": 0}

        from audiobook_generator.utils.existing_chapters_loader import load_chapters_from_run_folder

        run_folder = self._select_text_run_folder()
        if run_folder:
            chapters = load_chapters_from_run_folder(run_folder)
            logger.info("Text-first mode: found %d chapters in %s", len(chapters), run_folder)
            for chapter in chapters:
                self._check_chapter_text_first(chapter, store, counters)
        else:
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
        try:
            text = Path(chapter.text_path).read_text(encoding="utf-8")
        except Exception as exc:
            logger.error("Cannot read text file %s: %s", chapter.text_path, exc)
            return

        from audiobook_generator.utils.existing_chapters_loader import split_text_into_chunks
        from audiobook_generator.utils.sentence_hash import sentence_hash as _sentence_hash

        sentences = split_text_into_chunks(text, self.language.split("-")[0])
        chapter_key = chapter.chapter_key
        logger.info("Chapter '%s': %d sentences to check.", chapter_key, len(sentences))

        for sentence in sentences:
            s_hash = _sentence_hash(sentence)
            audio_file = self._find_audio_file(chapter_key, s_hash)
            if audio_file is None:
                counters["skipped"] += 1
                continue
            self._check_one_file(audio_file, chapter_key, s_hash, sentence, store, counters)

    def _run_fs_fallback(self, store, counters: dict) -> None:
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
        # ── 0. Skip only user-resolved chunks when audio has not changed ───
        cache_row = store.get_cached_transcription_entry(chapter_key, sentence_hash)
        if cache_row and not self.force:
            existing_status = cache_row["status"]
            if existing_status == STATUS_RESOLVED:
                if self._is_cached_transcription_fresh(audio_file, cache_row["checked_at"]):
                    logger.debug(
                        "  [%s] Hash %s – status='%s', audio unchanged → skip user-resolved chunk.",
                        chapter_key, sentence_hash[:10], existing_status,
                    )
                    counters["skipped"] += 1
                    return
                else:
                    logger.debug(
                        "  [%s] Hash %s – resolved chunk audio regenerated (newer than last check), re-checking.",
                        chapter_key, sentence_hash[:10],
                    )
        elif cache_row and self.force:
            logger.debug(
                "  [%s] Hash %s – force enabled, ignoring cached status='%s'.",
                chapter_key, sentence_hash[:10], cache_row["status"],
            )

        # ── 1. Transcribe (cached raw if available, otherwise Whisper) ──────
        try:
            transcription = self._get_transcription(
                audio_file, chapter_key, sentence_hash, store
            )
        except Exception as exc:
            logger.error(
                "  [%s] Hash %s – transcription error: %s",
                chapter_key, sentence_hash[:10], exc,
            )
            counters["skipped"] += 1
            return

        # ── 2. Run checker pipeline ─────────────────────────────────────────
        # Re-fetch cache_row so it reflects any transcription data just written
        # (for checkers like ReferenceChecker that may use it).
        chunk_cache_row = store.get_cached_transcription_entry(chapter_key, sentence_hash)

        is_disputed = False
        similarity_agg: Optional[float] = None
        ref_score: Optional[float] = None
        ref_threshold: Optional[float] = None
        ref_status: Optional[str] = None
        ref_payload: Optional[dict] = None
        checker_results: dict[str, bool] = {}
        for checker in self._checkers:
            try:
                result = checker.check(
                    audio_file, original_text, transcription, chunk_cache_row
                )
            except Exception as exc:
                logger.warning(
                    "  [%s] Hash %s – checker %r error: %s",
                    chapter_key, sentence_hash[:10], checker.name, exc,
                )
                continue

            checker_results[checker.name] = not result.disputed
            if result.disputed:
                is_disputed = True

            # Collect per-checker extras (first non-None wins per field)
            if result.similarity is not None and similarity_agg is None:
                similarity_agg = result.similarity
            if result.reference_check_score is not None and ref_score is None:
                ref_score = result.reference_check_score
                ref_threshold = result.reference_check_threshold
                ref_status = result.reference_check_status
                ref_payload = result.reference_check_payload

        # ── 3. Determine similarity to store ───────────────────────────────
        # Store the actual whisper similarity (or 1.0 if no similarity checker).
        # Disputed state is now communicated via the DB `status` column, not by
        # forcing similarity=0.0, so boundary failures no longer need the hack.
        sim_to_store = similarity_agg if similarity_agg is not None else 1.0

        counters["checked"] += 1
        if is_disputed:
            counters["disputed"] += 1

        status_label = "DISPUTED" if is_disputed else "ok"
        logger.info(
            "  [%s] %s  sim=%.2f  %s | orig: %s",
            chapter_key, sentence_hash[:10], sim_to_store, status_label, original_text[:60],
        )

        # ── 4. Persist ──────────────────────────────────────────────────────
        ref_payload_json = (
            json.dumps(ref_payload, ensure_ascii=False) if ref_payload is not None else None
        )
        transcription_for_storage = self._normalize_transcription_for_storage(transcription)
        save_kwargs = dict(
            chapter_key=chapter_key,
            sentence_hash=sentence_hash,
            original_text=original_text,
            transcription=transcription_for_storage,
            similarity=sim_to_store,
            raw_transcription=transcription,
            reference_check_score=ref_score,
            reference_check_threshold=ref_threshold,
            reference_check_status=ref_status,
            reference_check_payload=ref_payload_json,
        )
        if is_disputed:
            store.save_disputed_chunk(force_status=self.force, **save_kwargs)
        else:
            store.save_checked_chunk(force_status=self.force, **save_kwargs)

        # ── 5. Persist per-checker pass/fail results ────────────────────────
        # Only write the generic fallback column for checkers that declare
        # uses_fallback_passed_column = True (e.g. first_word, last_word).
        # Checkers like whisper_similarity and reference already persist their
        # results in dedicated columns (similarity, reference_check_*); writing
        # a redundant fallback column for them would be misleading because the
        # fallback is later read with a fixed threshold, not the one at check time.
        for checker in self._checkers:
            if checker.name not in checker_results:
                continue
            if not type(checker).uses_fallback_passed_column:
                continue
            try:
                store.save_checker_result(
                    chapter_key, sentence_hash, checker.name, checker_results[checker.name]
                )
            except Exception as exc:
                logger.debug(
                    "  [%s] Hash %s – could not save checker result for %r: %s",
                    chapter_key, sentence_hash[:10], checker.name, exc,
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
    parser.add_argument(
        "--checkers", default=None,
        help="Comma-separated checker names to run (default: whisper_similarity,first_word,last_word).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run checks and overwrite stored status in the DB even for previously resolved chunks.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    import types
    _cfg = types.SimpleNamespace(
        audio_check_checkers=args.checkers,
        audio_check_force=args.force,
        audio_check_threshold=args.threshold,
        audio_reference_check_command=None,
        audio_reference_check_threshold=None,
        audio_reference_check_timeout=None,
        audio_reference_check_cache_dir=None,
        audio_reference_check_stress=None,
        language=args.language,
        output_folder=args.output_folder,
        ffmpeg_path=None,
        prepared_text_folder=None,
    )

    output_folder = Path(args.output_folder).resolve()
    store = _build_store(output_folder)
    checker = AudioChecker(
        output_folder=output_folder,
        model_size=args.model,
        language=args.language,
        threshold=args.threshold,
        device=args.device,
        config=_cfg,
    )
    checker.run(store)


if __name__ == "__main__":
    main()
