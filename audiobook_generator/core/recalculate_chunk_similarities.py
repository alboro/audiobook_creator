# Copyright (c) 2026 alboro <alboro@users.noreply.github.com>
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See: https://polyformproject.org/licenses/noncommercial/1.0.0/

"""Recalculate chunk similarities using the current pre-compare normalizer.

Use this script after upgrading to the fixed audio_checker (which now applies
pre_compare to BOTH original_text and raw_transcription before computing
similarity).  Existing cached rows were scored with the old algorithm —
raw_transcription digits were stripped instead of expanded, producing
artificially low similarity and false-disputed entries.

What the script does for every row in ``chunk_cache``:
  1. Takes ``raw_transcription`` (Whisper output, kept untouched).
  2. Applies the same normalizer chain used by audio_checker:
       simple_symbols → ru_initials → ru_abbreviations → ru_numbers → ru_proper_names
     to BOTH ``original_text`` and ``raw_transcription``.
  3. Recomputes similarity with ``_normalize_for_compare`` + ``SequenceMatcher``.
  4. Updates ``similarity`` and ``transcription`` (= normalized transcription) in
     the DB, preserving ``raw_transcription`` and ``status`` unchanged.

Rows whose status is 'resolved' are recalculated too — the similarity score is
updated so the Review UI can display the corrected value, but the 'resolved'
status itself is never changed.

Dry-run mode (--dry-run) prints changes without writing to the DB.

Usage:
    python -m audiobook_generator.core.recalculate_chunk_similarities \\
        /path/to/book_output [--language ru-RU] [--dry-run]
"""
from __future__ import annotations

import argparse
import logging
import sqlite3
from contextlib import closing
from difflib import SequenceMatcher
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Normalisation helpers (mirrors audio_checker logic exactly)
# ---------------------------------------------------------------------------

def _normalize_for_compare(text: str) -> str:
    """Keep only Cyrillic/Latin letters and spaces, lowercase, strip diacritics."""
    import re
    import unicodedata
    nfd = unicodedata.normalize("NFD", text.replace("+", ""))
    stripped = "".join(c for c in nfd if unicodedata.category(c) != "Mn")
    only_letters = re.sub(r"[^\w ]", " ", stripped.lower())
    only_letters = re.sub(r"[0-9_]", " ", only_letters)
    return re.sub(r"\s+", " ", only_letters).strip()


def _similarity(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


# ---------------------------------------------------------------------------
# Build normalizer chain
# ---------------------------------------------------------------------------

_PRE_COMPARE_STEPS = "simple_symbols,ru_initials,ru_abbreviations,ru_numbers,ru_proper_names"


def _build_normalizer(language: str = "ru-RU"):
    """Build the normalizer chain strictly from the project components.

    Returns a callable ``normalize(text: str) -> str``, or raises on failure.
    """
    import types
    import importlib
    from audiobook_generator.normalizers.base_normalizer import (
        NORMALIZER_REGISTRY,
        ChainNormalizer,
    )

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
            logger.warning("Step '%s' not found in registry — skipped.", step)
            continue
        mod_path, cls_name = entry
        mod = importlib.import_module(mod_path)
        cls = getattr(mod, cls_name)
        normalizers.append(cls(cfg))

    if not normalizers:
        raise RuntimeError("No normalizer steps could be loaded.")

    chain = ChainNormalizer(config=cfg, normalizers=normalizers, steps=steps)

    def _fn(text: str) -> str:
        try:
            return chain.normalize(text)
        except Exception as exc:
            logger.warning("Normalizer error (returning input unchanged): %s", exc)
            return text

    return _fn


# ---------------------------------------------------------------------------
# Core recalculation logic
# ---------------------------------------------------------------------------

def _iter_cache_rows(db_path: Path):
    """Yield all chunk_cache rows as sqlite3.Row objects."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    with closing(conn):
        yield from conn.execute(
            "SELECT chapter_key, sentence_hash, original_text, "
            "       raw_transcription, transcription, similarity, status "
            "FROM chunk_cache"
        ).fetchall()


def _update_row(db_path: Path, chapter_key: str, sentence_hash: str,
                new_similarity: float, new_transcription: str) -> None:
    """Write updated similarity and transcription (preserving raw_transcription / status)."""
    conn = sqlite3.connect(str(db_path))
    with closing(conn):
        conn.execute(
            """
            UPDATE chunk_cache
               SET similarity    = ?,
                   transcription = ?
             WHERE chapter_key   = ?
               AND sentence_hash = ?
            """,
            (new_similarity, new_transcription, chapter_key, sentence_hash),
        )
        conn.commit()


def recalculate(
    output_folder: str | Path,
    language: str = "ru-RU",
    dry_run: bool = False,
) -> dict[str, int]:
    """Main entry point.

    Returns counters: {updated, unchanged, skipped}.
    """
    output_folder = Path(output_folder)
    db_path = output_folder / "wav" / "_state" / "audio_chunks.sqlite3"

    if not db_path.exists():
        raise FileNotFoundError(f"DB not found: {db_path}")

    logger.info("Building pre-compare normalizer chain (language=%s)…", language)
    try:
        normalize = _build_normalizer(language)
    except Exception as exc:
        raise RuntimeError(f"Failed to build normalizer: {exc}") from exc

    counters = {"updated": 0, "unchanged": 0, "skipped": 0}
    rows = list(_iter_cache_rows(db_path))
    logger.info("Processing %d rows from chunk_cache…", len(rows))

    for row in rows:
        chapter_key   = row["chapter_key"]
        sentence_hash = row["sentence_hash"]
        original_text = row["original_text"] or ""
        raw_trans     = row["raw_transcription"] or row["transcription"] or ""
        old_sim       = row["similarity"]

        # Skip manual overrides (no raw Whisper text available)
        if raw_trans.startswith("[manual]"):
            logger.debug("[%s] %s — manual transcription, skip.", chapter_key, sentence_hash[:10])
            counters["skipped"] += 1
            continue

        # Skip rows without any transcription
        if not raw_trans.strip():
            logger.debug("[%s] %s — no transcription, skip.", chapter_key, sentence_hash[:10])
            counters["skipped"] += 1
            continue

        # Recompute similarity with pre_compare applied to both sides
        norm_orig  = _normalize_for_compare(normalize(original_text))
        norm_trans = _normalize_for_compare(normalize(raw_trans))
        new_sim    = _similarity(norm_orig, norm_trans)
        new_trans  = normalize(raw_trans)  # store expanded form as transcription

        delta = new_sim - (old_sim or 0.0)

        if abs(delta) < 1e-6 and new_trans == (row["transcription"] or ""):
            counters["unchanged"] += 1
            continue

        logger.info(
            "[%s] %s  sim %.3f → %.3f  (Δ%+.3f)  orig: %s",
            chapter_key, sentence_hash[:10],
            old_sim or 0.0, new_sim, delta,
            original_text[:60],
        )

        if not dry_run:
            _update_row(db_path, chapter_key, sentence_hash, new_sim, new_trans)

        counters["updated"] += 1

    action = "[DRY-RUN] would update" if dry_run else "updated"
    logger.info(
        "Done — %s %d, unchanged %d, skipped %d",
        action, counters["updated"], counters["unchanged"], counters["skipped"],
    )
    return counters


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Recalculate chunk_cache similarity scores using the current "
            "pre-compare normalizer (ru_numbers etc applied to both sides)."
        )
    )
    parser.add_argument(
        "output_folder",
        help="Book output folder that contains wav/_state/audio_chunks.sqlite3",
    )
    parser.add_argument(
        "--language", default="ru-RU",
        help="Language code for the normalizer chain (default: ru-RU).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would change without writing to the DB.",
    )
    parser.add_argument(
        "--log", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level (default: INFO).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    counters = recalculate(
        output_folder=args.output_folder,
        language=args.language,
        dry_run=args.dry_run,
    )

    if args.dry_run:
        print(f"\n[DRY-RUN] would update {counters['updated']} rows, "
              f"unchanged {counters['unchanged']}, skipped {counters['skipped']}")
    else:
        print(f"\nUpdated {counters['updated']} rows, "
              f"unchanged {counters['unchanged']}, skipped {counters['skipped']}")


if __name__ == "__main__":
    main()

