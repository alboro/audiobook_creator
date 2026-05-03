# Copyright (c) 2026 alboro <alboro@users.noreply.github.com>
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See: https://polyformproject.org/licenses/noncommercial/1.0.0/

"""Tests for /api/chunk_check_details — specifically that the correct
audio_check_threshold is used when evaluating per-checker pass/fail.

Regression for: whisper_similarity badge shows green at threshold 0.94
when similarity is 0.93, because UiConfig has no audio_check_threshold
attribute and the code was falling back to DEFAULT_THRESHOLD (0.70).
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from audiobook_generator.core.audio_chunk_store import AudioChunkStore
from audiobook_generator.ui.review_server import app, get_chunk_check_details


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_store_with_disputed_chunk(
    db_path: Path,
    chapter_key: str,
    sentence_hash: str,
    similarity: float,
) -> AudioChunkStore:
    """Populate a minimal chunk_cache row with the given similarity score."""
    store = AudioChunkStore(db_path)
    store.save_sentence_version(sentence_hash, "Some original text.")
    store.save_disputed_chunk(
        chapter_key=chapter_key,
        sentence_hash=sentence_hash,
        original_text="Some original text.",
        transcription="some original text",
        raw_transcription="some original text",
        similarity=similarity,
        reference_check_score=None,
        reference_check_threshold=None,
        reference_check_status=None,
        reference_check_payload=None,
    )
    return store


# ---------------------------------------------------------------------------
# _get_effective_cfg unit tests
# ---------------------------------------------------------------------------

def test_get_effective_cfg_reads_threshold_from_ini_when_review_config_lacks_it():
    """UiConfig (no audio_check_threshold) → threshold comes from INI."""
    from audiobook_generator.ui.review_server import _get_effective_cfg

    ui_config = SimpleNamespace(host="127.0.0.1", port=7861, review=True)
    original = getattr(app.state, "review_config", None)
    try:
        app.state.review_config = ui_config
        with patch(
            "audiobook_generator.ui.review_server.load_merged_ini",
            return_value={"audio_check_threshold": "0.94"},
        ):
            cfg = _get_effective_cfg()
        assert float(cfg.audio_check_threshold) == 0.94
    finally:
        app.state.review_config = original


def test_get_effective_cfg_keeps_threshold_when_review_config_has_it():
    """GeneralConfig with audio_check_threshold=0.85 → that value is preserved."""
    from audiobook_generator.ui.review_server import _get_effective_cfg

    full_config = SimpleNamespace(audio_check_threshold=0.85)
    original = getattr(app.state, "review_config", None)
    try:
        app.state.review_config = full_config
        cfg = _get_effective_cfg()
        assert float(cfg.audio_check_threshold) == 0.85
    finally:
        app.state.review_config = original


def test_get_effective_cfg_defaults_to_0_70_when_ini_has_no_threshold():
    """Neither review_config nor INI → DEFAULT_THRESHOLD 0.70."""
    from audiobook_generator.ui.review_server import _get_effective_cfg

    ui_config = SimpleNamespace(host="127.0.0.1", port=7861)
    original = getattr(app.state, "review_config", None)
    try:
        app.state.review_config = ui_config
        with patch(
            "audiobook_generator.ui.review_server.load_merged_ini",
            return_value={},
        ):
            cfg = _get_effective_cfg()
        assert float(cfg.audio_check_threshold) == 0.70
    finally:
        app.state.review_config = original


# ---------------------------------------------------------------------------
# /api/chunk_check_details integration tests
# ---------------------------------------------------------------------------

def _run_check_details(
    output_dir: Path,
    chapter_key: str,
    sentence_hash: str,
    review_config,
    ini_overrides: dict,
) -> dict:
    """Call get_chunk_check_details with the given review_config and INI mock."""
    original = getattr(app.state, "review_config", None)
    try:
        app.state.review_config = review_config
        with patch(
            "audiobook_generator.ui.review_server.load_merged_ini",
            return_value=ini_overrides,
        ):
            return asyncio.run(
                get_chunk_check_details(
                    dir=str(output_dir),
                    chapter_key=chapter_key,
                    hash=sentence_hash,
                )
            )
    finally:
        app.state.review_config = original


def test_whisper_similarity_badge_fails_when_sim_below_ini_threshold():
    """Regression: similarity 0.93 < threshold 0.94 → passed=False (red badge).

    Previously: UiConfig had no audio_check_threshold, so 0.70 was used and
    0.93 >= 0.70 returned True (green badge) — wrong.
    """
    with tempfile.TemporaryDirectory() as tmp:
        output_dir = Path(tmp)
        chapter_key = "0001_Test"
        sentence_hash = "abcdef123456"

        db_path = output_dir / "wav" / "_state" / "audio_chunks.sqlite3"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        _make_store_with_disputed_chunk(db_path, chapter_key, sentence_hash, similarity=0.93)

        # UiConfig-like object — no audio_check_threshold attribute.
        ui_config = SimpleNamespace(host="127.0.0.1", port=7861, review=True)

        result = _run_check_details(
            output_dir=output_dir,
            chapter_key=chapter_key,
            sentence_hash=sentence_hash,
            review_config=ui_config,
            ini_overrides={
                "audio_check_threshold": "0.94",
                "audio_check_checkers": "whisper_similarity",
            },
        )

        cr = result.get("checker_results", {})
        assert "whisper_similarity" in cr, "whisper_similarity checker missing from response"
        assert cr["whisper_similarity"]["passed"] is False, (
            f"Expected passed=False (sim=0.93 < threshold=0.94), got {cr['whisper_similarity']['passed']}"
        )
        assert abs(cr["whisper_similarity"]["score"] - 0.93) < 1e-9


def test_whisper_similarity_badge_passes_when_sim_above_ini_threshold():
    """similarity 0.93 >= threshold 0.90 → passed=True (green badge)."""
    with tempfile.TemporaryDirectory() as tmp:
        output_dir = Path(tmp)
        chapter_key = "0001_Test"
        sentence_hash = "abcdef654321"

        db_path = output_dir / "wav" / "_state" / "audio_chunks.sqlite3"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        _make_store_with_disputed_chunk(db_path, chapter_key, sentence_hash, similarity=0.93)

        ui_config = SimpleNamespace(host="127.0.0.1", port=7861, review=True)

        result = _run_check_details(
            output_dir=output_dir,
            chapter_key=chapter_key,
            sentence_hash=sentence_hash,
            review_config=ui_config,
            ini_overrides={
                "audio_check_threshold": "0.90",
                "audio_check_checkers": "whisper_similarity",
            },
        )

        cr = result.get("checker_results", {})
        assert cr["whisper_similarity"]["passed"] is True, (
            f"Expected passed=True (sim=0.93 >= threshold=0.90), got {cr['whisper_similarity']['passed']}"
        )


def test_whisper_similarity_badge_fails_when_full_config_threshold_used():
    """When review_config already has audio_check_threshold, it takes precedence over INI."""
    with tempfile.TemporaryDirectory() as tmp:
        output_dir = Path(tmp)
        chapter_key = "0001_Test"
        sentence_hash = "aabbcc998877"

        db_path = output_dir / "wav" / "_state" / "audio_chunks.sqlite3"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        _make_store_with_disputed_chunk(db_path, chapter_key, sentence_hash, similarity=0.93)

        # Full config — has its own threshold of 0.95, INI says 0.70 (should be ignored)
        full_config = SimpleNamespace(
            audio_check_threshold=0.95,
            audio_check_checkers="whisper_similarity",
        )

        result = _run_check_details(
            output_dir=output_dir,
            chapter_key=chapter_key,
            sentence_hash=sentence_hash,
            review_config=full_config,
            ini_overrides={"audio_check_threshold": "0.70"},
        )

        cr = result.get("checker_results", {})
        assert cr["whisper_similarity"]["passed"] is False, (
            f"Expected passed=False (sim=0.93 < config threshold=0.95), got {cr['whisper_similarity']['passed']}"
        )
