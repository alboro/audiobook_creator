# Copyright (c) 2026 alboro <alboro@users.noreply.github.com>
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See: https://polyformproject.org/licenses/noncommercial/1.0.0/

"""FastAPI-based Review UI server.

Replaces the Gradio review UI with a lightweight FastAPI + Alpine.js interface.

Run via:
    python main_ui.py --review [--host 127.0.0.1] [--port 7861]
"""
from __future__ import annotations

import re
import subprocess
import threading
from pathlib import Path
from typing import List, Optional
from urllib.parse import unquote, urlparse

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from audiobook_generator.config.ini_config_manager import load_merged_ini
from audiobook_generator.core.audio_chunk_store import AudioChunkStore
from audiobook_generator.core.chunked_audio_generator import split_sentences_with_voices
from audiobook_generator.ui.review_text_ops import apply_review_edit, normalize_chunk_eof_text
from audiobook_generator.utils.existing_chapters_loader import (
    ExistingChapter,
    find_latest_run_folder,
    load_chapters_from_run_folder,
)
from audiobook_generator.utils.sentence_hash import sentence_hash as _sentence_hash

app = FastAPI(title="Audiobook Review UI")

# ── Paths ──────────────────────────────────────────────────────────────────
_UI_DIR = Path(__file__).parent


# ── Helpers ────────────────────────────────────────────────────────────────

def _audio_db_path(output_dir: str) -> str:
    return str(_resolve_audio_root(output_dir) / "_state" / "audio_chunks.sqlite3")


def _config_audio_folder() -> Optional[str]:
    cfg = getattr(app.state, "review_config", None)
    return getattr(cfg, "audio_folder", None) if cfg else None


def _smb_url_to_local_path(audio_folder: str, mount_root: str | Path = "/Volumes") -> Optional[str]:
    if not audio_folder:
        return None
    parsed = urlparse(audio_folder)
    if parsed.scheme.lower() != "smb":
        return None
    parts = [unquote(part) for part in parsed.path.split("/") if part]
    if not parts:
        return None
    return str(Path(mount_root).joinpath(*parts))


def _resolve_audio_root(output_dir: str) -> Path:
    configured = _config_audio_folder()
    if configured:
        smb_local = _smb_url_to_local_path(configured)
        if smb_local and Path(smb_local).is_dir():
            return Path(smb_local)
        return Path(configured)
    return Path(output_dir) / "wav"


def _wav_duration(path: str) -> Optional[float]:
    """Read WAV duration in seconds by parsing the RIFF/WAVE header directly.

    Supports both PCM (format 1) and IEEE Float (format 3) WAV files,
    unlike Python's standard ``wave`` module which only handles PCM.
    """
    import struct
    try:
        with open(path, 'rb') as f:
            # RIFF header: "RIFF" <size> "WAVE"
            riff_id = f.read(4)
            if riff_id != b'RIFF':
                return None
            f.read(4)  # file size — skip
            wave_id = f.read(4)
            if wave_id != b'WAVE':
                return None

            sample_rate: Optional[int] = None
            num_channels: Optional[int] = None
            bits_per_sample: Optional[int] = None
            data_size: Optional[int] = None

            while True:
                chunk_id = f.read(4)
                if len(chunk_id) < 4:
                    break
                chunk_size_bytes = f.read(4)
                if len(chunk_size_bytes) < 4:
                    break
                chunk_size = struct.unpack('<I', chunk_size_bytes)[0]

                if chunk_id == b'fmt ':
                    fmt_data = f.read(chunk_size)
                    if len(fmt_data) < 16:
                        return None
                    # audio_format(2) num_channels(2) sample_rate(4) byte_rate(4) block_align(2) bits_per_sample(2)
                    num_channels = struct.unpack_from('<H', fmt_data, 2)[0]
                    sample_rate = struct.unpack_from('<I', fmt_data, 4)[0]
                    bits_per_sample = struct.unpack_from('<H', fmt_data, 14)[0]
                elif chunk_id == b'data':
                    data_size = chunk_size
                    break  # no need to read further
                else:
                    # Skip unknown chunk (pad to even byte boundary)
                    f.seek(chunk_size + (chunk_size & 1), 1)

            if sample_rate and num_channels and bits_per_sample and data_size is not None:
                bytes_per_sample = bits_per_sample // 8
                if bytes_per_sample > 0 and num_channels > 0 and sample_rate > 0:
                    frames = data_size // (num_channels * bytes_per_sample)
                    return frames / float(sample_rate)
            return None
    except Exception:
        return None


def _get_dur(h: str, path: Optional[str]) -> Optional[float]:
    """Return cached WAV duration for *hash* h, computing from *path* if needed."""
    cache = getattr(app.state, '_dur_cache', None)
    if cache is None:
        app.state._dur_cache = {}
        cache = app.state._dur_cache
    if h in cache:
        return cache[h]
    if path is None:
        return None
    dur = _wav_duration(path)
    if dur is not None:
        cache[h] = dur
    return dur


def _find_chunk_path(output_dir: str, chapter_key: str, s_hash: str) -> Optional[str]:
    chunks_dir = _resolve_audio_root(output_dir) / "chunks" / chapter_key
    if not chunks_dir.exists():
        return None
    for ext in ("wav", "mp3", "ogg", "m4a"):
        p = chunks_dir / f"{s_hash}.{ext}"
        if p.exists():
            return str(p)
    return None


# ── Routes ─────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    html = (_UI_DIR / "review_ui.html").read_text(encoding="utf-8")
    return HTMLResponse(html)


@app.get("/api/chapters")
async def get_chapters(dir: str):
    """Load chapters for an output directory."""
    output_path = Path(dir)
    if not output_path.exists():
        raise HTTPException(404, f"Directory not found: {dir}")

    run_folder = find_latest_run_folder(str(output_path))
    if not run_folder:
        raise HTTPException(404, "No run folder found in output directory")

    chapters = load_chapters_from_run_folder(run_folder, audio_root=_resolve_audio_root(dir))
    result = []
    for ch in chapters:
        result.append({
            "idx": ch.chapter_idx,
            "key": ch.chapter_key,
            "title": ch.title,
            "text_path": ch.text_path,
            "audio_status": ch.audio_status,
            "sentence_count": ch.sentence_count,
        })
    return result


def _config_voice_name2() -> Optional[str]:
    """Return voice_name2 from server config or INI, or None if not set."""
    cfg = getattr(app.state, "review_config", None)
    v2 = getattr(cfg, "voice_name2", None)
    if v2:
        return v2
    try:
        ini = load_merged_ini()
        raw = ini.get("voice_name2")
        if raw:
            return raw
    except Exception:
        pass
    return None


@app.get("/api/chunks")
async def get_chunks(dir: str, chapter_key: str, text_path: str):
    """Return chunks for a chapter with their audio status."""
    if not Path(text_path).exists():
        raise HTTPException(404, f"Text file not found: {text_path}")
    text = Path(text_path).read_text(encoding="utf-8")
    voice2 = _config_voice_name2()
    pairs = split_sentences_with_voices(text, "ru", voice2=voice2)

    # Load audio-check data from DB (if DB exists)
    retry_counts: dict[str, int] = {}
    check_statuses: dict[str, str | None] = {}
    check_similarity: dict[str, float | None] = {}
    db_path = _audio_db_path(dir)
    if Path(db_path).exists():
        try:
            store = AudioChunkStore(db_path)
            for _chunk, _ in pairs:
                h = _sentence_hash(_chunk)
                cnt = store.get_auto_deletion_count(h)
                if cnt:
                    retry_counts[h] = cnt
                row = store.get_cached_transcription_entry(chapter_key, h)
                if row:
                    check_statuses[h] = row["status"]
                    # similarity is not in get_cached_transcription_entry; fetch via full row
        except Exception:
            pass

    # Also fetch similarity for each chunk via a separate query if DB exists
    if Path(db_path).exists():
        try:
            import sqlite3 as _sqlite3
            from contextlib import closing as _closing
            with _closing(_sqlite3.connect(str(db_path), timeout=10)) as _conn:
                _conn.row_factory = _sqlite3.Row
                for _chunk, _ in pairs:
                    h = _sentence_hash(_chunk)
                    r = _conn.execute(
                        "SELECT similarity FROM chunk_cache WHERE chapter_key=? AND sentence_hash=?",
                        (chapter_key, h),
                    ).fetchone()
                    if r:
                        check_similarity[h] = r["similarity"]
        except Exception:
            pass

    result = []
    for i, (chunk, voice) in enumerate(pairs):
        h = _sentence_hash(chunk)
        audio_path = _find_chunk_path(dir, chapter_key, h)
        dur = _get_dur(h, audio_path) if audio_path else None
        file_mtime: float | None = None
        if audio_path:
            try:
                file_mtime = Path(audio_path).stat().st_mtime
            except OSError:
                pass
        result.append({
            "idx": i,
            "text": chunk,
            "hash": h,
            "has_audio": audio_path is not None,
            # voice is None for default voice; non-None means voice2 is used for this chunk
            "voice": voice,
            # how many times this chunk was auto-deleted and re-synthesised (0 = never)
            "auto_retry_count": retry_counts.get(h, 0),
            # WAV duration in seconds (null when audio not yet synthesised)
            "duration_s": round(dur, 3) if dur is not None else None,
            # Unix timestamp (seconds) of audio file last modification time
            "file_mtime": round(file_mtime) if file_mtime is not None else None,
            # audio-check status from DB: 'checked' | 'disputed' | 'resolved' | null
            "check_status": check_statuses.get(h),
            # whisper similarity score from last audio_check run (null if never checked)
            "check_similarity": check_similarity.get(h),
        })
    return result


@app.get("/api/audio")
async def get_audio(dir: str, chapter_key: str, hash: str):
    """Stream audio chunk file."""
    path = _find_chunk_path(dir, chapter_key, hash)
    if not path:
        raise HTTPException(404, "Audio not found")
    return FileResponse(path, media_type="audio/wav")


@app.get("/api/chapter_durations")
async def get_chapter_durations(dir: str):
    """Return total synthesised audio duration (seconds) for every chapter.

    Used by the Review UI to compute book-global chunk timecodes.
    Results are fast after the first call because WAV header reads are
    cached in-memory by sentence hash.
    """
    output_path = Path(dir)
    if not output_path.exists():
        raise HTTPException(404, f"Directory not found: {dir}")

    run_folder = find_latest_run_folder(str(output_path))
    if not run_folder:
        return []

    chapters = load_chapters_from_run_folder(run_folder, audio_root=_resolve_audio_root(dir))
    voice2 = _config_voice_name2()

    result = []
    for ch in chapters:
        total = 0.0
        if Path(ch.text_path).exists():
            try:
                text = Path(ch.text_path).read_text(encoding="utf-8")
                pairs = split_sentences_with_voices(text, "ru", voice2=voice2)
                for chunk_text, _ in pairs:
                    h = _sentence_hash(chunk_text)
                    audio_path = _find_chunk_path(dir, ch.chapter_key, h)
                    dur = _get_dur(h, audio_path) if audio_path else None
                    if dur is not None:
                        total += dur
            except Exception:
                pass
        result.append({
            "key": ch.chapter_key,
            "idx": ch.chapter_idx,
            "total_s": round(total, 2),
        })
    return result


@app.get("/api/history")
async def get_history(dir: str, hash: str):
    """Return version history for a sentence hash."""
    db_path = _audio_db_path(dir)
    if not Path(db_path).exists():
        return []
    store = AudioChunkStore(db_path)
    rows = store.get_sentence_predecessors(hash)
    return [
        {
            "hash": r["sentence_hash"],
            "text": r["sentence_text"],
            "replaced_by": r["replaced_by_hash"],
            "created_at": r["created_at"],
        }
        for r in rows
    ]


class SaveEditRequest(BaseModel):
    dir: str
    chapter_key: str
    text_path: str
    old_text: str
    new_text: str


@app.post("/api/save")
async def save_edit(req: SaveEditRequest):
    """Save edited sentence to text file and record in version history.

    If the old chunk had ``status='disputed'`` in the DB the new chunk inherits
    that status so it stays visible in the Review UI after re-synthesis.
    """
    new_text = normalize_chunk_eof_text(req.new_text)
    if not new_text.strip():
        raise HTTPException(400, "New text is empty")
    if req.old_text == new_text:
        return {"status": "no_change"}

    text_file = Path(req.text_path)
    if not text_file.exists():
        raise HTTPException(404, "Text file not found")

    old_hash = _sentence_hash(req.old_text)
    new_hash = _sentence_hash(new_text)

    full_text = text_file.read_text(encoding="utf-8")

    try:
        new_full_text = apply_review_edit(full_text, req.old_text, new_text)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    text_file.write_text(new_full_text, encoding="utf-8")

    marked_disputed = False
    db_path = _audio_db_path(req.dir)
    try:
        store = AudioChunkStore(db_path)
        store.save_sentence_version(old_hash, req.old_text, replaced_by_hash=new_hash)
        store.save_sentence_version(new_hash, new_text)

        # Propagate disputed status: if the original chunk was disputed, the
        # edited chunk should also start as disputed so it stays in the review
        # queue until the new audio is checked.
        old_row = store.get_chunk_cache_full_row(req.chapter_key, old_hash)
        if old_row and old_row["status"] == "disputed":
            store.save_disputed_chunk(
                chapter_key=req.chapter_key,
                sentence_hash=new_hash,
                original_text=new_text,
                transcription="",
                similarity=0.0,
            )
            marked_disputed = True
    except Exception as e:
        print(f"[review_server] DB error (non-fatal): {e}")

    return {"status": "ok", "old_hash": old_hash, "new_hash": new_hash,
            "marked_disputed": marked_disputed}


class DeleteRequest(BaseModel):
    dir: str
    chapter_key: str
    text_path: str
    text: str


class DeleteAudioRequest(BaseModel):
    dir: str
    chapter_key: str
    hash: str
    text: str | None = None


@app.post("/api/delete_audio")
async def delete_audio(req: DeleteAudioRequest):
    """Delete the audio file for a chunk (WAV/MP3/etc) without touching the text."""
    path = _find_chunk_path(req.dir, req.chapter_key, req.hash)
    if not path:
        raise HTTPException(404, "Audio file not found")
    try:
        Path(path).unlink()
    except OSError as e:
        raise HTTPException(500, f"Could not delete file: {e}")

    db_path = _audio_db_path(req.dir)
    store = AudioChunkStore(db_path)
    original_text = (req.text or "").strip() or store.get_latest_sentence_text(req.hash)
    if not original_text:
        original_text = f"[hash:{req.hash}]"

    store.mark_missing_audio_disputed(
        chapter_key=req.chapter_key,
        sentence_hash=req.hash,
        original_text=original_text,
    )

    return {
        "status": "ok",
        "deleted": path,
        "disputed": {
            "hash": req.hash,
            "original_text": original_text,
            "transcription": "[manual] audio deleted",
            "similarity": 0.0,
            "status": "disputed",
            "resolved": False,
        },
    }


@app.post("/api/delete")
async def delete_sentence(req: DeleteRequest):
    """Remove sentence from text file and record deletion."""
    text_file = Path(req.text_path)
    if not text_file.exists():
        raise HTTPException(404, "Text file not found")

    old_hash = _sentence_hash(req.text)
    full_text = text_file.read_text(encoding="utf-8")
    if req.text not in full_text:
        raise HTTPException(400, "Sentence not found in file")

    new_full_text = full_text.replace(req.text, "", 1)
    new_full_text = re.sub(r'\n{3,}', '\n\n', new_full_text).strip()
    text_file.write_text(new_full_text + "\n", encoding="utf-8")

    db_path = _audio_db_path(req.dir)
    try:
        store = AudioChunkStore(db_path)
        store.save_sentence_version(old_hash, req.text, replaced_by_hash=None)
    except Exception as e:
        print(f"[review_server] DB history error (non-fatal): {e}")

    return {"status": "ok", "deleted_hash": old_hash}


def _get_chapter_titles_file_path(output_dir: Optional[str] = None) -> Optional[Path]:
    """Return the chapter_titles_file path.

    Priority:
      1. chapter_titles_file from server config / merged INI (explicit override).
      2. Auto-computed fallback: <output_dir>/text/<latest_run>/chapter_titles.txt
         (created on demand when the user edits a title via the Review UI).
    """
    cfg = getattr(app.state, "review_config", None)
    path = getattr(cfg, "chapter_titles_file", None)
    if not path:
        try:
            ini = load_merged_ini()
            path = ini.get("chapter_titles_file")
        except Exception:
            pass
    if path:
        return Path(str(path)).expanduser()

    # Fallback: derive path from the latest run folder inside output_dir
    if output_dir:
        run_folder = find_latest_run_folder(output_dir)
        if run_folder:
            return run_folder / "chapter_titles.txt"
        # No run folder yet — place it directly in <output_dir>/text/
        text_dir = Path(output_dir) / "text"
        if text_dir.exists():
            return text_dir / "chapter_titles.txt"

    return None


@app.get("/api/chapter_titles")
async def get_chapter_titles(dir: str = ""):
    """Return the chapter_titles_file path and its current lines (0-indexed overrides)."""
    path = _get_chapter_titles_file_path(dir or None)
    if not path:
        return {"path": None, "titles": []}
    titles: list[str] = []
    if path.is_file():
        try:
            titles = path.read_text(encoding="utf-8").splitlines()
        except Exception:
            pass
    return {"path": str(path), "titles": titles}


class SaveChapterTitleRequest(BaseModel):
    dir: str = ""
    chapter_pos: int   # 0-based position in the sorted chapters list (matches _apply_chapter_title_overrides)
    title: str


@app.post("/api/chapter_title")
async def save_chapter_title(req: SaveChapterTitleRequest):
    """Write a chapter title override into chapter_titles_file (one title per line).

    Line index = chapter_pos (0-based position in the sorted chapters list), matching
    the same indexing used by _apply_chapter_title_overrides at packaging time.

    If chapter_titles_file is not configured in INI, the file is auto-created at
    <output_dir>/text/<latest_run>/chapter_titles.txt.
    """
    path = _get_chapter_titles_file_path(req.dir or None)
    if not path:
        raise HTTPException(
            400,
            "Cannot determine chapter_titles_file path: "
            "either set chapter_titles_file in [m4b] config or run --mode prepare first.",
        )

    existing: list[str] = []
    if path.is_file():
        try:
            existing = path.read_text(encoding="utf-8").splitlines()
        except Exception as e:
            raise HTTPException(500, f"Cannot read chapter_titles_file: {e}")

    line_idx = req.chapter_pos
    if line_idx < 0:
        raise HTTPException(400, "chapter_pos must be >= 0")

    while len(existing) <= line_idx:
        existing.append("")
    existing[line_idx] = req.title.strip()

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(existing) + "\n", encoding="utf-8")
    except Exception as e:
        raise HTTPException(500, f"Cannot write chapter_titles_file: {e}")

    return {"status": "ok", "path": str(path), "line_idx": line_idx}


@app.get("/api/settings")
async def get_settings():
    """Return server-side config values useful for the Review UI."""
    cfg = getattr(app.state, "review_config", None)
    threshold = getattr(cfg, "audio_check_threshold", None)
    if threshold is None:
        # Fall back to reading INI files directly (same priority chain as main.py)
        try:
            ini = load_merged_ini()
            raw = ini.get("audio_check_threshold")
            if raw is not None:
                threshold = float(raw)
        except Exception:
            pass
    if threshold is None:
        threshold = 0.70
    try:
        threshold = float(threshold)
    except (TypeError, ValueError):
        threshold = 0.70

    # All known checker names (from registry) — used by UI filter bar.
    # Configured checkers (subset actually run) are separate.
    checker_names = _get_all_checker_names()

    return {"audio_check_threshold": threshold, "checker_names": checker_names}


def _get_configured_checker_names() -> list[str]:
    """Return ordered checker names from config, falling back to DEFAULT_CHECKERS.

    Used by the checker pipeline to know *which* checkers to actually run.
    """
    from audiobook_generator.core.audio_checkers.base_audio_chunk_checker import DEFAULT_CHECKERS
    cfg = getattr(app.state, "review_config", None)
    spec = getattr(cfg, "audio_check_checkers", None)
    if not spec:
        try:
            ini = load_merged_ini()
            spec = ini.get("audio_check_checkers")
        except Exception:
            pass
    spec = (spec or DEFAULT_CHECKERS).strip()
    return [n.strip() for n in spec.split(",") if n.strip()]


def _get_all_checker_names() -> list[str]:
    """Return all checker names registered in AUDIO_CHECKER_REGISTRY.

    Used by the UI filter bar so every known checker is shown regardless of
    which ones are active in the current config.
    """
    from audiobook_generator.core.audio_checkers.base_audio_chunk_checker import (
        AUDIO_CHECKER_REGISTRY,
    )
    return list(AUDIO_CHECKER_REGISTRY.keys())


def _get_effective_cfg():
    """Return a config object guaranteed to have ``audio_check_threshold`` set.

    ``review_config`` may be a ``UiConfig`` (standalone review server) or a full
    ``GeneralConfig`` (launched from main.py).  ``UiConfig`` has no
    ``audio_check_threshold``, so we supplement it with the INI value to ensure
    checker classmethods (``evaluate_from_row``) always receive the real threshold
    instead of silently falling back to the 0.70 default.
    """
    from types import SimpleNamespace
    cfg = getattr(app.state, "review_config", None)

    # Fast path: config already provides the threshold.
    if getattr(cfg, "audio_check_threshold", None) is not None:
        return cfg

    # Resolve from INI with the same fallback chain used by /api/settings.
    threshold: float = 0.70
    try:
        ini = load_merged_ini()
        raw = ini.get("audio_check_threshold")
        if raw is not None:
            threshold = float(raw)
    except Exception:
        pass

    # Wrap: keep all existing attrs from cfg (if any) and add audio_check_threshold.
    base = vars(cfg) if cfg is not None else {}
    return SimpleNamespace(**base, audio_check_threshold=threshold)


def _get_checker_class(checker_name: str):
    """Return the checker *class* for *checker_name*, or None if unknown."""
    from audiobook_generator.core.audio_checkers.base_audio_chunk_checker import (
        AUDIO_CHECKER_REGISTRY,
    )
    import importlib

    entry = AUDIO_CHECKER_REGISTRY.get(checker_name)
    if not entry:
        return None
    mod_path, cls_name = entry
    try:
        mod = importlib.import_module(mod_path)
        return getattr(mod, cls_name)
    except Exception:
        return None


def _run_checker_on_demand(
    dir: str,
    chapter_key: str,
    sentence_hash: str,
    checker_name: str,
    original_text: str,
    raw_transcription: str,
    cache_row_dict: Optional[dict],
) -> Optional[bool]:
    """Instantiate a single checker and run it against cached transcription.

    Returns True (passed), False (failed), or None if the checker cannot run.
    """
    from audiobook_generator.core.audio_checkers.base_audio_chunk_checker import (
        AUDIO_CHECKER_REGISTRY,
    )
    import importlib

    entry = AUDIO_CHECKER_REGISTRY.get(checker_name)
    if not entry:
        return None

    cfg = _get_effective_cfg()
    if cfg is None:
        return None

    mod_path, cls_name = entry
    try:
        mod = importlib.import_module(mod_path)
        cls = getattr(mod, cls_name)
        checker = cls(cfg)
    except Exception as exc:
        print(f"[review_server] cannot load checker {checker_name!r}: {exc}")
        return None

    audio_path = _find_chunk_path(dir, chapter_key, sentence_hash)
    audio_file = Path(audio_path) if audio_path else Path(f"/nonexistent/{sentence_hash}")

    try:
        result = checker.check(audio_file, original_text, raw_transcription, cache_row_dict)
        return not result.disputed
    except Exception as exc:
        print(f"[review_server] on-demand checker {checker_name!r} error: {exc}")
        return None


@app.get("/api/chunk_check_details")
async def get_chunk_check_details(dir: str, chapter_key: str, hash: str):
    """Return per-checker pass/fail results for a single chunk.

    For each checker configured in ``audio_check_checkers``:
    - Reads the dedicated ``checker_<name>_passed`` column from DB.
    - If the column is NULL, derives the result from existing score columns
      (``similarity`` for *whisper_similarity*, ``reference_check_score`` for
      *reference*) or, when transcription is cached, runs the checker on-demand.
    - Stores freshly-computed results back into the DB for future fast reads.

    Returns::

        {
          "checker_results": {
            "<checker_name>": {"passed": true|false|null, "score": float|null}
          }
        }
    """
    db_path = _audio_db_path(dir)
    if not Path(db_path).exists():
        return {"checker_results": {}}

    store = AudioChunkStore(db_path)
    checker_names = _get_all_checker_names()

    cache_row = store.get_chunk_cache_full_row(chapter_key, hash)
    if not cache_row:
        return {"checker_results": {}}

    cfg = _get_effective_cfg()

    # Convert sqlite3.Row to plain dict once
    cache_dict: dict = dict(cache_row)

    # Resolve raw transcription for on-demand checker execution
    raw_trans: Optional[str] = cache_dict.get("raw_transcription") or None
    if not raw_trans:
        fallback = (cache_dict.get("transcription") or "").strip()
        if fallback and not fallback.startswith("[manual]"):
            raw_trans = fallback

    final: dict[str, dict] = {}
    for name in checker_names:
        checker_cls = _get_checker_class(name)
        if checker_cls is None:
            final[name] = {"passed": None, "score": None}
            continue

        # Each checker class knows how to derive pass/fail from its own columns.
        passed: Optional[bool] = checker_cls.evaluate_from_row(cache_dict, cfg)
        score: Optional[float] = checker_cls.score_from_row(cache_dict, cfg)

        # If still undetermined and a transcription is cached, run on-demand.
        if passed is None and raw_trans:
            original_text = cache_dict.get("original_text") or ""
            if original_text:
                computed = _run_checker_on_demand(
                    dir, chapter_key, hash, name, original_text, raw_trans, cache_dict
                )
                if computed is not None:
                    passed = computed
                    # Persist only for checkers that use the fallback column
                    # (whisper_similarity and reference derive from own columns).
                    if checker_cls.uses_fallback_passed_column:
                        store.save_checker_result(chapter_key, hash, name, passed)

        final[name] = {"passed": passed, "score": score}

    return {"checker_results": final}


@app.get("/api/disputed")
async def get_disputed(dir: str, chapter_key: str, threshold: float = 0.70):
    """Return chunks whose main status is ``disputed``.

    *threshold* is accepted for backward compatibility but intentionally ignored.

    Each row now includes ``checker_results``: a dict mapping checker name to
    ``{"passed": bool|null}`` computed via ``evaluate_from_row``.  This lets the
    UI filter the disputed list by which checker(s) flagged each chunk.
    """
    db_path = _audio_db_path(dir)
    if not Path(db_path).exists():
        return []
    store = AudioChunkStore(db_path)
    rows = store.get_disputed_chunks(chapter_key, threshold=threshold)
    cfg = _get_effective_cfg()
    checker_names = _get_all_checker_names()
    checker_classes = [(n, _get_checker_class(n)) for n in checker_names]
    result = []
    for r in rows:
        checker_results = {}
        for name, cls in checker_classes:
            if cls is None:
                continue
            try:
                passed = cls.evaluate_from_row(r, cfg)
            except Exception:
                passed = None
            checker_results[name] = {"passed": passed}
        result.append(
            {
                "hash": r["sentence_hash"],
                "original_text": r["original_text"],
                "transcription": r["transcription"],
                "similarity": r["similarity"],
                "reference_check_score": r["reference_check_score"],
                "reference_check_threshold": r["reference_check_threshold"],
                "reference_check_status": r["reference_check_status"],
                "reference_check_payload": r["reference_check_payload"],
                "checked_at": r["checked_at"],
                "status": r["status"],
                "resolved": False,  # resolved rows are excluded by the dynamic query
                "checker_results": checker_results,
            }
        )
    return result


class ResolveDisputedRequest(BaseModel):
    dir: str
    chapter_key: str
    hash: str


@app.post("/api/disputed/resolve")
async def resolve_disputed(req: ResolveDisputedRequest):
    """Mark a disputed chunk as resolved."""
    db_path = _audio_db_path(req.dir)
    if not Path(db_path).exists():
        raise HTTPException(404, "DB not found")
    store = AudioChunkStore(db_path)
    store.resolve_disputed_chunk(req.chapter_key, req.hash)
    return {"status": "ok"}


_synthesis_threads: dict[str, threading.Thread] = {}


def _read_book_source(output_dir: str) -> Optional[str]:
    """Return the original book file path saved by AudiobookGenerator, or None."""
    p = Path(output_dir) / "_state" / "book_source.txt"
    if p.exists():
        path = p.read_text(encoding="utf-8").strip()
        if path:
            return path
    return None


@app.post("/api/synthesize")
async def synthesize(dir: str, chapter_key: str, text_path: str):
    """Trigger background re-synthesis for missing audio chunks (--mode audio_chunks)."""
    key = f"{dir}:{chapter_key}"
    if key in _synthesis_threads and _synthesis_threads[key].is_alive():
        return {"status": "already_running"}

    book_source = _read_book_source(dir)
    if not book_source:
        raise HTTPException(
            400,
            "Cannot determine source book file. "
            "Run --mode prepare or --mode audio at least once so _state/book_source.txt is created.",
        )

    project_root = Path(__file__).parent.parent.parent

    def run():
        print(f"[review_server] audio_chunks synthesis started: {chapter_key}")
        try:
            cmd = [
                ".venv/bin/python", "main.py",
                book_source,
                dir,
                "--mode", "audio_chunks",
            ]
            subprocess.run(cmd, cwd=str(project_root))
        except Exception as e:
            print(f"[review_server] Synthesis error: {e}")

    t = threading.Thread(target=run, daemon=True)
    _synthesis_threads[key] = t
    t.start()
    return {"status": "started"}


# ── Entry point ────────────────────────────────────────────────────────────

def host_review_ui_fastapi(config):
    app.state.review_config = config
    host = getattr(config, "host", "127.0.0.1")
    port = getattr(config, "port", None) or 7861
    print(f"Review UI → http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="warning")
