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

from audiobook_generator.core.audio_chunk_store import AudioChunkStore
from audiobook_generator.ui.review_text_ops import apply_review_edit
from audiobook_generator.utils.existing_chapters_loader import (
    ExistingChapter,
    find_latest_run_folder,
    load_chapters_from_run_folder,
    split_text_into_chunks,
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


@app.get("/api/chunks")
async def get_chunks(dir: str, chapter_key: str, text_path: str):
    """Return chunks for a chapter with their audio status."""
    if not Path(text_path).exists():
        raise HTTPException(404, f"Text file not found: {text_path}")
    text = Path(text_path).read_text(encoding="utf-8")
    chunks = split_text_into_chunks(text, "ru")
    result = []
    for i, chunk in enumerate(chunks):
        h = _sentence_hash(chunk)
        audio_path = _find_chunk_path(dir, chapter_key, h)
        result.append({
            "idx": i,
            "text": chunk,
            "hash": h,
            "has_audio": audio_path is not None,
        })
    return result


@app.get("/api/audio")
async def get_audio(dir: str, chapter_key: str, hash: str):
    """Stream audio chunk file."""
    path = _find_chunk_path(dir, chapter_key, hash)
    if not path:
        raise HTTPException(404, "Audio not found")
    return FileResponse(path, media_type="audio/wav")


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
    """Save edited sentence to text file and record in version history."""
    if not req.new_text.strip():
        raise HTTPException(400, "New text is empty")
    if req.old_text == req.new_text:
        return {"status": "no_change"}

    text_file = Path(req.text_path)
    if not text_file.exists():
        raise HTTPException(404, "Text file not found")

    old_hash = _sentence_hash(req.old_text)
    new_hash = _sentence_hash(req.new_text)

    full_text = text_file.read_text(encoding="utf-8")
    if req.old_text not in full_text:
        raise HTTPException(400, "Original text not found in file (may have been modified)")

    try:
        new_full_text = apply_review_edit(full_text, req.old_text, req.new_text)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    text_file.write_text(new_full_text, encoding="utf-8")

    db_path = _audio_db_path(req.dir)
    try:
        store = AudioChunkStore(db_path)
        store.save_sentence_version(old_hash, req.old_text, replaced_by_hash=new_hash)
        store.save_sentence_version(new_hash, req.new_text)
    except Exception as e:
        print(f"[review_server] DB history error (non-fatal): {e}")

    return {"status": "ok", "old_hash": old_hash, "new_hash": new_hash}


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


@app.get("/api/disputed")
async def get_disputed(dir: str, chapter_key: str):
    """Return disputed chunks (low transcription similarity) for a chapter."""
    db_path = _audio_db_path(dir)
    if not Path(db_path).exists():
        return []
    store = AudioChunkStore(db_path)
    rows = store.get_disputed_chunks(chapter_key)
    return [
        {
            "hash": r["sentence_hash"],
            "original_text": r["original_text"],
            "transcription": r["transcription"],
            "similarity": r["similarity"],
            "checked_at": r["checked_at"],
            "status": r["status"],
            "resolved": r["status"] == "resolved",
        }
        for r in rows
    ]


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

