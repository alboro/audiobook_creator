# Copyright (c) 2026 alboro <alboro@users.noreply.github.com>
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See: https://polyformproject.org/licenses/noncommercial/1.0.0/

"""Review UI - Gradio-based interface for reviewing and editing generated chapters.

This UI allows users to:
- Load existing chapters from an output directory
- View chapter text with sentence-level chunk highlighting
- Play individual sentence audio chunks
- Play historical (old) audio versions to compare
- View version history of edited sentences
- Edit sentences and trigger re-synthesis
"""

from __future__ import annotations

import os
import re
import sqlite3
from datetime import datetime, UTC
from pathlib import Path
from typing import List, Optional, Tuple

import gradio as gr

from audiobook_generator.core.audio_chunk_store import AudioChunkStore
from audiobook_generator.ui.review_text_ops import apply_review_edit
from audiobook_generator.utils.existing_chapters_loader import (
    ExistingChapter,
    find_latest_run_folder,
    load_chapters_from_run_folder,
    split_text_into_chunks,
)
from audiobook_generator.utils.sentence_hash import sentence_hash as _sentence_hash


# Global state (simplified approach for Gradio 5.x compatibility)
_current_chapters: List[ExistingChapter] = []
_current_chapter_key: str = ""
_current_chunks: List[str] = []
_current_full_text: str = ""  # Original full chapter text for accurate editing
_audio_db_path: str = ""
_selected_sentence_idx: int = -1  # Track selected sentence index properly
_current_output_dir: str = ""  # Store output dir for use in button callbacks

# Playback state
_playback_active: bool = False
_playback_current_idx: int = -1
_playback_disputed_mode: bool = False  # True = only disputed chunks playback
_current_history_versions: list = []  # List of (old_hash, old_text, replaced_by_hash) from get_history


def get_audio_chunk_path(
    output_dir: str,
    sentence_hash: str,
    chapter_key: str,
) -> Optional[str]:
    """Find the audio chunk file for a given sentence hash and chapter.

    Chunks live in output/wav/chunks/<chapter_key>/<hash>.<ext>.
    """
    chunks_dir = Path(output_dir) / "wav" / "chunks" / chapter_key

    if not chunks_dir.exists():
        return None

    for ext in ["wav", "mp3", "ogg", "m4a"]:
        chunk_path = chunks_dir / f"{sentence_hash}.{ext}"
        if chunk_path.exists():
            return str(chunk_path)

    return None


def get_sentence_versions_from_db(
    db_path: str,
    current_hash: str,
) -> List[Tuple[str, str, str]]:
    """Get version history for a sentence — all OLD versions replaced to reach current.

    Returns list of (old_hash, old_text, replaced_by_hash) ordered newest → oldest.
    """
    if not db_path or not Path(db_path).exists():
        return []

    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT sentence_hash, sentence_text, replaced_by_hash
            FROM sentence_text_versions
            WHERE replaced_by_hash = ?
            ORDER BY created_at DESC
            """,
            (current_hash,),
        ).fetchall()
        conn.close()

        return [(r["sentence_hash"], r["sentence_text"], r["replaced_by_hash"]) for r in rows]
    except sqlite3.Error:
        return []


def build_review_ui():
    """Build the Gradio review UI."""

    with gr.Blocks(analytics_enabled=False, title="Chapter Review") as ui:
        gr.Markdown("# Chapter Review UI")
        gr.Markdown("Load an output directory to review and edit chapters.")

        # Output directory input
        with gr.Row():
            output_dir_input = gr.Textbox(
                label="Output Directory",
                placeholder="/path/to/book_output",
                scale=3,
            )
            load_btn = gr.Button("Load Chapters", variant="primary", scale=1)

        gr.Markdown("---")

        # Chapter list and text view
        with gr.Row():
            with gr.Column(scale=1, min_width=200):
                chapter_list = gr.DataFrame(
                    label="Chapters",
                    headers=["#", "Title", "Status"],
                    datatype=["number", "str", "str"],
                    interactive=True,
                    max_height=500,
                    wrap=True,
                )

            with gr.Column(scale=2):
                chapter_title = gr.Markdown("**Select a chapter**")

                full_text_display = gr.Textbox(
                    label="Full Chapter Text",
                    lines=8,
                    interactive=False,
                )

                gr.Markdown("### Sentences (chunks for TTS)")
                gr.Markdown("*Click on a sentence to see actions*")

                sentence_list = gr.DataFrame(
                    label="Sentences",
                    headers=["#", "Sentence Preview", "Hash"],
                    datatype=["number", "str", "str"],
                    interactive=True,
                    max_height=300,
                    wrap=True,
                )

        gr.Markdown("---")

        # Sentence action buttons
        with gr.Row():
            play_btn = gr.Button("▶ Play Audio", variant="secondary")
            play_all_btn = gr.Button("▶▶ Play All", variant="secondary")
            play_disputed_btn = gr.Button("🔴 Play Disputed", variant="secondary")
            history_btn = gr.Button("📜 History", variant="secondary")
            edit_btn = gr.Button("✏️ Edit", variant="secondary")
            delete_btn = gr.Button("🗑️ Delete", variant="secondary", elem_id="delete_btn")
            synthesize_btn = gr.Button("🎙️ Synthesize", variant="primary")
            action_status = gr.Textbox(label="Status", interactive=False, lines=2)

        # Playback navigation
        with gr.Row(visible=False) as playback_controls:
            prev_btn = gr.Button("⏮️ Prev", variant="secondary")
            next_btn = gr.Button("⏭️ Next", variant="secondary")
            playback_status = gr.Textbox(label="Playing", interactive=False, lines=1)

        # Audio player with autoplay - automatically plays loaded audio
        audio_player = gr.Audio(label="Sentence Audio", visible=False, interactive=False, autoplay=True)

        gr.Markdown("---")

        # History panel
        with gr.Group(visible=False) as history_panel:
            gr.Markdown("### Version History")
            history_list = gr.DataFrame(
                label="Previous Versions",
                headers=["Version", "Text Preview", "Run"],
                datatype=["number", "str", "str"],
                interactive=True,
                max_height=200,
                wrap=True,
            )
            preview_text = gr.Textbox(label="Selected Version Preview", interactive=False, lines=3)
            with gr.Row():
                play_old_audio_btn = gr.Button("▶ Play Old Audio", variant="secondary")
                restore_btn = gr.Button("🔄 Restore", variant="primary")
                close_history_btn = gr.Button("Close")
            old_audio_player = gr.Audio(label="Old Version Audio", visible=False, interactive=False, autoplay=True)

        # Edit panel
        with gr.Group(visible=False) as edit_panel:
            gr.Markdown("### Edit Sentence")
            edit_textarea = gr.Textbox(
                label="Edit Text",
                lines=3,
                interactive=True,
            )
            with gr.Row():
                save_edit_btn = gr.Button("💾 Save", variant="primary")
                cancel_edit_btn = gr.Button("Cancel")

        # Hidden state for sentence index
        selected_sentence_idx = gr.Number(visible=False, value=-1)

        def load_chapters(output_dir: str):
            """Load chapters from the output directory."""
            global _current_chapters, _current_chapter_key, _current_chunks, _audio_db_path, _current_output_dir

            _current_output_dir = output_dir  # Store for later use in callbacks
            print(f"[DEBUG] load_chapters: output_dir={output_dir}")

            if not output_dir:
                return gr.update(value=[]), gr.update(value=""), gr.update(value=""), \
                       gr.update(), gr.update()

            output_path = Path(output_dir)
            if not output_path.exists():
                return gr.update(value=[]), gr.update(value=""), gr.update(value=""), \
                       gr.update(), gr.update()

            run_folder = find_latest_run_folder(str(output_path))
            if not run_folder:
                return gr.update(value=[]), gr.update(value=""), gr.update(value=""), \
                       gr.update(), gr.update()

            chapters = load_chapters_from_run_folder(run_folder)
            _current_chapters = chapters
            _current_chunks = []
            _audio_db_path = str(output_path / "wav" / "_state" / "audio_chunks.sqlite3")

            list_data = []
            for ch in chapters:
                audio_icon = "🔊" if ch.audio_status == "synthesized" else "⏳" if ch.audio_status == "pending" else "🔧" if ch.audio_status == "partial" else "📄"
                status_text = f"{audio_icon} {ch.sentence_count} sentences"
                list_data.append([ch.chapter_idx, ch.title, status_text])

            return (
                gr.update(value=list_data),
                gr.update(value=""),
                gr.update(value=""),
                gr.update(),
                gr.update(value=""),
            )

        def select_chapter(evt: gr.SelectData):
            """Handle chapter selection."""
            global _current_chapters, _current_chapter_key, _current_chunks, _current_full_text, _selected_sentence_idx

            _selected_sentence_idx = -1  # Reset sentence selection on chapter change
            row_idx = evt.index[0] if isinstance(evt.index, (tuple, list)) else evt.index
            print(f"[DEBUG] select_chapter: row_idx={row_idx}, chapters len={len(_current_chapters)}")

            if not _current_chapters or row_idx >= len(_current_chapters):
                return gr.update(), gr.update(), gr.update()

            chapter = _current_chapters[row_idx]
            _current_chapter_key = chapter.chapter_key

            text = ""
            if Path(chapter.text_path).exists():
                text = Path(chapter.text_path).read_text(encoding="utf-8")

            _current_full_text = text  # Store original text for editing
            chunks = split_text_into_chunks(text, "ru")
            _current_chunks = chunks
            print(f"[DEBUG] select_chapter: _current_chunks len={len(chunks)}, first_chunk={chunks[0][:50] if chunks else 'EMPTY'}...")

            sentence_data = []
            for i, chunk in enumerate(chunks):
                h = _sentence_hash(chunk)[:8]
                preview = chunk[:60] + "..." if len(chunk) > 60 else chunk
                sentence_data.append([i + 1, preview, h])

            return (
                gr.update(value=f"### {chapter.title}"),
                gr.update(value=text),
                gr.update(value=sentence_data),
                gr.update(value=row_idx),
            )

        def select_sentence(evt: gr.SelectData):
            """Handle sentence selection - store index."""
            global _selected_sentence_idx
            row_idx = evt.index[0] if isinstance(evt.index, (tuple, list)) else evt.index
            _selected_sentence_idx = row_idx
            print(f"[DEBUG] select_sentence: index={row_idx}, _current_chunks len={len(_current_chunks)}")
            return gr.update(value=row_idx)

        def get_selected_sentence_text():
            """Get the currently selected sentence text."""
            global _current_chunks, _selected_sentence_idx
            idx = _selected_sentence_idx
            if idx >= 0 and idx < len(_current_chunks):
                return _current_chunks[idx]
            return ""

        def play_sentence_audio():
            """Play audio for a sentence (FS-based)."""
            global _current_chunks, _current_chapter_key, _selected_sentence_idx, _current_output_dir

            print(f"[DEBUG] play_sentence_audio: _selected_sentence_idx={_selected_sentence_idx}")
            idx = _selected_sentence_idx

            if idx < 0 or idx >= len(_current_chunks) or not _current_output_dir:
                return gr.update(visible=False, value=None), gr.update(value="No sentence selected")

            sentence_text = _current_chunks[idx]
            s_hash = _sentence_hash(sentence_text)
            audio_path = get_audio_chunk_path(_current_output_dir, s_hash, _current_chapter_key)
            print(f"[DEBUG] play_sentence_audio: hash={s_hash}, path={audio_path}")

            if audio_path and Path(audio_path).exists():
                return (
                    gr.update(visible=True, value=audio_path),
                    gr.update(value=f"▶ Playing: {sentence_text[:60]}...")
                )
            else:
                return (
                    gr.update(visible=False, value=None),
                    gr.update(value=f"🔇 Audio not found for hash {s_hash[:8]}")
                )

        def get_history():
            """Get history data for display — previous versions that were replaced."""
            global _current_chunks, _audio_db_path, _selected_sentence_idx, _current_history_versions

            print(f"[DEBUG] get_history: _selected_sentence_idx={_selected_sentence_idx}")
            idx = _selected_sentence_idx
            if idx < 0 or idx >= len(_current_chunks):
                return gr.update(value=[]), gr.update(value="No sentence selected")

            sentence_text = _current_chunks[idx]
            current_hash = _sentence_hash(sentence_text)

            versions = get_sentence_versions_from_db(_audio_db_path, current_hash)
            _current_history_versions = versions

            if not versions:
                return (
                    gr.update(value=[[current_hash[:8], sentence_text[:80], "current"]]),
                    gr.update(value="No previous versions — this is the original text"),
                )

            list_data = []
            for old_hash, old_text, replaced_by in versions:
                preview = old_text[:60] + "..." if len(old_text) > 60 else old_text
                list_data.append([old_hash[:8], preview, f"→ {replaced_by[:8] if replaced_by else 'DELETED'}"])

            return (
                gr.update(value=list_data),
                gr.update(value=f"Found {len(versions)} previous version(s)"),
            )

        def select_history_version(evt: gr.SelectData):
            """Show full text of the selected historical version in preview."""
            global _current_history_versions
            try:
                row_idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
                if 0 <= row_idx < len(_current_history_versions):
                    _, old_text, _ = _current_history_versions[row_idx]
                    return gr.update(value=old_text)
            except Exception:
                pass
            return gr.update(value="")

        def play_old_audio():
            """Play audio for the selected historical version."""
            global _current_history_versions, _current_chapter_key, _current_output_dir
            # We identify the selected row by the preview text — instead we rely on the last
            # select event which updates _current_history_selected_idx.
            # Simpler: the most recently selected version is kept in preview_text.
            # We'll use _current_history_selected_idx set by select_history_version.
            idx = getattr(play_old_audio, "_selected_idx", -1)
            if idx < 0 or idx >= len(_current_history_versions) or not _current_output_dir:
                return gr.update(visible=False, value=None), gr.update(value="No history version selected")

            old_hash, old_text, _ = _current_history_versions[idx]
            audio_path = get_audio_chunk_path(_current_output_dir, old_hash, _current_chapter_key)
            if audio_path and Path(audio_path).exists():
                return (
                    gr.update(visible=True, value=audio_path),
                    gr.update(value=f"▶ Old version: {old_text[:60]}..."),
                )
            return (
                gr.update(visible=False, value=None),
                gr.update(value=f"🔇 Old audio not found (hash {old_hash[:8]})")
            )

        # We need to track which history row is selected for play_old_audio.
        _history_selected_idx: list = [-1]  # mutable container to share state

        def select_history_version_for_play(evt: gr.SelectData):
            """Track index AND show text preview."""
            global _current_history_versions
            try:
                row_idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
                _history_selected_idx[0] = row_idx
                if 0 <= row_idx < len(_current_history_versions):
                    _, old_text, _ = _current_history_versions[row_idx]
                    return gr.update(value=old_text)
            except Exception:
                pass
            return gr.update(value="")

        def play_old_audio_fn():
            """Play audio for the currently selected history version."""
            global _current_history_versions, _current_chapter_key, _current_output_dir
            idx = _history_selected_idx[0]
            if idx < 0 or idx >= len(_current_history_versions) or not _current_output_dir:
                return gr.update(visible=False, value=None), gr.update(value="No history version selected")

            old_hash, old_text, _ = _current_history_versions[idx]
            audio_path = get_audio_chunk_path(_current_output_dir, old_hash, _current_chapter_key)
            if audio_path and Path(audio_path).exists():
                return (
                    gr.update(visible=True, value=audio_path),
                    gr.update(value=f"▶ Old version ({old_hash[:8]}): {old_text[:50]}..."),
                )
            return (
                gr.update(visible=False, value=None),
                gr.update(value=f"🔇 Old audio not found (hash {old_hash[:8]}). Synthesised after this edit?"),
            )

        def show_edit_panel():
            """Show edit panel."""
            global _current_chunks, _selected_sentence_idx

            print(f"[DEBUG] show_edit_panel: _selected_sentence_idx={_selected_sentence_idx}")
            idx = _selected_sentence_idx
            if idx < 0 or idx >= len(_current_chunks):
                return gr.update(), gr.update()

            return (
                gr.update(value=_current_chunks[idx]),
                gr.update(visible=True),
            )

        def save_edit(new_text: str):
            """Save an edited sentence to text file and record version history."""
            global _current_chapters, _current_chapter_key, _current_chunks, _current_full_text, _audio_db_path, _selected_sentence_idx, _current_output_dir

            idx = _selected_sentence_idx
            output_dir = _current_output_dir

            if not new_text:
                return gr.update(value="No text provided"), gr.update(visible=False)

            if idx < 0 or idx >= len(_current_chunks) or not output_dir:
                return gr.update(value="Invalid state"), gr.update(visible=False)

            old_text = _current_chunks[idx]
            if new_text == old_text:
                return gr.update(value="No changes"), gr.update(visible=False)

            old_hash = _sentence_hash(old_text)
            new_hash = _sentence_hash(new_text)
            print(f"[DEBUG] save_edit: old_hash={old_hash}, new_hash={new_hash}")

            # Update text file
            chapter = next((ch for ch in _current_chapters if ch.chapter_key == _current_chapter_key), None)
            if not chapter or not Path(chapter.text_path).exists():
                return gr.update(value="Chapter text file not found"), gr.update(visible=False)

            try:
                new_full_text = apply_review_edit(_current_full_text, old_text, new_text)
            except ValueError as exc:
                return gr.update(value=str(exc)), gr.update(visible=False), gr.update()
            Path(chapter.text_path).write_text(new_full_text, encoding="utf-8")
            _current_full_text = new_full_text
            print(f"[DEBUG] save_edit: text file updated: {chapter.text_path}")

            # Record version history in DB
            print(f"[DEBUG] save_edit: _audio_db_path={_audio_db_path}")
            if _audio_db_path:
                try:
                    store = AudioChunkStore(_audio_db_path)
                    store.save_sentence_version(old_hash, old_text, replaced_by_hash=new_hash)
                    store.save_sentence_version(new_hash, new_text)
                    print(f"[DEBUG] save_edit: version history saved OK")
                except Exception as e:
                    import traceback
                    print(f"[DEBUG] save_edit: DB error: {e}\n{traceback.format_exc()}")

            # Re-split to reflect any chunk-count changes
            new_chunks = split_text_into_chunks(new_full_text, "ru")
            _current_chunks = new_chunks

            sentence_data = [
                [i + 1, (c[:60] + "..." if len(c) > 60 else c), _sentence_hash(c)[:8]]
                for i, c in enumerate(_current_chunks)
            ]
            return gr.update(value="✅ Saved!"), gr.update(visible=False), gr.update(value=sentence_data)

        def delete_sentence():
            """Delete a sentence — remove from text file and record in version history."""
            global _current_chapters, _current_chapter_key, _current_chunks, _current_full_text, _audio_db_path, _selected_sentence_idx, _current_output_dir

            idx = _selected_sentence_idx
            output_dir = _current_output_dir

            if idx < 0 or idx >= len(_current_chunks) or not output_dir:
                return gr.update(value="Invalid state"), gr.update(visible=False), gr.update()

            old_text = _current_chunks[idx]
            old_hash = _sentence_hash(old_text)
            print(f"[DEBUG] delete_sentence: old_hash={old_hash}, old_text={old_text[:50]}...")

            # Update text file — remove the sentence
            chapter = next((ch for ch in _current_chapters if ch.chapter_key == _current_chapter_key), None)
            if not chapter or not Path(chapter.text_path).exists():
                return gr.update(value="Chapter text file not found"), gr.update(visible=False), gr.update()

            new_full_text = _current_full_text.replace(old_text, "", 1)
            new_full_text = re.sub(r'\n\n+', '\n\n', new_full_text)
            new_full_text = re.sub(r'  +', ' ', new_full_text)
            Path(chapter.text_path).write_text(new_full_text, encoding="utf-8")
            _current_full_text = new_full_text
            print(f"[DEBUG] delete_sentence: text file updated")

            # Record deletion in version history (replaced_by_hash=None → deleted)
            print(f"[DEBUG] delete_sentence: _audio_db_path={_audio_db_path}")
            if _audio_db_path:
                try:
                    store = AudioChunkStore(_audio_db_path)
                    store.save_sentence_version(old_hash, old_text, replaced_by_hash=None)
                    print(f"[DEBUG] delete_sentence: version history saved OK")
                except Exception as e:
                    import traceback
                    print(f"[DEBUG] delete_sentence: DB error: {e}\n{traceback.format_exc()}")

            # Update local state
            _current_chunks.pop(idx)
            _selected_sentence_idx = -1

            sentence_data = [
                [i + 1, (c[:60] + "..." if len(c) > 60 else c), _sentence_hash(c)[:8]]
                for i, c in enumerate(_current_chunks)
            ]
            return (
                gr.update(value=f"🗑️ Deleted sentence (hash: {old_hash[:8]})"),
                gr.update(visible=False),
                gr.update(value=sentence_data),
            )

        def synthesize_chapter():
            """Trigger re-synthesis of the current chapter by running audio mode."""
            global _current_chapters, _current_chapter_key, _current_chunks, _audio_db_path, _selected_sentence_idx, _current_output_dir

            output_dir = _current_output_dir
            if not output_dir:
                return gr.update(value="No output directory loaded")

            chapter = None
            for ch in _current_chapters:
                if ch.chapter_key == _current_chapter_key:
                    chapter = ch
                    break

            if not chapter:
                return gr.update(value="No chapter selected")

            # Run the audio generation in a subprocess
            import subprocess
            import threading

            def run_synthesis():
                print(f"[DEBUG] synthesize_chapter: starting synthesis for {chapter.chapter_key}")
                try:
                    # Do NOT use capture_output=True: TTS progress logs (Submitting polling TTS job,
                    # Polling TTS job, Downloading completed TTS audio for job) must stream to the
                    # parent console in real time so the user sees progress.
                    cmd = [
                        ".venv/bin/python", "main.py",
                        "--mode", "audio",
                    ]
                    if config and getattr(config, "local", None):
                        cmd += ["--config", config.local]
                    cmd += ["--input_file", chapter.text_path]
                    result = subprocess.run(
                        cmd,
                        cwd=Path(__file__).parent.parent.parent,
                        # stdout/stderr inherited from parent → visible in server console.
                    )
                    print(f"[DEBUG] synthesize_chapter: done, returncode={result.returncode}")
                except Exception as e:
                    print(f"[DEBUG] synthesize_chapter: error: {e}")

            thread = threading.Thread(target=run_synthesis)
            thread.start()

            return gr.update(value=f"🎙️ Synthesis started for chapter {chapter.chapter_idx}...")

        def _build_sentence_data_with_marker(idx_to_mark: int):
            """Rebuild sentence_data list marking the currently playing row with ▶."""
            data = []
            for i, chunk in enumerate(_current_chunks):
                h = _sentence_hash(chunk)[:8]
                preview = chunk[:60] + "..." if len(chunk) > 60 else chunk
                if i == idx_to_mark:
                    preview = f"▶ {preview}"
                data.append([i + 1, preview, h])
            return data

        def _load_chunk_at(idx: int):
            """Load chunk at idx into player UI. Returns (controls, status, audio, sel_idx, sentence_list)."""
            global _playback_current_idx, _selected_sentence_idx
            if idx < 0 or idx >= len(_current_chunks):
                _playback_current_idx = -1
                return (
                    gr.update(visible=False),
                    gr.update(value="Finished!"),
                    gr.update(value=None, autoplay=False),
                    gr.update(value=-1),
                    gr.update(value=_build_sentence_data_with_marker(-1)),
                )
            _playback_current_idx = idx
            _selected_sentence_idx = idx
            sentence_text = _current_chunks[idx]
            s_hash = _sentence_hash(sentence_text)
            audio_path = get_audio_chunk_path(_current_output_dir, s_hash, _current_chapter_key)
            status = f"▶ {idx + 1}/{len(_current_chunks)}: {sentence_text[:60]}..."
            sdata = _build_sentence_data_with_marker(idx)
            if audio_path and Path(audio_path).exists():
                return (
                    gr.update(visible=True),
                    gr.update(value=status),
                    gr.update(value=audio_path, autoplay=True, visible=True),
                    gr.update(value=idx),
                    gr.update(value=sdata),
                )
            else:
                return (
                    gr.update(visible=True),
                    gr.update(value=f"🔇 {idx + 1}/{len(_current_chunks)}: Audio not found"),
                    gr.update(value=None, autoplay=False),
                    gr.update(value=idx),
                    gr.update(value=sdata),
                )

        def start_playback_all():
            """Start sequential playback. Uses selected sentence as starting point,
            otherwise starts from 0. Activates continuous chained playback."""
            global _playback_active, _playback_current_idx, _selected_sentence_idx, _playback_disputed_mode
            if not _current_chunks:
                return (
                    gr.update(visible=False),
                    gr.update(value="No chunks to play"),
                    gr.update(value=None, autoplay=False),
                    gr.update(value=-1),
                    gr.update(),
                )
            _playback_active = True
            _playback_disputed_mode = False
            start_idx = _selected_sentence_idx if _selected_sentence_idx >= 0 else 0
            return _load_chunk_at(start_idx)

        # --- disputed chunks helpers ---
        _disputed_hashes: set = set()  # hashes of disputed chunks for current chapter

        def _load_disputed_hashes():
            """Load disputed chunk hashes from DB for current chapter (not resolved only)."""
            _disputed_hashes.clear()
            if not _audio_db_path or not _current_chapter_key:
                return
            try:
                import sqlite3
                conn = sqlite3.connect(_audio_db_path)
                rows = conn.execute(
                    "SELECT sentence_hash FROM disputed_chunks WHERE chapter_key = ? AND resolved = 0",
                    (_current_chapter_key,),
                ).fetchall()
                conn.close()
                for row in rows:
                    _disputed_hashes.add(row[0])
            except Exception as exc:
                print(f"[DEBUG] _load_disputed_hashes: error: {exc}")

        def start_playback_disputed():
            """Play only unresolved disputed chunks for the current chapter."""
            global _playback_active, _playback_current_idx, _current_chunks, _selected_sentence_idx, _playback_disputed_mode

            _load_disputed_hashes()

            if not _disputed_hashes:
                return (
                    gr.update(visible=False),
                    gr.update(value="🔴 No disputed chunks found for this chapter"),
                    gr.update(value=None, autoplay=False),
                    gr.update(value=-1),
                    gr.update(),
                )

            # Find indices of disputed sentences in _current_chunks
            disputed_indices = [
                i for i, chunk in enumerate(_current_chunks)
                if _sentence_hash(chunk) in _disputed_hashes
            ]
            if not disputed_indices:
                return (
                    gr.update(visible=False),
                    gr.update(value="🔴 Disputed hashes found in DB but not matched in current text"),
                    gr.update(value=None, autoplay=False),
                    gr.update(value=-1),
                    gr.update(),
                )

            _playback_active = True
            _playback_disputed_mode = True
            # Load first disputed chunk
            return _load_chunk_at(disputed_indices[0])

        def play_next_disputed():
            """Advance to the next disputed chunk."""
            global _playback_active

            _load_disputed_hashes()
            if not _disputed_hashes:
                _playback_active = False
                return (
                    gr.update(visible=False),
                    gr.update(value="Finished (no more disputed)!"),
                    gr.update(value=None, autoplay=False),
                    gr.update(value=-1),
                    gr.update(value=_build_sentence_data_with_marker(-1)),
                )

            disputed_indices = sorted(
                i for i, chunk in enumerate(_current_chunks)
                if _sentence_hash(chunk) in _disputed_hashes
            )
            # Find next disputed index after current
            next_idx = None
            for di in disputed_indices:
                if di > _playback_current_idx:
                    next_idx = di
                    break

            if next_idx is None:
                _playback_active = False
                return (
                    gr.update(visible=False),
                    gr.update(value="🔴 All disputed chunks played!"),
                    gr.update(value=None, autoplay=False),
                    gr.update(value=-1),
                    gr.update(value=_build_sentence_data_with_marker(-1)),
                )
            _playback_active = True
            return _load_chunk_at(next_idx)

        def play_next_sentence():
            """Advance to next sentence and play it. Keeps continuous-playback flag."""
            global _playback_active
            _playback_active = True
            next_idx = _playback_current_idx + 1
            if next_idx >= len(_current_chunks):
                _playback_active = False
                return (
                    gr.update(visible=False),
                    gr.update(value="Finished!"),
                    gr.update(value=None, autoplay=False),
                    gr.update(value=-1),
                    gr.update(value=_build_sentence_data_with_marker(-1)),
                )
            return _load_chunk_at(next_idx)

        def play_prev_sentence():
            """Go to previous sentence and play it."""
            global _playback_active
            _playback_active = True
            prev_idx = max(0, _playback_current_idx - 1)
            return _load_chunk_at(prev_idx)

        def on_audio_stop():
            """Called when audio player finishes (or is paused). If continuous playback
            is active, advance to the next sentence automatically."""
            if not _playback_active:
                # Not in chained mode - do not advance
                return (
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                )
            # Disputed mode: jump to next disputed chunk
            if _playback_disputed_mode:
                return play_next_disputed()
            # Normal mode: next sequential chunk
            next_idx = _playback_current_idx + 1
            if next_idx >= len(_current_chunks):
                # End of chapter
                return (
                    gr.update(visible=False),
                    gr.update(value="Finished!"),
                    gr.update(value=None, autoplay=False),
                    gr.update(value=-1),
                    gr.update(value=_build_sentence_data_with_marker(-1)),
                )
            return _load_chunk_at(next_idx)

        # ============================================================
        # Event bindings - defined after all functions
        # ============================================================

        # Chapter loading and selection
        load_btn.click(
            fn=load_chapters,
            inputs=[output_dir_input],
            outputs=[chapter_list, chapter_title, full_text_display, sentence_list, action_status],
        )

        chapter_list.select(
            fn=select_chapter,
            inputs=[],
            outputs=[chapter_title, full_text_display, sentence_list, selected_sentence_idx],
        )

        sentence_list.select(
            fn=select_sentence,
            inputs=[],
            outputs=[selected_sentence_idx],
        )

        play_btn.click(
            fn=play_sentence_audio,
            inputs=[],
            outputs=[audio_player, action_status],
        )

        # History
        history_btn.click(
            fn=lambda: gr.update(visible=True),
            inputs=[],
            outputs=[history_panel],
        )

        history_btn.click(
            fn=get_history,
            inputs=[],
            outputs=[history_list, action_status],
        )

        history_list.select(
            fn=select_history_version_for_play,
            inputs=[],
            outputs=[preview_text],
        )

        play_old_audio_btn.click(
            fn=play_old_audio_fn,
            inputs=[],
            outputs=[old_audio_player, action_status],
        )

        # Edit
        edit_btn.click(
            fn=show_edit_panel,
            inputs=[],
            outputs=[edit_textarea, edit_panel],
        )

        save_edit_btn.click(
            fn=save_edit,
            inputs=[edit_textarea],
            outputs=[action_status, edit_panel, sentence_list],
        )

        cancel_edit_btn.click(
            fn=lambda: gr.update(visible=False),
            outputs=[edit_panel],
        )

        close_history_btn.click(
            fn=lambda: gr.update(visible=False),
            outputs=[history_panel],
        )

        # Delete
        delete_btn.click(
            fn=delete_sentence,
            inputs=[],
            outputs=[action_status, edit_panel, sentence_list],
        )

        # Synthesize
        synthesize_btn.click(
            fn=synthesize_chapter,
            inputs=[],
            outputs=[action_status],
        )

        # Playback
        play_all_btn.click(
            fn=start_playback_all,
            inputs=[],
            outputs=[playback_controls, playback_status, audio_player, selected_sentence_idx, sentence_list],
        )

        play_disputed_btn.click(
            fn=start_playback_disputed,
            inputs=[],
            outputs=[playback_controls, playback_status, audio_player, selected_sentence_idx, sentence_list],
        )

        next_btn.click(
            fn=play_next_sentence,
            inputs=[],
            outputs=[playback_controls, playback_status, audio_player, selected_sentence_idx, sentence_list],
        )

        prev_btn.click(
            fn=play_prev_sentence,
            inputs=[],
            outputs=[playback_controls, playback_status, audio_player, selected_sentence_idx, sentence_list],
        )

        # When the audio player finishes (fires .stop event), automatically advance
        # to the next sentence if continuous playback mode is active.
        audio_player.stop(
            fn=on_audio_stop,
            inputs=[],
            outputs=[playback_controls, playback_status, audio_player, selected_sentence_idx, sentence_list],
        )

    return ui


def host_review_ui(config=None):
    """Host the review UI."""
    ui = build_review_ui()
    host = config.host if config else "0.0.0.0"
    port = config.port if config else 7861
    # Allow audio files from any path - users load chapters from various directories
    ui.launch(server_name=host, server_port=port, allowed_paths=["/Users/aldem/Documents/books"])
