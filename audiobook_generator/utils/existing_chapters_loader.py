# Copyright (c) 2026 alboro <alboro@users.noreply.github.com>
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See: https://polyformproject.org/licenses/noncommercial/1.0.0/

"""Utilities for loading existing chapters from an output directory.

When the user sets an output directory that already contains generated chapters,
this module provides functions to discover and load those chapters.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from audiobook_generator.utils.sentence_hash import sentence_hash as _sentence_hash

_AUDIO_EXTENSIONS = ["wav", "mp3", "ogg", "m4a"]


@dataclass
class ExistingChapter:
    """Represents a chapter loaded from an existing output directory."""
    chapter_idx: int          # 1-based index
    chapter_key: str          # safe directory name like "0001_Chapter_Title"
    title: str                # original title extracted from filename
    text_path: str            # path to the .txt file
    sentence_count: int       # number of sentences (chunks)
    audio_status: str         # "synthesized", "pending", "partial", "none"


def find_latest_run_folder(output_dir: str | Path) -> Optional[Path]:
    """Find the latest run folder (e.g., '003') in the text/ subdirectory.

    Returns the path to the latest run folder, or None if no runs exist.
    """
    output_dir = Path(output_dir)
    text_dir = output_dir / "text"

    if not text_dir.exists():
        return None

    # Find all run folders (numbered directories like "001", "002", etc.)
    run_folders = []
    for item in text_dir.iterdir():
        if item.is_dir() and re.match(r'^\d{3}$', item.name):
            run_folders.append(item.name)

    if not run_folders:
        return None

    run_folders.sort()
    return text_dir / run_folders[-1]


def _chapter_audio_status(chapter_key: str, sentences: List[str], audio_root: Path) -> str:
    """Determine audio status for a chapter by scanning the filesystem.

    Checks wav/chunks/<chapter_key>/<hash>.<ext> for each sentence hash.
    """
    if not sentences:
        return "none"

    chunks_dir = audio_root / "chunks" / chapter_key
    if not chunks_dir.exists():
        return "none"

    found = 0
    for s in sentences:
        h = _sentence_hash(s)
        for ext in _AUDIO_EXTENSIONS:
            if (chunks_dir / f"{h}.{ext}").exists():
                found += 1
                break

    if found == 0:
        return "none"
    if found == len(sentences):
        return "synthesized"
    return "partial"


def load_chapters_from_run_folder(run_folder: Path, audio_root: Path | None = None) -> List[ExistingChapter]:
    """Load chapter metadata from a run folder.

    Scans the run folder for .txt files and extracts chapter information.
    Audio synthesis status is determined by checking chunk files on disk.
    """
    chapters: List[ExistingChapter] = []

    if not run_folder.exists():
        return chapters

    # run_folder is output/text/NNN — go two levels up to get output root
    output_root = run_folder.parent.parent
    audio_root = Path(audio_root) if audio_root is not None else (output_root / "wav")

    for txt_file in run_folder.glob("*.txt"):
        match = re.match(r'^(\d{4})_(.+)\.txt$', txt_file.name)
        if not match:
            continue

        chapter_idx = int(match.group(1))
        title = match.group(2).replace("_", " ")
        chapter_key = txt_file.stem  # "0001_Title"

        sentences: List[str] = []
        try:
            text = txt_file.read_text(encoding="utf-8")
            from audiobook_generator.core.chunked_audio_generator import split_into_sentences
            sentences = split_into_sentences(text, "ru")
        except Exception:
            pass

        audio_status = _chapter_audio_status(chapter_key, sentences, audio_root)

        chapters.append(ExistingChapter(
            chapter_idx=chapter_idx,
            chapter_key=chapter_key,
            title=title,
            text_path=str(txt_file),
            sentence_count=len(sentences),
            audio_status=audio_status,
        ))

    chapters.sort(key=lambda c: c.chapter_idx)
    return chapters


def get_full_text_for_chapter(text_path: str) -> Optional[str]:
    """Load full chapter text from a text file path."""
    try:
        return Path(text_path).read_text(encoding="utf-8")
    except Exception:
        return None


def split_text_into_chunks(text: str, language: str = "ru") -> List[str]:
    """Split text into sentence chunks, respecting quoted speech blocks.

    Quoted blocks («...» / "...") are further split into sub-sentences,
    mirroring the synthesis logic in ChunkedAudioGenerator.
    """
    from audiobook_generator.core.chunked_audio_generator import split_sentences_with_voices
    # Pass a sentinel voice2 so quote-splitting is triggered; discard voice tags.
    pairs = split_sentences_with_voices(text, language, voice2="__display__")
    return [sentence for sentence, _voice in pairs]
