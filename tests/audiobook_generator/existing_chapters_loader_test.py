# Copyright (c) 2026 alboro <alboro@users.noreply.github.com>
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See: https://polyformproject.org/licenses/noncommercial/1.0.0/

"""Tests for existing_chapters_loader module."""

import tempfile
from pathlib import Path
from types import SimpleNamespace

from audiobook_generator.utils.existing_chapters_loader import (
    find_latest_run_folder,
    load_chapters_from_run_folder,
    get_full_text_for_chapter,
    split_text_into_chunks,
)


def test_find_latest_run_folder():
    """Test finding the latest run folder."""
    with tempfile.TemporaryDirectory() as tmp:
        output_dir = Path(tmp)

        # No runs yet
        assert find_latest_run_folder(output_dir) is None

        # Create some run folders
        (output_dir / "text" / "001").mkdir(parents=True)
        (output_dir / "text" / "002").mkdir(parents=True)
        (output_dir / "text" / "010").mkdir(parents=True)

        latest = find_latest_run_folder(output_dir)
        assert latest is not None
        assert latest.name == "010"

        # Add a later run
        (output_dir / "text" / "003").mkdir(parents=True)
        latest = find_latest_run_folder(output_dir)
        assert latest is not None
        assert latest.name == "010"  # Still 010 as it's the latest

        # Add a much later run
        (output_dir / "text" / "999").mkdir(parents=True)
        latest = find_latest_run_folder(output_dir)
        assert latest is not None
        assert latest.name == "999"


def test_load_chapters_from_run_folder():
    """Test loading chapter metadata from a run folder."""
    with tempfile.TemporaryDirectory() as tmp:
        run_folder = Path(tmp) / "003"
        run_folder.mkdir(parents=True)

        # Create some chapter files
        (run_folder / "0001_First_Chapter.txt").write_text("First chapter content.", encoding="utf-8")
        (run_folder / "0002_Second_Chapter.txt").write_text("Second chapter.\n\nWith two sentences.", encoding="utf-8")
        (run_folder / "0003_Third_Chapter.txt").write_text("Third chapter content here.", encoding="utf-8")

        # Create non-chapter file that shouldn't be picked up
        (run_folder / "_state").mkdir()
        (run_folder / "_state" / "some_other_file.txt").write_text("ignored", encoding="utf-8")

        chapters = load_chapters_from_run_folder(run_folder)

        assert len(chapters) == 3
        assert chapters[0].chapter_idx == 1
        assert chapters[0].title == "First Chapter"
        assert chapters[0].text_path == str(run_folder / "0001_First_Chapter.txt")
        assert chapters[0].audio_status == "none"  # No DB

        assert chapters[1].chapter_idx == 2
        assert chapters[1].title == "Second Chapter"


def test_load_chapters_with_audio_status():
    """Test that audio status is derived from chunk files on disk (FS-based)."""
    with tempfile.TemporaryDirectory() as tmp:
        output_dir = Path(tmp)
        run_folder = output_dir / "text" / "003"
        run_folder.mkdir(parents=True)

        from audiobook_generator.utils.sentence_hash import sentence_hash

        # Chapter 1: single sentence "First.", chunk file present → synthesized
        (run_folder / "0001_First.txt").write_text("First.", encoding="utf-8")
        h1 = sentence_hash("First.")
        chunk_dir1 = output_dir / "wav" / "chunks" / "0001_First"
        chunk_dir1.mkdir(parents=True)
        (chunk_dir1 / f"{h1}.wav").write_bytes(b"dummy")

        # Chapter 2: single sentence "Second.", no chunk file → none
        (run_folder / "0002_Second.txt").write_text("Second.", encoding="utf-8")

        chapters = load_chapters_from_run_folder(run_folder)
        assert len(chapters) == 2

        ch1 = next(c for c in chapters if c.chapter_idx == 1)
        assert ch1.audio_status == "synthesized"

        ch2 = next(c for c in chapters if c.chapter_idx == 2)
        assert ch2.audio_status == "none"


def test_load_chapters_with_audio_status_from_override_audio_root():
    with tempfile.TemporaryDirectory() as tmp:
        output_dir = Path(tmp) / "book"
        run_folder = output_dir / "text" / "003"
        run_folder.mkdir(parents=True)
        external_audio_root = Path(tmp) / "external_audio"

        from audiobook_generator.utils.sentence_hash import sentence_hash

        (run_folder / "0001_First.txt").write_text("First.", encoding="utf-8")
        h1 = sentence_hash("First.")
        chunk_dir1 = external_audio_root / "chunks" / "0001_First"
        chunk_dir1.mkdir(parents=True)
        (chunk_dir1 / f"{h1}.wav").write_bytes(b"dummy")

        chapters = load_chapters_from_run_folder(run_folder, audio_root=external_audio_root)
        assert len(chapters) == 1
        assert chapters[0].audio_status == "synthesized"


def test_split_text_into_chunks():
    """Test splitting text into chunks for display."""
    text = "First sentence. Second sentence! Third sentence? Fourth sentence."
    chunks = split_text_into_chunks(text, "en")

    assert len(chunks) == 4
    assert chunks[0] == "First sentence."
    assert chunks[1] == "Second sentence!"


def test_get_full_text_for_chapter():
    """Test loading full text from a chapter file."""
    with tempfile.TemporaryDirectory() as tmp:
        txt_file = Path(tmp) / "0001_Test.txt"
        txt_file.write_text("Hello world. This is a test.", encoding="utf-8")

        text = get_full_text_for_chapter(str(txt_file))
        assert text == "Hello world. This is a test."

        # Non-existent file
        text = get_full_text_for_chapter(str(Path(tmp) / "nonexistent.txt"))
        assert text is None


def test_review_server_uses_audio_folder_override_for_chunk_and_db_paths():
    with tempfile.TemporaryDirectory() as tmp:
        audio_root = Path(tmp) / "mounted_audio"
        chapter_key = "0001_Test"
        sentence_hash = "abc123"
        chunk_dir = audio_root / "chunks" / chapter_key
        chunk_dir.mkdir(parents=True)
        chunk_path = chunk_dir / f"{sentence_hash}.wav"
        chunk_path.write_bytes(b"dummy")

        state_dir = audio_root / "_state"
        state_dir.mkdir(parents=True)

        from audiobook_generator.ui import review_server

        old_cfg = getattr(review_server.app.state, "review_config", None)
        review_server.app.state.review_config = SimpleNamespace(audio_folder=str(audio_root))
        try:
            assert review_server._find_chunk_path("/ignored/output", chapter_key, sentence_hash) == str(chunk_path)
            assert review_server._audio_db_path("/ignored/output") == str(state_dir / "audio_chunks.sqlite3")
        finally:
            review_server.app.state.review_config = old_cfg


if __name__ == "__main__":
    test_find_latest_run_folder()
    test_load_chapters_from_run_folder()
    test_load_chapters_with_audio_status()
    test_split_text_into_chunks()
    test_get_full_text_for_chapter()
    print("All tests passed!")
