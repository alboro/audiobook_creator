# Copyright (c) 2026 alboro <alboro@users.noreply.github.com>
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See: https://polyformproject.org/licenses/noncommercial/1.0.0/

import asyncio
import tempfile
from pathlib import Path

from audiobook_generator.core.audio_chunk_store import AudioChunkStore
from audiobook_generator.ui.review_server import DeleteAudioRequest, delete_audio


def test_delete_audio_marks_chunk_disputed():
    with tempfile.TemporaryDirectory() as tmp:
        output_dir = Path(tmp) / "book_output"
        chapter_key = "0001_Test"
        sentence_hash = "deadbeef1234"
        sentence_text = "Deleted audio sentence."

        audio_path = output_dir / "wav" / "chunks" / chapter_key / f"{sentence_hash}.wav"
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        audio_path.write_bytes(b"fake wav")

        db_path = output_dir / "wav" / "_state" / "audio_chunks.sqlite3"
        store = AudioChunkStore(db_path)
        store.save_sentence_version(sentence_hash, sentence_text)

        result = asyncio.run(
            delete_audio(
                DeleteAudioRequest(
                    dir=str(output_dir),
                    chapter_key=chapter_key,
                    hash=sentence_hash,
                    text=sentence_text,
                )
            )
        )

        assert result["status"] == "ok"
        assert not audio_path.exists()
        assert result["disputed"]["hash"] == sentence_hash
        assert result["disputed"]["transcription"] == "[manual] audio deleted"
        assert result["disputed"]["similarity"] == 0.0
        assert result["disputed"]["status"] == "disputed"

        rows = store.get_disputed_chunks(chapter_key)
        assert len(rows) == 1
        assert rows[0]["sentence_hash"] == sentence_hash
        assert rows[0]["original_text"] == sentence_text
        assert rows[0]["raw_transcription"] is None
        assert rows[0]["status"] == "disputed"

