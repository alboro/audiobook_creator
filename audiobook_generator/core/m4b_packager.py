# Copyright (c) 2026 alboro <alboro@users.noreply.github.com>
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See: https://polyformproject.org/licenses/noncommercial/1.0.0/

import logging
import mimetypes
import os
import subprocess
import tempfile

from pydub import AudioSegment

from audiobook_generator.utils.filename_sanitizer import make_safe_filename

logger = logging.getLogger(__name__)


def _escape_ffmetadata(value: str) -> str:
    escaped = value.replace("\\", "\\\\")
    escaped = escaped.replace(";", "\\;").replace("#", "\\#").replace("=", "\\=")
    return escaped.replace("\n", " ")


def _cover_suffix(media_type: str) -> str:
    guessed = mimetypes.guess_extension(media_type or "")
    return guessed or ".jpg"


def package_m4b(
    chapter_files,
    chapter_titles,
    book_title,
    book_author,
    output_dir,
    ffmpeg_path="ffmpeg",
    output_filename=None,
    bitrate="64k",
    cover=None,
):
    if not chapter_files:
        raise ValueError("No chapter files available for m4b packaging")

    output_name = output_filename
    if output_name:
        if not output_name.endswith(".m4b"):
            output_name += ".m4b"
    else:
        output_name = make_safe_filename(
            title=book_title,
            idx=None,
            output_dir=output_dir,
            ext=".m4b",
            collision_check=False,
        )

    output_path = os.path.join(output_dir, output_name)

    with tempfile.TemporaryDirectory(prefix="eta_m4b_") as temp_dir:
        concat_path = os.path.join(temp_dir, "chapters.txt")
        metadata_path = os.path.join(temp_dir, "metadata.txt")
        cover_path = None

        with open(concat_path, "w", encoding="utf-8") as concat_file:
            for chapter_file in chapter_files:
                chapter_path = os.path.abspath(chapter_file).replace("\\", "/")
                concat_file.write(f"file '{chapter_path}'\n")

        start_time = 0
        metadata_lines = [
            ";FFMETADATA1",
            f"title={_escape_ffmetadata(book_title)}",
            f"artist={_escape_ffmetadata(book_author)}",
            f"album={_escape_ffmetadata(book_title)}",
        ]

        for chapter_title, chapter_file in zip(chapter_titles, chapter_files):
            duration_ms = len(AudioSegment.from_file(chapter_file))
            chapter_end = start_time + duration_ms
            metadata_lines.extend(
                [
                    "[CHAPTER]",
                    "TIMEBASE=1/1000",
                    f"START={start_time}",
                    f"END={chapter_end}",
                    f"title={_escape_ffmetadata(chapter_title.replace('_', ' '))}",
                ]
            )
            start_time = chapter_end

        with open(metadata_path, "w", encoding="utf-8") as metadata_file:
            metadata_file.write("\n".join(metadata_lines) + "\n")

        command = [
            ffmpeg_path,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            concat_path,
            "-i",
            metadata_path,
            "-map",
            "0:a",
            "-map_metadata",
            "1",
            "-c:a",
            "aac",
            "-b:a",
            bitrate,
            "-movflags",
            "+faststart",
        ]

        if cover:
            cover_bytes, media_type = cover
            cover_path = os.path.join(temp_dir, f"cover{_cover_suffix(media_type)}")
            with open(cover_path, "wb") as cover_file:
                cover_file.write(cover_bytes)
            command.extend(
                [
                    "-i",
                    cover_path,
                    "-map",
                    "2:v",
                    "-c:v",
                    "copy",
                    "-disposition:v:0",
                    "attached_pic",
                ]
            )

        command.append(output_path)

        logger.info("Packaging audiobook to m4b: %s", output_path)
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "ffmpeg failed to package m4b: "
                f"{completed.stderr.strip() or completed.stdout.strip()}"
            )

    return output_path
