# Copyright (c) 2026 alboro <alboro@users.noreply.github.com>
# Licensed under the PolyForm Noncommercial License 1.0.0.

"""ReferenceChecker — external CLI-based audio quality check."""

from __future__ import annotations

import json
import logging
import os
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from audiobook_generator.core.audio_checkers.base_audio_chunk_checker import (
    BaseAudioChunkChecker,
    CheckResult,
)

logger = logging.getLogger(__name__)


class ReferenceChecker(BaseAudioChunkChecker):
    """Run an external reference-audio checker and mark the chunk suspicious
    when the returned score exceeds ``audio_reference_check_threshold``.

    The checker is *skipped silently* when ``audio_reference_check_command``
    is not set in config — inclusion in ``audio_check_checkers`` alone is not
    enough to activate it.

    Config keys:
        audio_reference_check_command   – path / command to the external tool
                                          (required for this checker to do anything).
        audio_reference_check_threshold – score threshold; ``None`` = only measure.
        audio_reference_check_timeout   – subprocess timeout in seconds (default 120).
        audio_reference_check_cache_dir – cache directory for the external tool.
        audio_reference_check_stress    – stress handling mode (default "preserve").
    """

    name = "reference"

    # Pass/fail is derived from the existing ``reference_check_status`` /
    # ``reference_check_score`` columns — no fallback column needed.
    uses_fallback_passed_column = False

    @classmethod
    def evaluate_from_row(cls, row: dict, config) -> Optional[bool]:
        status = row.get("reference_check_status")
        if status is None:
            return None
        if status == "ok":
            return True
        if status == "suspicious":
            return False
        if status == "measured":
            # Threshold was None when the check ran; re-evaluate with the
            # threshold currently in config (if any).
            score = row.get("reference_check_score")
            if score is None:
                return None
            try:
                threshold = float(
                    getattr(config, "audio_reference_check_threshold", None) or 0
                )
            except (TypeError, ValueError):
                return None
            return float(score) <= threshold if threshold > 0 else None
        # "error" or unknown — cannot determine pass/fail
        return None

    @classmethod
    def score_from_row(cls, row: dict, config) -> Optional[float]:
        score = row.get("reference_check_score")
        return float(score) if score is not None else None

    def __init__(self, config):
        super().__init__(config)
        self._command = getattr(config, "audio_reference_check_command", None) or ""
        self._threshold = self._coerce_float(
            getattr(config, "audio_reference_check_threshold", None)
        )
        self._timeout = self._coerce_timeout(
            getattr(config, "audio_reference_check_timeout", None)
        )
        self._cache_dir = getattr(config, "audio_reference_check_cache_dir", None)
        self._stress = getattr(config, "audio_reference_check_stress", None) or "preserve"
        self._ffmpeg_path = getattr(config, "ffmpeg_path", None)
        self._output_folder = Path(getattr(config, "output_folder", None) or ".")
        self._language = (getattr(config, "language", None) or "ru").split("-")[0]

    # ------------------------------------------------------------------
    @staticmethod
    def _coerce_float(value) -> Optional[float]:
        if value in (None, ""):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _coerce_timeout(value) -> int:
        if value in (None, ""):
            return 120
        try:
            return max(1, int(value))
        except (TypeError, ValueError):
            return 120

    def _command_parts(self) -> list[str]:
        command = self._command.strip()
        if not command:
            return []
        command_path = command.strip("\"'")
        if os.path.exists(command_path):
            return [command_path]
        return shlex.split(command, posix=(os.name != "nt"))

    # ------------------------------------------------------------------

    def check(
        self,
        audio_file: Path,
        original_text: str,
        transcription: str,
        chunk_cache_row: Optional[dict],
    ) -> CheckResult:
        if not self._command:
            # Checker is in the list but command is not configured — skip quietly.
            return CheckResult(disputed=False, reference_check_threshold=self._threshold)

        text_file = None
        try:
            parts = self._command_parts()
            if not parts:
                raise RuntimeError("audio_reference_check_command is empty")

            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", suffix=".txt", delete=False
            ) as tmp:
                tmp.write(original_text)
                text_file = tmp.name

            cache_dir = str(self._cache_dir) if self._cache_dir else str(
                self._output_folder / "wav" / "_reference_cache"
            )

            cmd = [
                *parts,
                "reference-compare",
                "--audio", str(audio_file),
                "--text-file", text_file,
                "--language", self._language,
                "--json",
                "--reference-stress", str(self._stress),
                "--cache-dir", cache_dir,
            ]
            if self._ffmpeg_path:
                cmd.extend(["--ffmpeg-path", str(self._ffmpeg_path)])

            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self._timeout,
                check=False,
            )
            stdout = proc.stdout.strip()
            if proc.returncode != 0:
                raise RuntimeError(
                    f"reference checker exited {proc.returncode}: "
                    f"{proc.stderr.strip() or stdout}"
                )
            if not stdout:
                raise RuntimeError("reference checker returned empty stdout")

            payload = json.loads(stdout.splitlines()[-1])
            score = payload.get("score")
            if score is None:
                raise RuntimeError("reference checker JSON has no 'score' field")

            score_f = float(score)
            if self._threshold is None:
                status = "measured"
                disputed = False
            else:
                disputed = score_f > self._threshold
                status = "suspicious" if disputed else "ok"

            return CheckResult(
                disputed=disputed,
                reference_check_score=score_f,
                reference_check_threshold=self._threshold,
                reference_check_status=status,
                reference_check_payload=payload,
            )

        except Exception as exc:
            logger.warning("reference-check error for %s: %s", audio_file.name, exc)
            return CheckResult(
                disputed=False,
                reference_check_score=None,
                reference_check_threshold=self._threshold,
                reference_check_status="error",
                reference_check_payload={"error": str(exc)},
            )
        finally:
            if text_file:
                try:
                    Path(text_file).unlink(missing_ok=True)
                except Exception:
                    pass

