from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from audiobook_generator.normalizers.base_normalizer import ChainNormalizer
from audiobook_generator.utils.change_report import (
    build_change_blocks,
    build_unified_diff,
    render_change_report,
)
from audiobook_generator.core.progress_store import NormalizationProgressStore

logger = logging.getLogger(__name__)


class NormalizationPipelineRunner:
    def __init__(self, *, config, artifact_dir: str | Path):
        self.config = config
        self.artifact_dir = Path(artifact_dir)
        self.steps_root = self.artifact_dir / "_normalizer_steps"
        self.steps_root.mkdir(parents=True, exist_ok=True)
        # Use per-run state path if set by AudiobookGenerator, else fall back to legacy location.
        _state_path = getattr(config, 'normalization_state_path', None)
        if not _state_path:
            _state_path = Path(self.config.output_folder) / "_state" / "normalization_progress.sqlite3"
        self.store = NormalizationProgressStore(Path(_state_path))
        self.chapter_key = self.artifact_dir.name
        self.step_summaries: list[dict[str, object]] = []

    def run(self, normalizer, text: str, chapter_title: str) -> tuple[str, list[tuple[str, str]]]:
        if not normalizer:
            return text, []

        steps = self._resolve_steps(normalizer)
        total_steps = len(steps)
        logger.info(
            "Normalization pipeline for '%s': %d steps, %d chars",
            chapter_title, total_steps, len(text),
        )
        current_text = text
        trace: list[tuple[str, str]] = []

        for step_index, (step_name, step_normalizer) in enumerate(steps, start=1):
            current_text = self._run_step(
                step_normalizer=step_normalizer,
                step_name=step_name,
                step_index=step_index,
                total_steps=total_steps,
                input_text=current_text,
                chapter_title=chapter_title,
            )
            trace.append((step_name, current_text))

        self._write_pipeline_summary()
        return current_text, trace

    def _resolve_steps(self, normalizer):
        if isinstance(normalizer, ChainNormalizer):
            return normalizer.iter_steps()
        return [(normalizer.get_step_name(), normalizer)]

    def _run_step(self, *, step_normalizer, step_name, step_index, total_steps=None, input_text, chapter_title):
        step_dir = self.steps_root / f"{step_index:02d}_{self._sanitize_name(step_name)}"
        step_dir.mkdir(parents=True, exist_ok=True)

        input_hash = self._hash_text(input_text)
        config_hash = self._hash_json(step_normalizer.get_resume_signature())
        output_path = step_dir / "output.txt"

        self._write_text_if_missing(step_dir / "input.txt", input_text)
        self._write_named_artifacts(
            step_dir,
            step_normalizer.get_step_artifacts(input_text, chapter_title=chapter_title),
        )

        existing_step = self.store.get_step_record(
            chapter_key=self.chapter_key,
            step_index=step_index,
            input_hash=input_hash,
            config_hash=config_hash,
        )
        if (
            existing_step
            and existing_step["status"] == "success"
            and output_path.is_file()
        ):
            logger.info(
                "  [%d/%s] %s — skipped (cached)",
                step_index, total_steps or '?', step_name,
            )
            cached_output = output_path.read_text(encoding="utf-8")
            self.step_summaries.append(
                {
                    "step_index": step_index,
                    "step_name": step_name,
                    "changed": input_text != cached_output,
                    "change_blocks": len(build_change_blocks(input_text, cached_output)) if step_normalizer.should_log_changes() else None,
                    "step_dir": str(step_dir),
                }
            )
            return cached_output

        self.store.upsert_step(
            chapter_key=self.chapter_key,
            step_index=step_index,
            step_name=step_name,
            input_hash=input_hash,
            config_hash=config_hash,
            status="running",
            output_path=str(output_path),
        )

        logger.info(
            "  [%d/%s] %s — running...",
            step_index, total_steps or '?', step_name,
        )

        try:
            if step_normalizer.supports_chunked_resume():
                normalized = self._run_chunked_step(
                    step_normalizer=step_normalizer,
                    step_name=step_name,
                    step_index=step_index,
                    input_text=input_text,
                    input_hash=input_hash,
                    config_hash=config_hash,
                    step_dir=step_dir,
                    chapter_title=chapter_title,
                )
            else:
                normalized = step_normalizer.normalize(input_text, chapter_title=chapter_title)

            output_path.write_text(normalized, encoding="utf-8", newline="\n")
            # Only create change artifacts if logging is enabled for this normalizer
            if step_normalizer.should_log_changes():
                self._write_named_artifacts(
                    step_dir,
                    self._build_change_artifacts(
                        input_text=input_text,
                        output_text=normalized,
                        title=f"{step_name} changes",
                    ),
                )
            self._write_named_artifacts(
                step_dir,
                step_normalizer.get_post_step_artifacts(
                    input_text=input_text,
                    output_text=normalized,
                    chapter_title=chapter_title,
                ),
            )
            self.step_summaries.append(
                {
                    "step_index": step_index,
                    "step_name": step_name,
                    "changed": input_text != normalized,
                    "change_blocks": len(build_change_blocks(input_text, normalized)) if step_normalizer.should_log_changes() else None,
                    "step_dir": str(step_dir),
                }
            )
            self.store.upsert_step(
                chapter_key=self.chapter_key,
                step_index=step_index,
                step_name=step_name,
                input_hash=input_hash,
                config_hash=config_hash,
                status="success",
                output_path=str(output_path),
            )
            return normalized
        except Exception as exc:
            self.store.upsert_step(
                chapter_key=self.chapter_key,
                step_index=step_index,
                step_name=step_name,
                input_hash=input_hash,
                config_hash=config_hash,
                status="failed",
                output_path=str(output_path),
                error_message=str(exc),
            )
            raise

    def _run_chunked_step(
        self,
        *,
        step_normalizer,
        step_name,
        step_index,
        input_text,
        input_hash,
        config_hash,
        step_dir,
        chapter_title,
    ):
        units = step_normalizer.plan_processing_units(input_text, chapter_title=chapter_title)
        chunks_dir = step_dir / "chunks"
        chunks_dir.mkdir(parents=True, exist_ok=True)
        results: list[str] = []

        for unit_index, unit in enumerate(units, start=1):
            unit_dir = chunks_dir / f"{unit_index:04d}"
            unit_dir.mkdir(parents=True, exist_ok=True)
            unit_output_path = unit_dir / "output.txt"
            self._write_text_if_missing(unit_dir / "input.txt", unit)
            self._write_named_artifacts(
                unit_dir,
                step_normalizer.get_unit_artifacts(
                    unit,
                    chapter_title=chapter_title,
                    unit_index=unit_index,
                    unit_count=len(units),
                ),
            )

            existing_unit = self.store.get_unit_record(
                chapter_key=self.chapter_key,
                step_index=step_index,
                unit_index=unit_index,
                input_hash=input_hash,
                config_hash=config_hash,
            )
            if (
                existing_unit
                and existing_unit["status"] == "success"
                and unit_output_path.is_file()
            ):
                logger.info(
                    "Resuming chapter '%s': reusing %s chunk %s/%s",
                    chapter_title,
                    step_name,
                    unit_index,
                    len(units),
                )
                results.append(unit_output_path.read_text(encoding="utf-8"))
                continue

            self.store.upsert_unit(
                chapter_key=self.chapter_key,
                step_index=step_index,
                unit_index=unit_index,
                step_name=step_name,
                input_hash=input_hash,
                config_hash=config_hash,
                status="running",
                output_path=str(unit_output_path),
            )

            try:
                unit_result = step_normalizer.process_unit(
                    unit,
                    chapter_title=chapter_title,
                    unit_index=unit_index,
                    unit_count=len(units),
                )
                unit_output_path.write_text(unit_result, encoding="utf-8", newline="\n")
                # Only create change artifacts if logging is enabled for this normalizer
                if step_normalizer.should_log_changes():
                    self._write_named_artifacts(
                        unit_dir,
                        self._build_change_artifacts(
                            input_text=unit,
                            output_text=unit_result,
                            title=f"{step_name} chunk {unit_index} changes",
                        ),
                    )
                self.store.upsert_unit(
                    chapter_key=self.chapter_key,
                    step_index=step_index,
                    unit_index=unit_index,
                    step_name=step_name,
                    input_hash=input_hash,
                    config_hash=config_hash,
                    status="success",
                    output_path=str(unit_output_path),
                )
                results.append(unit_result)
            except Exception as exc:
                self.store.upsert_unit(
                    chapter_key=self.chapter_key,
                    step_index=step_index,
                    unit_index=unit_index,
                    step_name=step_name,
                    input_hash=input_hash,
                    config_hash=config_hash,
                    status="failed",
                    output_path=str(unit_output_path),
                    error_message=str(exc),
                )
                raise

        merged = step_normalizer.merge_processed_units(results, chapter_title=chapter_title)
        return merged

    @staticmethod
    def _sanitize_name(name: str) -> str:
        return name.replace(" ", "_").replace("/", "_")

    @staticmethod
    def _hash_text(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def _hash_json(data) -> str:
        return hashlib.sha256(
            json.dumps(data, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _write_text_if_missing(path: Path, text: str):
        if path.exists():
            return
        path.write_text(text, encoding="utf-8", newline="\n")

    @staticmethod
    def _write_named_artifacts(base_dir: Path, artifacts: dict[str, str]):
        if not artifacts:
            return
        for relative_name, content in artifacts.items():
            path = base_dir / relative_name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8", newline="\n")

    @staticmethod
    def _build_change_artifacts(*, input_text: str, output_text: str, title: str) -> dict[str, str]:
        return {
            "90_changes.md": render_change_report(input_text, output_text, title=title),
            "91_changes.diff": build_unified_diff(
                input_text,
                output_text,
                fromfile="input.txt",
                tofile="output.txt",
            ),
        }

    def _write_pipeline_summary(self):
        if not self.step_summaries:
            return

        lines = ["# Normalizer Change Summary", ""]
        for item in self.step_summaries:
            lines.append(f"## {item['step_index']:02d}. {item['step_name']}")
            lines.append("")
            lines.append(f"- changed: {'yes' if item['changed'] else 'no'}")
            cb = item['change_blocks']
            lines.append(f"- change_blocks: {cb if cb is not None else 'n/a (logging disabled)'}")
            lines.append(f"- step_dir: {item['step_dir']}")
            lines.append("")

        summary_path = self.artifact_dir / "00_normalizer_change_summary.md"
        summary_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
