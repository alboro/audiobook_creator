# Copyright (c) 2026 alboro <alboro@users.noreply.github.com>
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See: https://polyformproject.org/licenses/noncommercial/1.0.0/

"""INI config manager for epub_to_audiobook.

Provides read/write support for INI-format configuration files using Python's
built-in configparser.  All setting definitions live here so they are not
duplicated across main.py, general_config.py, and other layers.

Merge priority when loading: CLI args > INI file > built-in defaults.
"""

from __future__ import annotations

import configparser
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Canonical section assignments for every config field.
# This is the single source of truth for which section a setting belongs to.
# ---------------------------------------------------------------------------
FIELD_SECTIONS: Dict[str, str] = {
    # [general] ----------------------------------------------------------------
    "mode": "general",
    "language": "general",
    "log": "general",
    "no_prompt": "general",
    "worker_count": "general",
    "use_pydub_merge": "general",
    "chapter_start": "general",
    "chapter_end": "general",
    "newline_mode": "general",
    "title_mode": "general",
    "chapter_mode": "general",
    "search_and_replace_file": "general",
    "output_text": "general",
    "prepared_text_folder": "general",
    "force_new_run": "general",
    # [tts] --------------------------------------------------------------------
    "tts": "tts",
    "voice_name": "tts",
    "voice_name2": "tts",
    "model_name": "tts",
    "output_format": "tts",
    "speed": "tts",
    "instructions": "tts",
    "tts_trailing_strip_chars": "tts",
    "tts_trim_silence": "tts",
    "tts_chunk_smooth_join": "tts",
    "tts_chunk_smooth_join_ms": "tts",
    "tts_chunk_dc_remove": "tts",
    "tts_chunk_merge_gap_ms": "tts",
    "tts_chunk_declick_start": "tts",
    "tts_chunk_declick_start_ms": "tts",
    "tts_chunk_declick_fade_ms": "tts",
    "tts_log_text": "tts",
    # [tts.openai] -------------------------------------------------------------
    "openai_api_key": "tts.openai",
    "openai_base_url": "tts.openai",
    "openai_max_chars": "tts.openai",
    "openai_enable_polling": "tts.openai",
    "openai_submit_url": "tts.openai",
    "openai_status_url_template": "tts.openai",
    "openai_download_url_template": "tts.openai",
    "openai_job_id_path": "tts.openai",
    "openai_job_status_path": "tts.openai",
    "openai_job_download_url_path": "tts.openai",
    "openai_job_done_values": "tts.openai",
    "openai_job_failed_values": "tts.openai",
    "openai_poll_interval": "tts.openai",
    "openai_poll_timeout": "tts.openai",
    "openai_poll_request_timeout": "tts.openai",
    "openai_poll_max_errors": "tts.openai",
    "openai_submit_omit_fields": "tts.openai",
    "openai_submit_extra_fields": "tts.openai",
    # [tts.azure_edge] ---------------------------------------------------------
    "break_duration": "tts.azure_edge",
    # [tts.edge] ---------------------------------------------------------------
    "voice_rate": "tts.edge",
    "voice_volume": "tts.edge",
    "voice_pitch": "tts.edge",
    "proxy": "tts.edge",
    # [tts.piper] --------------------------------------------------------------
    "piper_path": "tts.piper",
    "piper_docker_image": "tts.piper",
    "piper_speaker": "tts.piper",
    "piper_noise_scale": "tts.piper",
    "piper_noise_w_scale": "tts.piper",
    "piper_length_scale": "tts.piper",
    "piper_sentence_silence": "tts.piper",
    # [tts.qwen] ---------------------------------------------------------------
    "qwen_api_key": "tts.qwen",
    "qwen_language_type": "tts.qwen",
    "qwen_stream": "tts.qwen",
    "qwen_request_timeout": "tts.qwen",
    # [tts.gemini] -------------------------------------------------------------
    "gemini_api_key": "tts.gemini",
    "gemini_sample_rate": "tts.gemini",
    "gemini_channels": "tts.gemini",
    "gemini_audio_encoding": "tts.gemini",
    "gemini_temperature": "tts.gemini",
    "gemini_speaker_map": "tts.gemini",
    # [tts.kokoro] -------------------------------------------------------------
    "kokoro_base_url": "tts.kokoro",
    "kokoro_volume_multiplier": "tts.kokoro",
    # [normalize] --------------------------------------------------------------
    "normalize": "normalize",
    "normalize_steps": "normalize",
    "normalize_provider": "normalize",
    "normalize_model": "normalize",
    "normalize_api_key": "normalize",
    "normalize_base_url": "normalize",
    "normalize_model": "normalize",
    "normalize_system_prompt": "normalize",
    "normalize_system_prompt_file": "normalize",
    "normalize_user_prompt_file": "normalize",
    "normalize_tts_safe_max_chars": "normalize",
    "normalize_tts_safe_comma_as_period": "normalize",
    "normalize_tts_pronunciation_overrides_file": "normalize",
    "normalize_tts_pronunciation_overrides_words": "normalize",
    "normalize_pronunciation_lexicon_db": "normalize",
    "normalize_stress_ambiguity_file": "normalize",
    "normalize_stress_ambiguity_system_prompt": "normalize",
    "normalize_safe_split_system_prompt": "normalize",
    "normalize_reasoning_effort": "normalize",
    "normalize_tsnorm_stress_yo": "normalize",
    "normalize_tsnorm_stress_monosyllabic": "normalize",
    "normalize_tsnorm_min_word_length": "normalize",
    "normalize_stress_paradox_words": "normalize",
    "normalize_log_changes": "normalize",
    # [audio_check] ------------------------------------------------------------
    "audio_check_model": "audio_check",
    "audio_check_threshold": "audio_check",
    "audio_check_device": "audio_check",
    "audio_auto_check_threshold": "audio_check",
    "audio_auto_retry": "audio_check",
    "audio_reference_check_command": "audio_check",
    "audio_reference_check_threshold": "audio_check",
    "audio_reference_check_timeout": "audio_check",
    "audio_reference_check_cache_dir": "audio_check",
    "audio_reference_check_stress": "audio_check",
    # [m4b] --------------------------------------------------------------------
    "package_m4b": "m4b",
    "chunked_audio": "m4b",
    "chunked_audio_no_db": "m4b",
    "audio_worker_interval": "m4b",
    "audio_folder": "m4b",
    "m4b_filename": "m4b",
    "m4b_bitrate": "m4b",
    "chapter_titles_file": "m4b",
    "cover_image": "m4b",
    "ffmpeg_path": "m4b",
}

# Fields that are boolean flags (no value, just presence).
BOOL_FIELDS = {
    "no_prompt", "use_pydub_merge", "output_text", "force_new_run",
    "normalize", "normalize_log_changes",
    "openai_enable_polling",
    "normalize_tsnorm_stress_yo", "normalize_tsnorm_stress_monosyllabic",
    "qwen_stream", "package_m4b", "chunked_audio", "chunked_audio_no_db",
    "tts_log_text", "tts_chunk_smooth_join", "tts_chunk_dc_remove",
    "tts_chunk_declick_start",
}


def load_ini(path: str | Path) -> Dict[str, Any]:
    """Parse an INI file and return a flat {field_name: value} dict.

    Section names are stripped; field names must be unique across sections
    (which they are, by design in FIELD_SECTIONS).
    """
    cp = configparser.ConfigParser(interpolation=None)
    cp.read(str(path), encoding="utf-8")
    result: Dict[str, Any] = {}
    for section in cp.sections():
        for key, raw_value in cp.items(section):
            result[key] = raw_value
    return result


def _project_root() -> Path:
    """Return the project root directory (two levels up from this file)."""
    return Path(__file__).resolve().parents[2]


def discover_ini_files(input_file: Optional[str] = None, explicit_config: Optional[str] = None) -> list:
    """Return ordered list of INI file paths to load (lowest to highest priority).

    Priority order (later overrides earlier):
      1. Global user config:   ~/.config/epub_to_audiobook/config.ini
      2. Project-local config: <project_root>/config.local.ini
      3. Per-book config:      <book_dir>/<book_stem>.ini  (or audiobook.ini as fallback)
      4. Explicit --config:    the path passed on CLI

    CLI args always win over all INI files (handled by merge_ini_into_args).
    """
    files = []

    # 1. Global user config
    global_cfg = Path.home() / ".config" / "epub_to_audiobook" / "config.ini"
    if global_cfg.is_file():
        files.append(global_cfg)
        logger.info("Auto-loaded global config: %s", global_cfg)

    # 2. Project-local config (next to main.py, gitignored)
    project_local = _project_root() / "config.local.ini"
    if project_local.is_file():
        files.append(project_local)
        logger.info("Auto-loaded project-local config: %s", project_local)

    # 2. Per-book config (next to the input file)
    if input_file:
        book_path = Path(input_file).expanduser().resolve()
        book_dir = book_path.parent
        candidates = [
            book_dir / (book_path.stem + ".ini"),
            book_dir / "audiobook.ini",
        ]
        for candidate in candidates:
            if candidate.is_file():
                files.append(candidate)
                logger.info("Auto-loaded per-book config: %s", candidate)
                break  # only the first match

    # 3. Explicit --config
    if explicit_config:
        cfg_path = Path(explicit_config)
        if cfg_path.is_dir() and input_file:
            cfg_path = cfg_path / (Path(input_file).stem + ".ini")
        if cfg_path.is_file():
            files.append(cfg_path)
        else:
            import sys
            print(f"ERROR: Config file not found: {cfg_path}", file=sys.stderr)
            sys.exit(1)

    return files


def load_merged_ini(input_file: Optional[str] = None, explicit_config: Optional[str] = None) -> Dict[str, Any]:
    """Load and merge all discovered INI files into one flat dict.

    Later files override earlier ones.
    """
    merged: Dict[str, Any] = {}
    for path in discover_ini_files(input_file, explicit_config):
        merged.update(load_ini(path))
    return merged


def save_ini(path: str | Path, config, *, include_paths: bool = True) -> None:
    """Write a GeneralConfig snapshot to an INI file.

    Only non-None, non-False fields are written to keep the file readable.
    Sensitive fields like API keys are included but the file is local/private.
    """
    cp = configparser.ConfigParser(interpolation=None)

    # Always write resolved paths at the top.
    if include_paths:
        cp.add_section("paths")
        if getattr(config, "input_file", None):
            cp.set("paths", "input_file", str(config.input_file))
        if getattr(config, "output_folder", None):
            cp.set("paths", "output_folder", str(config.output_folder))

    for field, section in FIELD_SECTIONS.items():
        value = getattr(config, field, None)
        if value is None:
            continue
        if value is False and field in BOOL_FIELDS:
            continue  # skip False flags to reduce noise
        if not cp.has_section(section):
            cp.add_section(section)
        cp.set(section, field, str(value))

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(str(path), "w", encoding="utf-8") as fh:
        cp.write(fh)
    logger.debug("Config snapshot written to %s", path)


def merge_ini_into_args(args: Any, ini_values: Dict[str, Any]) -> None:
    """Merge INI-loaded values into an argparse Namespace (in-place).

    CLI-provided values take priority: a field is only set from INI if the
    current argparse value is None (i.e. the user did not pass it on the CLI).
    """
    for key, raw_value in ini_values.items():
        current = getattr(args, key, None)
        if current is not None:
            # For bool flags (action="store_true"), argparse defaults to False when
            # the flag is absent — that is indistinguishable from the user not passing
            # it.  We should still let the INI value win in that case.
            # Only skip if the flag was explicitly set to True by the user.
            if key in BOOL_FIELDS:
                if current is True:
                    continue  # user explicitly passed --flag, CLI wins
                # current is False → may be argparse default, let INI override
            else:
                continue  # CLI wins for non-bool fields
        if isinstance(raw_value, str):
            if key in BOOL_FIELDS:
                # Boolean fields: coerce "true"/"1"/"yes" → True, "false"/"0"/"no" → False
                if raw_value.lower() in ("true", "yes", "1"):
                    setattr(args, key, True)
                else:
                    setattr(args, key, False)
            else:
                # Non-boolean fields: try numeric coercion, then keep as string
                stripped = raw_value.strip()
                try:
                    coerced: Any = int(stripped)
                except ValueError:
                    try:
                        coerced = float(stripped)
                    except ValueError:
                        coerced = raw_value
                setattr(args, key, coerced)
        else:
            setattr(args, key, raw_value)
