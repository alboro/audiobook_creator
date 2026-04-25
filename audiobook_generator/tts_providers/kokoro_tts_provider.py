# Based on https://github.com/kroryan/epub_to_audiobook
# Adapted for this fork by alboro <alboro@users.noreply.github.com>

"""
Kokoro TTS Provider
Supports Kokoro-FastAPI (https://github.com/remsky/Kokoro-FastAPI) with voice mixing.

Adapted from kroryan/epub_to_audiobook fork.
"""

from __future__ import annotations

import io
import logging
import math
import re
from typing import Dict, List, Optional, Tuple, Union

import requests
from pydub import AudioSegment

from audiobook_generator.config.general_config import GeneralConfig
from audiobook_generator.core.audio_tags import AudioTags
from audiobook_generator.tts_providers.base_tts_provider import BaseTTSProvider
from audiobook_generator.utils.utils import merge_audio_segments, set_audio_tags, split_text

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://localhost:8880"
DEFAULT_MODEL = "kokoro"
DEFAULT_VOICE = "af_heart"
DEFAULT_OUTPUT_FORMAT = "mp3"
DEFAULT_SPEED = 1.0
DEFAULT_VOLUME_MULTIPLIER = 1.0
DEFAULT_MAX_CHARS = 2000
USD_PER_1000_CHAR = 0.0  # Free (self-hosted)


def get_kokoro_supported_output_formats() -> List[str]:
    return ["mp3", "opus", "aac", "flac", "wav", "pcm"]


def get_kokoro_supported_voices() -> List[str]:
    """Default voice list; live list is fetched from Kokoro server at runtime."""
    return [
        # American English
        "af_bella", "af_sky", "af_heart", "af_nicole", "af_sarah", "af_emma",
        "am_adam", "am_daniel", "am_michael", "am_liam",
        # British English
        "bf_emma", "bf_sarah", "bf_nicole", "bf_sky",
        "bm_lewis", "bm_george", "bm_william", "bm_james",
        # Spanish
        "ef_dora", "ef_sarah", "ef_maria", "ef_isabella",
        "em_alex", "em_carlos", "em_diego", "em_miguel",
    ]


def get_kokoro_supported_models() -> List[str]:
    return ["kokoro", "tts-1", "tts-1-hd"]


def get_kokoro_supported_languages() -> Dict[str, str]:
    return {
        "": "Auto-detect",
        "a": "English (US)",
        "b": "British English",
        "e": "Spanish",
        "f": "French",
        "h": "Hindi",
        "i": "Italian",
        "p": "Portuguese",
        "j": "Japanese",
        "z": "Chinese",
    }


class KokoroTTSProvider(BaseTTSProvider):
    def __init__(self, config: GeneralConfig):
        config.model_name = config.model_name or DEFAULT_MODEL
        config.voice_name = config.voice_name or DEFAULT_VOICE
        config.output_format = (config.output_format or DEFAULT_OUTPUT_FORMAT).lower()
        config.speed = config.speed or DEFAULT_SPEED

        self.base_url = (getattr(config, "kokoro_base_url", None) or DEFAULT_BASE_URL).rstrip("/")
        self.volume_multiplier = float(getattr(config, "kokoro_volume_multiplier", None) or DEFAULT_VOLUME_MULTIPLIER)
        self.price = USD_PER_1000_CHAR
        self._max_chars = DEFAULT_MAX_CHARS
        self._headers = {
            "Authorization": "Bearer fake-key",
            "Content-Type": "application/json",
        }

        super().__init__(config)

    def __str__(self) -> str:
        return (
            f"KokoroTTSProvider(model={self.config.model_name}, voice={self.config.voice_name}, "
            f"output_format={self.config.output_format}, speed={self.config.speed}, base_url={self.base_url})"
        )

    def validate_config(self):
        if self.config.output_format not in get_kokoro_supported_output_formats():
            raise ValueError(
                f"Kokoro: Unsupported output format '{self.config.output_format}'. "
                f"Supported: {get_kokoro_supported_output_formats()}"
            )
        speed = float(self.config.speed)
        if speed < 0.25 or speed > 4.0:
            raise ValueError(f"Kokoro: speed must be between 0.25 and 4.0, got {speed}")

    def text_to_speech(self, text: str, output_file: str, audio_tags: AudioTags):
        if not text.strip():
            logger.warning("KokoroTTS: Empty text received; skipping chunk synthesis.")
            return

        chunks = split_text(text, self._max_chars, self.config.language)
        audio_segments: List[io.BytesIO] = []
        chunk_ids: List[str] = []

        for index, chunk in enumerate(chunks, 1):
            chunk_id = f"chapter-{audio_tags.idx}_{audio_tags.title}_chunk_{index}_of_{len(chunks)}"
            logger.info("KokoroTTS: Processing %s (length=%s)", chunk_id, len(chunk))

            audio_bytes = self._synthesize(chunk)
            audio_segments.append(io.BytesIO(audio_bytes))
            chunk_ids.append(chunk_id)

        merge_audio_segments(
            audio_segments,
            output_file,
            self.get_output_file_extension(),
            chunk_ids,
            True,
        )
        set_audio_tags(output_file, audio_tags)

    def estimate_cost(self, total_chars: int) -> float:
        return 0.0  # Kokoro is self-hosted and free

    def get_break_string(self) -> str:
        return "   "  # Kokoro uses spaces as breaks

    def get_output_file_extension(self) -> str:
        return self.config.output_format

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _synthesize(self, text: str) -> bytes:
        """Send one chunk to Kokoro API and return raw audio bytes."""
        url = f"{self.base_url}/v1/audio/speech"
        payload: Dict = {
            "model": self.config.model_name,
            "voice": self.config.voice_name,
            "input": text.replace(self.get_break_string(), "\n\n").strip(),
            "speed": float(self.config.speed),
            "response_format": self.config.output_format,
            "stream": False,
            "volume_multiplier": self.volume_multiplier,
        }
        # Optional lang_code
        lang = getattr(self.config, "language", "") or ""
        if lang:
            payload["lang_code"] = lang[:1]  # Kokoro uses single-char codes

        resp = requests.post(url, json=payload, headers=self._headers, timeout=120)
        if resp.status_code != 200:
            raise RuntimeError(
                f"KokoroTTS: API error {resp.status_code} — {resp.text[:200]}"
            )
        return resp.content

    def fetch_voices(self) -> List[str]:
        """Fetch the live voice list from the Kokoro server."""
        try:
            url = f"{self.base_url}/v1/audio/voices"
            resp = requests.get(url, timeout=10)
            if resp.status_code != 200:
                return get_kokoro_supported_voices()
            data = resp.json()
            voices = data.get("voices", data) if isinstance(data, dict) else data
            result = []
            for v in voices:
                if isinstance(v, dict):
                    result.append(v.get("id") or v.get("name") or str(v))
                elif isinstance(v, str):
                    result.append(v)
            return result or get_kokoro_supported_voices()
        except Exception as exc:
            logger.warning("KokoroTTS: Could not fetch voices from server: %s", exc)
            return get_kokoro_supported_voices()

