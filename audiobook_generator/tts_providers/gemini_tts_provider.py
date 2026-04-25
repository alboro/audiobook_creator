# Based on https://github.com/7enChan/reson
# Adapted for this fork by alboro <alboro@users.noreply.github.com>

from __future__ import annotations

import base64
import io
import json
import logging
import math
import os
from typing import TYPE_CHECKING, Dict, List, Optional

try:
    from google import genai
    from google.genai import types
except ImportError:  # pragma: no cover - optional dependency
    genai = None  # type: ignore
    types = None  # type: ignore

from pydub import AudioSegment

from audiobook_generator.config.general_config import GeneralConfig
from audiobook_generator.core.audio_tags import AudioTags
from audiobook_generator.tts_providers.base_tts_provider import BaseTTSProvider
from audiobook_generator.utils.utils import (
    merge_audio_segments,
    set_audio_tags,
    split_text,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-2.5-pro-preview-tts"
DEFAULT_VOICE = "Kore"
DEFAULT_SAMPLE_RATE = 24000
DEFAULT_CHANNELS = 1
DEFAULT_OUTPUT_FORMAT = "wav"
DEFAULT_BREAK_STRING = " @BRK#"
DEFAULT_MAX_CHARS = 1800
DEFAULT_PRICE_PER_1000_CHARS = 0.0
DEFAULT_TEMPERATURE = 0.2

_SUPPORTED_VOICES = [
    "Zephyr",
    "Puck",
    "Charon",
    "Kore",
    "Fenrir",
    "Leda",
    "Orus",
    "Aoede",
    "Callirrhoe",
    "Autonoe",
    "Enceladus",
    "Iapetus",
    "Umbriel",
    "Algieba",
    "Despina",
    "Erinome",
    "Algenib",
    "Rasalgethi",
    "Laomedeia",
    "Achernar",
    "Alnilam",
    "Schedar",
    "Gacrux",
    "Pulcherrima",
    "Achird",
    "Zubenelgenubi",
    "Vindemiatrix",
    "Sadachbia",
    "Sadaltager",
    "Sulafat",
]

_SUPPORTED_OUTPUT_FORMATS = {"wav", "mp3", "flac", "ogg", "opus", "aac"}
_SUPPORTED_ENCODINGS = {
    "pcm16": 2,
    "linear16": 2,
    "pcm_s16le": 2,
    "pcm24": 3,
    "pcm_s24le": 3,
    "pcm32": 4,
    "pcm_s32le": 4,
}
_SUPPORTED_MODELS = [
    "gemini-2.5-pro-preview-tts",
    "gemini-2.5-flash-preview-tts",
]


def get_gemini_supported_voices() -> List[str]:
    return list(_SUPPORTED_VOICES)


def get_gemini_supported_output_formats() -> List[str]:
    return sorted(_SUPPORTED_OUTPUT_FORMATS)


def get_gemini_supported_models() -> List[str]:
    return list(_SUPPORTED_MODELS)


class GeminiTTSProvider(BaseTTSProvider):
    def __init__(self, config: GeneralConfig):
        config.model_name = config.model_name or DEFAULT_MODEL
        config.output_format = (config.output_format or DEFAULT_OUTPUT_FORMAT).lower()
        config.language = config.language or "en-US"
        config.voice_name = config.voice_name or DEFAULT_VOICE

        self.sample_rate = config.gemini_sample_rate or DEFAULT_SAMPLE_RATE
        self.channels = config.gemini_channels or DEFAULT_CHANNELS
        self.audio_encoding = (config.gemini_audio_encoding or "pcm16").lower()
        self.sample_width = self._resolve_sample_width(self.audio_encoding)
        self.price = DEFAULT_PRICE_PER_1000_CHARS
        self._speaker_map = self._parse_speaker_map(config.gemini_speaker_map)
        self._max_chars = DEFAULT_MAX_CHARS if not config.language.startswith("zh") else 1200
        config.gemini_temperature = (
            config.gemini_temperature
            if config.gemini_temperature is not None
            else DEFAULT_TEMPERATURE
        )
        self.temperature = max(0.0, min(1.0, float(config.gemini_temperature)))

        api_key = config.gemini_api_key or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GeminiTTSProvider: GOOGLE_API_KEY environment variable or --gemini_api_key is required."
            )

        if genai is None:
            raise ImportError(
                "GeminiTTSProvider: google-genai>=1.0.0 is required. Install it via 'pip install google-genai'."
            )

        self.client = genai.Client(api_key=api_key)

        super().__init__(config)

    def __str__(self) -> str:
        return (
            f"GeminiTTSProvider(model={self.config.model_name}, voice={self.config.voice_name}, "
            f"output_format={self.config.output_format}, sample_rate={self.sample_rate}, channels={self.channels}, "
            f"temperature={self.temperature})"
        )

    def validate_config(self):
        if self.config.output_format not in _SUPPORTED_OUTPUT_FORMATS:
            raise ValueError(
                f"GeminiTTS: Unsupported output format: {self.config.output_format}. "
                f"Supported formats: {sorted(_SUPPORTED_OUTPUT_FORMATS)}"
            )

        if self.sample_rate <= 0:
            raise ValueError("GeminiTTS: sample rate must be positive")

        if self.channels not in (1, 2):
            raise ValueError("GeminiTTS: only mono or stereo channels are supported")

        if self.config.voice_name and self.config.voice_name not in _SUPPORTED_VOICES:
            raise ValueError(
                f"GeminiTTS: Unsupported voice name: {self.config.voice_name}. "
                f"Supported voices: {_SUPPORTED_VOICES}"
            )

        for speaker, voice in self._speaker_map.items():
            if voice not in _SUPPORTED_VOICES:
                raise ValueError(
                    f"GeminiTTS: Unsupported voice name '{voice}' for speaker '{speaker}'. "
                    f"Supported voices: {_SUPPORTED_VOICES}"
                )

    def text_to_speech(self, text: str, output_file: str, audio_tags: AudioTags):
        if not text.strip():
            logger.warning("GeminiTTS: Received empty text chunk, skipping synthesis")
            return

        chunks = split_text(text, self._max_chars, self.config.language)
        audio_segments: List[io.BytesIO] = []
        chunk_ids: List[str] = []

        for index, chunk in enumerate(chunks, 1):
            chunk_id = f"chapter-{audio_tags.idx}_{audio_tags.title}_chunk_{index}_of_{len(chunks)}"
            logger.info("GeminiTTS: Processing %s (length=%s)", chunk_id, len(chunk))
            prepared_prompt = self._prepare_prompt(chunk)

            try:
                response = self.client.models.generate_content(
                    model=self.config.model_name,
                    contents=prepared_prompt,
                    config=self._build_generate_config(),
                )
            except Exception as exc:  # pragma: no cover - network call
                logger.exception("GeminiTTS: API call failed for %s", chunk_id)
                raise exc

            pcm_bytes = self._extract_pcm_bytes(response, chunk_id)
            segment_buffer = self._encode_pcm_to_segment(pcm_bytes)
            audio_segments.append(segment_buffer)
            chunk_ids.append(chunk_id)

        merge_audio_segments(
            audio_segments,
            output_file,
            self.get_output_file_extension(),
            chunk_ids,
            True,  # Gemini payloads require re-encoding; force pydub merge
        )

        set_audio_tags(output_file, audio_tags)

    def estimate_cost(self, total_chars):
        return math.ceil(total_chars / 1000) * self.price

    def get_break_string(self):
        return DEFAULT_BREAK_STRING

    def get_output_file_extension(self):
        return self.config.output_format

    def _build_generate_config(self) -> types.GenerateContentConfig:
        return types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=self._build_speech_config(),
            temperature=self.temperature,
        )

    def _build_speech_config(self) -> types.SpeechConfig:
        if self._speaker_map:
            speaker_configs = [
                types.SpeakerVoiceConfig(
                    speaker=speaker,
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)
                    ),
                )
                for speaker, voice in self._speaker_map.items()
            ]
            return types.SpeechConfig(
                multi_speaker_voice_config=types.MultiSpeakerVoiceConfig(
                    speaker_voice_configs=speaker_configs
                )
            )

        return types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=self.config.voice_name)
            )
        )

    def _prepare_prompt(self, chunk: str) -> str:
        cleaned = chunk.replace(self.get_break_string(), "\n\n")
        if self.config.instructions:
            return f"{self.config.instructions.strip()}\n\n{cleaned}"
        return cleaned

    def _extract_pcm_bytes(self, response, chunk_id: str) -> bytes:
        for candidate in getattr(response, "candidates", []):
            content = getattr(candidate, "content", None)
            if not content:
                continue
            for part in getattr(content, "parts", []):
                inline = getattr(part, "inline_data", None)
                if inline and getattr(inline, "data", None):
                    data = inline.data
                    if isinstance(data, str):
                        try:
                            return base64.b64decode(data)
                        except Exception as exc:  # pragma: no cover - defensive decode
                            logger.error("GeminiTTS: Failed to decode base64 audio for %s: %s", chunk_id, exc)
                            raise
                    return data
        raise RuntimeError(f"GeminiTTS: No audio payload returned for {chunk_id}")

    def _encode_pcm_to_segment(self, pcm_bytes: bytes) -> io.BytesIO:
        audio = AudioSegment(
            data=pcm_bytes,
            sample_width=self.sample_width,
            frame_rate=self.sample_rate,
            channels=self.channels,
        )
        buffer = io.BytesIO()
        audio.export(buffer, format=self.get_output_file_extension())
        buffer.seek(0)
        return buffer

    @staticmethod
    def _parse_speaker_map(raw: Optional[str]) -> Dict[str, str]:
        if not raw:
            return {}
        if isinstance(raw, dict):
            mapping = raw
        else:
            try:
                mapping = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "GeminiTTS: --gemini_speaker_map must be a valid JSON object string"
                ) from exc
        if not isinstance(mapping, dict):
            raise ValueError("GeminiTTS: speaker map must be a JSON object")
        return {str(key): str(value) for key, value in mapping.items()}

    @staticmethod
    def _resolve_sample_width(encoding: str) -> int:
        if encoding not in _SUPPORTED_ENCODINGS:
            raise ValueError(
                f"GeminiTTS: Unsupported audio encoding '{encoding}'. Supported encodings: {sorted(_SUPPORTED_ENCODINGS)}"
            )
        return _SUPPORTED_ENCODINGS[encoding]