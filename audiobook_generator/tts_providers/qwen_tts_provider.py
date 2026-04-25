# Based on https://github.com/7enChan/reson
# Adapted for this fork by alboro <alboro@users.noreply.github.com>

import base64
import io
import logging
import math
import os
import time
from typing import Dict, List, Optional

import requests
try:
    import dashscope  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    dashscope = None

from audiobook_generator.config.general_config import GeneralConfig
from audiobook_generator.core.audio_tags import AudioTags
from audiobook_generator.tts_providers.base_tts_provider import BaseTTSProvider
from audiobook_generator.utils.utils import (
    merge_audio_segments,
    set_audio_tags,
    split_text,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "qwen3-tts-flash"
DEFAULT_VOICE = "Cherry"
DEFAULT_OUTPUT_FORMAT = "wav"
DEFAULT_LANGUAGE_TYPE = "Chinese"
DEFAULT_BREAK_STRING = " @BRK#"
DEFAULT_MAX_INPUT_CHARS = 550
DEFAULT_TIMEOUT_SECONDS = 30
MAX_RETRIES = 4
RETRY_BACKOFF_SECONDS = 2
# Aliyun pricing: 0.8 RMB per 10,000 characters.
# Approximate conversion to USD using 1 RMB ≈ 0.14 USD (2025-09 w/ buffer).
USD_PER_1000_CHAR = 0.0112

_SUPPORTED_MODELS = [
    "qwen3-tts-flash",
    "qwen3-tts-flash-2025-09-18",
]

_SUPPORTED_VOICES = [
    "Cherry",
    "Ethan",
    "Nofish",
    "Jennifer",
    "Ryan",
    "Katerina",
    "Elias",
    "Jada",
    "Dylan",
    "Sunny",
    "li",
    "Marcus",
    "Roy",
    "Peter",
    "Rocky",
    "Kiki",
    "Eric",
]

_SUPPORTED_LANGUAGE_TYPES = [
    "Chinese",
    "English",
    "Spanish",
    "Russian",
    "Italian",
    "French",
    "Korean",
    "Japanese",
    "German",
    "Portuguese",
]

_LANGUAGE_ALIAS: Dict[str, str] = {
    "zh": "Chinese",
    "zh-cn": "Chinese",
    "zh-cn-beijing": "Chinese",
    "zh-cn-shanghai": "Chinese",
    "zh-cn-sichuan": "Chinese",
    "zh-cn-nanjing": "Chinese",
    "zh-cn-tianjin": "Chinese",
    "zh-cn-minnan": "Chinese",
    "zh-cn-cantonese": "Chinese",
    "zh-cn-guangxi": "Chinese",
    "zh-tw": "Chinese",
    "zh-hk": "Chinese",
    "en": "English",
    "en-us": "English",
    "en-gb": "English",
    "es": "Spanish",
    "ru": "Russian",
    "it": "Italian",
    "fr": "French",
    "ko": "Korean",
    "ja": "Japanese",
    "de": "German",
    "pt": "Portuguese",
}


def get_qwen_supported_models() -> List[str]:
    return list(_SUPPORTED_MODELS)


def get_qwen_supported_voices() -> List[str]:
    return list(_SUPPORTED_VOICES)


def get_qwen_supported_language_types() -> List[str]:
    return list(_SUPPORTED_LANGUAGE_TYPES)


class Qwen3TTSProvider(BaseTTSProvider):
    def __init__(self, config: GeneralConfig):
        config.model_name = config.model_name or DEFAULT_MODEL
        config.voice_name = config.voice_name or DEFAULT_VOICE
        config.output_format = (config.output_format or DEFAULT_OUTPUT_FORMAT).lower()
        config.language = config.language or "zh-CN"

        self._language_type = self._resolve_language_type(
            config.qwen_language_type,
            config.language,
        )
        self._stream = bool(config.qwen_stream)
        if self._stream:
            logger.warning("Qwen3TTSProvider: Streaming mode enabled; collected audio will be reassembled locally.")
        self._timeout = self._resolve_timeout(config.qwen_request_timeout)
        self.price = USD_PER_1000_CHAR
        self._max_chars = DEFAULT_MAX_INPUT_CHARS

        self._api_key = config.qwen_api_key or os.environ.get("DASHSCOPE_API_KEY")
        if not self._api_key:
            raise ValueError(
                "Qwen3TTSProvider: DASHSCOPE_API_KEY environment variable or --qwen_api_key is required."
            )

        if dashscope is None:
            raise ImportError(
                "Qwen3TTSProvider: dashscope>=1.24.6 is required. Install it via 'pip install dashscope>=1.24.6'."
            )

        super().__init__(config)

    def __str__(self) -> str:
        return (
            f"Qwen3TTSProvider(model={self.config.model_name}, voice={self.config.voice_name}, "
            f"language_type={self._language_type}, stream={self._stream}, output_format={self.config.output_format})"
        )

    def validate_config(self):
        if self.config.model_name not in _SUPPORTED_MODELS:
            raise ValueError(
                f"Qwen3TTS: Unsupported model '{self.config.model_name}'. Supported models: {_SUPPORTED_MODELS}"
            )
        if self.config.voice_name not in _SUPPORTED_VOICES:
            raise ValueError(
                f"Qwen3TTS: Unsupported voice '{self.config.voice_name}'. Supported voices: {_SUPPORTED_VOICES}"
            )
        if self._language_type not in _SUPPORTED_LANGUAGE_TYPES:
            raise ValueError(
                f"Qwen3TTS: Unsupported language type '{self._language_type}'. Supported types: {_SUPPORTED_LANGUAGE_TYPES}"
            )
        if self.config.output_format != DEFAULT_OUTPUT_FORMAT:
            raise ValueError(
                "Qwen3TTS: Only 'wav' output format is supported at the moment. Please set --output_format wav."
            )

    def text_to_speech(self, text: str, output_file: str, audio_tags: AudioTags):
        if not text.strip():
            logger.warning("Qwen3TTS: Empty text received; skipping chunk synthesis.")
            return

        chunks = split_text(text, self._max_chars, self.config.language)
        audio_segments: List[io.BytesIO] = []
        chunk_ids: List[str] = []

        for index, chunk in enumerate(chunks, 1):
            chunk_id = f"chapter-{audio_tags.idx}_{audio_tags.title}_chunk_{index}_of_{len(chunks)}"
            logger.info("Qwen3TTS: Processing %s (length=%s)", chunk_id, len(chunk))
            prepared_text = self._prepare_text(chunk)

            segment = self._synthesize_with_retry(prepared_text, chunk_id)
            audio_segments.append(segment)
            chunk_ids.append(chunk_id)

        merge_audio_segments(
            audio_segments,
            output_file,
            self.get_output_file_extension(),
            chunk_ids,
            True,  # ensure wav data is normalized via pydub
        )
        set_audio_tags(output_file, audio_tags)

    def estimate_cost(self, total_chars: int) -> float:
        return math.ceil(total_chars / 1000) * self.price

    def get_break_string(self):
        return DEFAULT_BREAK_STRING

    def get_output_file_extension(self):
        return self.config.output_format

    def _prepare_text(self, text: str) -> str:
        return text.replace(self.get_break_string(), "\n\n").strip()

    def _synthesize_with_retry(self, text: str, chunk_id: str) -> io.BytesIO:
        last_error: Optional[Exception] = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                return self._synthesize(text)
            except Exception as exc:  # pragma: no cover - network interaction
                last_error = exc
                logger.warning(
                    "Qwen3TTS: Attempt %s/%s failed for %s due to %s", attempt, MAX_RETRIES, chunk_id, exc
                )
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_BACKOFF_SECONDS * attempt)
        logger.error("Qwen3TTS: Exhausted retries for %s", chunk_id)
        if last_error:
            raise last_error
        raise RuntimeError(f"Qwen3TTS: Failed to synthesize chunk {chunk_id}")

    def _synthesize(self, text: str) -> io.BytesIO:
        if self._stream:
            return self._synthesize_streaming(text)
        return self._synthesize_non_streaming(text)

    def _synthesize_non_streaming(self, text: str) -> io.BytesIO:
        response = dashscope.MultiModalConversation.call(
            model=self.config.model_name,
            api_key=self._api_key,
            text=text,
            voice=self.config.voice_name,
            language_type=self._language_type,
            stream=False,
        )
        audio_url = getattr(response.output.audio, "url", None)
        if not audio_url:
            raise RuntimeError("Qwen3TTS: Response did not contain an audio URL.")

        logger.debug("Qwen3TTS: Downloading audio from %s", audio_url)
        audio_bytes = self._download_audio(audio_url)
        buffer = io.BytesIO(audio_bytes)
        buffer.seek(0)
        return buffer

    def _synthesize_streaming(self, text: str) -> io.BytesIO:
        buffer = io.BytesIO()
        audio_url: Optional[str] = None
        for chunk in dashscope.MultiModalConversation.call(
            model=self.config.model_name,
            api_key=self._api_key,
            text=text,
            voice=self.config.voice_name,
            language_type=self._language_type,
            stream=True,
        ):
            audio = getattr(chunk.output, "audio", None)
            if not audio:
                continue
            if getattr(audio, "data", None):
                buffer.write(base64.b64decode(audio.data))
            if getattr(audio, "url", None):
                audio_url = audio.url

        if buffer.tell() == 0:
            if not audio_url:
                raise RuntimeError("Qwen3TTS: Streaming response contained no audio data.")
            logger.debug("Qwen3TTS: Streaming fallback download from %s", audio_url)
            buffer = io.BytesIO(self._download_audio(audio_url))

        buffer.seek(0)
        return buffer

    def _download_audio(self, url: str) -> bytes:
        response = requests.get(url, timeout=self._timeout)
        response.raise_for_status()
        return response.content

    @staticmethod
    def _resolve_timeout(raw_timeout: Optional[int]) -> int:
        try:
            timeout = int(raw_timeout)
            if timeout <= 0:
                raise ValueError
            return timeout
        except Exception:
            return DEFAULT_TIMEOUT_SECONDS

    @staticmethod
    def _resolve_language_type(explicit: Optional[str], locale: Optional[str]) -> str:
        if explicit:
            return explicit
        if not locale:
            return DEFAULT_LANGUAGE_TYPE
        normalized = locale.lower()
        return _LANGUAGE_ALIAS.get(normalized, DEFAULT_LANGUAGE_TYPE)