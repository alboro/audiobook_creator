from typing import List
import logging

from audiobook_generator.config.general_config import GeneralConfig

logger = logging.getLogger(__name__)

TTS_AZURE = "azure"
TTS_OPENAI = "openai"
TTS_EDGE = "edge"
TTS_PIPER = "piper"
TTS_QWEN = "qwen"
TTS_GEMINI = "gemini"
TTS_KOKORO = "kokoro"


class BaseTTSProvider:  # Base interface for TTS providers
    def __init__(self, config: GeneralConfig):
        self.config = config
        self.validate_config()

    def __str__(self) -> str:
        return f"{self.config}"

    def validate_config(self):
        raise NotImplementedError

    def text_to_speech(self, *args, **kwargs):
        raise NotImplementedError

    def estimate_cost(self, total_chars):
        raise NotImplementedError

    def get_break_string(self):
        raise NotImplementedError

    def get_output_file_extension(self):
        raise NotImplementedError

    def prepare_tts_text(self, text: str) -> str:
        """Strip configured trailing characters from text before sending to TTS.

        Controlled by ``tts_trailing_strip_chars`` config option (default: ``"."``).
        Pass an empty string to disable stripping.

        If ``tts_log_text`` is enabled in config, logs the final text that will
        be sent to the TTS service.
        """
        strip_chars = getattr(self.config, "tts_trailing_strip_chars", None)
        if strip_chars is None:
            strip_chars = "."
        if strip_chars:
            result = text.rstrip(strip_chars)
        else:
            result = text
        if getattr(self.config, "tts_log_text", False):
            logger.info(
                "TTS INPUT TEXT (%d chars):\n%s\n%s",
                len(result),
                "─" * 60,
                result,
            )
        return result


# Common support methods for all TTS providers
def get_supported_tts_providers() -> List[str]:
    return [TTS_AZURE, TTS_OPENAI, TTS_EDGE, TTS_PIPER, TTS_QWEN, TTS_GEMINI, TTS_KOKORO]


def get_tts_provider(config) -> BaseTTSProvider:
    if config.tts == TTS_AZURE:
        from audiobook_generator.tts_providers.azure_tts_provider import AzureTTSProvider

        return AzureTTSProvider(config)
    elif config.tts == TTS_OPENAI:
        from audiobook_generator.tts_providers.openai_tts_provider import OpenAITTSProvider

        return OpenAITTSProvider(config)
    elif config.tts == TTS_EDGE:
        from audiobook_generator.tts_providers.edge_tts_provider import EdgeTTSProvider

        return EdgeTTSProvider(config)
    elif config.tts == TTS_PIPER:
        from audiobook_generator.tts_providers.piper_tts_provider import PiperTTSProvider

        return PiperTTSProvider(config)
    elif config.tts == TTS_QWEN:
        from audiobook_generator.tts_providers.qwen_tts_provider import Qwen3TTSProvider

        return Qwen3TTSProvider(config)
    elif config.tts == TTS_GEMINI:
        from audiobook_generator.tts_providers.gemini_tts_provider import GeminiTTSProvider

        return GeminiTTSProvider(config)
    elif config.tts == TTS_KOKORO:
        from audiobook_generator.tts_providers.kokoro_tts_provider import KokoroTTSProvider

        return KokoroTTSProvider(config)
    else:
        raise ValueError(f"Invalid TTS provider: {config.tts}")
