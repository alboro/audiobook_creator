import unittest
from unittest.mock import patch

from audiobook_generator.tts_providers.base_tts_provider import get_tts_provider
from audiobook_generator.tts_providers.openai_tts_provider import OpenAITTSProvider
from tests.test_utils import get_openai_config


class TestOpenAiTtsProvider(unittest.TestCase):

    def test_missing_env_var_keys(self):
        """Without OPENAI_API_KEY or openai_base_url, provider initialises with a 'dummy'
        API key (compatible with self-hosted OpenAI-API-compatible servers)."""
        with patch.dict('os.environ', {}, clear=True):
            config = get_openai_config()
            # Should not raise — provider falls back to 'dummy' key for local servers.
            tts_provider = get_tts_provider(config)
            self.assertIsInstance(tts_provider, OpenAITTSProvider)
            self.assertEqual(tts_provider.api_key, "dummy")

    @patch.dict('os.environ', {'OPENAI_API_KEY': 'fake_key'})
    def test_estimate_cost(self):
        config = get_openai_config()
        tts_provider = get_tts_provider(config)
        self.assertIsInstance(tts_provider, OpenAITTSProvider)
        self.assertEqual(tts_provider.estimate_cost(1000000), 15)

    @patch.dict('os.environ', {'OPENAI_API_KEY': 'fake_key'})
    def test_default_args(self):
        config = get_openai_config()
        config.model_name = None
        config.voice_name = None
        config.output_format = None
        tts_provider = get_tts_provider(config)
        self.assertIsInstance(tts_provider, OpenAITTSProvider)
        self.assertEqual(tts_provider.config.model_name, "gpt-4o-mini-tts")
        self.assertEqual(tts_provider.config.voice_name, "alloy")
        self.assertEqual(tts_provider.config.output_format, "mp3")

    @patch.dict('os.environ', {}, clear=True)
    def test_openai_compatible_base_url_without_real_key(self):
        config = get_openai_config()
        config.openai_base_url = "http://127.0.0.1:8000/v1"
        tts_provider = get_tts_provider(config)
        self.assertIsInstance(tts_provider, OpenAITTSProvider)
        self.assertEqual(tts_provider.api_key, "dummy")

    @patch.dict('os.environ', {'OPENAI_API_KEY': 'fake_key'})
    def test_polling_requires_submit_url(self):
        config = get_openai_config()
        config.openai_enable_polling = True
        config.openai_submit_url = None
        config.openai_status_url_template = "https://example.com/jobs/{job_id}"
        with self.assertRaises(ValueError):
            get_tts_provider(config)


if __name__ == '__main__':
    unittest.main()
