import unittest
from unittest.mock import patch, MagicMock
from main import handle_args
from audiobook_generator.config.general_config import GeneralConfig
from audiobook_generator.core.audiobook_generator import AudiobookGenerator


class TestHandleArgs(unittest.TestCase):

    # Test azure arguments
    @patch('sys.argv', ['program', 'input_file.epub', 'output_folder', '--mode', 'all', '--tts', 'azure'])
    def test_azure_args(self):
        config = handle_args()
        self.assertEqual(config.tts, 'azure')

    # Test openai arguments
    @patch('sys.argv', ['program', 'input_file.epub', 'output_folder', '--mode', 'all', '--tts', 'openai'])
    def test_openai_args(self):
        config = handle_args()
        self.assertEqual(config.tts, 'openai')

    @patch(
        'sys.argv',
        [
            'program',
            'input_file.epub',
            'output_folder',
            '--mode', 'all',
            '--tts',
            'openai',
            '--normalize',
            '--package_m4b',
            '--openai_enable_polling',
            '--openai_submit_url',
            'https://example.com/jobs',
            '--openai_status_url_template',
            'https://example.com/jobs/{job_id}',
        ],
    )
    def test_optional_mvp_flags(self):
        config = handle_args()
        self.assertTrue(config.normalize)
        self.assertTrue(config.package_m4b)
        self.assertTrue(config.openai_enable_polling)

    @patch('sys.argv', ['program', 'input_file.epub', 'output_folder', '--mode', 'prepare'])
    def test_mode_prepare(self):
        config = handle_args()
        self.assertEqual(config.mode, 'prepare')

    @patch('sys.argv', ['program', 'input_file.epub', 'output_folder', '--mode', 'audio', '--tts', 'edge'])
    def test_mode_audio(self):
        config = handle_args()
        self.assertEqual(config.mode, 'audio')

    @patch('sys.argv', ['program', 'input_file.epub', 'output_folder', '--mode', 'package', '--tts', 'edge'])
    def test_mode_package(self):
        config = handle_args()
        self.assertEqual(config.mode, 'package')

    @patch('sys.argv', ['program', 'input_file.epub', 'output_folder', '--mode', 'all', '--tts', 'edge'])
    def test_mode_all(self):
        config = handle_args()
        self.assertEqual(config.mode, 'all')

    @patch('sys.argv', ['program', '/some/path/MyBook.epub', '--mode', 'prepare'])
    def test_default_output_folder(self):
        config = handle_args()
        self.assertIsNotNone(config.output_folder)
        self.assertTrue(config.output_folder.endswith('MyBook'))

    # Test unsupported TTS provider
    @patch('sys.argv', ['program', 'input_file.epub', 'output_folder', '--mode', 'all', '--tts', 'unsupported_tts'])
    def test_unsupported_tts(self):
        with self.assertRaises(SystemExit):  # argparse exits with SystemExit on error
            handle_args()

    # Test missing required --mode argument
    @patch('sys.argv', ['program', 'input_file.epub', 'output_folder', '--tts', 'azure'])
    def test_missing_mode(self):
        with self.assertRaises(SystemExit):
            handle_args()

    # Test missing required input_file argument
    @patch('sys.argv', ['program', '--mode', 'all', '--tts', 'azure'])
    def test_missing_input_file(self):
        with self.assertRaises(SystemExit):
            handle_args()

    # Test invalid log level argument
    @patch('sys.argv', ['program', 'input_file.epub', 'output_folder', '--mode', 'all', '--log', 'INVALID_LOG_LEVEL'])
    def test_invalid_log_level(self):
        with self.assertRaises(SystemExit):
            handle_args()


class TestModeImpliesFlags(unittest.TestCase):
    """Tests that --mode all/prepare auto-enables normalize+package_m4b when steps are configured."""

    def _make_config(self, mode, normalize=False, normalize_steps=None, normalize_provider=None, package_m4b=False):
        config = GeneralConfig(None)
        config.input_file = 'input.epub'
        config.output_folder = 'output'
        config.mode = mode
        config.normalize = normalize
        config.normalize_steps = normalize_steps
        config.normalize_provider = normalize_provider
        config.package_m4b = package_m4b
        config.no_prompt = True
        config.tts = 'edge'
        return config

    def _apply_mode(self, config):
        """Replicate the mode-application logic from AudiobookGenerator.run()."""
        gen = AudiobookGenerator.__new__(AudiobookGenerator)
        gen.config = config
        mode = config.mode
        if mode == 'prepare':
            config.prepare_text = True
            config.package_m4b = False
            if not config.normalize and (config.normalize_steps or config.normalize_provider):
                config.normalize = True
        elif mode == 'audio':
            config.prepare_text = False
            config.package_m4b = False
            config.normalize = False
        elif mode == 'all':
            config.prepare_text = False
            config.package_m4b = True
            if not config.normalize and (config.normalize_steps or config.normalize_provider):
                config.normalize = True
        return config

    def test_all_mode_enables_package_m4b(self):
        config = self._make_config('all')
        config = self._apply_mode(config)
        self.assertTrue(config.package_m4b)

    def test_all_mode_enables_normalize_when_steps_set(self):
        config = self._make_config('all', normalize_steps='simple_symbols,ru_numbers')
        config = self._apply_mode(config)
        self.assertTrue(config.normalize)

    def test_all_mode_enables_normalize_when_provider_set(self):
        config = self._make_config('all', normalize_provider='openai')
        config = self._apply_mode(config)
        self.assertTrue(config.normalize)

    def test_all_mode_does_not_enable_normalize_without_steps(self):
        """Without steps or provider, normalize should NOT be auto-enabled."""
        config = self._make_config('all')
        config = self._apply_mode(config)
        self.assertFalse(config.normalize)

    def test_prepare_mode_enables_normalize_when_steps_set(self):
        config = self._make_config('prepare', normalize_steps='simple_symbols')
        config = self._apply_mode(config)
        self.assertTrue(config.normalize)
        self.assertTrue(config.prepare_text)
        self.assertFalse(config.package_m4b)

    def test_explicit_normalize_flag_respected(self):
        """If user explicitly passes --normalize, it should stay True regardless."""
        config = self._make_config('all', normalize=True)
        config = self._apply_mode(config)
        self.assertTrue(config.normalize)

    def test_audio_mode_disables_normalize(self):
        config = self._make_config('audio', normalize=True, normalize_steps='simple_symbols')
        config = self._apply_mode(config)
        self.assertFalse(config.normalize)


if __name__ == '__main__':
    unittest.main()


class TestHandleArgs(unittest.TestCase):

    # Test azure arguments
    @patch('sys.argv', ['program', 'input_file.epub', 'output_folder', '--mode', 'all', '--tts', 'azure'])
    def test_azure_args(self):
        config = handle_args()
        self.assertEqual(config.tts, 'azure')

    # Test openai arguments
    @patch('sys.argv', ['program', 'input_file.epub', 'output_folder', '--mode', 'all', '--tts', 'openai'])
    def test_openai_args(self):
        config = handle_args()
        self.assertEqual(config.tts, 'openai')

    @patch(
        'sys.argv',
        [
            'program',
            'input_file.epub',
            'output_folder',
            '--mode', 'all',
            '--tts',
            'openai',
            '--normalize',
            '--package_m4b',
            '--openai_enable_polling',
            '--openai_submit_url',
            'https://example.com/jobs',
            '--openai_status_url_template',
            'https://example.com/jobs/{job_id}',
        ],
    )
    def test_optional_mvp_flags(self):
        config = handle_args()
        self.assertTrue(config.normalize)
        self.assertTrue(config.package_m4b)
        self.assertTrue(config.openai_enable_polling)

    @patch('sys.argv', ['program', 'input_file.epub', 'output_folder', '--mode', 'prepare'])
    def test_mode_prepare(self):
        config = handle_args()
        self.assertEqual(config.mode, 'prepare')

    @patch('sys.argv', ['program', 'input_file.epub', 'output_folder', '--mode', 'audio', '--tts', 'edge'])
    def test_mode_audio(self):
        config = handle_args()
        self.assertEqual(config.mode, 'audio')

    @patch('sys.argv', ['program', 'input_file.epub', 'output_folder', '--mode', 'package', '--tts', 'edge'])
    def test_mode_package(self):
        config = handle_args()
        self.assertEqual(config.mode, 'package')

    @patch('sys.argv', ['program', 'input_file.epub', 'output_folder', '--mode', 'all', '--tts', 'edge'])
    def test_mode_all(self):
        config = handle_args()
        self.assertEqual(config.mode, 'all')

    @patch('sys.argv', ['program', '/some/path/MyBook.epub', '--mode', 'prepare'])
    def test_default_output_folder(self):
        config = handle_args()
        self.assertIsNotNone(config.output_folder)
        self.assertTrue(config.output_folder.endswith('MyBook'))

    # Test unsupported TTS provider
    @patch('sys.argv', ['program', 'input_file.epub', 'output_folder', '--mode', 'all', '--tts', 'unsupported_tts'])
    def test_unsupported_tts(self):
        with self.assertRaises(SystemExit):  # argparse exits with SystemExit on error
            handle_args()

    # Test missing required --mode argument
    @patch('sys.argv', ['program', 'input_file.epub', 'output_folder', '--tts', 'azure'])
    def test_missing_mode(self):
        with self.assertRaises(SystemExit):
            handle_args()

    # Test missing required input_file argument
    @patch('sys.argv', ['program', '--mode', 'all', '--tts', 'azure'])
    def test_missing_input_file(self):
        with self.assertRaises(SystemExit):
            handle_args()

    # Test invalid log level argument
    @patch('sys.argv', ['program', 'input_file.epub', 'output_folder', '--mode', 'all', '--log', 'INVALID_LOG_LEVEL'])
    def test_invalid_log_level(self):
        with self.assertRaises(SystemExit):
            handle_args()


if __name__ == '__main__':
    unittest.main()
