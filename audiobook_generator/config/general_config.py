class GeneralConfig:
    def __init__(self, args):
        # General arguments
        self.input_file = getattr(args, 'input_file', None)
        self.output_folder = getattr(args, 'output_folder', None)

        # Default output_folder: a directory named after the book, next to the input file.
        # e.g. /path/to/MyBook.epub  →  /path/to/MyBook/
        if not self.output_folder and self.input_file:
            from pathlib import Path
            _input = Path(self.input_file).expanduser().resolve()
            self.output_folder = str(_input.parent / _input.stem)

        # Generation mode: prepare | audio | package | all
        self.mode = getattr(args, 'mode', None)
        self.output_text = getattr(args, 'output_text', None)
        self.prepared_text_folder = getattr(args, 'prepared_text_folder', None)
        self.log = getattr(args, 'log', None)
        self.log_file = None
        self.no_prompt = getattr(args, 'no_prompt', None)
        self.worker_count = getattr(args, 'worker_count', None)
        self.use_pydub_merge = getattr(args, 'use_pydub_merge', None)
        self.force_new_run = getattr(args, 'force_new_run', None)
        self.package_m4b = getattr(args, 'package_m4b', None)
        self.chunked_audio = getattr(args, 'chunked_audio', None)
        self.audio_folder = getattr(args, 'audio_folder', None)
        self.m4b_filename = getattr(args, 'm4b_filename', None)
        self.m4b_bitrate = getattr(args, 'm4b_bitrate', None)
        self.chapter_titles_file = getattr(args, 'chapter_titles_file', None)
        self.cover_image = getattr(args, 'cover_image', None)
        self.ffmpeg_path = getattr(args, 'ffmpeg_path', None)

        # Book parser specific arguments
        self.title_mode = getattr(args, 'title_mode', None)
        self.chapter_mode = getattr(args, 'chapter_mode', None)
        self.newline_mode = getattr(args, 'newline_mode', None)
        self.chapter_start = getattr(args, 'chapter_start', None)
        self.chapter_end = getattr(args, 'chapter_end', None)
        self.search_and_replace_file = getattr(args, 'search_and_replace_file', None)

        # TTS provider: common arguments
        self.tts = getattr(args, 'tts', None)
        self.language = getattr(args, 'language', None)
        self.voice_name = getattr(args, 'voice_name', None)
        self.voice_name2 = getattr(args, 'voice_name2', None)
        self.output_format = getattr(args, 'output_format', None)
        self.model_name = getattr(args, 'model_name', None)
        self.tts_trailing_strip_chars = getattr(args, 'tts_trailing_strip_chars', None)
        self.tts_log_text = getattr(args, 'tts_log_text', False) or False
        _trim = getattr(args, 'tts_trim_silence', None)
        # Default is True; only explicitly setting "false" (string or bool) disables it.
        self.tts_trim_silence = str(_trim).lower() != 'false' if _trim is not None else True
        # Smooth chunk joining: apply short fade-in/fade-out at chunk boundaries to eliminate
        # crackling.  Default: True.  Disable for lowest-latency pure-concatenation.
        _smooth = getattr(args, 'tts_chunk_smooth_join', None)
        self.tts_chunk_smooth_join = str(_smooth).lower() != 'false' if _smooth is not None else True
        _smooth_ms = getattr(args, 'tts_chunk_smooth_join_ms', None)
        try:
            self.tts_chunk_smooth_join_ms = int(_smooth_ms) if _smooth_ms is not None else 30
        except (ValueError, TypeError):
            self.tts_chunk_smooth_join_ms = 30
        self.openai_api_key = getattr(args, 'openai_api_key', None)
        self.openai_base_url = getattr(args, 'openai_base_url', None)
        self.openai_max_chars = getattr(args, 'openai_max_chars', None)
        self.openai_enable_polling = getattr(args, 'openai_enable_polling', None)
        self.openai_submit_url = getattr(args, 'openai_submit_url', None)
        self.openai_status_url_template = getattr(args, 'openai_status_url_template', None)
        self.openai_download_url_template = getattr(args, 'openai_download_url_template', None)
        self.openai_job_id_path = getattr(args, 'openai_job_id_path', None)
        self.openai_job_status_path = getattr(args, 'openai_job_status_path', None)
        self.openai_job_download_url_path = getattr(args, 'openai_job_download_url_path', None)
        self.openai_job_done_values = getattr(args, 'openai_job_done_values', None)
        self.openai_job_failed_values = getattr(args, 'openai_job_failed_values', None)
        self.openai_poll_interval = getattr(args, 'openai_poll_interval', None)
        self.openai_poll_timeout = getattr(args, 'openai_poll_timeout', None)
        self.openai_poll_request_timeout = getattr(args, 'openai_poll_request_timeout', None)
        self.openai_poll_max_errors = getattr(args, 'openai_poll_max_errors', None)
        self.openai_submit_omit_fields = getattr(args, 'openai_submit_omit_fields', None)
        self.openai_submit_extra_fields = getattr(args, 'openai_submit_extra_fields', None)

        # OpenAI specific arguments
        self.instructions = getattr(args, 'instructions', None)
        self.speed = getattr(args, 'speed', None)

        # Normalizer specific arguments
        self.normalize = getattr(args, 'normalize', None)
        self.normalize_steps = getattr(args, 'normalize_steps', None)
        self.normalize_provider = getattr(args, 'normalize_provider', None)
        self.normalize_model = getattr(args, 'normalize_model', None)
        self.normalize_system_prompt = getattr(args, 'normalize_system_prompt', None)
        self.normalize_prompt_file = getattr(args, 'normalize_prompt_file', None)
        self.normalize_system_prompt_file = getattr(args, 'normalize_system_prompt_file', None)
        self.normalize_user_prompt_file = getattr(args, 'normalize_user_prompt_file', None)
        self.normalize_api_key = getattr(args, 'normalize_api_key', None)
        self.normalize_base_url = getattr(args, 'normalize_base_url', None)
        self.normalize_max_chars = getattr(args, 'normalize_max_chars', None)
        self.normalize_tts_safe_max_chars = getattr(args, 'normalize_tts_safe_max_chars', None)
        self.normalize_tts_safe_comma_as_period = getattr(args, 'normalize_tts_safe_comma_as_period', None)
        self.normalize_tts_pronunciation_overrides_file = (
            getattr(args, 'normalize_tts_pronunciation_overrides_file', None)
            or getattr(args, 'normalize_pronunciation_exceptions_file', None)
        )
        self.normalize_pronunciation_exceptions_file = self.normalize_tts_pronunciation_overrides_file
        self.normalize_tts_pronunciation_overrides_words = getattr(
            args, 'normalize_tts_pronunciation_overrides_words', None
        )
        self.normalize_pronunciation_lexicon_db = getattr(
            args, 'normalize_pronunciation_lexicon_db', None
        )
        # normalize_stress_exceptions_file: removed (replaced by normalize_stress_paradox_words)
        self.normalize_stress_ambiguity_file = getattr(
            args, 'normalize_stress_ambiguity_file', None
        )
        self.normalize_tsnorm_stress_yo = getattr(args, 'normalize_tsnorm_stress_yo', None)
        self.normalize_tsnorm_stress_monosyllabic = getattr(
            args, 'normalize_tsnorm_stress_monosyllabic', None
        )
        self.normalize_tsnorm_min_word_length = getattr(
            args, 'normalize_tsnorm_min_word_length', None
        )
        self.normalize_stress_paradox_words = getattr(
            args, 'normalize_stress_paradox_words', None
        )
        self.normalize_log_changes = getattr(args, 'normalize_log_changes', None)
        self.normalize_stress_ambiguity_system_prompt = getattr(
            args, 'normalize_stress_ambiguity_system_prompt', None
        )
        self.normalize_safe_split_system_prompt = getattr(
            args, 'normalize_safe_split_system_prompt', None
        )

        # TTS provider: Azure & Edge TTS specific arguments
        self.break_duration = getattr(args, 'break_duration', None)

        # TTS provider: Edge specific arguments
        self.voice_rate = getattr(args, 'voice_rate', None)
        self.voice_volume = getattr(args, 'voice_volume', None)
        self.voice_pitch = getattr(args, 'voice_pitch', None)
        self.proxy = getattr(args, 'proxy', None)

        # TTS provider: Piper specific arguments
        self.piper_path = getattr(args, 'piper_path', None)
        self.piper_docker_image = getattr(args, 'piper_docker_image', None)
        self.piper_speaker = getattr(args, 'piper_speaker', None)
        self.piper_noise_scale = getattr(args, 'piper_noise_scale', None)
        self.piper_noise_w_scale = getattr(args, 'piper_noise_w_scale', None)
        self.piper_length_scale = getattr(args, 'piper_length_scale', None)
        self.piper_sentence_silence = getattr(args, 'piper_sentence_silence', None)

        # TTS provider: Qwen3 specific arguments
        self.qwen_api_key = getattr(args, 'qwen_api_key', None)
        self.qwen_language_type = getattr(args, 'qwen_language_type', None)
        self.qwen_stream = getattr(args, 'qwen_stream', None)
        self.qwen_request_timeout = getattr(args, 'qwen_request_timeout', None)

        # TTS provider: Gemini specific arguments
        self.gemini_api_key = getattr(args, 'gemini_api_key', None)
        self.gemini_sample_rate = getattr(args, 'gemini_sample_rate', None)
        self.gemini_channels = getattr(args, 'gemini_channels', None)
        self.gemini_audio_encoding = getattr(args, 'gemini_audio_encoding', None)
        self.gemini_temperature = getattr(args, 'gemini_temperature', None)
        self.gemini_speaker_map = getattr(args, 'gemini_speaker_map', None)

        # TTS provider: Kokoro specific arguments
        self.kokoro_base_url = getattr(args, 'kokoro_base_url', None)
        self.kokoro_volume_multiplier = getattr(args, 'kokoro_volume_multiplier', None)

        # Audio check specific arguments
        self.audio_check_model = getattr(args, 'audio_check_model', None)
        self.audio_check_threshold = getattr(args, 'audio_check_threshold', None)
        self.audio_check_device = getattr(args, 'audio_check_device', None)

        # Dynamic per-normalizer model overrides: normalize_{step}_model
        # These are set by merge_ini_into_args for any key in INI that matches the pattern.
        for _key, _val in vars(args).items() if hasattr(args, '__dict__') else []:
            if _key.startswith("normalize_") and _key.endswith("_model") and _key != "normalize_model":
                setattr(self, _key, _val)

        # --- Internal runtime fields (set by AudiobookGenerator, not from CLI) ---
        # Sequential run index string, e.g. "001".  Set before workers start.
        self.current_run_index: str | None = None
        # Path to normalization state SQLite file (overrides default _state/ location).
        self.normalization_state_path: str | None = None
        # Set by run() based on --mode; not exposed as CLI flags.
        self.prepare_text: bool = False
        self.preview: bool = False

    def __str__(self):
        return ",\n".join(f"{key}={value}" for key, value in self.__dict__.items())
