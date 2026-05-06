class GeneralConfig:
    def __init__(self, args):
        # ------------------------------------------------------------------ #
        # Internal helper: read a field from args (already INI-merged),       #
        # apply optional type coercion, and fall back to a built-in default.  #
        #                                                                      #
        # coerce=int / float  — try numeric conversion; return default on err  #
        # coerce=bool         — handles Python bool, "true"/"false" strings    #
        #   default=True  → anything that is NOT explicitly 'false'/'0'/'no'  #
        #   default=False → only explicit 'true'/'1'/'yes' returns True       #
        # ------------------------------------------------------------------ #
        def _get(field, coerce=None, default=None):
            val = getattr(args, field, None)
            if val is None:
                return default
            if coerce is None:
                return val
            if coerce is int:
                try:
                    return int(val)
                except (TypeError, ValueError):
                    return default
            if coerce is float:
                try:
                    return float(val)
                except (TypeError, ValueError):
                    return default
            if coerce is bool:
                if isinstance(val, bool):
                    return val
                s = str(val).lower()
                if default is True:
                    return s not in ('false', '0', 'no')
                return s in ('true', '1', 'yes')
            return coerce(val)

        # General arguments
        self.input_file = _get('input_file')
        self.output_folder = _get('output_folder')

        # Default output_folder: a directory named after the book, next to the input file.
        # e.g. /path/to/MyBook.epub  →  /path/to/MyBook/
        if not self.output_folder and self.input_file:
            from pathlib import Path
            _input = Path(self.input_file).expanduser().resolve()
            self.output_folder = str(_input.parent / _input.stem)

        # Generation mode: prepare | audio | package | all
        self.mode = _get('mode')
        self.output_text = _get('output_text')
        self.prepared_text_folder = _get('prepared_text_folder')
        self.log = _get('log')
        self.log_file = None
        self.no_prompt = _get('no_prompt')
        self.worker_count = _get('worker_count')
        self.use_pydub_merge = _get('use_pydub_merge')
        self.force_new_run = _get('force_new_run')
        self.package_m4b = _get('package_m4b')
        self.chunked_audio = _get('chunked_audio')
        self.chunked_audio_no_db = _get('chunked_audio_no_db', bool, False)
        self.audio_worker_interval = _get('audio_worker_interval')
        self.audio_folder = _get('audio_folder')
        self.m4b_filename = _get('m4b_filename')
        self.m4b_bitrate = _get('m4b_bitrate')
        self.chapter_titles_file = _get('chapter_titles_file')
        self.cover_image = _get('cover_image')
        self.ffmpeg_path = _get('ffmpeg_path')

        # Book parser specific arguments
        self.title_mode = _get('title_mode')
        self.chapter_mode = _get('chapter_mode')
        self.newline_mode = _get('newline_mode')
        self.chapter_start = _get('chapter_start')
        self.chapter_end = _get('chapter_end')
        self.search_and_replace_file = _get('search_and_replace_file')

        # TTS provider: common arguments
        self.tts = _get('tts')
        self.language = _get('language')
        self.voice_name = _get('voice_name')
        self.voice_name2 = _get('voice_name2')
        self.output_format = _get('output_format')
        self.model_name = _get('model_name')
        self.tts_trailing_strip_chars = _get('tts_trailing_strip_chars')
        self.tts_log_text = _get('tts_log_text', bool, False)
        # Default is True; only explicitly setting "false" (string or bool) disables it.
        self.tts_trim_silence = _get('tts_trim_silence', bool, True)
        # Smooth chunk joining: crossfade chunk boundaries to eliminate crackling.
        # Default: True. Disable for lowest-latency pure-concatenation.
        self.tts_chunk_smooth_join = _get('tts_chunk_smooth_join', bool, True)
        self.tts_chunk_smooth_join_ms = _get('tts_chunk_smooth_join_ms', int, 30)
        # DC offset removal: subtract per-chunk mean before merging (default: True).
        self.tts_chunk_dc_remove = _get('tts_chunk_dc_remove', bool, True)
        # Silence gap between chunks at merge time, ms (default: 0 = disabled).
        self.tts_chunk_merge_gap_ms = _get('tts_chunk_merge_gap_ms', int, 0)
        # CosyVoice can emit a short click/burst at the beginning of generated
        # chunks. This optional pass removes only the chunk head before merging.
        self.tts_chunk_declick_start = _get('tts_chunk_declick_start', bool, False)
        self.tts_chunk_declick_start_ms = _get('tts_chunk_declick_start_ms', int, 10)
        self.tts_chunk_declick_fade_ms = _get('tts_chunk_declick_fade_ms', int, 6)
        # Low-frequency preamble detection and removal.  Some TTS engines
        # (e.g., CosyVoice) can emit a short low-frequency "breath/ock" burst
        # before the first actual phoneme.  Enabling this pass detects and
        # removes that preamble by analysing ZCR and spectral concentration.
        self.tts_chunk_declick_lf_preamble = _get('tts_chunk_declick_lf_preamble', bool, False)
        self.tts_chunk_declick_lf_preamble_fade_ms = _get('tts_chunk_declick_lf_preamble_fade_ms', int, 8)

        self.openai_api_key = _get('openai_api_key')
        self.openai_base_url = _get('openai_base_url')
        self.openai_max_chars = _get('openai_max_chars')
        self.openai_enable_polling = _get('openai_enable_polling')
        self.openai_submit_url = _get('openai_submit_url')
        self.openai_status_url_template = _get('openai_status_url_template')
        self.openai_download_url_template = _get('openai_download_url_template')
        self.openai_job_id_path = _get('openai_job_id_path')
        self.openai_job_status_path = _get('openai_job_status_path')
        self.openai_job_download_url_path = _get('openai_job_download_url_path')
        self.openai_job_done_values = _get('openai_job_done_values')
        self.openai_job_failed_values = _get('openai_job_failed_values')
        self.openai_poll_interval = _get('openai_poll_interval')
        self.openai_poll_timeout = _get('openai_poll_timeout')
        self.openai_poll_request_timeout = _get('openai_poll_request_timeout')
        self.openai_poll_max_errors = _get('openai_poll_max_errors')
        self.openai_submit_omit_fields = _get('openai_submit_omit_fields')
        self.openai_submit_extra_fields = _get('openai_submit_extra_fields')

        # OpenAI specific arguments
        self.instructions = _get('instructions')
        self.speed = _get('speed')

        # Normalizer specific arguments
        self.normalize = _get('normalize')
        self.normalize_steps = _get('normalize_steps')
        self.normalize_provider = _get('normalize_provider')
        self.normalize_model = _get('normalize_model')
        self.normalize_system_prompt = _get('normalize_system_prompt')
        self.normalize_prompt_file = _get('normalize_prompt_file')
        self.normalize_system_prompt_file = _get('normalize_system_prompt_file')
        self.normalize_user_prompt_file = _get('normalize_user_prompt_file')
        self.normalize_api_key = _get('normalize_api_key')
        self.normalize_base_url = _get('normalize_base_url')
        self.normalize_max_chars = _get('normalize_max_chars')
        self.normalize_tts_safe_max_chars = _get('normalize_tts_safe_max_chars')
        self.normalize_tts_safe_comma_as_period = _get('normalize_tts_safe_comma_as_period')
        self.normalize_tts_pronunciation_overrides_file = (
            _get('normalize_tts_pronunciation_overrides_file')
            or _get('normalize_pronunciation_exceptions_file')
        )
        self.normalize_pronunciation_exceptions_file = self.normalize_tts_pronunciation_overrides_file
        # tts_hard_consonants normalizer — new canonical keys (де→дэ / те→тэ etc.)
        # Falls back to the old tts_pronunciation_overrides keys in the normalizer itself.
        self.normalize_tts_hard_consonants_file = (
            _get('normalize_tts_hard_consonants_file')
            or self.normalize_tts_pronunciation_overrides_file
        )
        self.normalize_tts_hard_consonants_words = _get('normalize_tts_hard_consonants_words')
        self.normalize_pronunciation_lexicon_db = _get('normalize_pronunciation_lexicon_db')
        # normalize_stress_exceptions_file: removed (replaced by normalize_stress_paradox_words)
        self.normalize_stress_ambiguity_file = _get('normalize_stress_ambiguity_file')
        self.normalize_tsnorm_stress_yo = _get('normalize_tsnorm_stress_yo')
        self.normalize_tsnorm_stress_monosyllabic = _get('normalize_tsnorm_stress_monosyllabic')
        self.normalize_tsnorm_min_word_length = _get('normalize_tsnorm_min_word_length')
        self.normalize_stress_paradox_words = _get('normalize_stress_paradox_words')
        self.normalize_log_changes = _get('normalize_log_changes')
        self.normalize_stress_ambiguity_system_prompt = _get('normalize_stress_ambiguity_system_prompt')
        self.normalize_safe_split_system_prompt = _get('normalize_safe_split_system_prompt')
        self.normalize_reasoning_effort = _get('normalize_reasoning_effort')

        # TTS provider: Azure & Edge TTS specific arguments
        self.break_duration = _get('break_duration')

        # TTS provider: Edge specific arguments
        self.voice_rate = _get('voice_rate')
        self.voice_volume = _get('voice_volume')
        self.voice_pitch = _get('voice_pitch')
        self.proxy = _get('proxy')

        # TTS provider: Piper specific arguments
        self.piper_path = _get('piper_path')
        self.piper_docker_image = _get('piper_docker_image')
        self.piper_speaker = _get('piper_speaker')
        self.piper_noise_scale = _get('piper_noise_scale')
        self.piper_noise_w_scale = _get('piper_noise_w_scale')
        self.piper_length_scale = _get('piper_length_scale')
        self.piper_sentence_silence = _get('piper_sentence_silence')

        # TTS provider: Qwen3 specific arguments
        self.qwen_api_key = _get('qwen_api_key')
        self.qwen_language_type = _get('qwen_language_type')
        self.qwen_stream = _get('qwen_stream')
        self.qwen_request_timeout = _get('qwen_request_timeout')

        # TTS provider: Gemini specific arguments
        self.gemini_api_key = _get('gemini_api_key')
        self.gemini_sample_rate = _get('gemini_sample_rate')
        self.gemini_channels = _get('gemini_channels')
        self.gemini_audio_encoding = _get('gemini_audio_encoding')
        self.gemini_temperature = _get('gemini_temperature')
        self.gemini_speaker_map = _get('gemini_speaker_map')

        # TTS provider: Kokoro specific arguments
        self.kokoro_base_url = _get('kokoro_base_url')
        self.kokoro_volume_multiplier = _get('kokoro_volume_multiplier')

        # Audio check specific arguments
        self.audio_check_model = _get('audio_check_model')
        self.audio_check_threshold = _get('audio_check_threshold', float)
        self.audio_check_device = _get('audio_check_device')
        self.audio_check_force = _get('audio_check_force')
        # Comma-separated list of checker names to run (see AUDIO_CHECKER_REGISTRY).
        # Default: whisper_similarity,first_word,last_word
        # Full set:  whisper_similarity,first_word,last_word,reference,transcription_artifacts
        self.audio_check_checkers = _get('audio_check_checkers')
        # TranscriptionArtifactsChecker: comma-separated substrings to find in
        # Whisper transcription.  Chunk is marked disputed if any substring is
        # found (case-insensitive).  Example: "точка,очка"
        self.audio_checker_transcription_artifacts = _get('audio_checker_transcription_artifacts')
        # Auto-loop thresholds (used by --mode audio_auto).
        self.audio_auto_check_threshold = _get('audio_auto_check_threshold', float)
        self.audio_auto_retry = _get('audio_auto_retry', int)
        self.audio_reference_check_command = _get('audio_reference_check_command')
        self.audio_reference_check_threshold = _get('audio_reference_check_threshold', float)
        self.audio_reference_check_timeout = _get('audio_reference_check_timeout', int)
        self.audio_reference_check_cache_dir = _get('audio_reference_check_cache_dir')
        self.audio_reference_check_stress = _get('audio_reference_check_stress')

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
