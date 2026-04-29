import argparse
from pathlib import Path

from audiobook_generator.config.general_config import GeneralConfig
from audiobook_generator.core.audiobook_generator import AudiobookGenerator
from audiobook_generator.normalizers.base_normalizer import get_supported_normalizers
from audiobook_generator.tts_providers.base_tts_provider import (
    get_supported_tts_providers,
)
from audiobook_generator.utils.log_handler import setup_logging, generate_unique_log_path


def handle_args():
    parser = argparse.ArgumentParser(description="Convert text book to audiobook")
    parser.add_argument("input_file", help="Path to the input book file (EPUB or FB2)")
    parser.add_argument(
        "output_folder",
        nargs="?",
        default=None,
        help=(
            "Output folder (optional). "
            "Default: a directory named after the book file, placed next to it. "
            "e.g. /books/MyBook.epub → /books/MyBook/"
        ),
    )
    parser.add_argument(
        "--config",
        default=None,
        metavar="PATH",
        help=(
            "Path to an INI config file. Settings from the file are merged with CLI args; "
            "explicit CLI args always take priority."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["prepare", "audio", "audio_chunks", "audio_worker", "package", "all", "audio_check", "audio_auto"],
        required=True,
        help=(
            "Generation stage to run:\n"
            "  prepare      — parse book, normalize text, write per-chapter .txt files for review;\n"
            "  audio        — synthesize audio from per-chapter .txt (--prepared_text_folder) or raw book text;\n"
            "                 when chunked_audio=true, skips re-merging chapters whose WAV is already up-to-date;\n"
            "  audio_chunks — synthesise per-sentence chunk files only (no chapter merge).\n"
            "                 Useful to synthesise first and merge with --mode audio afterwards;\n"
            "  audio_worker — like audio_chunks but runs in an infinite loop, sleeping\n"
            "                 audio_worker_interval seconds between passes.\n"
            "                 Keeps synthesising any new/missing chunks as they appear.\n"
            "                 Stop with Ctrl+C — current chunk finishes before exit;\n"
            "  package      — package existing chapter audio files in output_folder into a single .m4b;\n"
            "  all          — normalize + synthesize + package in one pass (full pipeline).\n"
            "  audio_check  — transcribe existing audio chunks locally with Whisper and mark\n"
                 "                 mismatches as disputed in the DB for manual review.\n"
                 "  audio_auto   — auto-loop: synthesise → check → delete failed → repeat\n"
                 "                 up to audio_auto_retry times until all chunks pass\n"
                 "                 audio_auto_check_threshold (default: 0.78)."
        ),
    )
    parser.add_argument(
        "--audio_check_model",
        default=None,
        help="Whisper model size for --mode audio_check (default: small). Options: tiny, base, small, medium, large-v3.",
    )
    parser.add_argument(
        "--audio_check_threshold",
        type=float,
        default=None,
        help="Similarity threshold below which a chunk is marked disputed (default: 0.70).",
    )
    parser.add_argument(
        "--audio_check_device",
        default=None,
        help="Inference device for Whisper: cpu or cuda (default: cpu).",
    )
    parser.add_argument(
        "--audio_auto_check_threshold",
        type=float,
        default=None,
        help=(
            "Similarity threshold for --mode audio_auto: chunks below this are deleted and "
            "re-synthesised automatically (default: 0.78).  Lower than audio_check_threshold "
            "so only clearly bad chunks are retried unattended."
        ),
    )
    parser.add_argument(
        "--audio_auto_retry",
        type=int,
        default=None,
        help="Maximum number of re-synthesis attempts per chunk in --mode audio_auto (default: 5).",
    )
    parser.add_argument(
        "--tts",
        choices=get_supported_tts_providers(),
        default=None,
        help="Choose TTS provider (default: azure). azure: Azure Cognitive Services, openai: OpenAI TTS API. When using azure, environment variables MS_TTS_KEY and MS_TTS_REGION must be set. When using openai, environment variable OPENAI_API_KEY must be set.",
    )
    parser.add_argument(
        "--log",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Log level (default: INFO), can be DEBUG, INFO, WARNING, ERROR, CRITICAL",
    )
    parser.add_argument(
        "--no_prompt",
        action="store_true",
        help="Don't ask the user if they wish to continue after estimating the cloud cost for TTS. Useful for scripting.",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Language for the text-to-speech service (default: en-US). For Azure TTS (--tts=azure), check https://learn.microsoft.com/en-us/azure/ai-services/speech-service/language-support?tabs=tts#text-to-speech for supported languages. For OpenAI TTS (--tts=openai), their API detects the language automatically. But setting this will also help on splitting the text into chunks with different strategies in this tool, especially for Chinese characters. For Chinese books, use zh-CN, zh-TW, or zh-HK.",
    )
    parser.add_argument(
        "--newline_mode",
        choices=["single", "double", "none"],
        default=None,
        help="Choose the mode of detecting new paragraphs: 'single', 'double', or 'none'. (default: double)",
    )
    parser.add_argument(
        "--title_mode",
        choices=["auto", "tag_text", "first_few"],
        default=None,
        help="Choose the parse mode for chapter title. (default: auto)",
    )
    parser.add_argument(
        "--chapter_mode",
        choices=["documents", "toc_sections"],
        default=None,
        help=(
            "Choose how book content is grouped into chapters. "
            "'documents' keeps one chapter per XHTML document (EPUB) or per leaf section (FB2). "
            "'toc_sections' groups EPUB documents by top-level table-of-contents sections when possible. "
            "(default: documents)"
        ),
    )
    parser.add_argument(
        "--chapter_start",
        default=None,
        type=int,
        help="Chapter start index (default: 1, starting from 1)",
    )
    parser.add_argument(
        "--chapter_end",
        default=None,
        type=int,
        help="Chapter end index (default: -1, meaning to the last chapter)",
    )
    parser.add_argument(
        "--output_text",
        action="store_true",
        help="Enable Output Text. This will export a plain text file for each chapter specified and write the files to the output folder specified.",
    )
    parser.add_argument(
        "--prepared_text_folder",
        help="Use reviewed per-chapter .txt files from this folder as the TTS source instead of the raw text extracted from the EPUB.",
    )

    parser.add_argument(
        "--search_and_replace_file",
        default=None,
        help="""Path to a file that contains 1 regex replace per line, to help with fixing pronunciations, etc. The format is:
        <search>==<replace>
        Note that you may have to specify word boundaries, to avoid replacing parts of words.
        """,
    )

    parser.add_argument(
        "--worker_count",
        type=int,
        default=None,
        help="Specifies the number of parallel workers to use for audiobook generation. (default: 1)",
    )

    parser.add_argument(
        "--use_pydub_merge",
        action="store_true",
        help="Use pydub to merge audio segments of one chapter into single file instead of direct write. "
        "Currently only supported for OpenAI and Azure TTS. "
        "Direct write is faster but might skip audio segments if formats differ. "
        "Pydub merge is slower but more reliable for different audio formats. It requires ffmpeg to be installed first. "
        "You can use this option to avoid the issue of skipping audio segments in some cases. "
        "However, it's recommended to use direct write for most cases as it's faster. "
        "Only use this option if you encounter issues with direct write.",
    )
    parser.add_argument(
        "--package_m4b",
        action="store_true",
        help="Package generated chapter audio files into a single m4b audiobook with chapter markers.",
    )
    parser.add_argument(
        "--force_new_run",
        action="store_true",
        help="Force creating a new run directory (002, 003, etc.) instead of attempting to resume previous incomplete work.",
    )
    parser.add_argument(
        "--chunked_audio",
        action="store_true",
        help=(
            "Enable sentence-level chunked TTS generation with SQLite resume. "
            "Each sentence is synthesised independently; already-synthesised chunks "
            "are reused on reruns. Changed sentences are re-synthesised and old chunks "
            "are marked as superseded (not deleted). "
            "Chunks are merged into the chapter audio file after synthesis."
        ),
    )
    parser.add_argument(
        "--chunked_audio_no_db",
        action="store_true",
        help=(
            "Disable all SQLite DB writes during chunked synthesis. "
            "No sentence text history is recorded. "
            "Incompatible with --mode audio_check (which requires the DB). "
            "Useful when you only need the audio files and do not intend to use "
            "the Review UI or audio quality check features."
        ),
    )
    parser.add_argument(
        "--audio_worker_interval",
        type=int,
        default=None,
        metavar="SECONDS",
        help=(
            "Seconds to sleep between synthesis passes in --mode audio_worker. "
            "Default: 30. Can also be set in INI as [m4b] audio_worker_interval = N."
        ),
    )
    parser.add_argument(
        "--audio_folder",
        default=None,
        help=(
            "Explicit folder containing chapter audio files for --mode package. "
            "Can also be set in INI as [m4b] audio_folder = ... . "
            "If not set, the tool auto-detects output_folder/wav/ and then falls back to output_folder itself. "
            "On macOS you may also pass an smb://host/share/... URL if that share is mounted under /Volumes."
        ),
    )
    parser.add_argument(
        "--m4b_filename",
        help="Optional output filename for the packaged m4b file.",
    )
    parser.add_argument(
        "--m4b_bitrate",
        default="64k",
        help="AAC bitrate for m4b packaging (default: 64k).",
    )
    parser.add_argument(
        "--chapter_titles_file",
        default=None,
        help=(
            "Optional UTF-8 text file with one chapter title per line, in packaging order. "
            "Non-empty lines override chapter marker titles in the final m4b."
        ),
    )
    parser.add_argument(
        "--cover_image",
        default=None,
        help="Optional path to a cover image file (jpg/png/webp, etc.) used for m4b packaging.",
    )
    parser.add_argument(
        "--ffmpeg_path",
        default="ffmpeg",
        help="Path to ffmpeg binary used for m4b packaging.",
    )

    parser.add_argument(
        "--voice_name",
        help="Various TTS providers has different voice names, look up for your provider settings.",
    )

    parser.add_argument(
        "--output_format",
        help="Output format for the text-to-speech service. Supported format depends on selected TTS provider",
    )

    parser.add_argument(
        "--model_name",
        help="Various TTS providers has different neural model names",
    )

    parser.add_argument(
        "--tts_trailing_strip_chars",
        default=None,
        help="Characters to strip from the end of each TTS text chunk (default: '.', set to '' to disable).",
    )
    parser.add_argument(
        "--tts_chunk_declick_start",
        action="store_const",
        const=True,
        default=None,
        help="Remove a short detected click/burst from the beginning of each chunk before merging.",
    )
    parser.add_argument(
        "--tts_chunk_declick_start_ms",
        type=int,
        default=None,
        help="Milliseconds to remove from the beginning of a chunk when a start click is detected.",
    )
    parser.add_argument(
        "--tts_chunk_declick_fade_ms",
        type=int,
        default=None,
        help="Fade-in length after start de-click trimming.",
    )

    openai_tts_group = parser.add_argument_group(title="openai specific")
    openai_tts_group.add_argument(
        "--speed",
        default=1.0,
        type=float,
        help="The speed of the generated audio. Select a value from 0.25 to 4.0. 1.0 is the default.",
    )

    openai_tts_group.add_argument(
        "--instructions",
        help="Instructions for the TTS model. Only supported for 'gpt-4o-mini-tts' model.",
    )
    openai_tts_group.add_argument(
        "--openai_api_key",
        help="Optional API key override for OpenAI-compatible TTS servers.",
    )
    openai_tts_group.add_argument(
        "--openai_base_url",
        help="Optional base URL override for OpenAI-compatible TTS servers.",
    )
    openai_tts_group.add_argument(
        "--openai_max_chars",
        default=None,
        type=int,
        help="Local chunk size before sending text to the OpenAI TTS provider. Set to 0 or a negative value to disable local chunking. Default: 1800.",
    )
    openai_tts_group.add_argument(
        "--openai_enable_polling",
        action="store_const",
        const=True,
        default=None,
        help="Use submit/poll/download workflow instead of standard synchronous OpenAI TTS response handling.",
    )
    openai_tts_group.add_argument(
        "--openai_submit_url",
        help="Submit endpoint for polling-based OpenAI-compatible TTS servers.",
    )
    openai_tts_group.add_argument(
        "--openai_status_url_template",
        help="Status URL template for polling-based TTS servers, for example '/tts/jobs/{job_id}'.",
    )
    openai_tts_group.add_argument(
        "--openai_download_url_template",
        help="Optional download URL template for completed jobs, for example '/tts/jobs/{job_id}/audio'.",
    )
    openai_tts_group.add_argument(
        "--openai_job_id_path",
        default=None,
        help="Dot path to job id in submit response JSON (default: id).",
    )
    openai_tts_group.add_argument(
        "--openai_job_status_path",
        default=None,
        help="Dot path to job status in polling response JSON (default: status).",
    )
    openai_tts_group.add_argument(
        "--openai_job_download_url_path",
        default=None,
        help="Dot path to download URL in polling response JSON (default: download_url).",
    )
    openai_tts_group.add_argument(
        "--openai_job_done_values",
        default=None,
        help="Comma-separated status values that mean the polling job is complete.",
    )
    openai_tts_group.add_argument(
        "--openai_job_failed_values",
        default=None,
        help="Comma-separated status values that mean the polling job has failed.",
    )
    openai_tts_group.add_argument(
        "--openai_poll_interval",
        default=None,
        type=int,
        help="Polling interval in seconds for job-based OpenAI-compatible TTS servers. Default: 120.",
    )
    openai_tts_group.add_argument(
        "--openai_poll_timeout",
        default=None,
        type=int,
        help="Maximum time in seconds to wait for a polling TTS job before failing. Default: 14400.",
    )
    openai_tts_group.add_argument(
        "--openai_poll_request_timeout",
        default=None,
        type=int,
        help="HTTP timeout in seconds for each individual polling or download request. Default: 120.",
    )
    openai_tts_group.add_argument(
        "--openai_poll_max_errors",
        default=None,
        type=int,
        help="How many consecutive transient polling/download HTTP errors to tolerate before failing. Default: 10.",
    )

    normalizer_group = parser.add_argument_group(title="normalizer specific")
    normalizer_group.add_argument(
        "--normalize",
        action="store_const",
        const=True,
        default=None,
        help="Normalize chapter text before sending it to TTS.",
    )
    normalizer_group.add_argument(
        "--normalize_steps",
        help=(
            "Comma-separated normalizer steps to apply in order. "
            "When set, --normalize_provider is ignored. "
            "Example: simple_symbols,ru_initials,ru_numbers,ru_llm_stress_ambiguity,ru_llm_proper_nouns_pronunciation,tts_llm_safe_split"
        ),
    )
    normalizer_group.add_argument(
        "--normalize_provider",
        choices=get_supported_normalizers(),
        default=None,
        help=(
            "Single-step normalizer shorthand when --normalize_steps is not set (default: openai). "
            "Superseded by --normalize_steps when both are given."
        ),
    )
    normalizer_group.add_argument(
        "--normalize_model",
        help="Model name for the LLM normalizer.",
    )
    normalizer_group.add_argument(
        "--normalize_prompt_file",
        help="Optional text file with a custom system prompt for the normalizer. Kept for backwards compatibility.",
    )
    normalizer_group.add_argument(
        "--normalize_system_prompt",
        help="System prompt for the openai normalizer (inline text, overrides normalize_system_prompt_file).",
    )
    normalizer_group.add_argument(
        "--normalize_system_prompt_file",
        help="Optional text file with a custom system prompt for the normalizer (legacy; prefer normalize_system_prompt).",
    )
    normalizer_group.add_argument(
        "--normalize_user_prompt_file",
        help="Optional text file with a custom user prompt template for the normalizer. Available placeholders: {chapter_title}, {text}.",
    )
    normalizer_group.add_argument(
        "--normalize_api_key",
        help="Optional API key override for the normalizer endpoint.",
    )
    normalizer_group.add_argument(
        "--normalize_base_url",
        help="Optional base URL override for the normalizer endpoint.",
    )
    normalizer_group.add_argument(
        "--normalize_max_chars",
        default=4000,
        type=int,
        help="Maximum characters per normalization request. Use a negative value to disable local splitting.",
    )
    normalizer_group.add_argument(
        "--normalize_tts_safe_max_chars",
        default=180,
        type=int,
        help="Maximum characters per sentence for the deterministic tts_safe_split normalizer (default: 180).",
    )
    normalizer_group.add_argument(
        "--normalize_pronunciation_exceptions_file",
        help=(
            "Deprecated alias for --normalize_tts_pronunciation_overrides_file. "
            "Use --normalize_tts_pronunciation_overrides_file instead."
        ),
    )
    normalizer_group.add_argument(
        "--normalize_tts_pronunciation_overrides_file",
        help=(
            "Optional UTF-8 file with per-line XTTS pronunciation overrides in the form "
            "'source==replacement'. Use this with tts_pronunciation_overrides."
        ),
    )
    normalizer_group.add_argument(
        "--normalize_pronunciation_lexicon_db",
        help=(
            "Optional SQLite path for the shared pronunciation/stress lexicon. "
            "If omitted, a cached project-local database is created automatically."
        ),
    )
    normalizer_group.add_argument(
        "--normalize_tts_pronunciation_overrides_words",
        help=(
            "Inline TTS pronunciation overrides as 'word=replacement,word2=replacement2'. "
            "Overrides the built-in defaults entirely when set."
        ),
    )
    normalizer_group.add_argument(
        "--normalize_stress_ambiguity_file",
        help=(
            "Optional UTF-8 file with per-line ambiguity variants in the form "
            "'source==variant1|variant2'. Variants may use combining acute accents or "
            "Silero-style plus notation such as 'б+едыбед+ы'. Use this with ru_llm_stress_ambiguity."
        ),
    )
    normalizer_group.add_argument(
        "--normalize_stress_ambiguity_system_prompt",
        help="Override the LLM system prompt for ru_llm_stress_ambiguity normalizer.",
    )
    normalizer_group.add_argument(
        "--normalize_safe_split_system_prompt",
        help="Override the LLM system prompt for tts_llm_safe_split normalizer.",
    )
    normalizer_group.add_argument(
        "--normalize_tsnorm_stress_yo",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When ru_tsnorm is enabled, restore or keep 'ё' where the backend can determine it (default: on).",
    )
    normalizer_group.add_argument(
        "--normalize_tsnorm_stress_monosyllabic",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="When ru_tsnorm is enabled, also add stress marks to monosyllabic words (default: off).",
    )
    normalizer_group.add_argument(
        "--normalize_tsnorm_min_word_length",
        default=2,
        type=int,
        help="Minimum token length for ru_tsnorm stress processing (default: 2).",
    )

    edge_tts_group = parser.add_argument_group(title="edge specific")
    edge_tts_group.add_argument(
        "--voice_rate",
        help="""
            Speaking rate of the text. Valid relative values range from -50%%(--xxx='-50%%') to +100%%. 
            For negative value use format --arg=value,
        """,
    )

    edge_tts_group.add_argument(
        "--voice_volume",
        help="""
            Volume level of the speaking voice. Valid relative values floor to -100%%.
            For negative value use format --arg=value,
        """,
    )

    edge_tts_group.add_argument(
        "--voice_pitch",
        help="""
            Baseline pitch for the text.Valid relative values like -80Hz,+50Hz, pitch changes should be within 0.5 to 1.5 times the original audio.
            For negative value use format --arg=value,
        """,
    )

    edge_tts_group.add_argument(
        "--proxy",
        help="Proxy server for the TTS provider. Format: http://[username:password@]proxy.server:port",
    )

    azure_edge_tts_group = parser.add_argument_group(title="azure/edge specific")
    azure_edge_tts_group.add_argument(
        "--break_duration",
        default="1250",
        help="Break duration in milliseconds for the different paragraphs or sections (default: 1250, means 1.25 s). Valid values range from 0 to 5000 milliseconds for Azure TTS.",
    )

    piper_tts_group = parser.add_argument_group(title="piper specific")
    piper_tts_group.add_argument(
        "--piper_path",
        default="piper",
        help="Path to the Piper TTS executable",
    )
    piper_tts_group.add_argument(
        "--piper_docker_image",
        default="lscr.io/linuxserver/piper:latest",
        help="Piper Docker image name (if using Docker)",
    )
    piper_tts_group.add_argument(
        "--piper_speaker",
        default=0,
        help="Piper speaker id, used for multi-speaker models",
    )
    piper_tts_group.add_argument(
        "--piper_sentence_silence",
        default=0.2,
        help="Seconds of silence after each sentence",
    )
    piper_tts_group.add_argument(
        "--piper_length_scale",
        default=1.0,
        help="Phoneme length, a.k.a. speaking rate",
    )

    qwen_tts_group = parser.add_argument_group(title="qwen specific")
    qwen_tts_group.add_argument(
        "--qwen_api_key",
        default=None,
        help="Aliyun DashScope API key for Qwen3 TTS. Can also be set via DASHSCOPE_API_KEY env variable.",
    )
    qwen_tts_group.add_argument(
        "--qwen_language_type",
        default=None,
        help=(
            "Language type hint for Qwen3 TTS (e.g. Russian, Chinese, English). "
            "Inferred from --language if not set. "
            "Supported: Chinese, English, Spanish, Russian, Italian, French, Korean, Japanese, German, Portuguese."
        ),
    )
    qwen_tts_group.add_argument(
        "--qwen_stream",
        action="store_true",
        default=False,
        help="Enable streaming synthesis for Qwen3 TTS (default: False).",
    )
    qwen_tts_group.add_argument(
        "--qwen_request_timeout",
        type=int,
        default=30,
        help="Request timeout in seconds for Qwen3 TTS API calls (default: 30).",
    )

    gemini_tts_group = parser.add_argument_group(title="gemini specific")
    gemini_tts_group.add_argument(
        "--gemini_api_key",
        default=None,
        help="Google AI API key for Gemini TTS. Can also be set via GOOGLE_API_KEY env variable.",
    )
    gemini_tts_group.add_argument(
        "--gemini_sample_rate",
        type=int,
        default=24000,
        help="PCM sample rate in Hz for Gemini TTS audio output (default: 24000).",
    )
    gemini_tts_group.add_argument(
        "--gemini_channels",
        type=int,
        default=1,
        choices=[1, 2],
        help="Number of audio channels for Gemini TTS output: 1=mono, 2=stereo (default: 1).",
    )
    gemini_tts_group.add_argument(
        "--gemini_audio_encoding",
        default="pcm16",
        help="PCM encoding returned by Gemini API (default: pcm16). Supported: pcm16, pcm24, pcm32.",
    )
    gemini_tts_group.add_argument(
        "--gemini_temperature",
        type=float,
        default=0.2,
        help="Sampling temperature for Gemini TTS (0.0–1.0, default: 0.2).",
    )
    gemini_tts_group.add_argument(
        "--gemini_speaker_map",
        default=None,
        help='JSON object mapping speaker names to Gemini voices, e.g. \'{"Alice":"Kore","Bob":"Puck"}\'. Used for multi-speaker synthesis.',
    )

    kokoro_tts_group = parser.add_argument_group(title="kokoro specific")
    kokoro_tts_group.add_argument(
        "--kokoro_base_url",
        default="http://localhost:8880",
        help="Base URL of the Kokoro-FastAPI server (default: http://localhost:8880).",
    )
    kokoro_tts_group.add_argument(
        "--kokoro_volume_multiplier",
        type=float,
        default=1.0,
        help="Volume multiplier for Kokoro TTS output (default: 1.0).",
    )

    args = parser.parse_args()

    # Auto-discover and merge INI configs (CLI args always take priority).
    # Order: global ~/.config/epub_to_audiobook/config.ini
    #        → per-book <book_dir>/<book_stem>.ini
    #        → explicit --config
    from audiobook_generator.config.ini_config_manager import load_merged_ini, merge_ini_into_args
    ini_values = load_merged_ini(
        input_file=getattr(args, "input_file", None),
        explicit_config=getattr(args, "config", None),
    )
    if ini_values:
        merge_ini_into_args(args, ini_values)

    # Apply defaults for args that weren't set via CLI or INI
    if not getattr(args, "tts", None):
        args.tts = get_supported_tts_providers()[0]  # default: azure
    if not getattr(args, "language", None):
        args.language = "en-US"
    if not getattr(args, "newline_mode", None):
        args.newline_mode = "double"
    if not getattr(args, "title_mode", None):
        args.title_mode = "auto"
    if not getattr(args, "chapter_mode", None):
        args.chapter_mode = "documents"
    if getattr(args, "chapter_start", None) is None:
        args.chapter_start = 1
    if getattr(args, "chapter_end", None) is None:
        args.chapter_end = -1
    if getattr(args, "worker_count", None) is None:
        args.worker_count = 1
    if not getattr(args, "normalize", None):
        args.normalize = False
    if not getattr(args, "openai_enable_polling", None):
        args.openai_enable_polling = False
    if getattr(args, "openai_max_chars", None) is None:
        args.openai_max_chars = 1800
    if not getattr(args, "normalize_provider", None):
        args.normalize_provider = "openai"

    return GeneralConfig(args)


def _run_audio_auto(config):
    """Auto-loop mode: synthesise → check → delete failed chunks → repeat.

    Runs up to ``audio_auto_retry`` iterations.  In each iteration:
      1. Synthesise any missing/deleted chunks (audio_chunks mode, skips existing files).
      2. Transcribe all chunks with Whisper and record similarity scores.
      3. Delete chunk audio files whose similarity < ``audio_auto_check_threshold``.
         Each deletion is recorded in the DB so the Review UI can show retry counts.
      4. If no chunks were deleted → all pass; stop.
      5. After ``audio_auto_retry`` retries → warn and stop (remaining bad chunks
         stay visible in the Review UI as disputed for manual inspection).
    """
    import copy
    import logging as _logging
    from pathlib import Path as _Path

    from audiobook_generator.core.audio_checker import AudioChecker
    from audiobook_generator.core.audio_chunk_store import AudioChunkStore
    from audiobook_generator.core.audiobook_generator import AudiobookGenerator

    _log = _logging.getLogger(__name__)

    auto_threshold = float(getattr(config, "audio_auto_check_threshold", None) or 0.78)
    max_retry = int(getattr(config, "audio_auto_retry", None) or 5)

    output_folder = _Path(config.output_folder).resolve()
    db_path = output_folder / "wav" / "_state" / "audio_chunks.sqlite3"

    _log.info(
        "=== audio_auto: threshold=%.2f  max_retry=%d ===",
        auto_threshold, max_retry,
    )

    _AUDIO_EXTS = ["wav", "mp3", "ogg", "opus", "m4a", "flac"]
    chunks_root = output_folder / "wav" / "chunks"

    for attempt in range(1, max_retry + 2):   # attempt 1 … max_retry+1
        _log.info("--- audio_auto: attempt %d ---", attempt)

        # ── Step 1: Synthesise ────────────────────────────────────────────────
        synth_cfg = copy.copy(config)
        synth_cfg.mode = "audio_chunks"
        synth_cfg.package_m4b = False
        synth_cfg.normalize = False
        synth_cfg.chunked_audio = True
        AudiobookGenerator(synth_cfg).run()

        # ── Step 2: audio_check ───────────────────────────────────────────────
        if not db_path.exists():
            raise FileNotFoundError(
                f"No audio DB at {db_path}. Run --mode audio_chunks first."
            )
        store = AudioChunkStore(db_path)
        checker = AudioChecker(
            output_folder=output_folder,
            model_size=getattr(config, "audio_check_model", None) or "small",
            language=getattr(config, "language", "ru"),
            threshold=auto_threshold,
            device=getattr(config, "audio_check_device", None) or "cpu",
            config=config,
        )
        checker.run(store)

        # ── Step 3: Find failed chunks ────────────────────────────────────────
        failed_rows = store.get_all_failed_chunks(auto_threshold)
        # Keep only rows whose audio file actually exists on disk
        failed: list[tuple] = []
        for row in failed_rows:
            chapter_key = row["chapter_key"]
            s_hash = row["sentence_hash"]
            similarity = row["similarity"]
            for ext in _AUDIO_EXTS:
                p = chunks_root / chapter_key / f"{s_hash}.{ext}"
                if p.exists():
                    failed.append((chapter_key, s_hash, similarity, str(p)))
                    break

        if not failed:
            _log.info(
                "=== audio_auto: ✅ all chunks pass threshold %.2f after %d attempt(s) ===",
                auto_threshold, attempt,
            )
            break

        if attempt > max_retry:
            _log.warning(
                "=== audio_auto: ⚠ max retries (%d) reached; "
                "%d chunk(s) still below threshold %.2f — manual review recommended ===",
                max_retry, len(failed), auto_threshold,
            )
            break

        # ── Step 4: Delete failed chunks & record in history ──────────────────
        deleted = 0
        for chapter_key, s_hash, similarity, audio_path in failed:
            try:
                _Path(audio_path).unlink()
                store.record_auto_deletion(chapter_key, s_hash, similarity, auto_threshold)
                deleted += 1
                _log.info(
                    "  Deleted %s/%s (sim=%.2f < %.2f) — queued for re-synthesis",
                    chapter_key, s_hash[:10], similarity, auto_threshold,
                )
            except Exception as exc:
                _log.warning("  Failed to delete %s: %s", audio_path, exc)

        _log.info(
            "--- audio_auto: deleted %d chunk(s), starting attempt %d ---",
            deleted, attempt + 1,
        )

    # Final audio_check at the normal threshold so the Review UI reflects the
    # true (possibly lower) threshold configured by the user.
    normal_threshold = float(getattr(config, "audio_check_threshold", None) or 0.70)
    if normal_threshold != auto_threshold:
        _log.info(
            "=== audio_auto: running final audio_check at normal threshold %.2f ===",
            normal_threshold,
        )
        store2 = AudioChunkStore(db_path)
        checker2 = AudioChecker(
            output_folder=output_folder,
            model_size=getattr(config, "audio_check_model", None) or "small",
            language=getattr(config, "language", "ru"),
            threshold=normal_threshold,
            device=getattr(config, "audio_check_device", None) or "cpu",
            config=config,
        )
        checker2.run(store2)


def _run_audio_check(config):
    """Run audio_check mode: transcribe chunks and mark disputed ones."""
    from audiobook_generator.core.audio_checker import AudioChecker
    from audiobook_generator.core.audio_chunk_store import AudioChunkStore

    output_folder = Path(config.output_folder).resolve()
    db_path = output_folder / "wav" / "_state" / "audio_chunks.sqlite3"
    if not db_path.exists():
        raise FileNotFoundError(
            f"No audio DB found at {db_path}. "
            "Run --mode audio first to synthesise audio chunks."
        )
    store = AudioChunkStore(db_path)

    checker = AudioChecker(
        output_folder=output_folder,
        model_size=getattr(config, "audio_check_model", None) or "small",
        language=getattr(config, "language", "ru"),
        threshold=getattr(config, "audio_check_threshold", None) or 0.70,
        device=getattr(config, "audio_check_device", None) or "cpu",
        config=config,
    )
    checker.run(store)


def _run_audio_worker(config):
    """Run audio_worker mode: synthesise missing chunks in an infinite loop."""
    import copy
    import logging
    import time

    _log = logging.getLogger(__name__)

    poll_interval = int(getattr(config, "audio_worker_interval", None) or 30)

    print(f"\n[audio_worker] Book : {config.output_folder}")
    print(f"[audio_worker] Interval: {poll_interval}s between passes")
    print("[audio_worker] Press Ctrl+C to stop (current chunk will finish first).\n")

    pass_num = 0
    try:
        while True:
            pass_num += 1
            _log.info("=== audio_worker: pass %d ===", pass_num)

            synth_cfg = copy.copy(config)
            synth_cfg.mode = "audio_chunks"
            synth_cfg.package_m4b = False
            synth_cfg.normalize = False
            synth_cfg.no_prompt = True

            try:
                from audiobook_generator.core.audiobook_generator import AudiobookGenerator
                AudiobookGenerator(synth_cfg).run()
            except KeyboardInterrupt:
                raise  # propagate to outer handler
            except Exception as exc:
                _log.error("audio_worker: pass %d error: %s", pass_num, exc)

            _log.info(
                "=== audio_worker: pass %d done — sleeping %ds ===",
                pass_num, poll_interval,
            )
            time.sleep(poll_interval)

    except KeyboardInterrupt:
        print(f"\n[audio_worker] Stopped after {pass_num} pass(es).")


def main(config=None, log_file=None):
    if not config: # config passed from UI, or uses args if CLI
        config = handle_args()

    if log_file:
        # If log_file is provided (e.g., from UI), use it directly as a Path object.
        # The UI passes an absolute path string.
        effective_log_file = Path(log_file)
    else:
        # Otherwise (e.g., CLI usage without a specific log file from UI),
        # keep logs inside the selected output folder so each run stays self-contained.
        base_dir = Path(config.output_folder) if getattr(config, "output_folder", None) else None
        effective_log_file = generate_unique_log_path("EtA", base_dir=base_dir)
    
    # Ensure config.log_file is updated, as it's used by AudiobookGenerator for worker processes.
    config.log_file = effective_log_file

    setup_logging(config.log, str(effective_log_file))

    if getattr(config, "mode", None) == "audio_check":
        _run_audio_check(config)
        return

    if getattr(config, "mode", None) == "audio_auto":
        _run_audio_auto(config)
        return

    if getattr(config, "mode", None) == "audio_worker":
        _run_audio_worker(config)
        return

    AudiobookGenerator(config).run()


if __name__ == "__main__":
    main()
