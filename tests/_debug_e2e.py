"""Temporary debug script for e2e test (delete after use)."""
import logging
import os
import tempfile
import shutil
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

logging.basicConfig(level=logging.WARNING)

EPUB_PATH = Path(__file__).parent.parent / "examples" / "The_Life_and_Adventures_of_Robinson_Crusoe.epub"


def _make_dummy_wav(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample_rate = 22050
    n_frames = sample_rate // 10
    with wave.open(str(output_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * n_frames)


class _MockTTSProvider:
    def __init__(self, config):
        self.config = config

    def validate_config(self):
        pass

    def text_to_speech(self, text, output_path, audio_tags=None):
        print(f"  [MockTTS] text_to_speech: {output_path}")
        _make_dummy_wav(Path(output_path))

    def estimate_cost(self, total_chars):
        return 0.0

    def get_break_string(self):
        return "\n\n"

    def get_output_file_extension(self):
        return "wav"

    def prepare_tts_text(self, text):
        return text


def build_args(**overrides):
    values = dict(
        input_file=str(EPUB_PATH),
        output_folder=None,
        mode=None,
        chapter_end=2,
        chapter_start=1,
        output_text=False,
        prepare_text=False,
        prepared_text_folder=None,
        log="DEBUG",
        no_prompt=True,
        worker_count=1,
        use_pydub_merge=False,
        force_new_run=False,
        package_m4b=False,
        chunked_audio=True,
        audio_folder=None,
        m4b_filename=None,
        m4b_bitrate="64k",
        ffmpeg_path="ffmpeg",
        title_mode="auto",
        chapter_mode="documents",
        newline_mode="double",
        search_and_replace_file="",
        tts="openai",
        language="ru-RU",
        voice_name="reference",
        output_format="wav",
        model_name="xtts_v2",
        openai_api_key=None,
        openai_base_url=None,
        openai_max_chars=0,
        openai_enable_polling=False,
        openai_submit_url=None,
        openai_status_url_template=None,
        openai_download_url_template=None,
        openai_job_id_path="id",
        openai_job_status_path="status",
        openai_job_download_url_path="download_url",
        openai_job_done_values="done,completed",
        openai_job_failed_values="failed,error",
        openai_poll_interval=5,
        openai_poll_timeout=60,
        openai_poll_request_timeout=60,
        openai_poll_max_errors=3,
        instructions=None,
        speed=1.0,
        normalize=False,
        normalize_steps=None,
        normalize_provider=None,
        normalize_model=None,
        normalize_prompt_file=None,
        normalize_system_prompt_file=None,
        normalize_user_prompt_file=None,
        normalize_api_key=None,
        normalize_base_url=None,
        normalize_max_chars=4000,
        normalize_tts_safe_max_chars=180,
        normalize_pronunciation_exceptions_file=None,
        normalize_tts_pronunciation_overrides_file=None,
        normalize_pronunciation_lexicon_db=None,
        normalize_stress_ambiguity_file=None,
        normalize_tsnorm_stress_yo=True,
        normalize_tsnorm_stress_monosyllabic=False,
        normalize_tsnorm_min_word_length=2,
        normalize_stress_paradox_words=None,
        normalize_log_changes=False,
        normalize_stress_ambiguity_system_prompt=None,
        normalize_safe_split_system_prompt=None,
        break_duration="1250",
        voice_rate=None,
        voice_volume=None,
        voice_pitch=None,
        proxy=None,
        piper_path="piper",
        piper_docker_image="piper:latest",
        piper_speaker=0,
        piper_noise_scale=None,
        piper_noise_w_scale=None,
        piper_length_scale=1.0,
        piper_sentence_silence=0.2,
        qwen_api_key=None,
        qwen_language_type=None,
        qwen_stream=False,
        qwen_request_timeout=60,
        gemini_api_key=None,
        gemini_sample_rate=24000,
        gemini_channels=1,
        gemini_audio_encoding="LINEAR16",
        gemini_temperature=0.7,
        gemini_speaker_map=None,
        kokoro_base_url=None,
        kokoro_volume_multiplier=None,
    )
    values.update(overrides)
    return MagicMock(**values)


if __name__ == "__main__":
    from audiobook_generator.config.general_config import GeneralConfig
    from audiobook_generator.core.audiobook_generator import AudiobookGenerator

    tmp = Path(tempfile.mkdtemp(prefix="eta_debug_"))
    output_dir = tmp / "book_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"tmp={tmp}")

    try:
        # Step 1: prepare
        args1 = build_args(output_folder=str(output_dir), mode="prepare")
        config1 = GeneralConfig(args1)
        AudiobookGenerator(config1).run()
        txt_files = sorted((output_dir / "text" / "001").glob("*.txt"))
        print(f"Prepare: {len(txt_files)} txt files")

        # Step 2: audio (enable error logging to see exceptions)
        logging.getLogger("audiobook_generator").setLevel(logging.ERROR)
        args2 = build_args(output_folder=str(output_dir), mode="audio", chunked_audio=True)
        config2 = GeneralConfig(args2)

        with patch(
            "audiobook_generator.core.audiobook_generator.get_tts_provider",
            side_effect=lambda cfg: _MockTTSProvider(cfg),
        ):
            AudiobookGenerator(config2).run()

        db_path = output_dir / "wav" / "_state" / "audio_chunks.sqlite3"
        print(f"DB exists: {db_path.exists()}")
        print("Files in wav/:")
        for root, dirs, files in os.walk(output_dir / "wav"):
            for f in files:
                print(f"  {Path(root)/f}")

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

