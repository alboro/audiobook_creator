from unittest.mock import MagicMock
from audiobook_generator.config.general_config import GeneralConfig


def get_azure_config():
    args = MagicMock(
        input_file='examples/The_Life_and_Adventures_of_Robinson_Crusoe.epub',
        output_folder='output',
        output_text=False,
        title_mode='auto',
        chapter_mode='documents',
        log='INFO',
        newline_mode='double',
        chapter_start=1,
        chapter_end=-1,
        search_and_replace_file=None,
        tts='azure',
        language='en-US',
        voice_name='en-US-GuyNeural',
        output_format='audio-24khz-48kbitrate-mono-mp3',
        model_name='',
        break_duration='1250'
    )
    return GeneralConfig(args)


def get_openai_config():
    args = MagicMock(
        input_file='examples/The_Life_and_Adventures_of_Robinson_Crusoe.epub',
        output_folder='output',
        output_text=False,
        title_mode='auto',
        chapter_mode='documents',
        log='INFO',
        newline_mode='double',
        chapter_start=1,
        chapter_end=-1,
        search_and_replace_file=None,
        tts='openai',
        language='en-US',
        voice_name='echo',
        output_format='mp3',
        model_name='tts-1',
        speed=1.0,
        instructions=None,
        openai_api_key=None,
        openai_base_url=None,
        openai_max_chars=4000,
        openai_enable_polling=False,
        openai_submit_url=None,
        openai_status_url_template=None,
        openai_download_url_template=None,
        openai_job_id_path=None,
        openai_job_status_path=None,
        openai_job_download_url_path=None,
        openai_job_done_values=None,
        openai_job_failed_values=None,
        openai_poll_interval=2,
        openai_poll_timeout=300,
        openai_poll_request_timeout=30,
        openai_poll_max_errors=10,
    )
    return GeneralConfig(args)
