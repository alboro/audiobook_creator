import io
import logging
import math
import os
import time
from urllib.parse import urljoin

from openai import OpenAI
import requests

from audiobook_generator.core.audio_tags import AudioTags
from audiobook_generator.config.general_config import GeneralConfig
from audiobook_generator.utils.chunk_boundaries import split_text_on_explicit_chunk_boundaries
from audiobook_generator.utils.utils import split_text, set_audio_tags, merge_audio_segments
from audiobook_generator.tts_providers.base_tts_provider import BaseTTSProvider


logger = logging.getLogger(__name__)
TRANSIENT_HTTP_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}


def get_openai_supported_output_formats():
    return ["mp3", "aac", "flac", "opus", "wav"]

def get_openai_supported_voices():
    return ["alloy", "ash", "ballad", "coral", "echo", "fable", "onyx", "nova", "sage", "shimmer", "verse"]

def get_openai_supported_models():
    return ["gpt-4o-mini-tts", "tts-1", "tts-1-hd"]

def get_openai_instructions_example():
    return """Voice Affect: Calm, composed, and reassuring. Competent and in control, instilling trust.
Tone: Sincere, empathetic, with genuine concern for the customer and understanding of the situation.
Pacing: Slower during the apology to allow for clarity and processing. Faster when offering solutions to signal action and resolution.
Emotions: Calm reassurance, empathy, and gratitude.
Pronunciation: Clear, precise: Ensures clarity, especially with key details. Focus on key words like 'refund' and 'patience.' 
Pauses: Before and after the apology to give space for processing the apology."""

def get_price(model):
    # https://platform.openai.com/docs/pricing#transcription-and-speech-generation
    if model == "tts-1": # $15 per 1 mil chars
        return 0.015
    elif model == "tts-1-hd": # $30 per 1 mil chars
        return 0.03
    elif model == "gpt-4o-mini-tts": # $12 per 1 mil tokens (not chars, as 1 token is ~4 chars)
        return 0.003 # TODO: this could be very wrong for Chinese. Not sure how openai calculates the audio token count.
    else:
        logger.debug(f"OpenAI: Unsupported model name: {model}, unable to retrieve the price")
        return 0.0


class OpenAITTSProvider(BaseTTSProvider):
    def __init__(self, config: GeneralConfig):
        config.model_name = config.model_name or "gpt-4o-mini-tts" # default to this model as it's the cheapest
        config.voice_name = config.voice_name or "alloy"
        config.speed = config.speed or 1.0
        config.instructions = config.instructions or None
        config.output_format = config.output_format or "mp3"
        config.openai_max_chars = 1800 if config.openai_max_chars is None else config.openai_max_chars
        config.openai_job_id_path = config.openai_job_id_path or "id"
        config.openai_job_status_path = config.openai_job_status_path or "status"
        config.openai_job_download_url_path = config.openai_job_download_url_path or "download_url"
        config.openai_job_done_values = config.openai_job_done_values or "done,completed,succeeded,success"
        config.openai_job_failed_values = config.openai_job_failed_values or "failed,error,cancelled"
        config.openai_poll_interval = config.openai_poll_interval or 120
        config.openai_poll_timeout = config.openai_poll_timeout or 14400
        config.openai_poll_request_timeout = config.openai_poll_request_timeout or 120
        config.openai_poll_max_errors = config.openai_poll_max_errors or 10

        self.price = get_price(config.model_name)
        super().__init__(config)

        self.base_url = config.openai_base_url or os.getenv("OPENAI_BASE_URL")
        self.api_key = config.openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            self.api_key = "dummy"

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            max_retries=4,
        )
        self.http_session = self._build_http_session()

    def __str__(self) -> str:
        return super().__str__()

    def text_to_speech(self, text: str, output_file: str, audio_tags: AudioTags):
        max_chars = self.config.openai_max_chars
        if max_chars is not None and max_chars <= 0:
            text_chunks = split_text_on_explicit_chunk_boundaries(text, min_chars=1)
        else:
            text_chunks = split_text(text, max_chars, self.config.language)

        audio_segments = []
        chunk_ids = []

        for i, chunk in enumerate(text_chunks, 1):
            chunk_id = f"chapter-{audio_tags.idx}_{audio_tags.title}_chunk_{i}_of_{len(text_chunks)}"
            logger.info(
                f"Processing {chunk_id}, length={len(chunk)}"
            )
            logger.debug(
                f"Processing {chunk_id}, length={len(chunk)}, text=[{chunk}]"
            )

            audio_segments.append(io.BytesIO(self._synthesize_chunk(chunk, chunk_id)))
            chunk_ids.append(chunk_id)

        # Use utility function to merge audio segments
        merge_audio_segments(audio_segments, output_file, self.config.output_format, chunk_ids, self.config.use_pydub_merge)

        set_audio_tags(output_file, audio_tags)

    def get_break_string(self):
        return "   "

    def get_output_file_extension(self):
        return self.config.output_format

    def validate_config(self):
        if self.config.output_format not in get_openai_supported_output_formats():
            raise ValueError(f"OpenAI: Unsupported output format: {self.config.output_format}")
        if self.config.speed < 0.25 or self.config.speed > 4.0:
            raise ValueError(f"OpenAI: Unsupported speed: {self.config.speed}")
        if self.config.instructions and len(self.config.instructions) > 0 and self.config.model_name != "gpt-4o-mini-tts":
            raise ValueError(f"OpenAI: Instructions are only supported for 'gpt-4o-mini-tts' model")
        if self.config.openai_poll_request_timeout <= 0:
            raise ValueError("OpenAI: openai_poll_request_timeout must be positive")
        if self.config.openai_poll_max_errors < 0:
            raise ValueError("OpenAI: openai_poll_max_errors must be zero or positive")
        if self.config.openai_enable_polling:
            if not self.config.openai_submit_url:
                raise ValueError("OpenAI polling mode requires --openai-submit-url")
            if not self.config.openai_status_url_template:
                raise ValueError("OpenAI polling mode requires --openai-status-url-template")

    def estimate_cost(self, total_chars):
        return math.ceil(total_chars / 1000) * self.price

    def _synthesize_chunk(self, chunk: str, chunk_id: str) -> bytes:
        if self.config.openai_enable_polling:
            return self._synthesize_chunk_with_polling(chunk, chunk_id)
        return self._synthesize_chunk_sync(chunk)

    def _synthesize_chunk_sync(self, chunk: str) -> bytes:
        response = self.client.audio.speech.create(
            model=self.config.model_name,
            voice=self.config.voice_name,
            speed=self.config.speed,
            instructions=self.config.instructions,
            input=chunk,
            response_format=self.config.output_format,
        )

        logger.debug(
            "Remote server response: status_code=%s, size=%s bytes, content=%s...",
            response.response.status_code,
            len(response.content),
            response.content[:128],
        )
        return response.content

    def _synthesize_chunk_with_polling(self, chunk: str, chunk_id: str) -> bytes:
        submit_url = self._resolve_url(self.config.openai_submit_url)
        payload = {
            "model": self.config.model_name,
            "voice": self.config.voice_name,
            "speed": self.config.speed,
            "instructions": self.config.instructions,
            "input": chunk,
            "response_format": self.config.output_format,
        }
        # Remove fields listed in openai_submit_omit_fields (comma-separated)
        omit_raw = getattr(self.config, "openai_submit_omit_fields", None)
        if omit_raw:
            for field in (f.strip() for f in omit_raw.split(",") if f.strip()):
                payload.pop(field, None)
        # Merge extra fields from openai_submit_extra_fields (JSON string)
        extra_raw = getattr(self.config, "openai_submit_extra_fields", None)
        if extra_raw:
            import json as _json
            try:
                extra = _json.loads(extra_raw)
                if isinstance(extra, dict):
                    payload.update(extra)
            except Exception as exc:
                logger.warning("openai_submit_extra_fields: failed to parse JSON: %s", exc)
        # Drop None values (not sent to server)
        payload = {k: v for k, v in payload.items() if v is not None}
        logger.info("Submitting polling TTS job for %s to %s", chunk_id, submit_url)
        submit_response = self._request_with_retries(
            "POST",
            submit_url,
            operation=f"submit TTS job for {chunk_id}",
            timeout=self.config.openai_poll_request_timeout,
            json=payload,
        )
        submit_json = submit_response.json()
        job_id = self._extract_json_path(submit_json, self.config.openai_job_id_path)
        if not job_id:
            raise RuntimeError(
                f"Polling TTS submit response did not include job id at path '{self.config.openai_job_id_path}'"
            )

        done_values = self._split_csv(self.config.openai_job_done_values)
        failed_values = self._split_csv(self.config.openai_job_failed_values)
        poll_started = time.monotonic()
        last_status = None

        while True:
            if time.monotonic() - poll_started > self.config.openai_poll_timeout:
                raise TimeoutError(
                    f"Polling TTS job '{job_id}' did not finish within {self.config.openai_poll_timeout} seconds"
                )

            status_url = self._format_template(self.config.openai_status_url_template, job_id)
            logger.info("Polling TTS job %s at %s", job_id, status_url)
            status_response = self._request_with_retries(
                "GET",
                status_url,
                operation=f"poll TTS job {job_id}",
                timeout=self.config.openai_poll_request_timeout,
            )
            status_json = status_response.json()

            status_value = str(
                self._extract_json_path(status_json, self.config.openai_job_status_path)
            ).strip().lower()
            if status_value and status_value != last_status:
                logger.info("Polling TTS job %s status changed to '%s'", job_id, status_value)
                last_status = status_value

            if status_value in done_values:
                download_url = None
                if self.config.openai_download_url_template:
                    download_url = self._format_template(
                        self.config.openai_download_url_template,
                        job_id,
                    )
                else:
                    download_url = self._extract_json_path(
                        status_json,
                        self.config.openai_job_download_url_path,
                    )
                if not download_url:
                    raise RuntimeError(
                        "Polling TTS job finished but no download URL was found. "
                        f"Set --openai-download-url-template or adjust --openai-job-download-url-path."
                    )
                logger.info("Downloading completed TTS audio for job %s from %s", job_id, download_url)
                download_response = self._request_with_retries(
                    "GET",
                    self._resolve_url(download_url),
                    operation=f"download TTS audio for job {job_id}",
                    timeout=max(600, self.config.openai_poll_request_timeout),
                )
                return download_response.content

            if status_value in failed_values:
                error_detail = (
                    self._extract_json_path(status_json, "error")
                    or self._extract_json_path(status_json, "detail")
                    or self._extract_json_path(status_json, "message")
                )
                if isinstance(error_detail, (dict, list)):
                    import json as _json
                    error_detail = _json.dumps(error_detail, ensure_ascii=False)
                suffix = f": {error_detail}" if error_detail else ""
                raise RuntimeError(
                    f"Polling TTS job '{job_id}' failed with status '{status_value}'{suffix}"
                )

            time.sleep(self.config.openai_poll_interval)

    def _build_http_session(self) -> requests.Session:
        session = requests.Session()
        if self.api_key:
            session.headers["Authorization"] = f"Bearer {self.api_key}"
        session.headers["Content-Type"] = "application/json"
        return session

    def _reset_http_session(self) -> None:
        try:
            self.http_session.close()
        except Exception:
            pass
        self.http_session = self._build_http_session()

    def _request_with_retries(self, method: str, url: str, *, operation: str, timeout: int, **kwargs):
        max_attempts = max(1, int(self.config.openai_poll_max_errors) + 1)
        headers = dict(kwargs.pop("headers", {}) or {})
        headers.setdefault("Connection", "close")
        headers.setdefault("Cache-Control", "no-cache")
        kwargs["headers"] = headers

        last_error = None
        for attempt in range(1, max_attempts + 1):
            try:
                response = self.http_session.request(method, url, timeout=timeout, **kwargs)
                if response.status_code in TRANSIENT_HTTP_STATUS_CODES:
                    raise requests.HTTPError(
                        f"Transient HTTP {response.status_code} during {operation}",
                        response=response,
                    )
                response.raise_for_status()
                return response
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
                if attempt == max_attempts:
                    break
                self._reset_http_session()
                delay = min(
                    max(float(self.config.openai_poll_interval), 1.0),
                    0.5 * (2 ** (attempt - 1)),
                )
                logger.warning(
                    "%s failed on attempt %s/%s: %s. Retrying in %.1fs",
                    operation,
                    attempt,
                    max_attempts,
                    exc,
                    delay,
                )
                time.sleep(delay)

        raise RuntimeError(
            f"{operation} failed after {max_attempts} attempts: {last_error}"
        ) from last_error

    def _resolve_url(self, url: str) -> str:
        if url.startswith(("http://", "https://")) or not self.base_url:
            return url
        return urljoin(self.base_url.rstrip("/") + "/", url.lstrip("/"))

    @staticmethod
    def _extract_json_path(data, path: str):
        current = data
        for part in path.split("."):
            if isinstance(current, list):
                try:
                    current = current[int(part)]
                except (TypeError, ValueError, IndexError):
                    return None
            elif isinstance(current, dict):
                current = current.get(part)
            else:
                return None
            if current is None:
                return None
        return current

    @staticmethod
    def _split_csv(value: str):
        return {item.strip().lower() for item in value.split(",") if item.strip()}

    def _format_template(self, template: str, job_id: str) -> str:
        return self._resolve_url(template.format(job_id=job_id))
