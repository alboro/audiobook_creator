# macOS Setup

These steps were verified locally on macOS on April 14, 2026.

## Prerequisites

- Homebrew
- `ffmpeg`
- Python 3.12 or 3.13 preferred

Python 3.14 can install and run the project, but the `openai` package currently emits a compatibility warning because of legacy Pydantic V1 internals. For a quieter setup, prefer Python 3.12 or 3.13 if you already have one.

## Fast path

From the repository root:

```bash
bash scripts/setup_macos.sh
bash scripts/run_preview_macos.sh
```

The preview command uses the bundled `Robinson Crusoe` example and `edge` preview settings so it works without any API keys.

## Manual setup

```bash
brew install ffmpeg
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

## Verified preview command

```bash
.venv/bin/python main.py \
  "examples/The_Life_and_Adventures_of_Robinson_Crusoe.epub" \
  "/tmp/epub_to_audiobook_preview" \
  --preview \
  --tts edge \
  --voice_name en-US-AriaNeural \
  --no_prompt
```

## OpenAI-compatible example

Adjust the URLs and model names for your local services:

```bash
.venv/bin/python main.py \
  "examples/The_Life_and_Adventures_of_Robinson_Crusoe.epub" \
  "/tmp/epub_to_audiobook_out" \
  --tts openai \
  --model_name gpt-4o-mini-tts \
  --voice_name alloy \
  --output_format mp3 \
  --openai_base_url "http://127.0.0.1:8000/v1" \
  --openai_enable_polling \
  --openai_submit_url "/audio/speech" \
  --openai_status_url_template "/audio/speech/{job_id}" \
  --openai_job_id_path id \
  --openai_job_status_path status \
  --openai_job_download_url_path download_url \
  --openai_poll_interval 120 \
  --openai_poll_timeout 14400 \
  --openai_max_chars 0 \
  --normalize \
  --normalize_provider openai \
  --normalize_model gpt-4.1-mini \
  --normalize_base_url "http://127.0.0.1:1234/v1" \
  --normalize_max_chars 4000 \
  --package_m4b \
  --ffmpeg_path "$(command -v ffmpeg)" \
  --worker_count 1 \
  --no_prompt
```

## Notes

- `--openai_max_chars 0` disables local TTS chunking and sends whole chapters to the TTS provider.
- `--normalize_max_chars 4000` keeps the normalizer conservative. Increase it only if your LLM endpoint is happy with larger requests.
- Even in `--preview`, the `openai` provider initializes immediately. That means preview mode still expects either `OPENAI_API_KEY` or an explicit `--openai_base_url` plus the fork's dummy-key fallback path.
