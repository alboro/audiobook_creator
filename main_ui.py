import argparse

from audiobook_generator.config.ini_config_manager import load_merged_ini, merge_ini_into_args
from audiobook_generator.config.ui_config import UiConfig
from audiobook_generator.ui.web_ui import host_ui
from audiobook_generator.ui.review_server import host_review_ui_fastapi


def handle_args():
    parser = argparse.ArgumentParser(description="WebUI for Epub to Audiobook convertor")
    parser.add_argument("--host", default="127.0.0.1", help="Host address")
    parser.add_argument("--port", type=int, help="Port number (default: 7860 for generation, 7861 for review)")
    parser.add_argument("--review", action="store_true", help="Launch review UI instead of generation UI")
    parser.add_argument("--audio_folder", default=None, help="Optional audio folder override for review/package flows")

    ui_args = parser.parse_args()
    merge_ini_into_args(ui_args, load_merged_ini())
    return UiConfig(ui_args)

def main():
    config = handle_args()
    if config.review:
        host_review_ui_fastapi(config)
    else:
        host_ui(config)

if __name__ == "__main__":
    main()


