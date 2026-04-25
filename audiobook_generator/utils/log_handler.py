import logging
import datetime
import sys
from pathlib import Path

def get_formatter(is_worker):
    if is_worker:
        return logging.Formatter(
            "%(asctime)s - [Worker-%(process)d] - %(filename)s:%(lineno)d - %(funcName)s - %(levelname)s - %(message)s"
        )
    else:
        return logging.Formatter(
            "%(asctime)s - %(filename)s:%(lineno)d - %(funcName)s - %(levelname)s - %(message)s"
        )

def setup_logging(log_level, log_file=None, is_worker=False):
    formatter = get_formatter(is_worker)
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (AttributeError, OSError, ValueError):
                pass

    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    root_logger.setLevel(log_level)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    if not log_file:
        log_file = generate_unique_log_path("app") # Default prefix "app"

    # Ensure the directory for the log file exists
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(log_file, encoding="utf-8-sig")
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

def generate_unique_log_path(prefix: str, base_dir: str | Path | None = None) -> Path:
    """Generates a unique log file path with a timestamp."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = Path(base_dir) / "logs" if base_dir else Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / f"{prefix}_{timestamp}.log"
