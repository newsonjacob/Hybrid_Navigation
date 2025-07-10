import logging
import logging.config
from pathlib import Path
from typing import Optional

def setup_logging(log_file: Optional[str] = None, level: int = logging.INFO) -> None:
    """Configure root logging with optional file output."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    handlers = {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "level": level,
            "stream": "ext://sys.stdout",
        }
    }
    if log_file:
        handlers["file"] = {
            "class": "logging.FileHandler",
            "formatter": "default",
            "level": level,
            "filename": str(log_dir / log_file),
            "encoding": "utf-8",
            "mode": "w",
        }
        root_handlers = ["console", "file"]
    else:
        root_handlers = ["console"]

    logging_config = {
        "version": 1,
        "formatters": {
            "default": {"format": "%(asctime)s %(levelname)s: %(message)s"}
        },
        "handlers": handlers,
        "root": {"level": level, "handlers": root_handlers},
    }

    logging.config.dictConfig(logging_config)
