import logging
from pathlib import Path

def configure_file_logger(filename: str) -> logging.Logger:
    """Return a logger writing to logs/filename and set as root logger."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    logger = logging.getLogger(Path(filename).stem)
    logger.setLevel(logging.INFO)

    # Clear existing handlers to avoid duplicate messages
    logger.handlers.clear()

    handler = logging.FileHandler(log_dir / filename, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False

    # --- Set as root logger handlers and level ---
    logging.root.handlers = logger.handlers
    logging.root.setLevel(logger.level)

    return logger