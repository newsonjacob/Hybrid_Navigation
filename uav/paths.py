from pathlib import Path

"""Common filesystem paths used across modules."""

FLAGS_DIR = Path("flags")
FLAGS_DIR.mkdir(exist_ok=True)

STOP_FLAG_PATH = FLAGS_DIR / "stop.flag"

__all__ = ["FLAGS_DIR", "STOP_FLAG_PATH"]
