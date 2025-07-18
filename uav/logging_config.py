import logging
import logging.config
from pathlib import Path
from typing import Optional, Dict
from datetime import datetime

def setup_logging(
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    module_logs: Optional[Dict[str, str]] = None
) -> None:
    """ README:
        -----------
        Configure logging with a global log file and optional per-module logs.

        module_logs: dict like {"uav.nav_loop": "nav_output.log"}

        Modules can use:
            logger = logging.getLogger(__name__)

        --- How to use: ---

        Log globally:
            setup_logging(log_file="launch.log")
            logger = logging.getLogger("main")
            logger.info("Only written to logs/launch.log")

        Log to dedicated file:
            setup_logging(
                log_file="launch.log",
                module_files={"slam_bridge": "slam_output.log"}
            )
            logger = logging.getLogger("slam_bridge")

        Print to console:
            print("Only seen in terminal.")
    """

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Fallback global log
    if not log_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"default_log_{timestamp}.log"

    handlers = {}

    # Root/global file handler
    handlers["file"] = {
        "class": "logging.FileHandler",
        "formatter": "default",
        "level": level,
        "filename": str(log_dir / log_file),
        "encoding": "utf-8",
        "mode": "w",
    }

    # Define root and file_only logger names
    root_handlers = ["file"]
    loggers = {
        "main": {"level": level, "handlers": root_handlers, "propagate": False},
        "file_only": {"level": level, "handlers": ["file"], "propagate": False},
    }

    # Add custom per-module file handlers
    if module_logs:
        for mod, fname in module_logs.items():
            handler_name = f"{mod.replace('.', '_')}_file"
            handlers[handler_name] = {
                "class": "logging.FileHandler",
                "formatter": "default",
                "level": level,
                "filename": str(log_dir / fname),
                "encoding": "utf-8",
                "mode": "w"
            }
            loggers[mod] = {
                "level": level,
                "handlers": [handler_name],
                "propagate": False  # ✅ Prevent logs from leaking to root
            }

    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,  # ✅ Preserve third-party loggers unless explicitly overridden
        "formatters": {
            "default": {
                "format": "%(asctime)s %(levelname)s [%(name)s]: %(message)s"
            }
        },
        "handlers": handlers,
        "root": {"level": level, "handlers": root_handlers},
        "loggers": loggers,
    }

    logging.config.dictConfig(logging_config)

    # === Force per-module loggers to have correct handlers ===
    if module_logs:
        for mod, fname in module_logs.items():
            logger = logging.getLogger(mod)
            logger.setLevel(level)
            logger.propagate = False

            expected_path = str(log_dir / fname)
            attached_files = {
                getattr(h, 'baseFilename', None)
                for h in logger.handlers
                if isinstance(h, logging.FileHandler)
            }

            if expected_path not in attached_files:
                handler = logging.FileHandler(expected_path, mode="w", encoding="utf-8")
                formatter = logging.Formatter("%(asctime)s %(levelname)s [%(name)s]: %(message)s")
                handler.setFormatter(formatter)
                logger.addHandler(handler)

    # # === Optional: Print active loggers and handlers for debugging ===
    # for mod in module_logs or {}:
    #     logger = logging.getLogger(mod)
    #     for h in logger.handlers:
    #         print(f"[CHECK] Handler for {mod}: {h.baseFilename if hasattr(h, 'baseFilename') else h}")

    # # === Optional: Print active loggers and handlers for debugging ===
    # for name, obj in logging.root.manager.loggerDict.items():
    #     if isinstance(obj, logging.Logger):
    #         print(f"[LOGGER] {name} — handlers: {obj.handlers}")
    #     else:
    #         print(f"[PLACEHOLDER] {name}")



