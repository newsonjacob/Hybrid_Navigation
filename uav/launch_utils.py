"""Helper functions for launching external processes and waiting on events."""

import logging
import subprocess
import socket
import time
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple
from uav.paths import STOP_FLAG_PATH

try:
    import pygetwindow as gw
except Exception:  # pragma: no cover - platform without GUI
    gw = None  # type: ignore

# Default stop flag used to cancel wait operations. Tests or callers may
# monkeypatch this path.
STOP_FLAG: Path = STOP_FLAG_PATH

__all__ = [
    "wait_for_window",
    "wait_for_flag",
    "wait_for_port",
    "start_streamer",
    "launch_slam_backend",
    "resize_window",
    "STOP_FLAG",
]

def wait_for_window(title_substring: str, timeout: float = 20.0) -> bool:
    """Wait until a window containing ``title_substring`` appears."""
    logger = logging.getLogger(__name__)
    logger.info("Waiting for window containing title: '%s'...", title_substring)
    start_time = time.time()
    while time.time() - start_time < timeout:
        if STOP_FLAG.exists():
            logger.info("[WAIT] Stop flag detected - cancelling window wait")
            return False
        titles = gw.getAllTitles() if gw else []
        for title in titles:
            if title_substring.lower() in title.lower():
                logger.info("Window found: '%s'", title)
                return True
        time.sleep(0.5)
    logger.error("Timeout waiting for window with title containing: '%s'", title_substring)
    return False


def wait_for_flag(flag_path: Path, timeout: float = 15.0) -> bool:
    """Block until ``flag_path`` exists or the operation times out."""
    logger = logging.getLogger(__name__)
    logger.info("[WAIT] Waiting for flag: %s (timeout=%ss)", flag_path, timeout)
    start = time.time()
    while not flag_path.exists():
        if STOP_FLAG.exists():
            logger.info("[WAIT] Stop flag detected - cancelling flag wait")
            return False
        if time.time() - start > timeout:
            logger.error("[WAIT] Timeout waiting for flag: %s", flag_path)
            return False
        time.sleep(0.5)
    logger.info("[WAIT] Flag found: %s", flag_path)
    return True


def wait_for_port(host: str, port: int, timeout: float = 5.0) -> bool:
    """Wait until ``host:port`` accepts a TCP connection."""
    logger = logging.getLogger(__name__)
    logger.info("[WAIT] Waiting for port %s:%s to become available (timeout=%ss)", host, port, timeout)
    start_time = time.time()
    while time.time() - start_time < timeout:
        if STOP_FLAG.exists():
            logger.info("[WAIT] Stop flag detected - cancelling port wait")
            return False
        try:
            with socket.create_connection((host, port), timeout=1):
                logger.info("[WAIT] Port %s:%s is now accepting connections.", host, port)
                return True
        except OSError as e:
            logger.debug("[WAIT] Port %s:%s not ready yet: %s", host, port, e)
            time.sleep(0.2)
    logger.error("[WAIT] Timeout: Port %s:%s did not become ready.", host, port)
    return False


def start_streamer(host: str, port: int, stream_mode: str = "stereo", log_timestamp: Optional[str] = None) -> subprocess.Popen:
    """Launch the image streaming bridge for SLAM."""
    logger = logging.getLogger(__name__)
    logger.info("[LAUNCH] Starting SLAM image streamer on %s:%s with mode '%s'", host, port, stream_mode)
    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd()
    cmd = [
        "python",
        "slam_bridge/stream_airsim_image.py",
        "--host", host,
        "--port", str(port),
        "--mode", stream_mode,
    ]
    if log_timestamp:
        cmd += ["--log-timestamp", log_timestamp]
    proc = subprocess.Popen(cmd, env=env)
    logger.info("[LAUNCH] SLAM image streamer started (PID %s)", getattr(proc, "pid", "n/a"))
    time.sleep(2)
    return proc


def launch_slam_backend(receiver_host: str, receiver_port: int) -> subprocess.Popen:
    """Launch the ORB-SLAM backend under WSL."""
    import psutil

    logger = logging.getLogger(__name__)
    logger.info("[LAUNCH] Starting SLAM backend in WSL for receiver %s:%s", receiver_host, receiver_port)

    for conn in psutil.net_connections(kind='inet'):
        if conn.laddr.port == receiver_port and conn.status == psutil.CONN_ESTABLISHED:
            try:
                psutil.Process(conn.pid).kill()
                logger.warning("[LAUNCH] Killed process %s using port %s", conn.pid, receiver_port)
            except Exception as e:  # pragma: no cover - best effort
                logger.error("[LAUNCH] Failed to kill process on port %s: %s", receiver_port, e)

    slam_cmd = [
        "wsl", "bash", "-c",
        f"export POSE_RECEIVER_IP={receiver_host}; "
        f"export POSE_RECEIVER_PORT={receiver_port}; "
        "export SLAM_FLAG_DIR=/mnt/h/Documents/AirSimExperiments/Hybrid_Navigation/flags; "
        "export SLAM_LOG_DIR=/mnt/h/Documents/AirSimExperiments/Hybrid_Navigation/logs; "
        "export SLAM_IMAGE_DIR=/mnt/h/Documents/AirSimExperiments/Hybrid_Navigation/logs/images; "
        "cd /mnt/h/Documents/AirSimExperiments/Hybrid_Navigation/linux_slam/build && "
        "./app/tcp_slam_server ../Vocabulary/ORBvoc.txt ../app/rgbd_settings.yaml"
    ]
    logger.info("[LAUNCH] SLAM backend command: %s", ' '.join(slam_cmd))
    proc = subprocess.Popen(slam_cmd)
    logger.info("[LAUNCH] SLAM backend started (PID %s)", getattr(proc, "pid", "n/a"))
    return proc


def resize_window(title_substring: str, width: int, height: int) -> bool:
    """Resize the first window containing ``title_substring``."""
    logger = logging.getLogger(__name__)
    if not gw:
        logger.debug("pygetwindow not available; cannot resize window")
        return False
    try:
        for win in gw.getAllWindows():
            if title_substring.lower() in win.title.lower():
                win.resizeTo(width, height)
                logger.info("Resized window '%s' to %dx%d", win.title, width, height)
                return True
    except Exception as e:  # pragma: no cover - depends on OS
        logger.warning("Failed to resize window '%s': %s", title_substring, e)
    return False
