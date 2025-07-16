import logging
import subprocess
import socket
import time
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

try:
    import pygetwindow as gw
except Exception:  # pragma: no cover - platform without GUI
    gw = None  # type: ignore

# Default stop flag used to cancel wait operations. Tests or callers may
# monkeypatch this path.
STOP_FLAG: Path = Path("flags/stop.flag")

__all__ = [
    "wait_for_window",
    "wait_for_flag",
    "wait_for_port",
    "start_streamer",
    "launch_slam_backend",
    "record_slam_video",
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
        "cd /mnt/h/Documents/AirSimExperiments/Hybrid_Navigation/linux_slam/build && "
        "./app/tcp_slam_server ../Vocabulary/ORBvoc.txt ../app/rgbd_settings.yaml"
    ]
    logger.info("[LAUNCH] SLAM backend command: %s", ' '.join(slam_cmd))
    proc = subprocess.Popen(slam_cmd)
    logger.info("[LAUNCH] SLAM backend started (PID %s)", getattr(proc, "pid", "n/a"))
    return proc


def record_slam_video(window_substring: str = "ORB-SLAM2", duration: int = 60) -> Tuple[Optional[subprocess.Popen], Optional[str]]:
    """Record the SLAM visualization window using ``ffmpeg``."""
    logger = logging.getLogger(__name__)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path = f"analysis/slam_output_{timestamp}.mp4"
    try:
        time.sleep(1)
        window_title = None
        titles = gw.getAllTitles() if gw else []
        for title in titles:
            if window_substring in title:
                window_title = title
                break
        if not window_title:
            raise RuntimeError(f"Could not find window with title containing '{window_substring}'.")
        ffmpeg_cmd = [
            "ffmpeg",
            "-hide_banner", "-loglevel", "error",
            "-y",
            "-f", "gdigrab",
            "-framerate", "30",
            "-i", f"title={window_title}",
            "-t", str(duration),
            video_path,
        ]
        proc = subprocess.Popen(ffmpeg_cmd)
        logger.info("Started screen recording to %s", video_path)
        return proc, video_path
    except Exception as e:  # pragma: no cover - depends on system
        logger.warning("Screen recording failed to start: %s", e)
        return None, None