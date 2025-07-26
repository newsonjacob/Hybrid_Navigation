"""High-level launcher orchestrating AirSim, SLAM and the GUI."""
import logging
import os
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional
from uav.paths import FLAGS_DIR

from uav.logging_config import setup_logging
from uav.config import RETENTION_CONFIG
from uav.utils import retain_recent_files_config
from uav.cli import parse_args
from uav.config import load_app_config
from uav.interface import start_gui, exit_flag

from uav import launch_utils as lutils

# ---------------------------------------------------------------------------
# Flag paths are defined at import time so tests may monkeypatch them.
# ---------------------------------------------------------------------------
flags_dir = FLAGS_DIR
AIRSIM_READY_FLAG = FLAGS_DIR / "airsim_ready.flag"
SLAM_READY_FLAG = FLAGS_DIR / "slam_ready.flag"
SLAM_FAILED_FLAG = FLAGS_DIR / "slam_failed.flag"
START_NAV_FLAG = FLAGS_DIR / "start_nav.flag"
STOP_FLAG = FLAGS_DIR / "stop.flag"

# When a shutdown is requested we wait this long for the main process to exit
# before forcibly terminating it.
GRACE_TIME = 10

# keep lutils in sync with our stop flag
lutils.STOP_FLAG = STOP_FLAG

# Timestamp for log files; populated in ``init_logging_and_flags``
TIMESTAMP: Optional[str] = None
logger = logging.getLogger("Launch")

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def init_logging_and_flags() -> str:
    """Configure logging and return the timestamp used for log files."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    launch_log = f"launch_{timestamp}.log"
    modules_with_dedicated_logs = [
        "nav_loop",
        "slam_receiver",
        "slam_plotter",
        "utils",
        "pose_receiver",
        "perception",
    ]
    module_logs = {
        mod: f"{mod.replace('.', '_')}_{timestamp}.log" for mod in modules_with_dedicated_logs
    }
    setup_logging(log_file=launch_log, module_logs=module_logs, level=logging.DEBUG)

    global logger
    logger = logging.getLogger("Launch")
    logger.info("Logging to logs/%s", launch_log)

    return timestamp

# ---------------------------------------------------------------------------
# Helper wrappers delegating to ``uav.launch_utils`` so existing imports work.
# ---------------------------------------------------------------------------


def wait_for_window(title_substring: str, timeout: float = 20.0) -> bool:
    lutils.STOP_FLAG = STOP_FLAG
    return lutils.wait_for_window(title_substring, timeout)


def wait_for_flag(flag_path: Path, timeout: float = 15.0) -> bool:
    lutils.STOP_FLAG = STOP_FLAG
    return lutils.wait_for_flag(flag_path, timeout)


def wait_for_port(host: str, port: int, timeout: float = 5.0) -> bool:
    lutils.STOP_FLAG = STOP_FLAG
    return lutils.wait_for_port(host, port, timeout)


def start_streamer(host: str, port: int, stream_mode: str = "stereo") -> subprocess.Popen:
    return lutils.start_streamer(host, port, stream_mode, log_timestamp=TIMESTAMP)


def launch_slam_backend(receiver_host: str, receiver_port: int) -> subprocess.Popen:
    return lutils.launch_slam_backend(receiver_host, receiver_port)

# ---------------------------------------------------------------------------
# Shutdown helpers
# ---------------------------------------------------------------------------

def request_shutdown(proc: Optional[subprocess.Popen], logger: logging.Logger, timeout: float = GRACE_TIME) -> bool:
    """Signal ``main.py`` to stop and wait for it to exit.

    Returns ``True`` if the process exited gracefully within the timeout."""
    if proc is None:
        return True

    STOP_FLAG.touch()

    if getattr(proc, "poll", lambda: None)() is not None:
        return True

    try:
        logger.info("[SHUTDOWN] Waiting up to %.1fs for main process to exit...", timeout)
        proc.wait(timeout=timeout)
        logger.info("[SHUTDOWN] Main process exited gracefully.")
        return True
    except subprocess.TimeoutExpired:
        logger.warning("[SHUTDOWN] Main process did not exit within grace period.")
        return False

# ---------------------------------------------------------------------------
# Launcher dataclass managing subprocesses
# ---------------------------------------------------------------------------

@dataclass
class Launcher:
    logger: logging.Logger
    timestamp: str
    main_proc: Optional[subprocess.Popen] = None
    slam_proc: Optional[subprocess.Popen] = None
    stream_proc: Optional[subprocess.Popen] = None
    slam_video_path: Optional[str] = None

    def shutdown(self) -> None:
        """Terminate all started subprocesses and clean temporary files."""
        self.logger.info("[SHUTDOWN] Initiating shutdown sequence for all subprocesses.")

        # Ensure any running GUI exits when the simulation finishes
        exit_flag.set()

        if exit_flag.is_set() or STOP_FLAG.exists():
            self.logger.info("[SHUTDOWN] Shutdown signal detected. Proceeding with shutdown.")
        else:
            self.logger.info("[SHUTDOWN] No shutdown signal detected. Performing cleanup anyway.")

        if self.stream_proc is not None:
            self.logger.info("[SHUTDOWN] Terminating SLAM streamer (PID %s)", getattr(self.stream_proc, "pid", "n/a"))
            self.stream_proc.terminate()
            try:
                self.stream_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.logger.warning("[SHUTDOWN] Forcing SLAM streamer shutdown...")
                self.stream_proc.kill()

        if self.slam_proc is not None:
            self.logger.info("[SHUTDOWN] Terminating SLAM backend (PID %s)", getattr(self.slam_proc, "pid", "n/a"))
            self.slam_proc.terminate()
            try:
                self.slam_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.logger.warning("[SHUTDOWN] Forcing SLAM backend shutdown...")
                self.slam_proc.kill()

        graceful = request_shutdown(self.main_proc, self.logger)

        if self.main_proc is not None and getattr(self.main_proc, "poll", lambda: None)() is None:
            if not graceful:
                self.logger.info("[SHUTDOWN] Terminating main script (PID %s)", getattr(self.main_proc, "pid", "n/a"))
                self.main_proc.terminate()
                try:
                    self.main_proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.logger.warning("[SHUTDOWN] Forcing main script shutdown...")
                    self.main_proc.kill()

        pid_file = FLAGS_DIR / "ue4_sim.pid"
        if pid_file.exists():
            try:
                ue4_pid = int(pid_file.read_text())
                self.logger.info("[SHUTDOWN] Terminating UE4 simulation (PID %s)", ue4_pid)
                try:
                    subprocess.call(
                        ["taskkill", "/F", "/PID", str(ue4_pid)],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                except FileNotFoundError:
                    self.logger.warning("taskkill not available")
            except Exception as e:  # pragma: no cover - best effort
                self.logger.warning("[SHUTDOWN] Failed to terminate UE4 by PID: %s", e)
            finally:
                pid_file.unlink(missing_ok=True)
        else:
            for proc_name in ("Blocks.exe", "UE4Editor.exe"):
                try:
                    subprocess.call(
                        ["taskkill", "/F", "/IM", proc_name],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                except FileNotFoundError:
                    self.logger.warning("taskkill not available")

        if self.slam_video_path and not os.path.exists(self.slam_video_path):
            self.logger.warning("SLAM video file not created.")

# ---------------------------------------------------------------------------
# Helper functions used by tests and the CLI
# ---------------------------------------------------------------------------

def wait_for_start_flag() -> bool:
    """Block until the user initiates navigation via the GUI."""
    logger.info("Waiting for user to initiate navigation via GUI (flags/start_nav.flag)...")
    while not START_NAV_FLAG.exists():
        if STOP_FLAG.exists():
            logger.info("Stop flag detected before navigation started. Shutting down.")
            return False
        time.sleep(0.2)
    logger.info("Signaling navigation to begin...")
    return True


def wait_for_slam_done_flag(timeout: float = 15.0) -> bool:
    """Wait for the slam_done.flag to appear, or timeout."""
    slam_done_flag = FLAGS_DIR / "slam_done.flag"
    logger.info("[MAIN] Waiting for SLAM backend to finish (slam_done.flag)...")
    start = time.time()
    while not slam_done_flag.exists():
        if time.time() - start > timeout:
            logger.warning("[MAIN] Timeout waiting for slam_done.flag.")
            return False
        time.sleep(0.2)
    logger.info("[MAIN] slam_done.flag detected.")
    return True


def main(timestamp: str, selected_nav_mode: Optional[str] = None) -> bool:
    """Launch the simulation using the provided timestamp."""

    # --- Remove old flags if they exist ---
    for flag_name in ["slam_shutdown.flag", "stop.flag", "slam_done.flag"]:
        flag_path = FLAGS_DIR / flag_name
        if flag_path.exists():
            flag_path.unlink()
            logger.info(f"[MAIN] Removed stale {flag_name} at startup.")

    args = parse_args()
    if selected_nav_mode is not None:
        args.nav_mode = selected_nav_mode
    if not getattr(args, "stream_mode", None):
        args.stream_mode = "stereo"

    config = load_app_config(args.config)
    slam_server_host = args.slam_server_host or config.get("network", "slam_server_host", fallback="127.0.0.1")
    slam_server_port = int(args.slam_server_port or config.get("network", "slam_server_port", fallback="6000"))
    slam_receiver_host = args.slam_receiver_host or config.get("network", "slam_receiver_host", fallback="0.0.0.0")
    slam_receiver_port = int(args.slam_receiver_port or config.get("network", "slam_receiver_port", fallback="6001"))

    global TIMESTAMP
    TIMESTAMP = timestamp

    launcher = Launcher(logger=logger, timestamp=timestamp)

    try:
        logger.info("[MAIN] Launching Unreal Engine + main.py with nav_mode=%s", args.nav_mode)
        launcher.main_proc = subprocess.Popen([
            "python", "main.py", "--nav-mode", args.nav_mode,
            "--slam-server-host", slam_server_host,
            "--slam-server-port", str(slam_server_port),
            "--slam-receiver-host", slam_receiver_host,
            "--slam-receiver-port", str(slam_receiver_port),
            "--log-timestamp", timestamp,
        ])
        logger.info("[MAIN] main.py started (PID %s)", getattr(launcher.main_proc, "pid", "n/a"))
        logger.info("[MAIN] Waiting for AirSim to fully launch...")

        if not wait_for_flag(AIRSIM_READY_FLAG, timeout=20):
            logger.error("[MAIN] AirSim did not become ready or startup was cancelled.")
            launcher.shutdown()
            return False

        airsim_width = config.getint("window", "airsim_width", fallback=1280)
        airsim_height = config.getint("window", "airsim_height", fallback=720)
        lutils.resize_window("Blocks", airsim_width, airsim_height)

        if args.nav_mode == "slam":
            logger.info("[MAIN] Launching SLAM streamer and backend for SLAM mode.")
            launcher.stream_proc = start_streamer(slam_server_host, slam_server_port, args.stream_mode)
            logger.info("[MAIN] SLAM streamer process started.")

            from slam_bridge.slam_receiver import start_receiver
            logger.info("[MAIN] Starting SLAM receiver (Python pose receiver)...")
            start_receiver("0.0.0.0", 6001)
            logger.info("[MAIN] SLAM receiver started.")

            logger.info("[MAIN] Waiting for SLAM receiver port to become available...")
            if not wait_for_port("127.0.0.1", 6001):
                logger.error("[MAIN] SLAM receiver port never became ready.")
                launcher.shutdown()
                return False

            logger.info("[MAIN] Launching SLAM backend in WSL...")
            launcher.slam_proc = launch_slam_backend(slam_receiver_host, slam_receiver_port)
            logger.info("[MAIN] SLAM backend process started.")

            logger.info("[MAIN] Waiting for SLAM backend to signal readiness...")
            if not wait_for_flag(SLAM_READY_FLAG, timeout=30):
                logger.error("[MAIN] SLAM backend never received first image or startup was cancelled.")
                time.sleep(2)
                launcher.shutdown()
                return False

            logger.info("[MAIN] Waiting for Pangolin visualization window...")
            if not wait_for_window("ORB-SLAM2", timeout=20):
                logger.error("[MAIN] SLAM visualization window not detected or startup was cancelled.")
                launcher.shutdown()
                return False
            logger.info("[MAIN] Pangolin window found.")

            if SLAM_FAILED_FLAG.exists():
                logger.error("[MAIN] SLAM backend reported failure — aborting simulation.")
                launcher.shutdown()
                SLAM_FAILED_FLAG.unlink(missing_ok=True)
                return False

        logger.info("[MAIN] Waiting for user to initiate navigation via GUI...")
        if not wait_for_start_flag():
            logger.info("[MAIN] Navigation not started by user. Shutting down.")
            launcher.shutdown()
            return False

        logger.info("[MAIN] Waiting for main.py to finish or stop.flag to be set...")
        shutdown_requested = False
        while launcher.main_proc and getattr(launcher.main_proc, "poll", lambda: None)() is None:
            if STOP_FLAG.exists():
                logger.info("[MAIN] Stop flag detected. Initiating graceful shutdown...")
                break
            time.sleep(1)
        logger.info("[MAIN] main.py completed or terminated.")

        # Wait for SLAM backend to finish saving files
        wait_for_slam_done_flag(timeout=15.0)

        return True

    finally:
        logger.info("[MAIN] Final shutdown sequence.")
        launcher.shutdown()

# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def wait_for_nav_mode_and_launch(timestamp: str) -> None:
    logger.info("Waiting for user to select navigation mode and launch simulation...")
    while not (FLAGS_DIR / "nav_mode.flag").exists():
        if STOP_FLAG.exists():
            logger.info("Stop flag detected before simulation started. Shutting down.")
            return
        time.sleep(0.2)

    with open(FLAGS_DIR / "nav_mode.flag") as f:
        selected_nav_mode = f.read().strip()
    logger.info(f"User selected navigation mode: {selected_nav_mode}")

    try:
        main(timestamp, selected_nav_mode)
    finally:
        retain_recent_files_config(RETENTION_CONFIG)


def cli_main() -> None:
    "Entry point used when executing this module as a script."
    timestamp = init_logging_and_flags()

    args = parse_args()

    logger.info("[RESET] Clearing flags directory %s", FLAGS_DIR)
    for f in FLAGS_DIR.glob("*"):
        if f.is_file():
            f.unlink(missing_ok=True)

    for flag in [AIRSIM_READY_FLAG,
                 SLAM_READY_FLAG,
                 SLAM_FAILED_FLAG,
                 START_NAV_FLAG,
                 STOP_FLAG,
                 FLAGS_DIR / "nav_mode.flag",
                 ]:
        flag.unlink(missing_ok=True)

    param_refs = {"L": [0.0], "C": [0.0], "R": [0.0], "state": ["idle"]}

    sim_thread = threading.Thread(target=wait_for_nav_mode_and_launch, args=(timestamp,))
    sim_thread.start()

    # Start the GUI in the main thread (this blocks until GUI closes)
    start_gui(param_refs)

    # Wait for the simulation thread to finish
    STOP_FLAG.touch()
    sim_thread.join()

if __name__ == "__main__":  # pragma: no cover - manual execution
    cli_main()
