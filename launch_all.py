import subprocess
import time
import socket
import os
from pathlib import Path
import sys
from datetime import datetime
import webbrowser
import logging
from uav.logging_config import setup_logging
from uav.utils import retain_recent_logs
from uav.cli import parse_args

args = parse_args()

# --- Set default stream mode based on nav mode ---
if not hasattr(args, "stream_mode") or args.stream_mode is None:
    args.stream_mode = "stereo"  # Force stereo as default for all modes


def init_logging_and_flags():
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    launch_log = f"launch_{timestamp}.log"
    modules_with_dedicated_logs = [
        "nav_loop",
        "slam_receiver",
        "slam_plotter",
        "utils",
        "pose_receiver"
    ]
    module_logs = {
        mod: f"{mod.replace('.', '_')}_{timestamp}.log"
        for mod in modules_with_dedicated_logs
    }

    setup_logging(log_file=launch_log, module_logs=module_logs, level=logging.DEBUG)
    
    logger = logging.getLogger("Launch")
    logger.info(f"Logging to logs/{launch_log}")

    flags_dir = Path("flags")
    flags_dir.mkdir(exist_ok=True)

    return logger, launch_log, module_logs, timestamp

# === CALL LOGGING INIT FIRST ===
logger, launch_log, module_logs, timestamp = init_logging_and_flags()

# Now safe to import other logging-based modules
from uav.cli import parse_args
from uav.config import load_app_config
import pygetwindow as gw
from uav.interface import start_gui

# --- Flag paths ---
flags_dir = Path("flags")
flags_dir.mkdir(exist_ok=True)
AIRSIM_READY_FLAG = flags_dir / "airsim_ready.flag"
SLAM_READY_FLAG = flags_dir / "slam_ready.flag"
SLAM_FAILED_FLAG = flags_dir / "slam_failed.flag"
START_NAV_FLAG = flags_dir / "start_nav.flag"

def shutdown_all(main_proc=None, slam_proc=None, stream_proc=None, ffmpeg_proc=None, slam_video_path=None, sim_proc=None):
    """Terminate all subprocesses and clean temporary files.

    Parameters mirror the processes started by this launcher. Any ``None`` value
    is simply ignored. This function is safe to call multiple times.
    """
    logger.info("[SHUTDOWN] Initiating shutdown sequence for all subprocesses.")
    # --- CLEAN UP streamer ---
    if stream_proc is not None:
        logger.info("[SHUTDOWN] Terminating SLAM streamer (PID %s)", stream_proc.pid)
        stream_proc.terminate()
        try:
            stream_proc.wait(timeout=5)
            logger.info("[SHUTDOWN] SLAM streamer terminated cleanly.")
        except subprocess.TimeoutExpired:
            logger.warning("[SHUTDOWN] Forcing SLAM streamer shutdown...")
            stream_proc.kill()

    # --- CLEAN UP SLAM ---
    if slam_proc is not None:
        logger.info("[SHUTDOWN] Terminating SLAM backend (PID %s)", slam_proc.pid)
        slam_proc.terminate()
        try:
            slam_proc.wait(timeout=5)
            logger.info("[SHUTDOWN] SLAM backend terminated cleanly.")
        except subprocess.TimeoutExpired:
            logger.warning("[SHUTDOWN] Forcing SLAM backend shutdown...")
            slam_proc.kill()

    # --- CLEAN UP FFMPEG ---
    if isinstance(ffmpeg_proc, subprocess.Popen):
        logger.info("[SHUTDOWN] Terminating screen recording (PID %s)", ffmpeg_proc.pid)
        ffmpeg_proc.terminate()
        try:
            ffmpeg_proc.wait(timeout=5)
            logger.info("[SHUTDOWN] Screen recording terminated cleanly.")
        except subprocess.TimeoutExpired:
            logger.warning("[SHUTDOWN] Forcing screen recorder shutdown...")
            ffmpeg_proc.kill()

    # --- CLEAN UP main.py ---
    if main_proc is not None:
        logger.info("[SHUTDOWN] Terminating main script (PID %s)", main_proc.pid)
        main_proc.terminate()
        try:
            main_proc.wait(timeout=5)
            logger.info("[SHUTDOWN] main.py terminated cleanly.")
        except subprocess.TimeoutExpired:
            logger.warning("[SHUTDOWN] Forcing main script shutdown...")
            main_proc.kill()

    # --- CLEAN UP Unreal Engine (UE4) ---
    pid_file = flags_dir / "ue4_sim.pid"
    if pid_file.exists():
        try:
            ue4_pid = int(pid_file.read_text())
            logger.info("[SHUTDOWN] Terminating UE4 simulation (PID %s)", ue4_pid)
            subprocess.call(["taskkill", "/F", "/PID", str(ue4_pid)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            logger.warning("[SHUTDOWN] Failed to terminate UE4 by PID: %s", e)
        finally:
            pid_file.unlink(missing_ok=True)
    else:
        logger.warning("[SHUTDOWN] No UE4 PID file found — attempting forced shutdown via taskkill")
        subprocess.call(["taskkill", "/F", "/IM", "Blocks.exe"],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.call(["taskkill", "/F", "/IM", "UE4Editor.exe"],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


    # Check if video was saved
    if slam_video_path and not os.path.exists(slam_video_path):
        logger.warning("SLAM video file not created.")

    # Automatically open video if it exists
    # if slam_video_path and os.path.exists(slam_video_path):
    #     webbrowser.open(slam_video_path)

def wait_for_window(title_substring, timeout=20):
    """Wait until a window containing ``title_substring`` appears."""
    logger.info(f"Waiting for window containing title: '{title_substring}'...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        titles = gw.getAllTitles()
        for title in titles:
            if title_substring.lower() in title.lower():
                logger.info(f"Window found: '{title}'")
                return True
        time.sleep(0.5)
    raise TimeoutError(f"Timeout waiting for window with title containing: '{title_substring}'")

def wait_for_flag(flag_path, timeout=15):
    logger.info(f"[WAIT] Waiting for flag: {flag_path} (timeout={timeout}s)")
    start = time.time()
    while not os.path.exists(flag_path):
        elapsed = time.time() - start
        if elapsed > timeout:
            logger.error(f"[WAIT] Timeout waiting for flag: {flag_path}")
            return False
        time.sleep(0.5)
    logger.info(f"[WAIT] Flag found: {flag_path}")
    return True

def wait_for_port(host: str, port: int, timeout: float = 5.0):
    logger.info(f"[WAIT] Waiting for port {host}:{port} to become available (timeout={timeout}s)")
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            with socket.create_connection((host, port), timeout=1):
                logger.info(f"[WAIT] Port {host}:{port} is now accepting connections.")
                return True
        except OSError as e:
            logger.debug(f"[WAIT] Port {host}:{port} not ready yet: {e}")
            time.sleep(0.2)
    logger.error(f"[WAIT] Timeout: Port {host}:{port} did not become ready.")
    return False

def start_streamer(host: str, port: int, stream_mode: str = "stereo"):
    logger.info(f"[LAUNCH] Starting SLAM image streamer on {host}:{port} with mode '{stream_mode}'")
    import os
    os.environ["PYTHONPATH"] = os.getcwd()
    proc = subprocess.Popen([
        "python",
        "slam_bridge/stream_airsim_image.py",
        "--host", host,
        "--port", str(port),
        "--mode", stream_mode,
        "--log-timestamp", timestamp
    ])
    logger.info(f"[LAUNCH] SLAM image streamer started (PID {proc.pid})")
    time.sleep(2)
    return proc

def wait_for_port(host: str, port: int, timeout: float = 5.0):
    logger.info(f"[wait_for_port] Waiting for {host}:{port} to become available...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            with socket.create_connection((host, port), timeout=1):
                logger.info(f"[wait_for_port] {host}:{port} is now accepting connections.")
                return True
        except OSError:
            time.sleep(0.2)
    logger.error(f"[wait_for_port] Timeout: {host}:{port} did not become ready.")
    return False

import psutil

def launch_slam_backend(receiver_host: str, receiver_port: int):
    logger.info(f"[LAUNCH] Starting SLAM backend in WSL for receiver {receiver_host}:{receiver_port}")
    def kill_port(port: int):
        killed = 0
        for conn in psutil.net_connections(kind='inet'):
            if conn.laddr.port == port and conn.status == psutil.CONN_ESTABLISHED:
                try:
                    psutil.Process(conn.pid).kill()
                    logger.warning(f"[LAUNCH] Killed process {conn.pid} using port {port}")
                    killed += 1
                except Exception as e:
                    logger.error(f"[LAUNCH] Failed to kill process on port {port}: {e}")
        if killed == 0:
            logger.info(f"[LAUNCH] No established connections found on port {port}")

    kill_port(receiver_port)
    slam_cmd = [
        "wsl", "bash", "-c",
        f"export POSE_RECEIVER_IP={receiver_host}; "
        f"export POSE_RECEIVER_PORT={receiver_port}; "
        "cd /mnt/h/Documents/AirSimExperiments/Hybrid_Navigation/linux_slam/build && "
        "./app/tcp_slam_server ../Vocabulary/ORBvoc.txt ../app/rgbd_settings.yaml"
    ]
    logger.info(f"[LAUNCH] SLAM backend command: {' '.join(slam_cmd)}")
    proc = subprocess.Popen(slam_cmd)
    logger.info(f"[LAUNCH] SLAM backend started (PID {proc.pid})")
    return proc

def record_slam_video(window_substring: str = "ORB-SLAM2", duration: int = 60):
    """Record the SLAM visualization window using ``ffmpeg``.
    Returns a tuple of ``(process, video_path)`` or (None, None) if it fails.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path = f"analysis/slam_output_{timestamp}.mp4"

    try:
        import time
        import pygetwindow as gw

        # Wait briefly in case the window isn't ready yet
        time.sleep(1)

        window_title = None
        for title in gw.getAllTitles():
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

    except Exception as e:
        logger.warning("⚠️ Screen recording failed to start: %s", e)
        return None, None



def wait_for_start_flag():
    """Block until the user initiates navigation via the GUI."""
    logger.info(
        "Waiting for user to initiate navigation via GUI (flags/start_nav.flag)..."
    )
    while not START_NAV_FLAG.exists():
        if os.path.exists("flags/stop.flag"):
            logger.info("Stop flag detected before navigation started. Shutting down.")
            return False
        time.sleep(0.2)
    logger.info("Signaling navigation to begin...")
    return True

def main(timestamp):
    args = parse_args()
    config = load_app_config(args.config)
    slam_server_host = args.slam_server_host or config.get("network", "slam_server_host", fallback="127.0.0.1")
    slam_server_port = int(args.slam_server_port or config.get("network", "slam_server_port", fallback="6000"))
    slam_receiver_host = args.slam_receiver_host or config.get("network", "slam_receiver_host", fallback="0.0.0.0")
    slam_receiver_port = int(args.slam_receiver_port or config.get("network", "slam_receiver_port", fallback="6001"))

    main_proc = slam_proc = stream_proc = ffmpeg_proc = None
    slam_video_path = None

    try:
        logger.info(f"[MAIN] Launching Unreal Engine + main.py with nav_mode={args.nav_mode}")
        main_proc = subprocess.Popen([
            "python", "main.py", "--nav-mode", args.nav_mode,
            "--slam-server-host", slam_server_host,
            "--slam-server-port", str(slam_server_port),
            "--slam-receiver-host", slam_receiver_host,
            "--slam-receiver-port", str(slam_receiver_port),
            "--log-timestamp", timestamp,
        ])
        logger.info(f"[MAIN] main.py started (PID {main_proc.pid})")
        logger.info("[MAIN] Waiting for AirSim to fully launch...")

        if not wait_for_flag(AIRSIM_READY_FLAG, timeout=20):
            logger.error("[MAIN] AirSim did not become ready in time. Shutting down.")
            shutdown_all(main_proc)
            sys.exit(1)

        if args.nav_mode == "slam":
            logger.info("[MAIN] Launching SLAM streamer and backend for SLAM mode.")
            stream_proc = start_streamer(slam_server_host, slam_server_port, args.stream_mode)
            logger.info("[MAIN] SLAM streamer process started.")

            from slam_bridge.slam_receiver import start_receiver
            logger.info("[MAIN] Starting SLAM receiver (Python pose receiver)...")
            start_receiver("0.0.0.0", 6001)
            logger.info("[MAIN] SLAM receiver started.")

            logger.info("[MAIN] Waiting for SLAM receiver port to become available...")
            wait_for_port("0.0.0.0", 6001)

            logger.info("[MAIN] Launching SLAM backend in WSL...")
            slam_proc = launch_slam_backend(slam_receiver_host, slam_receiver_port)
            logger.info("[MAIN] SLAM backend process started.")

            logger.info("[MAIN] Waiting for SLAM backend to signal readiness...")
            if not wait_for_flag(SLAM_READY_FLAG, timeout=30):
                logger.error("[MAIN] SLAM backend never received first image — shutting down.")
                time.sleep(2)
                shutdown_all(main_proc, slam_proc, stream_proc)
                sys.exit(1)

            try:
                logger.info("[MAIN] Waiting for Pangolin visualization window...")
                wait_for_window("ORB-SLAM2", timeout=20)
                logger.info("[MAIN] Pangolin window found.")
            except TimeoutError as e:
                logger.error(f"[MAIN] {e}")
                logger.info("[MAIN] Shutting down due to missing SLAM visualization window...")
                shutdown_all(main_proc, slam_proc)
                sys.exit(1)

            ffmpeg_proc, slam_video_path = record_slam_video("ORB-SLAM2")
            if ffmpeg_proc:
                logger.info(f"[MAIN] Screen recording started (PID {ffmpeg_proc.pid})")
            else:
                logger.warning("[MAIN] Screen recording failed to start.")

            if os.path.exists(SLAM_FAILED_FLAG):
                logger.error("[MAIN] SLAM backend reported failure — aborting simulation.")
                shutdown_all(main_proc, slam_proc, stream_proc, ffmpeg_proc, slam_video_path)
                SLAM_FAILED_FLAG.unlink(missing_ok=True)
                sys.exit(1)

        logger.info("[MAIN] Waiting for user to initiate navigation via GUI...")
        if not wait_for_start_flag():
            logger.info("[MAIN] Navigation not started by user. Shutting down.")
            shutdown_all(main_proc, slam_proc, stream_proc, ffmpeg_proc, slam_video_path)
            sys.exit(0)

        logger.info("[MAIN] Waiting for main.py to finish...")
        main_proc.wait()
        logger.info("[MAIN] main.py completed.")

    finally:
        logger.info("[MAIN] Final shutdown sequence.")
        shutdown_all(main_proc, slam_proc, stream_proc, ffmpeg_proc, slam_video_path)

if __name__ == "__main__":
    # --- CLEANUP FLAGS FIRST ---
    for flag in [AIRSIM_READY_FLAG, SLAM_READY_FLAG, SLAM_FAILED_FLAG, START_NAV_FLAG, flags_dir / "stop.flag", flags_dir / "nav_mode.flag"]:
        try:
            flag.unlink()
        except FileNotFoundError:
            pass
        except PermissionError:
            logger.warning(f"Could not delete {flag} due to permission error.")

    # --- LAUNCH GUI ---
    param_refs = {
        "L": [0.0],
        "C": [0.0],
        "R": [0.0],
        "state": ["idle"]
    }
    start_gui(param_refs)

    # Wait for user to select nav mode and launch simulation
    logger.info("Waiting for user to select navigation mode and launch simulation...")
    while not os.path.exists("flags/nav_mode.flag"):
        if os.path.exists("flags/stop.flag"):
            logger.info("Stop flag detected before simulation started. Shutting down.")
            sys.exit(0)
        time.sleep(0.2)

    # Read selected navigation mode
    with open("flags/nav_mode.flag") as f:
        selected_nav_mode = f.read().strip()
    logger.info(f"User selected navigation mode: {selected_nav_mode}")

    # --- NOW CALL MAIN() TO LAUNCH THE SIMULATION ---
    try:
        main(timestamp)
    finally:
        retain_recent_logs("logs")
