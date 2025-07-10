import subprocess
import time
import socket
import os
from pathlib import Path
import sys
from datetime import datetime
import webbrowser
import logging
from uav.cli import parse_args
from uav.config import load_app_config
from uav.logging_config import setup_logging

import pygetwindow as gw

# --- Add GUI import ---
from uav.interface import start_gui

# --- Logging setup ---
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
logfile = log_dir / f"launch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
setup_logging(logfile.name)
logger = logging.getLogger(__name__)
logger.info(f"Logging to {logfile}")

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
    # --- CLEAN UP streamer ---
    if stream_proc is not None:
        logger.info("Terminating SLAM streamer")
        stream_proc.terminate()
        try:
            stream_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning("Forcing SLAM streamer shutdown...")
            stream_proc.kill()

    # --- CLEAN UP SLAM ---
    if slam_proc is not None:
        logger.info("Terminating SLAM backend")
        slam_proc.terminate()
        try:
            slam_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning("Forcing SLAM backend shutdown...")
            slam_proc.kill()

    # --- CLEAN UP FFMPEG ---
    if isinstance(ffmpeg_proc, subprocess.Popen):
        logger.info("Terminating screen recording")
        ffmpeg_proc.terminate()
        try:
            ffmpeg_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning("Forcing screen recorder shutdown...")
            ffmpeg_proc.kill()

    # --- CLEAN UP main.py ---
    if main_proc is not None:
        logger.info("Terminating main script")
        main_proc.terminate()
        try:
            main_proc.wait(timeout=5)
            logger.info("Reactive navigation loop complete.")
        except subprocess.TimeoutExpired:
            logger.warning("Forcing main script shutdown...")
            main_proc.kill()

    # --- CLEAN UP Unreal Engine (UE4) ---
    pid_file = flags_dir / "ue4_sim.pid"
    if pid_file.exists():
        try:
            ue4_pid = int(pid_file.read_text())
            logger.info(f"Terminating UE4 simulation (PID {ue4_pid})")
            subprocess.call(["taskkill", "/F", "/PID", str(ue4_pid)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            logger.warning(f"Failed to terminate UE4 by PID: {e}")
        finally:
            pid_file.unlink(missing_ok=True)
    else:
        logger.warning("No UE4 PID file found — attempting forced shutdown via taskkill")
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
    """Poll for the existence of ``flag_path`` up to ``timeout`` seconds."""
    logger.info(f"Waiting for {flag_path}...")
    start = time.time()
    while not os.path.exists(flag_path):
        if time.time() - start > timeout:
            logger.error(f"Timeout waiting for {flag_path}")
            return False
        time.sleep(0.5)
    logger.info(f"{flag_path} found.")
    return True


def start_streamer(host: str, port: int):
    """Start the Python image streamer used for SLAM communication."""
    proc = subprocess.Popen([
        "python",
        "slam_bridge/stream_airsim_image.py",
        "--host", host,
        "--port", str(port),
    ])
    logger.info("Started SLAM image streamer")
    # Give the streamer a moment to send the first frame
    time.sleep(2)
    return proc


def launch_slam_backend(receiver_host: str, receiver_port: int):
    """Launch the SLAM backend via WSL and return the process handle."""
    slam_cmd = [
        "wsl", "bash", "-c",
        f"export POSE_RECEIVER_IP={receiver_host}; "
        f"export POSE_RECEIVER_PORT={receiver_port}; "
        "cd /mnt/h/Documents/AirSimExperiments/Hybrid_Navigation/linux_slam/build && ./app/tcp_slam_server ../Vocabulary/ORBvoc.txt ../app/rgbd_settings.yaml",
    ]
    proc = subprocess.Popen(slam_cmd)
    logger.info("Started SLAM backend in WSL")
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

def main():
    args = parse_args()
    config = load_app_config(args.config)
    slam_server_host = args.slam_server_host or config.get("network", "slam_server_host", fallback="127.0.0.1")
    slam_server_port = int(args.slam_server_port or config.get("network", "slam_server_port", fallback="6000"))
    slam_receiver_host = args.slam_receiver_host or config.get("network", "slam_receiver_host", fallback="127.0.0.1")
    slam_receiver_port = int(args.slam_receiver_port or config.get("network", "slam_receiver_port", fallback="6001"))

    main_proc = slam_proc = stream_proc = ffmpeg_proc = None
    slam_video_path = None

    try:
        # --- STEP 1: Launch Unreal + main.py ---
        main_proc = subprocess.Popen([
            "python", "main.py", "--nav-mode", args.nav_mode,
            "--slam-server-host", slam_server_host,
            "--slam-server-port", str(slam_server_port),
            "--slam-receiver-host", slam_receiver_host,
            "--slam-receiver-port", str(slam_receiver_port),
        ])

        logger.info("Started Unreal Engine + main script (idle)")
        logger.info("Giving Unreal Engine time to finish loading map and camera...")

        # --- STEP 2: Wait for AirSim to fully launch ---
        if not wait_for_flag(AIRSIM_READY_FLAG, timeout=20):
            shutdown_all(main_proc)
            sys.exit(1)

        if args.nav_mode == "slam":
            # --- STEP 3: Launch image streamer BEFORE waiting for slam_ready.flag
            stream_proc = start_streamer(slam_server_host, slam_server_port)

            # --- STEP 4: Launch SLAM backend in WSL ---
            slam_proc = launch_slam_backend(slam_receiver_host, slam_receiver_port)

            # --- STEP 4c: Wait for slam_ready.flag (now that streamer can talk to SLAM)
            if not wait_for_flag(SLAM_READY_FLAG, timeout=30):
                logger.error("SLAM backend never received first image — shutting down.")
                time.sleep(2)  # Give some time for SLAM to log the error
                shutdown_all(main_proc, slam_proc, stream_proc)
                sys.exit(1)

            # --- STEP 5: Wait for Pangolin window to appear ---
            try:
                wait_for_window("ORB-SLAM2", timeout=20)
            except TimeoutError as e:
                logger.error(e)
                logger.info("Shutting down due to missing SLAM visualization window...")
                shutdown_all(main_proc, slam_proc)
                sys.exit(1)

            # --- STEP 6: Start screen recording ---
            ffmpeg_proc, slam_video_path = record_slam_video("ORB-SLAM2")

            # --- STEP 6.5: Abort if SLAM failed to receive first image ---
            if os.path.exists(SLAM_FAILED_FLAG):
                logger.error("SLAM backend reported failure — aborting simulation.")
                shutdown_all(main_proc, slam_proc, stream_proc, ffmpeg_proc, slam_video_path)
                SLAM_FAILED_FLAG.unlink(missing_ok=True)
                sys.exit(1)


        # --- STEP 7: Wait for user to initiate navigation via GUI ---
        if not wait_for_start_flag():
            shutdown_all(main_proc, slam_proc, stream_proc, ffmpeg_proc, slam_video_path)
            sys.exit(0)

        # --- STEP 8: Wait for main process to finish ---
        main_proc.wait()
        logger.info("main.py completed")

    finally:
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
    main()
