import subprocess
import time
import socket
import os
from pathlib import Path
import sys
from datetime import datetime
import webbrowser
import logging

import pygetwindow as gw

# --- Logging setup ---
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
logfile = log_dir / f"launch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler(logfile, mode='w', encoding='utf-8'), # Uncomment this line to log to a file
        logging.StreamHandler(sys.stdout) # Uncomment this line to also log to console
    ]
)
logging.info(f"Logging to {logfile}")

# --- Flag paths ---
flags_dir = Path("flags")
flags_dir.mkdir(exist_ok=True)
AIRSIM_READY_FLAG = flags_dir / "airsim_ready.flag"
SLAM_READY_FLAG = flags_dir / "slam_ready.flag"
SLAM_FAILED_FLAG = flags_dir / "slam_failed.flag"
START_NAV_FLAG = flags_dir / "start_nav.flag"

def shutdown_all(main_proc=None, slam_proc=None, stream_proc=None, ffmpeg_proc=None, slam_video_path=None, sim_proc=None):
    # --- CLEAN UP streamer ---
    if stream_proc is not None:
        logging.info("Terminating SLAM streamer")
        stream_proc.terminate()
        try:
            stream_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logging.warning("Forcing SLAM streamer shutdown...")
            stream_proc.kill()

    # --- CLEAN UP SLAM ---
    if slam_proc is not None:
        logging.info("Terminating SLAM backend")
        slam_proc.terminate()
        try:
            slam_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logging.warning("Forcing SLAM backend shutdown...")
            slam_proc.kill()

    # --- CLEAN UP FFMPEG ---
    if ffmpeg_proc is not None:
        logging.info("Terminating screen recording")
        ffmpeg_proc.terminate()
        try:
            ffmpeg_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logging.warning("Forcing screen recorder shutdown...")
            ffmpeg_proc.kill()

    # --- CLEAN UP main.py ---
    if main_proc is not None:
        logging.info("Terminating main script")
        main_proc.terminate()
        try:
            main_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logging.warning("Forcing main script shutdown...")
            main_proc.kill()

    # --- CLEAN UP Unreal Engine (UE4) ---
    pid_file = flags_dir / "ue4_sim.pid"
    if pid_file.exists():
        try:
            ue4_pid = int(pid_file.read_text())
            logging.info(f"Terminating UE4 simulation (PID {ue4_pid})")
            subprocess.call(["taskkill", "/F", "/PID", str(ue4_pid)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            logging.warning(f"Failed to terminate UE4 by PID: {e}")
        finally:
            pid_file.unlink(missing_ok=True)
    else:
        logging.warning("No UE4 PID file found — attempting forced shutdown via taskkill")
        subprocess.call(["taskkill", "/F", "/IM", "Blocks.exe"],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.call(["taskkill", "/F", "/IM", "UE4Editor.exe"],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


    # Check if video was saved
    if slam_video_path and not os.path.exists(slam_video_path):
        logging.warning("SLAM video file not created.")

    # Automatically open video if it exists
    # if slam_video_path and os.path.exists(slam_video_path):
    #     webbrowser.open(slam_video_path)

def wait_for_window(title_substring, timeout=20):
    logging.info(f"Waiting for window containing title: '{title_substring}'...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        titles = gw.getAllTitles()
        for title in titles:
            if title_substring.lower() in title.lower():
                logging.info(f"Window found: '{title}'")
                return True
        time.sleep(0.5)
    raise TimeoutError(f"Timeout waiting for window with title containing: '{title_substring}'")

def wait_for_flag(flag_path, timeout=15):
    logging.info(f"Waiting for {flag_path}...")
    start = time.time()
    while not os.path.exists(flag_path):
        if time.time() - start > timeout:
            logging.error(f"Timeout waiting for {flag_path}")
            return False
        time.sleep(0.5)
    logging.info(f"{flag_path} found.")
    return True

def wait_for_port(host, port, timeout=10):
    start_time = time.time()
    while time.time() - start_time < timeout:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1)
            if sock.connect_ex((host, port)) == 0:
                return True
        time.sleep(0.5)
    return False

def main():
    main_proc = slam_proc = stream_proc = ffmpeg_proc = None
    slam_video_path = None
    try:
        # --- STEP 1: Launch Unreal + main.py ---
        main_proc = subprocess.Popen(["python", "main.py", "--nav-mode", "slam"]) # Launch main.py with SLAM mode
        logging.info("Started Unreal Engine + main script (idle)")
        logging.info("Giving Unreal Engine time to finish loading map and camera...")

        # --- STEP 2: Wait for AirSim to fully launch ---
        if not wait_for_flag(AIRSIM_READY_FLAG, timeout=20):
            shutdown_all(main_proc)
            sys.exit(1)

        # --- STEP 3: Launch image streamer BEFORE waiting for slam_ready.flag
        stream_proc = subprocess.Popen(["python", "slam_bridge/stream_airsim_image.py"])
        logging.info("Started SLAM image streamer")
        
        # --- STEP 3.5: Wait a moment for first image to be sent ---
        time.sleep(2)

        # --- STEP 4: Launch SLAM backend in WSL ---
        slam_cmd = [
            "wsl", "bash", "-c",
            "cd /mnt/h/Documents/AirSimExperiments/Hybrid_Navigation/linux_slam/build && ./app/tcp_slam_server ../Vocabulary/ORBvoc.txt ../app/rgbd_settings.yaml"
        ]

        slam_proc = subprocess.Popen(slam_cmd)
        logging.info("Started SLAM backend in WSL")

        # --- STEP 4: Wait for port 6000 (TCP) and flag ---
        if not wait_for_port("127.0.0.1", 6000, timeout=10):
            logging.error("SLAM port 6000 not open.")
            shutdown_all(main_proc, slam_proc)
            sys.exit(1)


        # --- STEP 4c: Wait for slam_ready.flag (now that streamer can talk to SLAM)
        if not wait_for_flag(SLAM_READY_FLAG, timeout=15):
            logging.error("SLAM backend never received first image — shutting down.")
            shutdown_all(main_proc, slam_proc, stream_proc)
            sys.exit(1)

        # --- Debug: List all open window titles ---
        logging.debug("Current open windows:")
        for title in gw.getAllTitles():
            logging.debug("  - %s", title)

        # --- Wait for Pangolin window to appear ---
        try:
            wait_for_window("ORB-SLAM2", timeout=20)
        except TimeoutError as e:
            logging.error(e)
            logging.info("Shutting down due to missing SLAM visualization window...")
            shutdown_all(main_proc, slam_proc)
            sys.exit(1)

        # Generate timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        slam_video_path = f"analysis/slam_output_{timestamp}.mp4"

        # Screen capture command (adjust coordinates as needed)
        # Find the window title containing "ORB-SLAM2"
        window_title = None
        for title in gw.getAllTitles():
            if "ORB-SLAM2" in title:
                window_title = title
                break

        if not window_title:
            logging.error("Could not find window with title containing 'ORB-SLAM2'.")
            shutdown_all(main_proc, slam_proc)
            sys.exit(1)

        ffmpeg_cmd = [
            "ffmpeg",
            "-hide_banner",              # ✅ hides config/compile info
            "-loglevel", "error",        # ✅ only show actual errors (or use "warning" for minimal logs)
            "-y",
            "-f", "gdigrab",
            "-framerate", "30",
            "-i", f"title={window_title}",
            "-t", "60",
            slam_video_path
        ]

        try:
            ffmpeg_proc = subprocess.Popen(ffmpeg_cmd) # Start ffmpeg process
        except FileNotFoundError:
            logging.error("ffmpeg executable not found. Please install ffmpeg and ensure it is in your PATH.")
            shutdown_all(main_proc, slam_proc, stream_proc, None, slam_video_path)
            sys.exit(1)
        logging.info("Started screen recording to %s", slam_video_path)

        # --- STEP 6.5: Abort if SLAM failed to receive first image ---
        if os.path.exists(SLAM_FAILED_FLAG):
            logging.error("SLAM backend reported failure — aborting simulation.")
            shutdown_all(main_proc, slam_proc, stream_proc, ffmpeg_proc, slam_video_path)
            SLAM_FAILED_FLAG.unlink(missing_ok=True)
            sys.exit(1)

        # --- STEP 7: Touch start_nav.flag to begin navigation ---
        START_NAV_FLAG.touch()
        logging.info("Signaling navigation to begin...")

        # --- STEP 8: Wait for main process to finish ---
        main_proc.wait()
        logging.info("main.py completed")

    finally:
        shutdown_all(main_proc, slam_proc, stream_proc, ffmpeg_proc, slam_video_path)

if __name__ == "__main__":
    # --- CLEANUP FLAGS FIRST ---
    for flag in [AIRSIM_READY_FLAG, SLAM_READY_FLAG, SLAM_FAILED_FLAG, START_NAV_FLAG]:
        try:
            flag.unlink()
        except FileNotFoundError:
            pass
        except PermissionError:
            logging.warning(f"Could not delete {flag} due to permission error.")

    main()
