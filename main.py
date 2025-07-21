import threading
import time
import os
from pathlib import Path
from uav.paths import STOP_FLAG_PATH
import queue
from datetime import datetime
import cv2

from uav.cli import parse_args
from uav.sim_launcher import launch_sim
import airsim
from uav.utils import FLOW_STD_MAX, init_client
from uav.config import load_app_config
import logging
import argparse
import sys
from uav.logging_config import setup_logging

# Parse the timestamp early for logging
parser = argparse.ArgumentParser()
parser.add_argument("--log-timestamp", type=str, default=None)
args, remaining_argv = parser.parse_known_args()
timestamp = args.log_timestamp or datetime.now().strftime('%Y%m%d_%H%M%S')

# Fix sys.argv so parse_args still works
sys.argv = [sys.argv[0]] + remaining_argv

# Set up logging just like in launch_all.py
module_logs = {
    "nav_loop": f"nav_loop_{timestamp}.log",
    "slam_receiver": f"slam_receiver_{timestamp}.log",
    "pose_receiver": f"pose_receiver_{timestamp}.log",
    "slam_plotter": f"slam_plotter_{timestamp}.log",
    "pose_plotter": f"pose_plotter_{timestamp}.log"
}
setup_logging(log_file=f"main_{timestamp}.log", module_logs=module_logs, level=logging.DEBUG)
print(f"[main.py] Logging configured. Writing to logs/main_{timestamp}.log and module logs...")

# Then import the rest
logger = logging.getLogger("main")

# --- Flag paths ---
flags_dir = Path("flags")
flags_dir.mkdir(exist_ok=True)
START_FLAG_PATH = flags_dir / "start_nav.flag"
SETTINGS_PATH = r"C:\Users\Jacob\OneDrive\Documents\AirSim\settings.json"


def get_settings_path(args, config):
    try:
        return args.settings_path or config.get("paths", "settings")
    except Exception:
        return SETTINGS_PATH

def wait_for_nav_trigger():
    logger.info("[INFO] Waiting for navigation start flag...")
    while not START_FLAG_PATH.exists():
        time.sleep(1)
    logger.info("[INFO] Navigation start flag found. Beginning nav logic...")

def main() -> None:
    # Safety: fallback logging if nothing is configured
    ctx = None
    logger = logging.getLogger(__name__)
    if not logger.hasHandlers():
        from uav.logging_config import setup_logging
        setup_logging(log_file="fallback_main.log", module_logs={__name__: "fallback_module.log"})
        logger = logging.getLogger(__name__)
        logger.warning("Logger was missing handlers. Reconfigured logging as fallback.")

    from uav.nav_loop import setup_environment, start_perception_thread, navigation_loop, slam_navigation_loop, cleanup
    from slam_bridge.slam_receiver import start_receiver, stop_receiver, set_state_ref
    from slam_bridge.slam_plotter import plot_slam_trajectory

    args = parse_args()
    config = load_app_config(args.config)
    settings_path = get_settings_path(args, config)

    if args.slam_covariance_threshold is None:
        try:
            args.slam_covariance_threshold = config.getfloat(
                "slam", "covariance_threshold"
            )
        except Exception:
            pass
    if args.slam_inlier_threshold is None:
        try:
            args.slam_inlier_threshold = config.getint(
                "slam", "inlier_threshold"
            )
        except Exception:
            pass

    slam_server_host = args.slam_server_host or config.get("network", "slam_server_host", fallback="127.0.0.1")
    slam_server_port = int(args.slam_server_port or config.get("network", "slam_server_port", fallback="6000"))
    slam_receiver_host = "0.0.0.0"
    print(f"[main.py] SLAM receiver host resolved to: {slam_receiver_host}")

    slam_receiver_port = int(args.slam_receiver_port or config.get("network", "slam_receiver_port", fallback="6001"))

    # Add a nav_mode argument to your CLI parser (e.g., --nav-mode [slam|reactive])
    nav_mode = getattr(args, "nav_mode", "slam")  # Default to slam if not specified

    sim_process = launch_sim(args, settings_path, config)
    pid_path = Path("flags/ue4_sim.pid")
    pid_path.write_text(str(sim_process.pid))

    # Wait for the simulator to be ready before connecting the AirSim client
    max_attempts = 10
    for attempt in range(max_attempts):
        try:
            client = airsim.MultirotorClient()
            client.confirmConnection()
            break
        except Exception as e:
            logger.info(f"Waiting for simulator to be ready... (attempt {attempt+1}/{max_attempts})")
            time.sleep(2)
    else:
        logger.error("Failed to connect to AirSim simulator after multiple attempts.")
        cleanup(None, sim_process, None)
        return

    (flags_dir / "airsim_ready.flag").touch()
    logger.info("[INFO] AirSim + camera ready — flag set")

    logger.info("[TEST] nav_loop logger test")
    if nav_mode == "slam":
        receiver_thread = start_receiver(slam_receiver_host, slam_receiver_port)
        time.sleep(1) # Give the receiver time to start
        if STOP_FLAG_PATH.exists():
            logger.info("[main.py] Stop flag detected before navigation started. Exiting.")
            cleanup(client, sim_process, ctx)
            return
        wait_for_nav_trigger()
        init_client(client)

        ctx = None
        try:
            # import atexit
            # from slam_bridge.slam_plotter import save_interactive_plot
            # plotter_thread = threading.Thread(target=plot_slam_trajectory, daemon=True)
            # plotter_thread.start()
            # atexit.register(save_interactive_plot)

            ctx = setup_environment(args, client)
            set_state_ref(ctx.param_refs.state)
            start_perception_thread(ctx)
            slam_navigation_loop(args, client, ctx, config)
        finally:
            if receiver_thread:
                logger.info("[main.py] Stopping SLAM receiver thread...")
                stop_receiver()
                logger.info("[main.py] Waiting for SLAM receiver to finish...")
                receiver_thread.join()
                logger.info("[main.py] SLAM receiver thread joined successfully.")
            for flag in [flags_dir / "airsim_ready.flag", flags_dir / "start_nav.flag"]:
                try:
                    flag.unlink()
                except FileNotFoundError:
                    pass
            cleanup(client, sim_process, ctx if ctx is not None else None)
    elif nav_mode == "reactive":
        if STOP_FLAG_PATH.exists():
            logger.info("[main.py] Stop flag detected before navigation started. Exiting.")
            cleanup(client, sim_process, ctx)
            return

        wait_for_nav_trigger()
        init_client(client)

        ctx = setup_environment(args, client)
        start_perception_thread(ctx)

        # Logging: startup debug
        logger.info(f"🧭 Navigation loop starting with mode: {nav_mode}")
        logger.info(f"📌 Navigator object: {ctx.navigator}")
        logger.info(f"🗺️  Initial state: {ctx.param_refs.state[0]}")
        logger.info(f"📦 Perception thread running: {ctx.perception_thread.is_alive()}")

        # The navigation loop internally checks ``navigator.settling`` and
        # should proceed normally at this level.
        try:
            navigation_loop(args, client, ctx)

        finally:
            for flag in [flags_dir / "airsim_ready.flag", flags_dir / "start_nav.flag", flags_dir / "stop.flag"]:
                try:
                    flag.unlink()
                except FileNotFoundError:
                    pass

            # Close dummy log file to avoid warnings or open handles
            if ctx.log_file:
                try:
                    ctx.log_file.close()
                except Exception:
                    pass

            cleanup(client, sim_process, ctx)


    else:
        logger.error(f"Unknown navigation mode: {nav_mode}")
        cleanup(client, sim_process, ctx)

if __name__ == "__main__":
    main()
