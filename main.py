import threading
import time
import os
from pathlib import Path
from uav.paths import STOP_FLAG_PATH
from uav import launch_utils as lutils
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
    "pose_plotter": f"pose_plotter_{timestamp}.log",
    "perception": f"perception_{timestamp}.log",
    "utils": f"utils_{timestamp}.log",
}
setup_logging(log_file=f"main_{timestamp}.log", module_logs=module_logs, level=logging.DEBUG)
logger = logging.getLogger("main")
logger.info(
    f"[main.py] Logging configured. Writing to logs/main_{timestamp}.log and module logs..."
)

# --- Flag paths ---
flags_dir = Path("flags")
flags_dir.mkdir(exist_ok=True)
START_FLAG_PATH = flags_dir / "start_nav.flag"


def get_settings_path(args, config):
    """Resolve the path to the AirSim settings file."""
    if args.settings_path:
        return args.settings_path
    try:
        return config.get("paths", "settings")
    except Exception:
        return None


def get_arg_or_config(args, config, name, section, option, default=None):
    """Return ``args.name`` if set, otherwise value from ``config``.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line arguments.
    config : configparser.ConfigParser
        Loaded configuration object.
    name : str
        Attribute name on ``args``.
    section : str
        Section in the configuration file.
    option : str
        Option within ``section`` to read.
    default : Any, optional
        Value returned if neither argument nor config option is provided.
    """

    value = getattr(args, name, None)
    if value is not None:
        return value

    getter = config.get
    if isinstance(default, int):
        getter = config.getint
    elif isinstance(default, float):
        getter = config.getfloat

    try:
        if default is None:
            return getter(section, option)
        return getter(section, option, fallback=default)
    except Exception:
        return default

def main() -> None:
    # Safety: fallback logging if nothing is configured
    ctx = None
    logger = logging.getLogger(__name__)
    if not logger.hasHandlers():
        from uav.logging_config import setup_logging
        setup_logging(log_file="fallback_main.log", module_logs={__name__: "fallback_module.log"})
        logger = logging.getLogger(__name__)
        logger.warning("Logger was missing handlers. Reconfigured logging as fallback.")

    from uav.nav_runtime import setup_environment, start_perception_thread, cleanup
    from uav.nav_loop import navigation_loop, slam_navigation_loop
    from slam_bridge.slam_receiver import start_receiver, stop_receiver, set_state_ref
    from slam_bridge.slam_plotter import plot_slam_trajectory

    args = parse_args()
    config = load_app_config(args.config)
    settings_path = get_settings_path(args, config)

    args.slam_covariance_threshold = get_arg_or_config(
        args, config, "slam_covariance_threshold", "slam", "covariance_threshold", None
    )
    args.slam_inlier_threshold = get_arg_or_config(
        args, config, "slam_inlier_threshold", "slam", "inlier_threshold", None
    )

    slam_server_host = get_arg_or_config(
        args, config, "slam_server_host", "network", "slam_server_host", "127.0.0.1"
    )
    slam_server_port = int(
        get_arg_or_config(
            args, config, "slam_server_port", "network", "slam_server_port", 6000
        )
    )
    slam_receiver_host = "0.0.0.0"
    logger.info(f"[main.py] SLAM receiver host resolved to: {slam_receiver_host}")

    # Resolve the SLAM receiver port from command line or config
    slam_receiver_port = int(
        get_arg_or_config(
            args, config, "slam_receiver_port", "network", "slam_receiver_port", 6001
        )
    )

    # Resolve the pose source from command line or config
    pose_source = args.slam_pose_source
    if "--slam-pose-source" not in remaining_argv:
        try:
            pose_source = config.get("slam", "pose_source")
        except Exception:
            pose_source = args.slam_pose_source

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
        lutils.wait_for_flag(START_FLAG_PATH)
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
            slam_navigation_loop(args, client, ctx, config, pose_source=pose_source)
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
            logger.info("[main.py] SLAM navigation loop finished - Calling cleanup.")
            cleanup(client, sim_process, ctx if ctx is not None else None)
    elif nav_mode == "reactive":
        if STOP_FLAG_PATH.exists():
            logger.info("[main.py] Stop flag detected before navigation started. Exiting.")
            cleanup(client, sim_process, ctx)
            return

        lutils.wait_for_flag(START_FLAG_PATH)
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
