# uav/utils.py
"""Utility helpers for AirSim drone state and image processing."""
import math
import os
import fnmatch
import logging
import cv2
import numpy as np
import airsim
from datetime import datetime
from uav.config import FLOW_STD_MAX

logger = logging.getLogger(__name__)

def init_client(client):
    """Enable API control, arm the drone and confirm the connection."""
    try:
        client.enableApiControl(True)
        client.armDisarm(True)
        if hasattr(client, "confirmConnection"):
            client.confirmConnection()
    except Exception as e:
        logger.warning("Client init failed: %s", e)
    return client

def apply_clahe(gray_image):
    """Improve contrast of a grayscale image using CLAHE."""
    clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(4, 4))
    return clahe.apply(gray_image)


def get_yaw(orientation):
    """Return yaw angle in degrees from an AirSim quaternion."""
    return math.degrees(airsim.to_eularian_angles(orientation)[2])


def get_speed(velocity):
    """Compute the speed magnitude of a velocity vector."""
    return np.linalg.norm([velocity.x_val, velocity.y_val, velocity.z_val])


def get_drone_state(client):
    """Fetch position, yaw and speed from the AirSim client."""
    try:
        state = client.getMultirotorState()
    except Exception as e:
        logger.warning("State fetch error: %s", e)
        return airsim.Vector3r(0, 0, 0), 0.0, 0.0
    pos = state.kinematics_estimated.position
    ori = state.kinematics_estimated.orientation
    yaw = get_yaw(ori)
    vel = state.kinematics_estimated.linear_velocity
    speed = get_speed(vel)
    return pos, yaw, speed


def _timestamp_from_name(path: str) -> float:
    """Return a UNIX timestamp parsed from ``full_log_YYYYMMDD_HHMMSS.csv``.

    Falls back to the file's modification time if parsing fails.
    """
    name = os.path.basename(path)
    ts = name[len("full_log_"):-len(".csv")]
    try:
        dt = datetime.strptime(ts, "%Y%m%d_%H%M%S")
        return dt.timestamp()
    except Exception:
        return os.path.getmtime(path)


def retain_recent_logs(log_dir: str, keep: int = 2) -> None:
    """
    Keep only the ``keep`` most recent logs for each log type in the logs folder.
    Handles: full_log_*.csv, launch_*.log, slam_server_debug_*.log, pose_log_*.txt
    """
    log_patterns = [
        "full_log_*.csv",
        "launch_*.log",
        "slam_*.log",
        "pose_*.txt",
        "pose_*.log",
        "airsim_*.log",
    ]
    for pattern in log_patterns:
        try:
            files = [
                os.path.join(log_dir, f)
                for f in os.listdir(log_dir)
                if fnmatch.fnmatch(f, pattern)
            ]
        except FileNotFoundError:
            logger.warning("\u26A0\uFE0F Log directory '%s' not found.", log_dir)
            continue
        files.sort(key=_timestamp_from_name, reverse=True)
        for old_file in files[keep:]: # Keep the most recent `keep` files
            try:
                os.remove(old_file)
            except OSError as e:
                logger.warning("\u26A0\uFE0F Could not delete %s: %s", old_file, e)

def retain_recent_files(dir_path: str, pattern: str, keep: int = 5) -> None:
    """Keep only the ``keep`` most recent files matching ``pattern``.

    Parameters
    ----------
    dir_path : str
        Directory to search for files.
    pattern : str
        Glob pattern used to select files within ``dir_path``.
    keep : int, optional
        Number of recent files to preserve. Older files are removed.
    """
    try:
        files = [
            os.path.join(dir_path, f)
            for f in os.listdir(dir_path)
            if fnmatch.fnmatch(f, pattern)
        ]
    except FileNotFoundError:
        return

    files.sort(key=os.path.getmtime, reverse=True)

    for old_file in files[keep:]:
        try:
            os.remove(old_file)
        except OSError:
            pass

def retain_recent_views(view_dir: str, keep: int = 5) -> None:
    """Keep only the ``keep`` most recent ``flight_view_*.html`` files."""

    retain_recent_files(view_dir, "flight_view_*.html", keep)

def should_flat_wall_dodge(
    center_mag: float,
    probe_mag: float,
    probe_count: int,
    min_probe_features: int = 5,
    flow_std: float = 0.0,
    std_threshold: float = FLOW_STD_MAX,
) -> bool:
    """Return ``True`` when flow indicates a likely flat wall ahead."""

    if flow_std >= std_threshold:
        return False

    if probe_count < min_probe_features:
        return False

    return probe_mag < 0.5 and center_mag > 0.7
