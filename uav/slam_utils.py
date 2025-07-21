"""Helpers for interfacing with the SLAM backend."""

import logging
import subprocess
import sys
from typing import Tuple, Optional

import numpy as np
# from slam_bridge.slam_receiver import get_latest_pose, get_latest_covariance, get_latest_inliers  # Import functions from slam_receiver.py
import slam_bridge.slam_receiver as slam_receiver

# Public API
__all__ = [
    "is_slam_stable",
    "is_obstacle_ahead",
    "generate_pose_comparison_plot",
]
# You should define these thresholds as constants.
COVARIANCE_THRESHOLD = 1.0  # Example threshold for pose covariance
MIN_INLIERS_THRESHOLD = 50  # Example threshold for the minimum number of inliers

logger = logging.getLogger(__name__)

def is_slam_stable(
    covariance_threshold: Optional[float] = COVARIANCE_THRESHOLD,
    inlier_threshold: Optional[int] = MIN_INLIERS_THRESHOLD,
) -> bool:
    """Check the SLAM system stability.

    Parameters
    ----------
    covariance_threshold : float, optional
        Maximum allowed pose covariance.
    inlier_threshold : int, optional
        Minimum number of map inliers required.

    Returns
    -------
    bool
        ``True`` if the latest SLAM pose data is deemed stable.
    """
    if covariance_threshold is None:
        covariance_threshold = COVARIANCE_THRESHOLD
    if inlier_threshold is None:
        inlier_threshold = MIN_INLIERS_THRESHOLD

    pose = slam_receiver.get_latest_pose()
    
    if pose is None:
        logger.warning("[SLAM] No pose data available. SLAM is unstable.")
        return False
    
    # Get the covariance of the current pose
    pose_covariance = slam_receiver.get_latest_covariance()
    inliers = slam_receiver.get_latest_inliers()

    if pose_covariance is None or inliers is None:
        logger.warning(
            "[SLAM] Covariance or inlier data unavailable. SLAM is unstable."
        )
        return False
    
    # Check if pose covariance is within acceptable limits
    if pose_covariance > covariance_threshold:
        logger.warning("[SLAM] High pose covariance detected. SLAM is unstable.")
        return False
    
    # Check the number of inliers detected by SLAM

    if inliers < inlier_threshold:
        logger.warning("[SLAM] Low inlier count. SLAM is unstable.")
        return False
    
    return True

def generate_pose_comparison_plot() -> None:
    """Invoke the pose comparison plotting helper script.

    The helper script compares the logged SLAM trajectory against the
    ground truth and saves an HTML report.
    """
    logger.info("[Plotting] Generating pose comparison plot.")
    try:
        logger.info("[Plotting] Running pose_comparison_plotter.py script.")
        result = subprocess.run(
            [sys.executable, "slam_bridge/pose_comparison_plotter.py"],
            check=True,
            capture_output=True,
            text=True,
        )
        logger.info("[Plotting] Pose comparison plot generated.")
        logger.info(result.stdout)
    except subprocess.CalledProcessError as e:  # pragma: no cover - defensive
        logger.error(e.stderr)
