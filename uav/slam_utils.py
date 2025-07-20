# uav/slam_utils.py
import logging
import subprocess
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

def is_slam_stable():
    """
    Checks the stability of the SLAM system.
    
    Returns True if SLAM is stable, otherwise False.
    """
    pose = slam_receiver.get_latest_pose()
    
    if pose is None:
        logger.warning("[SLAM] No pose data available. SLAM is unstable.")
        return False
    
    # Get the covariance of the current pose
    pose_covariance = slam_receiver.get_latest_covariance()
    inliers = slam_receiver.get_latest_inliers()

    if pose_covariance is None or inliers is None:
        logger.warning(
            "[SLAM] Covariance or inlier data unavailable. Assuming stable."
        )
        return True
    
    # Check if pose covariance is within acceptable limits
    if pose_covariance > COVARIANCE_THRESHOLD:
        logger.warning("[SLAM] High pose covariance detected. SLAM is unstable.")
        return False
    
    # Check the number of inliers detected by SLAM

    if inliers < MIN_INLIERS_THRESHOLD:
        logger.warning("[SLAM] Low inlier count. SLAM is unstable.")
        return False
    
    return True


def is_obstacle_ahead(
    client, depth_threshold: float = 2.0, vehicle_name: str = "UAV"
) -> Tuple[bool, Optional[float]]:
    """Check if a depth obstacle is within ``depth_threshold`` meters."""
    from airsim import ImageRequest, ImageType

    logger.info("[Obstacle Check] Checking for obstacles ahead.")
    try:
        responses = client.simGetImages(
            [ImageRequest("oakd_camera", ImageType.DepthPlanar, True)],
            vehicle_name=vehicle_name,
        )
        if not responses or responses[0].height == 0:
            logger.error(
                "[Obstacle Check] No depth image received or image height is zero."
            )
            return False, None

        depth_image = airsim.get_pfm_array(responses[0])
        h, w = depth_image.shape
        cx, cy = w // 2, h // 2
        roi = depth_image[cy - 20 : cy + 20, cx - 20 : cx + 20]
        mean_depth = np.nanmean(roi)
        return mean_depth < depth_threshold, float(mean_depth)
    except Exception as e:  # pragma: no cover - defensive
        logger.error("[Obstacle Check] Depth read failed: %s", e)
        return False, None


def generate_pose_comparison_plot() -> None:
    """Invoke the pose comparison plotting helper script."""
    logger.info("[Plotting] Generating pose comparison plot.")
    try:
        logger.info("[Plotting] Running pose_comparison_plotter.py script.")
        result = subprocess.run(
            ["python", "slam_bridge/pose_comparison_plotter.py"],
            check=True,
            capture_output=True,
            text=True,
        )
        print("[Plotting] Pose comparison plot generated.")
        print(result.stdout)
    except subprocess.CalledProcessError as e:  # pragma: no cover - defensive
        logger.error("[Plotting] Failed to generate pose comparison plot.")
        print("[Plotting] Failed to generate plot:")
        print(e.stderr)
