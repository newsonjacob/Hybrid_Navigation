# uav/slam_utils.py
import logging
from slam_bridge.slam_receiver import get_latest_pose, get_latest_covariance, get_latest_inliers  # Import functions from slam_receiver.py

# You should define these thresholds as constants.
SOME_THRESHOLD = 1.0  # Example threshold for pose covariance
MIN_INLIERS_THRESHOLD = 50  # Example threshold for the minimum number of inliers

logger = logging.getLogger(__name__)

def is_slam_stable():
    """
    Checks the stability of the SLAM system.
    
    Returns True if SLAM is stable, otherwise False.
    """
    pose = get_latest_pose()
    
    if pose is None:
        logger.warning("[SLAM] No pose data available. SLAM is unstable.")
        return False
    
    # Get the covariance of the current pose
    pose_covariance = get_latest_covariance()
    
    if pose_covariance is None:
        logger.warning("[SLAM] Covariance data unavailable. SLAM is unstable.")
        return False
    
    # Check if pose covariance is within acceptable limits
    if pose_covariance > SOME_THRESHOLD:
        logger.warning("[SLAM] High pose covariance detected. SLAM is unstable.")
        return False
    
    # Check the number of inliers detected by SLAM
    inliers = get_latest_inliers()
    
    if inliers is None:
        logger.warning("[SLAM] Inlier count unavailable. SLAM is unstable.")
        return False
    
    if inliers < MIN_INLIERS_THRESHOLD:
        logger.warning("[SLAM] Low inlier count. SLAM is unstable.")
        return False
    
    return True
