"""SLAM initialization maneuvers for AirSim flights."""

import airsim
import time
import math
import logging

logger = logging.getLogger(__name__)

def run_slam_bootstrap(client, duration=8.0, vehicle_name="UAV"):
    """
    Perform rich camera motion to help SLAM initialize.
    Includes forward motion, lateral zigzag, and yaw rotations.
    The drone ends the manoeuvre facing forward (yaw=0).
    """
    logger.info("[SLAM_BOOT] Starting SLAM bootstrap motion...")

    start_time = time.time()
    elapsed = 0

    speed = 1.5  # m/s
    zigzag_amplitude = 2.0  # side-to-side in meters
    yaw_amplitude = 15  # degrees

    while elapsed < duration:
        t = time.time() - start_time
        elapsed = t

        # Zigzag sideways
        lateral = math.sin(t * 2.0) * speed  # oscillate
        forward = speed
        yaw = math.sin(t * 1.5) * yaw_amplitude

        vx = forward
        vy = lateral
        vz = 0  # keep altitude constant

        # Send velocity command
        client.moveByVelocityAsync(vx, vy, vz, duration=0.1, vehicle_name=vehicle_name)

        # Send yaw command (optional — adds angular variation)
        yaw_rad = math.radians(yaw)
        if hasattr(client, "rotateToYawAsync"):
            client.rotateToYawAsync(math.degrees(yaw_rad), vehicle_name=vehicle_name)
        else:
            # Older stubs used in tests may not implement this method
            logger.debug("rotateToYawAsync not available on client stub")

        time.sleep(0.1)

    # Stop the drone at the end and ensure it faces forward
    client.moveByVelocityAsync(0, 0, 0, 1.0, vehicle_name=vehicle_name)
    if hasattr(client, "rotateToYawAsync"):
        client.rotateToYawAsync(0, vehicle_name=vehicle_name)
    else:
        logger.debug("rotateToYawAsync not available on client stub")
    logger.info("[SLAM_BOOT] Bootstrap motion complete.")

