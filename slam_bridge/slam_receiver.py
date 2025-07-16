from typing import Optional, Tuple

from .pose_receiver import PoseReceiver

import logging
import time
import argparse
import csv
from pathlib import Path
import threading
import numpy as np

logger = logging.getLogger("slam_receiver")

HOST = "192.168.1.100"  # Default IP if not provided
PORT = 6001


class SlamReceiver:
    """Manage a :class:`PoseReceiver` and log SLAM poses."""

    def __init__(self, host: str = HOST, port: int = PORT) -> None:
        self.receiver = PoseReceiver(host, port)
        self.pose_writer = None
        self.pose_log_file = None
        self.airsim_client = None
        self.log_thread_running = False
        self._log_thread: Optional[threading.Thread] = None

    @property
    def port(self) -> int:
        return self.receiver.port

    def start(self) -> None:
        if self.receiver is not None:
            self.receiver.start()
            logger.info("[SLAMReceiver] PoseReceiver started.")
            (Path("flags") / "pose_receiver_ready.flag").touch()

            log_dir = Path("analysis")
            log_dir.mkdir(exist_ok=True)

            csv_path = log_dir / "pose_comparison.csv"
            self.pose_log_file = open(csv_path, "w", newline="")
            self.pose_writer = csv.writer(self.pose_log_file)
            self.pose_writer.writerow([
                "timestamp",
                "slam_x",
                "slam_y",
                "slam_z",
                "gt_x",
                "gt_y",
                "gt_z",
            ])

            try:
                import airsim
                logger.info("[SLAMReceiver] Connecting to AirSim client...")
                self.airsim_client = airsim.MultirotorClient()
                self.airsim_client.confirmConnection()
            except Exception as e:
                logger.error("[SLAMReceiver] Failed to connect to AirSim: %s", e)
                return

            self.log_thread_running = True
            self._log_thread = threading.Thread(
                target=self._log_pose_loop, daemon=True
            )
            self._log_thread.start()
            logger.info("[SLAMReceiver] Pose logging thread started.")

    def stop(self) -> None:
        if self.receiver is not None:
            self.receiver.stop()

            self.log_thread_running = False
            if self._log_thread:
                self._log_thread.join(timeout=1)

            if self.pose_log_file:
                self.pose_log_file.close()

    def _log_pose_loop(self) -> None:
        while self.log_thread_running:
            slam_pose = self.get_latest_pose()
            logger.info(f"[PoseLogger] Latest SLAM pose: {slam_pose}")

            if slam_pose is None:
                logger.warning("[PoseLogger] get_latest_pose() returned None")
                time.sleep(0.05)
                continue

            if not isinstance(slam_pose, (list, tuple, np.ndarray)) or len(
                slam_pose
            ) != 3:
                logger.error(
                    "[PoseLogger] Invalid SLAM pose format: %s", slam_pose
                )
                time.sleep(0.05)
                continue

            try:
                assert self.airsim_client is not None
                gt_pose = self.airsim_client.simGetVehiclePose().position
                timestamp = time.time()

                slam_x, slam_y, slam_z = slam_pose
                gt_x, gt_y, gt_z = gt_pose.x_val, gt_pose.y_val, gt_pose.z_val

                logger.debug(
                    "[PoseLogger] SLAM: (%.2f, %.2f, %.2f) | GT: (%.2f, %.2f, %.2f)",
                    slam_x,
                    slam_y,
                    slam_z,
                    gt_x,
                    gt_y,
                    gt_z,
                )

                assert self.pose_writer is not None
                assert self.pose_log_file is not None
                self.pose_writer.writerow(
                    [timestamp, slam_x, slam_y, slam_z, gt_x, gt_y, gt_z]
                )
                self.pose_log_file.flush()
            except Exception as e:
                logger.warning("[PoseLogger] Error while logging poses: %s", e)
            time.sleep(0.05)

    def get_latest_pose(self) -> Optional[Tuple[float, float, float]]:
        return self.receiver.get_latest_pose()
    
    def get_pose_history(self):
        return self.receiver.get_pose_history()

    def get_latest_inliers(self) -> Optional[int]:
        return self.receiver.get_latest_inliers()

    def get_latest_covariance(self) -> Optional[float]:
        return self.receiver.get_latest_covariance()
    
def get_latest_inliers() -> Optional[int]:
    if _manager is not None:
        return _manager.get_latest_inliers()
    return None

def get_latest_covariance() -> Optional[float]:
    if _manager is not None:
        return _manager.get_latest_covariance()
    return None

_manager: Optional[SlamReceiver] = None

def start_receiver(host: str = HOST, port: int = PORT) -> Optional[threading.Thread]:
    global _manager
    if _manager is None:
        _manager = SlamReceiver(host, port)
        _manager.start()
    return _manager._log_thread


def stop_receiver() -> None:
    """Stop and clear the global :class:`SlamReceiver` instance."""
    global _manager
    if _manager is not None:
        _manager.stop()
        _manager = None


def get_latest_pose() -> Optional[Tuple[float, float, float]]:
    if _manager is not None:
        matrix = _manager.get_latest_pose()
        logger.debug("[slam_receiver] Latest pose matrix: %s", matrix)
        if matrix is not None and isinstance(matrix, (list, tuple)) and len(matrix) == 3:
            return tuple(matrix)
    return None


def get_pose_history():
    if _manager is not None:
        return _manager.get_pose_history()
    return []

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SLAM pose receiver")
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args()

    start_receiver(args.host, args.port)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        stop_receiver()
