from typing import Optional, Tuple
from .pose_receiver import PoseReceiver
import logging
import time
import argparse
import csv
from pathlib import Path
import threading

logger = logging.getLogger(__name__)

HOST = "192.168.1.102"  # Default IP if not provided
PORT = 6001

_receiver: Optional[PoseReceiver] = None
pose_writer = None
pose_log_file = None
airsim_client = None
log_thread_running = False

def start_receiver(host: str = HOST, port: int = PORT) -> None:
    """Start the global PoseReceiver and begin logging SLAM vs GT poses."""
    global _receiver, pose_writer, pose_log_file, airsim_client, log_thread_running

    if _receiver is None:
        _receiver = PoseReceiver(host, port)
        _receiver.start()

        # Create analysis directory and CSV file
        log_dir = Path("analysis")
        log_dir.mkdir(exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        csv_path = log_dir / f"pose_comparison.csv"  # Can change to timestamped if desired
        pose_log_file = open(csv_path, "w", newline="")
        pose_writer = csv.writer(pose_log_file)
        pose_writer.writerow(["timestamp", "slam_x", "slam_y", "slam_z", "gt_x", "gt_y", "gt_z"])

        try:
            import airsim
            airsim_client = airsim.MultirotorClient()
            airsim_client.confirmConnection()
        except Exception as e:
            logger.error(f"[SLAMReceiver] Failed to connect to AirSim: {e}")
            return

        log_thread_running = True
        threading.Thread(target=_log_pose_loop, daemon=True).start()
        logger.info("[SLAMReceiver] Pose logging thread started.")


def _log_pose_loop():
    global log_thread_running, airsim_client, pose_writer, pose_log_file
    while log_thread_running:
        slam_pose = get_latest_pose()
        if slam_pose is None:
            time.sleep(0.05)
            continue
        try:
            gt_pose = airsim_client.simGetVehiclePose().position
            timestamp = time.time()

            slam_x, slam_y, slam_z = slam_pose
            gt_x, gt_y, gt_z = gt_pose.x_val, gt_pose.y_val, gt_pose.z_val

            logger.debug(f"[SLAMReceiver] SLAM: {slam_pose} | GT: ({gt_x:.2f}, {gt_y:.2f}, {gt_z:.2f})")

            pose_writer.writerow([timestamp, slam_x, slam_y, slam_z, gt_x, gt_y, gt_z])
            pose_log_file.flush()
        except Exception as e:
            logger.warning(f"[SLAMReceiver] Error while logging poses: {e}")
        time.sleep(0.05)


def stop_receiver() -> None:
    """Stop the PoseReceiver and logging thread."""
    global _receiver, log_thread_running, pose_log_file
    if _receiver is not None:
        _receiver.stop()
        _receiver = None

        log_thread_running = False
        if pose_log_file:
            pose_log_file.close()


def get_latest_pose() -> Optional[Tuple[float, float, float]]:
    """Return the latest SLAM pose."""
    if _receiver is not None:
        return _receiver.get_latest_pose()
    return None


def get_pose_history():
    if _receiver is not None:
        return _receiver.get_pose_history()
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
