import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Hybrid UAV Navigation")
    parser.add_argument("--manual-nudge", action="store_true", help="Enable manual nudge at frame 5 for testing")
    parser.add_argument("--map", choices=["reactive", "deliberative", "hybrid"], default="reactive", help="Which map to load")
    parser.add_argument("--ue4-path", default=None, help="Override the default path to the Unreal Engine executable")
    parser.add_argument("--settings-path", default=None, help="Path to AirSim settings.json")
    parser.add_argument("--config", default="config.ini", help="Path to config file with default paths")
    parser.add_argument("--goal-x", type=int, default=None, help="Distance from start to goal (X coordinate)")
    parser.add_argument("--goal-y", type=int, default=None, help="Distance from start to goal (Y coordinate)")
    parser.add_argument("--max-duration", type=int, default=None, help="Maximum simulation duration in seconds")
    parser.add_argument(
        "--nav-mode",
        choices=["slam", "reactive"],
        default="slam",
        help="Navigation mode: 'slam' for SLAM-based, 'reactive' for optical flow/reactive navigation (default: slam)"
    )
    parser.add_argument(
        "--stream-mode",
        choices=["rgbd", "stereo"],
        default="rgbd",
        help="Image streaming mode: 'rgbd' (RGB + Depth) or 'stereo' (Left + Right RGB)"
    )
    parser.add_argument("--slam-server-host", default=None, help="SLAM server IP or hostname")
    parser.add_argument("--slam-server-port", type=int, default=None, help="SLAM server TCP port")
    parser.add_argument("--slam-receiver-host", default=None, help="Pose receiver IP")
    parser.add_argument("--slam-receiver-port", type=int, default=None, help="Pose receiver TCP port")
    parser.add_argument("--slam-pose-source", choices=["slam", "airsim"], default="slam", 
                        help="Source of pose data for SLAM navigation: 'slam' uses the pose receiver; 'airsim' calls simGetVehiclePose",
    )
    parser.add_argument(
        "--slam-covariance-threshold",
        type=float,
        default=None,
        help="SLAM covariance threshold for stability check",
    )
    parser.add_argument(
        "--slam-inlier-threshold",
        type=int,
        default=None,
        help="Minimum inliers for SLAM stability",
    )
    parser.add_argument("--log-timestamp", type=str, help="Timestamp used to sync logging across modules")
    return parser.parse_args()
