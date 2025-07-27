# Configuration constants for UAV navigation

MAX_FLOW_MAG = 70.0 # Maximum flow magnitude for optical flow
MIN_FLOW_MAG = 1  # Minimum flow magnitude to consider valid
FLOW_HISTORY_SIZE = 3  # Number of recent flow measurements to keep
MAX_VECTOR_COMPONENT = 30.0 # Maximum vector component for flow vectors
MIN_FEATURES_PER_ZONE = 5 # Minimum features to consider a zone valid
MIN_PROBE_FEATURES = 5 # Minimum features to consider a zone valid
FLOW_STD_MAX = 50.0 # Maximum standard deviation for flow vectors
DEPTH_FILTER_DIST = 2  # Distance threshold for depth filtering

GRACE_PERIOD_SEC = 1.0 # Grace period for state transitions

# Toggle SLAM bootstrap manoeuvre used for initialization
ENABLE_SLAM_BOOTSTRAP = False

SLAM_YAW_OFFSET = 0.0
MAX_SIM_DURATION = 120  # seconds
GOAL_X = 48 # meters
GOAL_Y = 0  # meters
GOAL_THRESHOLD = 2  # Distance threshold for goal completion (meters)

FLOW_CAMERA = "0"  # camera used for optical flow
STEREO_LEFT_CAMERA = "front_left"  # left camera used for depth filtering
STEREO_RIGHT_CAMERA = "front_right"  # right camera used for depth filtering

TARGET_FPS = 20 
LOG_INTERVAL = 5 
VIDEO_FPS = 8.0 
VIDEO_SIZE = (1280, 720) # Resolution for video output
VIDEO_OUTPUT = 'flow_output.avi'

RETENTION_CONFIG = {
    "logs": [
        ("launch_*.log", 2),
        ("slam_*.log", 2),
        ("pose_*.txt", 2),
        ("pose_*.log", 2),
        ("main_*.log", 2),
        ("nav_*.log", 2),
        ("stream_*.log", 2),
        ("utils_*.log", 2),
        ("analyse_*.log", 2),
        ("perception_*.log", 2),
        ("tracking_*.log", 2),
    ],
    "flow_logs": [
        ("reactive_log_*.csv", 2),
        ("slam_log_*.csv", 2),
    ],
    "analysis": [
        ("flight_report_*.html", 2),
        ("slam_trajectory_*.html", 4),
        ("slam_error_*.mp4", 2),
        ("gt_trajectory_*.html", 2),
        ("pose_vs_*.html", 2),
        ("trajectory_flight_*.html", 2),
        ("performance_*.html", 2),
    ],
}

def load_app_config(config_path: str = "config.ini"):
    """Load application config from an INI file."""
    import configparser
    parser = configparser.ConfigParser()
    parser.read(config_path)
    return parser
