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

SLAM_YAW_OFFSET = 0.0
MAX_SIM_DURATION = 120  # seconds
GOAL_X = 46 # meters
GOAL_Y = 0  # meters
GOAL_THRESHOLD = 1  # Distance threshold for goal completion (meters)

FLOW_CAMERA = "0"  # camera used for optical flow
STEREO_LEFT_CAMERA = "front_left"  # left camera used for depth filtering
STEREO_RIGHT_CAMERA = "front_right"  # right camera used for depth filtering

TARGET_FPS = 20 
LOG_INTERVAL = 5 
VIDEO_FPS = 8.0 
VIDEO_SIZE = (1280, 720) # Resolution for video output
VIDEO_OUTPUT = 'flow_output.avi'

def load_app_config(config_path: str = "config.ini"):
    """Load application config from an INI file."""
    import configparser
    parser = configparser.ConfigParser()
    parser.read(config_path)
    return parser
