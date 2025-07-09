# slam_plotter.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from slam_bridge.slam_receiver import get_latest_pose
import time
import plotly.graph_objects as go
import logging
from slam_bridge.logging_helper import configure_file_logger

logger = configure_file_logger("slam_plotter.log")
logger.info("Script started.")

# Configure logging for plotting
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
timestamp = time.strftime("%Y%m%d_%H%M%S")
logfile = log_dir / f"pose_log_{timestamp}.txt"

# Remove all handlers associated with the root logger object.
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Create a dedicated logger for plot logging
plot_logger = logging.getLogger("plot_logger")
plot_logger.setLevel(logging.INFO)
plot_handler = logging.FileHandler(logfile, mode='a', encoding='utf-8')
plot_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
plot_logger.addHandler(plot_handler)
plot_logger.propagate = False  # Prevents double logging to root logger

import threading
x_vals, y_vals, z_vals = [], [], []
pose_lock = threading.Lock()

def plot_slam_trajectory():
    logger.info("Starting SLAM pose collection...")
    try:
        logger.info("Script is running...")
        while True:
            pose = get_latest_pose()
            if pose is not None:
                x, y, z = pose
                with pose_lock:
                    x_vals.append(x)
                    y_vals.append(y)
                    z_vals.append(z)
                # Log the pose data
                plot_logger.info(f"Pose received: x={x}, y={y}, z={z}") 
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        save_interactive_plot()

def save_interactive_plot():
    plot_dir = Path("analysis")
    plot_dir.mkdir(exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = plot_dir / f"slam_traj_{timestamp}.html"

    fig = go.Figure()
    with pose_lock:
        fig.add_trace(go.Scatter(y=x_vals, mode='lines', name='x'))
        fig.add_trace(go.Scatter(y=y_vals, mode='lines', name='y'))
        fig.add_trace(go.Scatter(y=z_vals, mode='lines', name='z'))

        resets = []
        prev = None
        for idx, (x, y, z) in enumerate(zip(x_vals, y_vals, z_vals)):
            if prev is not None and (
                abs(x - prev[0]) > 1 or abs(y - prev[1]) > 1 or abs(z - prev[2]) > 1
            ):
                resets.append(idx)
            prev = (x, y, z)
        for r in resets:
            fig.add_vline(x=r, line=dict(color="red", dash="dash"))
    fig.update_layout(
        title="SLAM Translation (x, y, z)",
        xaxis_title="Frame index",
        yaxis_title="Position (m)",
        template="plotly_dark"
    )
    fig.write_html(str(filename))
    logger.info("Saved interactive plot to %s", filename)

