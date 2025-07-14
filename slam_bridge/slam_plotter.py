# slam_plotter.py
import sys
import time
import logging
import threading
from pathlib import Path
import plotly.graph_objects as go

# --- Add project root to sys.path ---
sys.path.append(str(Path(__file__).resolve().parent.parent))

from slam_bridge.slam_receiver import get_latest_pose

# --- Global logger for this module ---
logger = logging.getLogger("slam_plotter")
logger.info("SLAM plotter script started.")

# --- Optional pose-specific log file ---
pose_log_dir = Path("logs")
pose_log_dir.mkdir(exist_ok=True)
pose_log_path = pose_log_dir / f"pose_log_{time.strftime('%Y%m%d_%H%M%S')}.txt"

pose_logger = logging.getLogger("slam_plotter.pose_log")
pose_logger.setLevel(logging.INFO)

pose_handler = logging.FileHandler(pose_log_path, mode="a", encoding="utf-8")
pose_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))

pose_logger.addHandler(pose_handler)
pose_logger.propagate = False  # Don't propagate to root

# --- Trajectory buffers ---
x_vals, y_vals, z_vals = [], [], []
pose_lock = threading.Lock()


def plot_slam_trajectory():
    logger.info("SLAM trajectory collection started.")
    try:
        while True:
            pose = get_latest_pose()
            if pose is not None:
                x, y, z = pose
                with pose_lock:
                    x_vals.append(x)
                    y_vals.append(y)
                    z_vals.append(z)
                pose_logger.info(f"Pose received: x={x:.2f}, y={y:.2f}, z={z:.2f}")
            time.sleep(0.05)
    except KeyboardInterrupt:
        logger.info("SLAM plotter interrupted by user.")
    finally:
        save_interactive_plot()


def save_interactive_plot():
    logger.info("Saving SLAM trajectory plot...")
    plot_dir = Path("analysis")
    plot_dir.mkdir(exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    plot_path = plot_dir / f"slam_traj_{timestamp}.html"

    fig = go.Figure()
    with pose_lock:
        fig.add_trace(go.Scatter(y=x_vals, mode='lines', name='x'))
        fig.add_trace(go.Scatter(y=y_vals, mode='lines', name='y'))
        fig.add_trace(go.Scatter(y=z_vals, mode='lines', name='z'))

        # Detect large resets in pose to mark potential SLAM resets
        prev = None
        for i, (x, y, z) in enumerate(zip(x_vals, y_vals, z_vals)):
            if prev and any(abs(a - b) > 1.0 for a, b in zip((x, y, z), prev)):
                fig.add_vline(x=i, line=dict(color="red", dash="dash"))
            prev = (x, y, z)

    fig.update_layout(
        title="SLAM Translation (x, y, z)",
        xaxis_title="Frame Index",
        yaxis_title="Position (m)",
        template="plotly_dark"
    )
    fig.write_html(str(plot_path))
    logger.info(f"Trajectory plot saved to: {plot_path}")
