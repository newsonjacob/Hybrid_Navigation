"""Generate an interactive 3D visualisation from ORB-SLAM trajectories."""

import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import sys
import logging
import io

# ========== Setup Logging FIRST ========== #

def get_timestamp_from_args():
    """Extract timestamp from command line args if available."""
    try:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--log-timestamp", type=str, default=None)
        args, _ = parser.parse_known_args()
        return args.log_timestamp
    except:
        return None

# Setup logging
timestamp = get_timestamp_from_args() or datetime.now().strftime('%Y%m%d_%H%M%S')
log_dir = Path("logs").expanduser().resolve()
log_dir.mkdir(exist_ok=True)
log_file_path = log_dir / f"SLAM_visual_{timestamp}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(str(log_file_path)),
        logging.StreamHandler(sys.stdout)
    ],
    force=True
)

logger = logging.getLogger("SLAM_visual")
logger.info("Starting SLAM trajectory visualisation generation")

def load_trajectory(file_path, cols=(0, 1, 2)):
    """Return an Nx3 array of positions from a trajectory file, ignoring comment lines."""
    try:
        with open(file_path, "r") as f:
            lines = [line for line in f if not line.lstrip().startswith("//") and line.strip()]
        if not lines:
            return np.empty((0, 3))
        df = pd.read_csv(io.StringIO("".join(lines)), sep=r"\s+", header=None)
        arr = df.iloc[:, list(cols)].values
        arr = arr[~np.isnan(arr).any(axis=1)]
        arr = arr[~np.isinf(arr).any(axis=1)]
        return arr
    except Exception as e:
        logger.error(f"Error loading trajectory from {file_path}: {e}")
        return np.empty((0, 3))

def load_mappoints(file_path):
    """Return Nx3 array of map points."""
    try:
        df = pd.read_csv(file_path, sep=" ", header=None, comment="/")
        arr = df.iloc[:, :3].values
        arr = arr[~np.isnan(arr).any(axis=1)]
        arr = arr[~np.isinf(arr).any(axis=1)]
        return arr
    except Exception as e:
        logger.error(f"Error loading map points from {file_path}: {e}")
        return np.empty((0, 3))

def main(args=None):
    parser = argparse.ArgumentParser(description="Generate interactive 3D visualisation from ORB-SLAM trajectories.")
    parser.add_argument(
        "base_path",
        nargs="?",
        default="linux_slam",
        help="Directory containing ORB-SLAM output files",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Path for the generated HTML file",
    )
    opts = parser.parse_args(args)

    base_path = Path(opts.base_path).expanduser().resolve()
    camera_file = log_dir / "CameraTrajectory.txt"
    keyframe_file = log_dir / "KeyFrameTrajectory.txt"
    map_file = log_dir / "MapPoints.txt"

    logger.info(f"Loading from: {camera_file}")
    logger.info(f"Loading from: {keyframe_file}")
    logger.info(f"Loading from: {map_file}")

    camera = load_trajectory(camera_file, cols=(1, 2, 3)) if camera_file.exists() else np.empty((0, 3))
    keyframe = load_trajectory(keyframe_file, cols=(1, 2, 3)) if keyframe_file.exists() else np.empty((0, 3))
    mappoints = load_mappoints(map_file) if map_file.exists() else np.empty((0, 3))

    # Reorient: Z → X, X → Y, Y → Z
    camera = camera[:, [2, 0, 1]]
    keyframe = keyframe[:, [2, 0, 1]]
    mappoints = mappoints[:, [2, 0, 1]]

    # Invert Y and Z axes
    for arr in [camera, keyframe, mappoints]:
        if arr.shape[0] > 0:
            arr[:, 1] *= -1
            arr[:, 2] *= -1

    # Custom axis ranges
    axis_ranges = {
        "x": [-5, 100],  # X (was Z)
        "y": [-15, 15],  # Y (was -X)
        "z": [-5, 20],   # Z (was -Y)
    }

    # Create interactive 3D plot
    fig = go.Figure()

    if camera.shape[0] > 0:
        fig.add_trace(go.Scatter3d(
            x=camera[:, 0], y=camera[:, 1], z=camera[:, 2],
            mode='lines', name='Camera Trajectory',
            line=dict(width=3, color='blue')
        ))

    if keyframe.shape[0] > 0:
        fig.add_trace(go.Scatter3d(
            x=keyframe[:, 0], y=keyframe[:, 1], z=keyframe[:, 2],
            mode='lines', name='Keyframe Trajectory',
            line=dict(width=3, color='red')
        ))

    if mappoints.shape[0] > 0:
        fig.add_trace(go.Scatter3d(
            x=mappoints[:, 0], y=mappoints[:, 1], z=mappoints[:, 2],
            mode='markers', name='Map Points',
            marker=dict(size=2, color='gray', opacity=0.5)
        ))

    fig.update_layout(
        title='SLAM Trajectory and Map Points',
        scene=dict(
            xaxis=dict(title='X (was Z)', range=axis_ranges["x"]),
            yaxis=dict(title='Y (was -X)', range=axis_ranges["y"]),
            zaxis=dict(title='Z (was -Y)', range=axis_ranges["z"]),
            aspectmode='manual',
            aspectratio=dict(x=105, y=30, z=25),
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        legend=dict(x=0.01, y=0.99)
    )

    # Output file
    output_file = Path(opts.output) if opts.output else base_path / "slam_trajectory_visualisation.html"
    output_file.parent.mkdir(exist_ok=True, parents=True)
    fig.write_html(str(output_file))
    logger.info(f"Saved visualization to: {output_file}")

    # --- Optional: Print analysis ---
    if camera.shape[0] > 1:
        dist = np.linalg.norm(np.diff(camera, axis=0), axis=1).sum()
        logger.info(f"Camera trajectory length: {dist:.2f} units")
        print(f"Camera trajectory length: {dist:.2f} units")
    if keyframe.shape[0] > 1:
        dist = np.linalg.norm(np.diff(keyframe, axis=0), axis=1).sum()
        logger.info(f"Keyframe trajectory length: {dist:.2f} units")
        print(f"Keyframe trajectory length: {dist:.2f} units")
    if mappoints.shape[0] > 0:
        logger.info(f"Map points count: {mappoints.shape[0]}")
        print(f"Map points count: {mappoints.shape[0]}")

if __name__ == "__main__":
    main()
