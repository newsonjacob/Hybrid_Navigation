"""Generate an interactive 3D visualisation from ORB-SLAM trajectories."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

def load_trajectory(file_path, cols=(1, 2, 3)):
    """Return an ``Nx3`` array of positions from a trajectory file."""

    return (
        pd.read_csv(file_path, sep=" ", header=None)
        .iloc[:, list(cols)]
        .values
    )

def main(args=None):
    """Entry point for the script."""

    parser = argparse.ArgumentParser(description=__doc__)
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

    base_path = Path(opts.base_path)
    base_path = base_path.expanduser().resolve()
    camera_file = base_path / "CameraTrajectory.txt"
    keyframe_file = base_path / "KeyFrameTrajectory.txt"
    map_file = base_path / "MapPoints.txt"

    # Load and reorient axes: Z→X, X→Y, Y→Z
    camera = load_trajectory(camera_file, cols=(2, 0, 1)) if camera_file.exists() else np.empty((0, 3))
    keyframe = load_trajectory(keyframe_file, cols=(2, 0, 1)) if keyframe_file.exists() else np.empty((0, 3))
    mappoints = load_trajectory(map_file, cols=(2, 0, 1)) if map_file.exists() else np.empty((0, 3))

    # Invert Z (was Y) and Y (was X)
    camera[:, 1] *= -1
    camera[:, 2] *= -1
    keyframe[:, 1] *= -1
    keyframe[:, 2] *= -1
    mappoints[:, 1] *= -1
    mappoints[:, 2] *= -1

    # Custom axis ranges
    axis_ranges = {
        "x": [-5, 100],  # X (was Z)
        "y": [-15, 15],  # Y (was -X)
        "z": [-5, 20],   # Z (was -Y)
    }

    # Create interactive 3D plot
    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=camera[:, 0], y=camera[:, 1], z=camera[:, 2],
        mode='lines', name='Camera Trajectory', line=dict(width=3)
    ))

    fig.add_trace(go.Scatter3d(
        x=keyframe[:, 0], y=keyframe[:, 1], z=keyframe[:, 2],
        mode='lines', name='Keyframe Trajectory', line=dict(width=3)
    ))

    fig.add_trace(go.Scatter3d(
        x=mappoints[:, 0], y=mappoints[:, 1], z=mappoints[:, 2],
        mode='markers', name='Map Points', marker=dict(size=1, color='gray')
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
    output_file = (
        Path(opts.output) if opts.output else base_path / "slam_trajectory_visualisation.html"
    )
    fig.write_html(str(output_file))
    print(f"Saved visualization to: {output_file}")

if __name__ == "__main__":
    main()
