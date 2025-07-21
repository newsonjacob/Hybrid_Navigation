"""Tools for plotting flight paths and scene obstacles in 3D."""

import argparse
import json
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments for the CLI."""
    parser = argparse.ArgumentParser(description="Visualise a flight path")
    parser.add_argument("output", help="Output HTML file")
    parser.add_argument("--log", help="CSV log containing pos_x/pos_y/pos_z")
    parser.add_argument("--obstacles", help="JSON file with obstacle list")
    return parser.parse_args(argv)


def find_alignment_marker(obstacles: List[Dict], marker_name: str = "PlayerStart_3") -> np.ndarray:
    """Return the location of an obstacle by name.

    Raises
    ------
    ValueError
        If the marker is not found.
    """
    for obj in obstacles:
        if obj.get("name") == marker_name:
            return np.asarray(obj.get("location", [0, 0, 0]), dtype=float)
    raise ValueError(f"Marker '{marker_name}' not found")


def draw_box(location, dimensions, rotation):
    """Return Plotly traces for a 3D box.

    Parameters
    ----------
    location : array-like
        XYZ centre of the box.
    dimensions : array-like
        Width, depth and height of the box.
    rotation : array-like
        Euler rotation angles in degrees (roll, pitch, yaw).
    """

    location = np.asarray(location, dtype=float)
    dims = np.asarray(dimensions, dtype=float)
    rot = np.asarray(rotation, dtype=float)

    # Compute vertices of the unit box centred on the origin
    half = dims / 2.0
    base_vertices = np.array([
        [-half[0], -half[1], -half[2]],
        [ half[0], -half[1], -half[2]],
        [ half[0],  half[1], -half[2]],
        [-half[0],  half[1], -half[2]],
        [-half[0], -half[1],  half[2]],
        [ half[0], -half[1],  half[2]],
        [ half[0],  half[1],  half[2]],
        [-half[0],  half[1],  half[2]],
    ])

    try:
        from scipy.spatial.transform import Rotation as R

        rot_matrix = R.from_euler("xyz", rot, degrees=True).as_matrix()
    except Exception:
        rot_matrix = np.eye(3)

    vertices = base_vertices @ rot_matrix.T + location
    x, y, z = vertices.T

    faces_i = [0, 0, 0, 3, 4, 4, 4, 7, 1, 2, 5, 6]
    faces_j = [1, 2, 3, 2, 5, 6, 7, 6, 5, 3, 6, 3]
    faces_k = [2, 3, 1, 0, 6, 7, 5, 4, 2, 6, 4, 7]

    mesh = go.Mesh3d(x=x, y=y, z=z, i=faces_i, j=faces_j, k=faces_k,
                     opacity=0.5, color="lightgrey", showscale=False)

    return [mesh]


def build_plot(
    telemetry: np.ndarray,
    obstacles: List[Dict],
    offset: np.ndarray,
    scale: float = 1.0,
    log=None,
    colour_by: str | None = None,
    orientations: np.ndarray | None = None,
) -> go.Figure:
    """Create a simple 3D plot of the flight path and obstacles.

    Parameters
    ----------
    telemetry : ndarray
        Nx3 array of XYZ positions in the local frame.
    obstacles : list
        Obstacle dictionaries describing the scene.
    offset : ndarray
        XYZ translation applied to ``telemetry``.
    scale : float, optional
        Multiplicative scale factor for the coordinates.
    log : DataFrame or dict, optional
        Flight log providing ``time`` and ``speed`` columns.
    colour_by : {'time', 'speed'}, optional
        If provided, colour the trajectory by this value.
    orientations : ndarray, optional
        Yaw angles in degrees for orientation arrows.
    """

    telemetry = np.asarray(telemetry, dtype=float)
    offset = np.asarray(offset, dtype=float)

    path = np.empty_like(telemetry, dtype=float)
    path[:, 0] = offset[0] + telemetry[:, 0] * scale
    path[:, 1] = offset[1] - telemetry[:, 1] * scale
    path[:, 2] = offset[2] + telemetry[:, 2] * scale

    line_opts: dict = {}
    if colour_by in ("time", "speed") and log is not None:
        try:
            values = log[colour_by]  # type: ignore[index]
            values = np.asarray(values, dtype=float)
            if len(values) == len(path):
                line_opts = {"color": values, "colorscale": "Viridis", "showscale": True}
        except Exception:
            pass

    traces = [
        go.Scatter3d(
            x=path[:, 0],
            y=path[:, 1],
            z=path[:, 2],
            mode="lines",
            name="path",
            line=line_opts,
        )
    ]

    for obs in obstacles:
        if obs.get("name", "").startswith("UCX"):
            continue
        loc = obs.get("location", [0, 0, 0])
        dims = obs.get("dimensions", [1, 1, 1])
        rot = obs.get("rotation", [0, 0, 0])
        # draw_box may be patched in tests to avoid heavy computation.
        try:
            box_traces = draw_box(loc, dims, rot)
            if isinstance(box_traces, list):
                traces.extend(box_traces)
        except Exception:
            # Ignore errors to keep the plot valid.
            pass

    if orientations is not None:
        try:
            yaws = np.asarray(orientations, dtype=float).reshape(-1)
            arrow_len = 0.5 * scale
            xs, ys, zs = [], [], []
            for pos, yaw in zip(path, yaws):
                dx = arrow_len * np.cos(np.radians(yaw))
                dy = arrow_len * np.sin(np.radians(yaw))
                xs.extend([pos[0], pos[0] + dx, None])
                ys.extend([pos[1], pos[1] + dy, None])
                zs.extend([pos[2], pos[2], None])
            traces.append(
                go.Scatter3d(
                    x=xs,
                    y=ys,
                    z=zs,
                    mode="lines",
                    name="orientation",
                    line=dict(color="orange"),
                )
            )
        except Exception:
            pass

    fig = go.Figure(traces)
    fig.update_layout(scene=dict(aspectmode="data"))
    return fig


def main(argv: List[str] | None = None) -> None:
    """Entry point for the ``python -m`` interface."""
    args = parse_args(argv)

    if args.log:
        df = pd.read_csv(args.log)
        telemetry = df[["pos_x", "pos_y", "pos_z"]].to_numpy(dtype=float)
        log = df
    else:
        telemetry = np.zeros((0, 3))
        log = None

    if args.obstacles:
        with open(args.obstacles, "r", encoding="utf-8") as f:
            obstacles = json.load(f)
    else:
        obstacles = []

    fig = build_plot(telemetry, obstacles, np.array([0, 0, 0]), log=log,
                     colour_by="time" if log is not None else None)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(args.output)


if __name__ == "__main__":
    main()
