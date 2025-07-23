"""Utilities for analyzing individual flight logs."""

import pandas as pd
import numpy as np
from typing import Dict

from pathlib import Path
import plotly.graph_objects as go


def parse_log(csv_path: str) -> Dict[str, float]:
    """Parse a flight log CSV and compute basic statistics.

    Parameters
    ----------
    csv_path : str
        Path to the CSV log produced during a run.

    Returns
    -------
    dict
        Dictionary containing frame count, collision count, travelled
        distance and average FPS/loop times along with a state histogram.
    """
    df = pd.read_csv(csv_path)

    frames = len(df)
    collisions = df.get("collided", pd.Series([0] * frames)).sum()

    if frames > 0:
        start = df.loc[0, ["pos_x", "pos_y", "pos_z"]].to_numpy(dtype=float)
        end = df.loc[frames - 1, ["pos_x", "pos_y", "pos_z"]].to_numpy(dtype=float)
        distance = float(np.linalg.norm(end - start))
    else:
        distance = 0.0

    fps_avg = float(df["fps"].mean()) if "fps" in df else float("nan")
    loop_avg = float(df["loop_s"].mean()) if "loop_s" in df else float("nan")

    states = (
        df["state"].value_counts().to_dict() if "state" in df else {}
    )

    return {
        "frames": int(frames),
        "collisions": int(collisions),
        "distance": distance,
        "fps_avg": fps_avg,
        "loop_avg": loop_avg,
        "states": {str(k): int(v) for k, v in states.items()},
    }


def align_path(path: np.ndarray, obstacles, *, scale: float = 1.0, marker_name: str = "PlayerStart_3") -> np.ndarray:
    """Align a local path to the world coordinate system using a marker.

    Parameters
    ----------
    path : ndarray
        Nx3 array of XYZ positions in the local AirSim frame.
    obstacles : list
        List of obstacle dictionaries containing at least ``name`` and
        ``location`` keys.
    scale : float
        Scale factor to apply to the coordinates.
    marker_name : str, optional
        Name of the obstacle to use as the origin marker.

    Returns
    -------
    ndarray
        Transformed path aligned to the simulation world.
    """
    marker = None
    for obj in obstacles:
        if obj.get("name") == marker_name:
            marker = np.asarray(obj.get("location", [0, 0, 0]), dtype=float)
            break
    if marker is None:
        raise ValueError(f"Alignment marker '{marker_name}' not found")

    p = np.asarray(path, dtype=float)
    aligned = np.empty_like(p, dtype=float)
    aligned[:, 0] = marker[0] + p[:, 0] * scale
    aligned[:, 1] = marker[1] - p[:, 1] * scale
    aligned[:, 2] = marker[2] + p[:, 2] * scale
    return aligned


def plot_state_histogram(stats: Dict, save_path: str) -> None:
    """Plot a histogram of state occurrences and save as HTML.

    Parameters
    ----------
    stats : dict
        Dictionary with a ``states`` entry mapping state names to counts.
    save_path : str
        Output HTML file path.
    """
    state_counts = stats.get("states", {})
    fig = go.Figure(
        go.Bar(x=list(state_counts.keys()), y=list(state_counts.values()))
    )
    fig.update_layout(xaxis_title="State", yaxis_title="Count")
    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out)


def plot_distance_over_time(csv_path: str, save_path: str) -> None:
    """Chart cumulative distance travelled over time and save to HTML."""
    df = pd.read_csv(csv_path)
    positions = df[["pos_x", "pos_y", "pos_z"]].to_numpy(dtype=float)
    if len(positions) == 0:
        cum_dist = np.array([0.0])
        x = [0]
        x_title = "Frame"
    else:
        deltas = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        cum_dist = np.concatenate([[0.0], deltas.cumsum()])
        if "time" in df.columns:
            x = df["time"]
            x_title = "Time (s)"
        else:
            x = np.arange(len(cum_dist))
            x_title = "Frame"

    fig = go.Figure(go.Scatter(x=x, y=cum_dist, mode="lines", name="distance"))
    fig.update_layout(xaxis_title=x_title, yaxis_title="Distance (m)")
    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out)
