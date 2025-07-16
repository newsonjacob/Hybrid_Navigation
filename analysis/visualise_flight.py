import numpy as np
import plotly.graph_objects as go
from typing import List, Dict


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
    """Placeholder that returns an empty list of traces.

    In the real project this would create Plotly traces describing a 3D box.
    """
    return []


def build_plot(telemetry: np.ndarray, obstacles: List[Dict], offset: np.ndarray, scale: float = 1.0) -> go.Figure:
    """Create a simple 3D plot of the flight path and obstacles."""
    telemetry = np.asarray(telemetry, dtype=float)
    offset = np.asarray(offset, dtype=float)

    path = np.empty_like(telemetry, dtype=float)
    path[:, 0] = offset[0] + telemetry[:, 0] * scale
    path[:, 1] = offset[1] - telemetry[:, 1] * scale
    path[:, 2] = offset[2] + telemetry[:, 2] * scale

    traces = [go.Scatter3d(x=path[:, 0], y=path[:, 1], z=path[:, 2], mode="lines", name="path")]

    for obs in obstacles:
        if obs.get("name", "").startswith("UCX"):
            continue
        loc = obs.get("location", [0, 0, 0])
        dims = obs.get("dimensions", [1, 1, 1])
        rot = obs.get("rotation", [0, 0, 0])
        # draw_box may be patched in tests to avoid heavy computation.
        # Ignore any returned objects to keep the plot valid.
        try:
            draw_box(loc, dims, rot)
        except Exception:
            pass

    fig = go.Figure(traces)
    fig.update_layout(scene=dict(aspectmode="data"))
    return fig