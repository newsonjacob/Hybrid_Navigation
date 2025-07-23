import argparse
from pathlib import Path
from typing import List, Union  # Add Union import

import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from .flight_review import parse_log, plot_state_histogram, plot_distance_over_time
from .visualise_flight import build_plot


def analyse_logs(
    log_paths: List[str],
    output: str,
    *,
    state_plot: str | None = None,
    distance_plot: str | None = None,
) -> None:
    """Parse ``log_paths`` and write an interactive HTML report."""
    dfs = []
    stats = []
    for p in log_paths:
        stats.append(parse_log(p))
        dfs.append(pd.read_csv(p))
    df = pd.concat(dfs, ignore_index=True)

    path = df[["pos_x", "pos_y", "pos_z"]].to_numpy(dtype=float)
    fig3d = build_plot(path, [], np.array([0, 0, 0]), log=df, colour_by="time")

    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "xy"}], [{"type": "xy", "colspan": 2}, None]],
        subplot_titles=("Trajectory", "Flow Magnitudes", "Speed / Performance"),
    )

    for trace in fig3d.data:
        fig.add_trace(trace, row=1, col=1)

    if {"time", "flow_left", "flow_center", "flow_right"}.issubset(df.columns):
        fig.add_trace(go.Scatter(x=df["time"], y=df["flow_left"], name="flow_left"), row=1, col=2)
        fig.add_trace(go.Scatter(x=df["time"], y=df["flow_center"], name="flow_center"), row=1, col=2)
        fig.add_trace(go.Scatter(x=df["time"], y=df["flow_right"], name="flow_right"), row=1, col=2)

    if "time" in df.columns and "speed" in df.columns:
        fig.add_trace(go.Scatter(x=df["time"], y=df["speed"], name="speed"), row=2, col=1)
    if "time" in df.columns and "cpu_percent" in df.columns:
        fig.add_trace(go.Scatter(x=df["time"], y=df["cpu_percent"], name="cpu %"), row=2, col=1)
    if "time" in df.columns and "mem_mb" in df.columns:
        fig.add_trace(go.Scatter(x=df["time"], y=df["mem_mb"], name="mem MB"), row=2, col=1)

    fig.update_layout(height=600, width=900)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output)

    if state_plot:
        combined = {}
        for s in stats:
            for k, v in s.get("states", {}).items():
                combined[k] = combined.get(k, 0) + v
        plot_state_histogram({"states": combined}, state_plot)

    if distance_plot and log_paths:
        plot_distance_over_time(log_paths[0], distance_plot)

    total_frames = sum(s["frames"] for s in stats)
    total_collisions = sum(s["collisions"] for s in stats)
    total_distance = sum(s["distance"] for s in stats)
    fps_vals = [s["fps_avg"] for s in stats if not np.isnan(s["fps_avg"])]
    loop_vals = [s["loop_avg"] for s in stats if not np.isnan(s["loop_avg"])]

    print(f"Frames: {total_frames}")
    print(f"Collisions: {total_collisions}")
    print(f"Distance travelled: {total_distance:.2f} m")
    if fps_vals:
        print(f"Average FPS: {np.mean(fps_vals):.2f}")
    if loop_vals:
        print(f"Average loop time: {np.mean(loop_vals):.3f}s")


def parse_args(argv: Union[List[str], None] = None) -> argparse.Namespace:  # Change | to Union
    parser = argparse.ArgumentParser(description="Analyze flight logs")
    parser.add_argument("logs", nargs="+", help="CSV log files")
    parser.add_argument("-o", "--output", default="analysis/flight_view.html", help="Output HTML file")
    parser.add_argument("--state-plot", help="Output HTML for state histogram")
    parser.add_argument("--distance-plot", help="Output HTML for distance over time")
    return parser.parse_args(argv)


def main(argv: Union[List[str], None] = None) -> None:  # Change | to Union None:  # Change | to Union
    args = parse_args(argv)
    analyse_logs(
        args.logs,
        args.output,
        state_plot=args.state_plot,
        distance_plot=args.distance_plot,
    )


if __name__ == "__main__":
    main()
