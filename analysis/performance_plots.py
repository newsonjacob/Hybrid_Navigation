"""Plot CPU and memory usage from a flight log."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from typing import Any

# Plotly is imported lazily inside build_plot so tests can provide stubs


def build_plot(df: pd.DataFrame) -> Any:
    """Return a Plotly figure showing CPU and memory usage.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing ``cpu_percent`` and ``memory_rss`` columns and
        either ``time`` or ``frame`` for the x-axis.
    """
    if "time" in df.columns:
        # Convert time to relative seconds from start
        start_time = df["time"].iloc[0]
        x = df["time"] - start_time
        x_title = "Time (seconds from start)"
    else:
        x = df.index
        x_title = "Frame"

    # Import plotly only when needed so tests can stub these modules
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    cpu = df.get("cpu_percent", pd.Series(dtype=float))
    mem = df.get("memory_rss", pd.Series(dtype=float)) / (1024 * 1024)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=x, y=cpu, name="CPU %"), secondary_y=False)
    fig.add_trace(go.Scatter(x=x, y=mem, name="Memory MB"), secondary_y=True)

    fig.update_layout(xaxis_title=x_title)
    fig.update_yaxes(title_text="CPU %", secondary_y=False)
    fig.update_yaxes(title_text="Memory (MB)", secondary_y=True)
    return fig


def plot_performance(log_path: str, out_path: str) -> None:
    """Read ``log_path`` CSV and write an interactive HTML plot to ``out_path``."""
    df = pd.read_csv(log_path)
    fig = build_plot(df)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot performance metrics from a log")
    parser.add_argument("log", help="CSV log file")
    parser.add_argument("-o", "--output", default="analysis/performance.html", help="Output HTML file")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    plot_performance(args.log, args.output)


if __name__ == "__main__":
    main()
