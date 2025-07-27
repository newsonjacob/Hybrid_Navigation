"""Plot CPU and memory usage from a flight log."""

from __future__ import annotations

import argparse
from pathlib import Path
import logging

import pandas as pd
from typing import Any

# Plotly is imported lazily inside build_plot so tests can provide stubs

logger = logging.getLogger("performance_plots")


def build_plot(df: pd.DataFrame) -> Any:
    """Return a Plotly figure showing CPU and memory usage."""
    
    # Check if DataFrame is empty
    if len(df) == 0:
        logger.warning("DataFrame is empty - cannot generate performance plot")
        # Return empty figure
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.update_layout(title="No Data Available for Performance Plot")
        return fig
    
    # Check if required columns exist
    if "time" not in df.columns or len(df) == 0:
        logger.warning("No time data available for performance plot")
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.update_layout(title="No Time Data Available")
        return fig
    
    # Convert time to relative seconds from start
    start_time = df["time"].iloc[0]
    x = df["time"] - start_time
    x_title = "Time (seconds from start)"

    # Import plotly only when needed so tests can stub these modules
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    cpu = df.get("cpu_percent", pd.Series(dtype=float))

    if "memory_rss" in df.columns:
        mem = df["memory_rss"] / (1024 * 1024)
    elif "memory_mb" in df.columns:
        mem = df["memory_mb"]
    else:
        mem = pd.Series(dtype=float)

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
