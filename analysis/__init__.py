"""Lightweight analysis helpers for tests."""

from .flight_review import (
    parse_log,
    align_path,
    plot_state_histogram,
    plot_distance_over_time,
)
from .summarise_runs import summarise_log
from . import visualise_flight
try:
    from .performance_plots import plot_performance
except Exception:  # pragma: no cover - optional dependency missing
    plot_performance = None

__all__ = [
    "parse_log",
    "align_path",
    "plot_state_histogram",
    "plot_distance_over_time",
    "summarise_log",
    "visualise_flight",
    "plot_performance",
]
