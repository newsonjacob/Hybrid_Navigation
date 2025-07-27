"""Compute simple SLAM performance metrics."""

from pathlib import Path
from typing import Dict
import pandas as pd
import numpy as np


def compute_metrics(log_path: str) -> Dict[str, float]:
    """Return RMSE and average covariance/inliers from a pose log.

    Parameters
    ----------
    log_path : str
        CSV file produced by :mod:`slam_bridge.slam_receiver`.

    Returns
    -------
    dict
        Dictionary containing frame count, RMSE, mean covariance and mean inliers.
    """
    df = pd.read_csv(log_path)
    frames = len(df)
    diff = df[["slam_x", "slam_y", "slam_z"]].to_numpy() - df[["gt_x", "gt_y", "gt_z"]].to_numpy()
    rmse = float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))) if frames else float("nan")
    cov_avg = float(df["covariance"].mean()) if "covariance" in df else float("nan")
    inlier_avg = float(df["inliers"].mean()) if "inliers" in df else float("nan")
    return {
        "frames": int(frames),
        "rmse": rmse,
        "covariance_avg": cov_avg,
        "inliers_avg": inlier_avg,
    }


def export_metrics(log_path: str, out_path: str) -> None:
    """Compute and write metrics to ``out_path`` in CSV format."""
    metrics = compute_metrics(log_path)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([metrics])
    df.to_csv(out, index=False)
