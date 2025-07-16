import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime
import sys
import logging

logger = logging.getLogger(__name__)

def main():
    log_path = Path("analysis/pose_comparison.csv")
    if not log_path.exists():
        logger.error(f"CSV file not found: {log_path}")
        sys.exit(1)

    if log_path.stat().st_size == 0:
        logger.error(f"CSV file is empty: {log_path}")
        sys.exit(1)

    df = pd.read_csv(log_path)
    if df.empty or df.isnull().all().all():
        logger.error(f"CSV file is empty or contains only null values.")
        sys.exit(1)

    # Plot both trajectories together
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=df['gt_x'], y=df['gt_y'], z=df['gt_z'],
        mode='lines', name='Ground Truth',
        line=dict(color='green')
    ))

    fig.add_trace(go.Scatter3d(
        x=df['slam_x'], y=df['slam_y'], z=df['slam_z'],
        mode='lines', name='SLAM Trajectory',
        line=dict(color='blue', width=3)
    ))

    fig.update_layout(
        title="SLAM vs Ground Truth Trajectory",
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Z (m)",
            aspectmode="data"
        ),
        legend=dict(x=0, y=1)
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    traj_path = Path(f"analysis/pose_vs_gt_plot_{ts}.html")
    fig.write_html(traj_path)
    logger.info(f"[✓] Saved 3D plot to: {traj_path}")

    # Call the other two plotting functions
    plot_ground_truth_only(df)
    plot_slam_only(df)

def plot_ground_truth_only(df, save_dir="analysis"):
    import plotly.graph_objects as go
    from pathlib import Path
    from datetime import datetime

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=df['gt_x'], y=df['gt_y'], z=df['gt_z'],
        mode='lines',
        name='Ground Truth',
        line=dict(color='green', width=4)
    ))
    fig.update_layout(
        title="Ground Truth Trajectory Only",
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Z (m)",
            aspectmode="data"
        )
    )
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(save_dir) / f"gt_trajectory_only_{ts}.html"
    fig.write_html(path)
    logger.info(f"[✓] Saved Ground Truth only plot to: {path}")

def plot_slam_only(df, save_dir="analysis"):
    import plotly.graph_objects as go
    from pathlib import Path
    from datetime import datetime

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=df['slam_x'], y=df['slam_y'], z=df['slam_z'],
        mode='lines',
        name='SLAM Trajectory',
        line=dict(color='blue', width=4)
    ))
    fig.update_layout(
        title="SLAM Trajectory Only",
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Z (m)",
            aspectmode="data"
        )
    )
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(save_dir) / f"slam_trajectory_only_{ts}.html"
    fig.write_html(path)
    logger.info(f"[✓] Saved SLAM only plot to: {path}")


if __name__ == "__main__":
    main()
