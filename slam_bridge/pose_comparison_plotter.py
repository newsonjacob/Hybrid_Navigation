# analysis/pose_comparison_plotter.py

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime
import sys

def main():
    log_path = Path("analysis/pose_comparison.csv")
    if not log_path.exists():
        print(f"[ERROR] CSV file not found: {log_path}")
        sys.exit(1)

    df = pd.read_csv(log_path)
    if df.empty or df.isnull().all().all():
        print(f"[ERROR] CSV file is empty or contains only null values.")
        sys.exit(1)

    # Compute SLAM error
    df['error'] = np.sqrt(
        (df['slam_x'] - df['gt_x'])**2 +
        (df['slam_y'] - df['gt_y'])**2 +
        (df['slam_z'] - df['gt_z'])**2
    )
    mean_error = df['error'].mean()
    max_error = df['error'].max()

    # Create trajectory figure
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

    fig.add_trace(go.Scatter3d(
        x=df['slam_x'], y=df['slam_y'], z=df['slam_z'],
        mode='lines',
        name='SLAM Error (colorized)',
        line=dict(
            color=df['error'],
            colorscale='Viridis',
            colorbar=dict(title='SLAM Error (m)'),
            width=6
        )
    ))

    fig.update_layout(
        title=f"SLAM vs Ground Truth Trajectory<br><sub>Mean error: {mean_error:.2f}m | Max error: {max_error:.2f}m</sub>",
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Z (m)",
            aspectmode="data"
        ),
        legend=dict(x=0, y=1)
    )

    # Error over time plot
    error_fig = go.Figure()
    error_fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['error'],
        mode='lines',
        name='SLAM Error (m)',
        line=dict(color='red')
    ))
    error_fig.update_layout(
        title="SLAM Error Over Time",
        xaxis_title="Timestamp (s)",
        yaxis_title="Error (meters)"
    )

    # Save to timestamped files
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    traj_path = Path(f"analysis/pose_vs_gt_plot_{ts}.html")
    err_path = Path(f"analysis/slam_error_plot_{ts}.html")
    fig.write_html(traj_path)
    error_fig.write_html(err_path)

    print(f"[✓] Saved 3D plot to: {traj_path}")
    print(f"[✓] Saved error plot to: {err_path}")
    print(f"[✓] Mean SLAM error: {mean_error:.2f} m | Max SLAM error: {max_error:.2f} m")

if __name__ == "__main__":
    main()
