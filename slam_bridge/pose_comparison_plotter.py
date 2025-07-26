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

    # Raw SLAM (no transformation)
    df['slam_x_raw'] = df['slam_x']
    df['slam_y_raw'] = df['slam_y']
    df['slam_z_raw'] = df['slam_z']

    # Transformed SLAM (no scale correction)
    df['slam_x_trans'] = df['slam_z']      # SLAM Z → Display X
    df['slam_y_trans'] = -df['slam_x']     # SLAM X → Display Y
    df['slam_z_trans'] = -df['slam_y']     # SLAM Y → Display Z

    # Scale-corrected SLAM (already present)
    df['slam_x_corrected'] = 0.7 * df['slam_z']    # SLAM Z → Display X
    df['slam_y_corrected'] = 0.68 * -df['slam_x']    # SLAM X → Display Y
    df['slam_z_corrected'] = 0.48 * -df['slam_y']    # SLAM Y → Display Z
    
    # Apply coordinate transformation to ground truth data  
    # Transform AirSim coordinates to match display coordinate system
    df['gt_x_corrected'] = df['gt_x']        # X stays the same
    df['gt_y_corrected'] = -df['gt_y']       # Invert Y axis
    df['gt_z_corrected'] = -df['gt_z']       # Invert Z axis
    
    logger.info("[COORD] Applied coordinate transformations:")
    logger.info("[COORD] SLAM: Z→X, X→Y, Y→Z")
    logger.info("[COORD] Ground Truth: Y and Z axes inverted")

    # Plot both trajectories together using corrected coordinates
    fig = go.Figure()

    # Ground Truth (corrected)
    fig.add_trace(go.Scatter3d(
        x=df['gt_x_corrected'], y=df['gt_y_corrected'], z=df['gt_z_corrected'],
        mode='lines', name='Ground Truth (Corrected)',
        line=dict(color='green')
    ))

    # SLAM Trajectory (scale-corrected)
    fig.add_trace(go.Scatter3d(
        x=df['slam_x_corrected'], y=df['slam_y_corrected'], z=df['slam_z_corrected'],
        mode='lines', name='SLAM (Scale Corrected)',
        line=dict(color='blue', width=3)
    ))

    # SLAM Trajectory (transformed, no scale)
    fig.add_trace(go.Scatter3d(
        x=df['slam_x_trans'], y=df['slam_y_trans'], z=df['slam_z_trans'],
        mode='lines', name='SLAM (Transformed, No Scale)',
        line=dict(color='orange', width=2, dash='dash')
    ))

    # SLAM Trajectory (raw)
    fig.add_trace(go.Scatter3d(
        x=df['slam_x_raw'], y=df['slam_y_raw'], z=df['slam_z_raw'],
        mode='lines', name='SLAM (Raw)',
        line=dict(color='red', width=1, dash='dot')
    ))

    fig.update_layout(
        title="SLAM vs Ground Truth Trajectory (Both Coordinate Corrected)",
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

    # Call the other plotting functions with corrected data
    plot_ground_truth_only(df)
    plot_slam_only(df)
    plot_translation_error(df)

def plot_ground_truth_only(df, save_dir="analysis"):
    import plotly.graph_objects as go
    from pathlib import Path
    from datetime import datetime

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=df['gt_x_corrected'], y=df['gt_y_corrected'], z=df['gt_z_corrected'],  # Use corrected coordinates
        mode='lines',
        name='Ground Truth (Corrected)',
        line=dict(color='green', width=4)
    ))
    fig.update_layout(
        title="Ground Truth Trajectory Only (Coordinate Corrected)",
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
    """Plot SLAM trajectory with corrected coordinates."""
    import plotly.graph_objects as go
    from pathlib import Path
    from datetime import datetime

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=df['slam_x_corrected'], y=df['slam_y_corrected'], z=df['slam_z_corrected'],
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

def plot_translation_error(df, save_dir="analysis"):
    """Plot translational error magnitude versus time using corrected coordinates."""
    import plotly.graph_objects as go
    from pathlib import Path
    from datetime import datetime

    # Calculate error using both corrected coordinate systems
    err = ((df['gt_x_corrected'] - df['slam_x_corrected']) ** 2 +
           (df['gt_y_corrected'] - df['slam_y_corrected']) ** 2 +
           (df['gt_z_corrected'] - df['slam_z_corrected']) ** 2) ** 0.5

    if 'timestamp' in df.columns:
        start = df['timestamp'].iloc[0]
        xvals = df['timestamp'] - start
        x_title = "Time (s)"
    else:
        xvals = df.index
        x_title = 'Index'

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xvals, y=err, mode='lines', name='error'))
    fig.update_layout(
        title="Translation Error Over Time",
        xaxis_title=x_title,
        yaxis_title="Error (m)",
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(save_dir) / f"slam_error_vs_time_{ts}.html"
    fig.write_html(path)
    logger.info(f"[✓] Saved translation error plot to: {path}")

if __name__ == "__main__":
    main()
