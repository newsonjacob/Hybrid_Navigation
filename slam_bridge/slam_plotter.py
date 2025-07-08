# slam_plotter.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from slam_bridge.slam_receiver import get_latest_pose
from pathlib import Path
import time
import plotly.graph_objects as go

x_vals, y_vals, z_vals = [], [], []

def plot_slam_trajectory():
    print("[SLAM Plotter] Collecting SLAM poses...")
    try:
        while True:
            pose = get_latest_pose()
            if pose is not None:
                x, y, z = pose
                x_vals.append(x)
                y_vals.append(y)
                z_vals.append(z)
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        save_interactive_plot()

def save_interactive_plot():
    plot_dir = Path("analysis")
    plot_dir.mkdir(exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = plot_dir / f"slam_traj_{timestamp}.html"

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=x_vals, mode='lines', name='x'))
    fig.add_trace(go.Scatter(y=y_vals, mode='lines', name='y'))
    fig.add_trace(go.Scatter(y=z_vals, mode='lines', name='z'))
    fig.update_layout(
        title="SLAM Translation (x, y, z)",
        xaxis_title="Frame index",
        yaxis_title="Position (m)",
        template="plotly_dark"
    )
    fig.write_html(str(filename))
    print(f"[SLAM Plotter] Saved interactive plot to {filename}")
