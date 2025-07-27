"""
Script to overlay multiple flight trajectories on a single 3D map visualization.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from mesh_utils import (
    add_environment_mesh_to_plot,
    extract_mesh_data,
    apply_mesh_corrections,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("overlay_trajectories")

# Try to import trimesh for mesh loading
try:
    import trimesh
    TRIMESH_AVAILABLE = True
    logger.info("✅ trimesh available for mesh loading")
except ImportError:
    TRIMESH_AVAILABLE = False
    logger.warning("⚠️ trimesh not available. Install with: pip install trimesh")

# Define trajectory colors for up to 10 simulations
TRAJECTORY_COLORS = [
    '#FF0000',  # Red
    '#0000FF',  # Blue
    '#00FF00',  # Green
    '#FF8C00',  # Orange
    '#8A2BE2',  # Purple
    '#FFD700',  # Gold
    '#FF1493',  # Deep Pink
    '#00CED1',  # Dark Turquoise
    '#32CD32',  # Lime Green
    '#FF4500',  # Red Orange
]

def load_environment_mesh():
    """Load the environment mesh with scaling."""
    if not TRIMESH_AVAILABLE:
        logger.warning("trimesh not available for mesh loading")
        return None
        
    try:
        import json

        # Paths to try for settings.json
        settings_paths = [
            "settings.json",
            "../settings.json",
            os.path.expanduser("~/Documents/AirSim/settings.json"),
            r"C:\Users\Jacob\Documents\AirSim\settings.json"
        ]
        
        settings = None
        for settings_path in settings_paths:
            if os.path.exists(settings_path):
                with open(settings_path, 'r') as f:
                    settings = json.load(f)
                logger.info(f"Loaded AirSim settings from: {settings_path}")
                break
        
        # Define mesh file candidates
        mesh_file_candidates = [
            r"H:\Documents\AirSimExperiments\Hybrid_Navigation\Map_Reactive\Map_Reactive.obj",
            settings.get("environment_mesh", "") if settings else "",
            "Map_Reactive.obj",
            "Map_Reactive/Map_Reactive.obj", 
            "../Map_Reactive.obj",
            "../Map_Reactive/Map_Reactive.obj",
        ]
        
        # Remove empty strings and duplicates
        mesh_file_candidates = list(filter(None, set(mesh_file_candidates)))
        
        # Load mesh
        mesh = None
        for mpath in mesh_file_candidates:
            if os.path.exists(mpath):
                logger.info(f"✅ Found mesh file at: {mpath}")
                try:
                    mesh = trimesh.load(mpath)
                    
                    # Apply scaling
                    scale_factor = 0.01  # Adjust based on your mesh size
                    if scale_factor != 1.0:
                        mesh.apply_scale(scale_factor)
                    
                    logger.info(f"✅ Successfully loaded and scaled mesh")
                    break
                    
                except Exception as e:
                    logger.warning(f"Failed to load mesh from {mpath}: {e}")
                    continue
        
        return mesh
        
    except Exception as e:
        logger.error(f"Error in load_environment_mesh: {e}")
        return None


def load_trajectory_from_csv(csv_path: str) -> Optional[tuple]:
    """Load trajectory data from a CSV log file."""
    try:
        df = pd.read_csv(csv_path)
        
        if not all(col in df.columns for col in ["pos_x", "pos_y", "pos_z"]):
            logger.error(f"Missing position columns in {csv_path}")
            return None
        
        # Extract trajectory points
        trajectory = df[["pos_x", "pos_y", "pos_z"]].to_numpy(dtype=float)
        
        # Fix AirSim Z-axis (negative Z = up → positive Z = up)
        if np.mean(trajectory[:, 2]) < 0:
            trajectory[:, 2] = -trajectory[:, 2]

        # # Invert Y-axis
        # trajectory[:, 1] = -trajectory[:, 1]
        
        # Extract metadata
        start_pos = trajectory[0] if len(trajectory) > 0 else np.array([0, 0, 0])
        end_pos = trajectory[-1] if len(trajectory) > 0 else np.array([0, 0, 0])
        
        # Calculate flight statistics
        total_distance = 0
        if len(trajectory) > 1:
            distances = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)
            total_distance = np.sum(distances)
        
        # Check for collisions
        collisions = df["collided"].sum() if "collided" in df.columns else 0
        
        # Flight duration
        duration = df["time"].max() - df["time"].min() if "time" in df.columns and len(df) > 0 else 0
        
        metadata = {
            "path": csv_path,
            "points": len(trajectory),
            "distance": total_distance,
            "duration": duration,
            "collisions": collisions,
            "start": start_pos,
            "end": end_pos
        }
        
        logger.info(f"✅ Loaded trajectory: {len(trajectory)} points, {total_distance:.2f}m, {duration:.1f}s")
        return trajectory, metadata
        
    except Exception as e:
        logger.error(f"Failed to load trajectory from {csv_path}: {e}")
        return None

def create_overlay_plot(trajectories: List[tuple], output_path: str):
    """Create a 3D plot with multiple overlaid trajectories."""
    fig = go.Figure()
    
    # Load and add environment mesh
    env_mesh = load_environment_mesh()
    if env_mesh is not None:
        add_environment_mesh_to_plot(fig, env_mesh, opacity=0.2)
    
    # Track overall bounds for setting plot ranges
    all_points = []
    
    # Add each trajectory
    for i, (trajectory, metadata) in enumerate(trajectories):
        color = TRAJECTORY_COLORS[i % len(TRAJECTORY_COLORS)]
        
        # Create trajectory name
        flight_name = f"Flight {i+1}"
        csv_name = Path(metadata["path"]).stem
        if "reactive_log_" in csv_name:
            timestamp = csv_name.replace("reactive_log_", "")
            flight_name = f"Flight {timestamp[:8]}_{timestamp[9:13]}"
        
        # Add trajectory line
        fig.add_trace(go.Scatter3d(
            x=trajectory[:, 0],
            y=trajectory[:, 1], 
            z=trajectory[:, 2],
            mode='lines+markers',
            line=dict(color=color, width=2),
            marker=dict(size=1, color=color),
            name=f"{flight_name}",
            hovertemplate=(
                f"<b>{flight_name}</b><br>"
                "X: %{x:.2f}m<br>"
                "Y: %{y:.2f}m<br>"
                "Z: %{z:.2f}m<br>"
                f"Distance: {metadata['distance']:.2f}m<br>"
                f"Duration: {metadata['duration']:.1f}s<br>"
                f"Collisions: {metadata['collisions']}<br>"
                "<extra></extra>"
            )
        ))
        
        # Add start marker
        fig.add_trace(go.Scatter3d(
            x=[trajectory[0, 0]],
            y=[trajectory[0, 1]],
            z=[trajectory[0, 2]],
            mode='markers',
            marker=dict(size=3, color=color, symbol='diamond'),
            name=f"{flight_name} Start",
            showlegend=False,
            hovertemplate=f"<b>{flight_name} START</b><br>X: %{{x:.2f}}m<br>Y: %{{y:.2f}}m<br>Z: %{{z:.2f}}m<extra></extra>"
        ))
        
        # Add end marker
        fig.add_trace(go.Scatter3d(
            x=[trajectory[-1, 0]],
            y=[trajectory[-1, 1]],
            z=[trajectory[-1, 2]],
            mode='markers',
            marker=dict(size=3, color=color, symbol='square'),
            name=f"{flight_name} End",
            showlegend=False,
            hovertemplate=f"<b>{flight_name} END</b><br>X: %{{x:.2f}}m<br>Y: %{{y:.2f}}m<br>Z: %{{z:.2f}}m<extra></extra>"
        ))
        
        all_points.extend(trajectory)
    
    # Set plot ranges
    if all_points:
        all_points = np.array(all_points)
        
        # Calculate ranges with padding
        flight_ranges = [
            [all_points[:, i].min(), all_points[:, i].max()] 
            for i in range(3)
        ]
        env_ranges = [[-0.5, 49.5], [-15, 15], [0, 10]]
        
        ranges = [
            [min(flight_ranges[i][0], env_ranges[i][0]) - (2 if i != 1 else 0),
             max(flight_ranges[i][1], env_ranges[i][1]) + (2 if i != 1 else 0)]
            for i in range(3)
        ]
        ranges[2][0] = max(ranges[2][0], 0)  # Don't go below ground
        
        fig.update_scenes(
            xaxis=dict(range=ranges[0], title="X Position (m)"),
            yaxis=dict(range=ranges[1], title="Y Position (m)"),
            zaxis=dict(range=ranges[2], title="Z Position (m)"),
            aspectmode='manual',
            aspectratio=dict(x=1, y=0.6, z=0.3)
        )
    
    # Update layout
    fig.update_layout(
        title=f"🚁 Multi-Flight Trajectory Overlay ({len(trajectories)} Flights)",
        width=1600,
        height=1200,
        scene=dict(
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)),
            bgcolor='white'
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.8)"
        )
    )
    
    # Save the plot
    fig.write_html(output_path)
    logger.info(f"✅ Saved overlay plot: {output_path}")
    
    # Print summary
    logger.info(f"\n📊 FLIGHT SUMMARY:")
    logger.info(f"{'Flight':<20} {'Distance':<12} {'Duration':<12} {'Collisions':<12}")
    logger.info("-" * 60)
    
    for i, (trajectory, metadata) in enumerate(trajectories):
        flight_name = f"Flight {i+1}"
        logger.info(f"{flight_name:<20} {metadata['distance']:<12.2f} {metadata['duration']:<12.1f} {metadata['collisions']:<12}")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Overlay multiple UAV flight trajectories on a 3D map")
    parser.add_argument(
        "logs", 
        nargs="+", 
        help="CSV log files to overlay (up to 10 recommended)"
    )
    parser.add_argument(
        "-o", "--output",
        default="analysis/trajectory_overlay.html",
        help="Output HTML file path"
    )
    parser.add_argument(
        "--max-flights",
        type=int,
        default=10,
        help="Maximum number of flights to overlay (default: 10)"
    )
    return parser.parse_args()

def main():
    """Main function."""
    try:
        args = parse_args()
        
        logger.info(f"🚀 Starting trajectory overlay analysis")
        logger.info(f"Input logs: {len(args.logs)} files")
        logger.info(f"Output: {args.output}")
        
        # Limit number of flights
        if len(args.logs) > args.max_flights:
            logger.warning(f"Too many flights ({len(args.logs)}), limiting to {args.max_flights}")
            args.logs = args.logs[:args.max_flights]
        
        # Load all trajectories
        trajectories = []
        for log_path in args.logs:
            if not os.path.exists(log_path):
                logger.error(f"File not found: {log_path}")
                continue
            
            result = load_trajectory_from_csv(log_path)
            if result is not None:
                trajectories.append(result)
            else:
                logger.warning(f"Skipping {log_path} due to loading error")
        
        if not trajectories:
            logger.error("No valid trajectories loaded!")
            sys.exit(1)
        
        logger.info(f"Successfully loaded {len(trajectories)} trajectories")
        
        # Create output directory
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        
        # Create overlay plot
        create_overlay_plot(trajectories, args.output)
        
        logger.info("✅ Trajectory overlay completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Overlay analysis failed: {e}")
        import traceback
        logger.error("Full error traceback:")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()