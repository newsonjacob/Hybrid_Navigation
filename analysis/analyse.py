import argparse
from pathlib import Path
from typing import List, Union
import sys
import os
from datetime import datetime
import logging

import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# ========== Setup Logging FIRST ========== #

def get_timestamp_from_args():
    """Extract timestamp from command line args if available."""
    try:
        # Quick parse just for timestamp
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--log-timestamp", type=str, default=None)
        args, _ = parser.parse_known_args()
        return args.log_timestamp
    except:
        return None

# Setup logging once at module level
timestamp = get_timestamp_from_args() or datetime.now().strftime('%Y%m%d_%H%M%S')
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file_path = log_dir / f"analyse_{timestamp}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(str(log_file_path)),
        logging.StreamHandler(sys.stdout)
    ],
    force=True
)

logger = logging.getLogger("analyse")
logger.info(f"Analysis logging configured - writing to {log_file_path}")

# ========== Now Safe to Import with Logger ========== #

# Add trimesh import for mesh loading - AFTER logger is created
try:
    import trimesh
    TRIMESH_AVAILABLE = True
    logger.info("[OK] trimesh available for mesh loading")
except ImportError:
    TRIMESH_AVAILABLE = False
    logger.warning("⚠️ trimesh not available. Install with: pip install trimesh")

# Import flight analysis modules
try:
    from .flight_review import (
        parse_log,
        plot_state_histogram,
        plot_distance_over_time,
    )
    from .visualise_flight import build_plot
    logger.info("✅ Imported modules using relative imports")
except ImportError as e:
    logger.warning(f"Relative imports failed: {e}")
    try:
        from flight_review import (
            parse_log,
            plot_state_histogram,
            plot_distance_over_time,
        )
        from visualise_flight import build_plot
        logger.info("[OK] Imported modules using direct imports")
    except ImportError as e:
        logger.error(f"❌ Failed to import required modules: {e}")
        logger.error("Make sure flight_review.py and visualise_flight.py are in the analysis directory")
        sys.exit(1)

# Import mesh utilities
from .mesh_utils import (
    add_environment_mesh_to_plot,
    extract_mesh_data,
    apply_mesh_corrections,
)


# ========== Analysis Functions ========== #
# These functions handle the main analysis logic.

def analyse_logs(log_file_path, output_dir):
    """Analyze navigation logs and generate reports."""
    try:
        df = pd.read_csv(log_file_path)
        logger.info(f"[DEBUG] Loaded CSV with {len(df)} rows and {len(df.columns)} columns")
        
        # Check if dataframe is empty
        if df.empty:
            logger.error(f"[ERROR] Log file is empty: {log_file_path}")
            return
            
        if len(df) == 0:
            logger.error(f"[ERROR] No data rows in log file: {log_file_path}")
            return
            
        # Check if required columns exist
        required_columns = ["time", "frame", "state"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.warning(f"Missing expected columns: {missing_columns}")
        
        # Now safe to access data
        start_time = df["time"].iloc[0]
        logger.info(f"[OK] Analysis starting - first timestamp: {start_time}")
        
        # Convert time to relative seconds from start
        if "time" in df.columns:
            start_time = df["time"].iloc[0]
            df["time_relative"] = df["time"] - start_time
            time_col = "time_relative"
            time_label = "Time (seconds from start)"
        else:
            time_col = "time"
            time_label = "Time"

        path = df[["pos_x", "pos_y", "pos_z"]].to_numpy(dtype=float)

        # Generate separate 3D trajectory file
        output = Path(output_dir)
        generate_3d_trajectory(path, df, output, time_col)

        # Basic summary statistics for testing environments
        stats = [
            {
                "frames": len(df),
                "collisions": int(df.get("collision", pd.Series(0)).sum()),
                "distance": float(df.get("pos_x", pd.Series()).diff().abs().sum()),
                "fps_avg": float(df.get("fps", pd.Series()).mean()),
                "loop_avg": float(df.get("loop_s", pd.Series()).mean()),
            }
        ]

        fig = make_subplots(
            rows=4,
            cols=1,
            specs=[
                [{"type": "xy"}],
                [{"type": "xy"}],
                [{"type": "xy", "secondary_y": True}],  # Enable secondary y-axis for row 3
                [{"type": "table"}]
            ],
            subplot_titles=( 
                "Flow Magnitudes", 
                "Speed",
                "Performance",
                "Flight Summary"
            ),
            row_heights=[0.25, 0.25, 0.25, 0.25],  # Give table equal height with other plots
            vertical_spacing=0.08  # Adjust spacing between subplots
        )

        # Add flow magnitudes to row 1 (now works because it's "xy" type)
        if {time_col, "flow_left", "flow_center", "flow_right"}.issubset(df.columns):
            fig.add_trace(go.Scatter(
                x=df[time_col], 
                y=df["flow_left"], 
                name="flow_left", 
                line=dict(color='red')
            ), row=1, col=1)  
            fig.add_trace(go.Scatter(
                x=df[time_col], 
                y=df["flow_center"], 
                name="flow_center",
                line=dict(color='blue')
            ), row=1, col=1)  
            fig.add_trace(go.Scatter(
                x=df[time_col], 
                y=df["flow_right"], 
                name="flow_right",
                line=dict(color='orange')
            ), row=1, col=1)

        # Add speed
        if time_col in df.columns and "speed" in df.columns:
            fig.add_trace(go.Scatter(
                x=df[time_col], 
                y=df["speed"], 
                name="speed",
                line=dict(color='blue')
            ), row=2, col=1)

        # Add performance metrics
        if time_col in df.columns and "cpu_percent" in df.columns:
            fig.add_trace(go.Scatter(
                x=df[time_col], 
                y=df["cpu_percent"], 
                name="CPU %",
                line=dict(color='red')
            ), row=3, col=1, secondary_y=False)  # Primary y-axis
            
        if time_col in df.columns and "memory_rss" in df.columns:
            fig.add_trace(go.Scatter(
                x=df[time_col], 
                y=df["memory_rss"] / (1024 * 1024),
                name="Memory (MB)",
                line=dict(color='purple')
            ), row=3, col=1, secondary_y=True)  # Secondary y-axis

        # Calculate summary statistics
        total_frames = sum(s["frames"] for s in stats)
        total_collisions = sum(s["collisions"] for s in stats) -1  # Subtract 1 from total collision count
        total_distance = sum(s["distance"] for s in stats)
        fps_vals = [s["fps_avg"] for s in stats if not np.isnan(s["fps_avg"])]
        loop_vals = [s["loop_avg"] for s in stats if not np.isnan(s["loop_avg"])]

        # Calculate CPU and memory statistics
        cpu_vals = df["cpu_percent"].dropna() if "cpu_percent" in df.columns else pd.Series(dtype=float)
        memory_vals = df["memory_rss"].dropna() / (1024 * 1024) if "memory_rss" in df.columns else pd.Series(dtype=float)  # Convert to MB

        # Calculate flight duration
        flight_duration = 0
        if time_col in df.columns and len(df) > 0:
            flight_duration = df[time_col].max()

        # Create summary statistics table
        summary_data = [
            ["Flight Duration", f"{flight_duration:.2f} seconds"],
            ["Total Frames", f"{total_frames:,}"],
            ["Collisions", f"{total_collisions}"],
            ["Distance Travelled", f"{total_distance:.2f} m"],
        ]
        
        if fps_vals:
            summary_data.append(["Average FPS", f"{np.mean(fps_vals):.2f}"])
        if loop_vals:
            summary_data.append(["Average Loop Time", f"{np.mean(loop_vals):.3f}s"])
        
        # Add CPU statistics
        if len(cpu_vals) > 0:
            summary_data.append(["Average CPU", f"{cpu_vals.mean():.1f}%"])
            summary_data.append(["Peak CPU", f"{cpu_vals.max():.1f}%"])
        
        # Add memory statistics
        if len(memory_vals) > 0:
            summary_data.append(["Average Memory", f"{memory_vals.mean():.1f} MB"])
            summary_data.append(["Peak Memory", f"{memory_vals.max():.1f} MB"])
        
        fig.add_trace(go.Table(
            header=dict(
                values=["Metric", "Value"],
                fill_color='lightblue',
                align='left',
                font=dict(size=14, color='darkblue')
            ),
            cells=dict(
                values=list(zip(*summary_data)),  # Transpose the data
                fill_color='white',
                align='left',
                font=dict(size=12)
            )
        ), row=4, col=1)

        fig.update_layout(
            height=1800,  # Increased from 1500 to 1800 for more table space
            width=1000,
            title="🚁 UAV Flight Analysis Report",
            showlegend=True
        )
        
        # Update axis labels
        fig.update_xaxes(title_text=time_label, row=1, col=1)
        fig.update_yaxes(title_text="Flow Magnitude", row=1, col=1)
        fig.update_xaxes(title_text=time_label, row=2, col=1)
        fig.update_yaxes(title_text="Speed (m/s)", row=2, col=1)
        fig.update_xaxes(title_text=time_label, row=3, col=1)
        fig.update_yaxes(title_text="CPU (%)", row=3, col=1, secondary_y=False)
        fig.update_yaxes(title_text="Memory (MB)", row=3, col=1, secondary_y=True)

        Path(output).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(output)

        # Still print to console as well
        logger.info(f"Flight Duration: {flight_duration:.2f} seconds")
        logger.info(f"Frames: {total_frames}")
        logger.info(f"Collisions: {total_collisions}")
        logger.info(f"Distance travelled: {total_distance:.2f} m")
        if fps_vals:
            logger.info(f"Average FPS: {np.mean(fps_vals):.2f}")
        if loop_vals:
            logger.info(f"Average loop time: {np.mean(loop_vals):.3f}s")

    except Exception as e:
        logger.error("[ERROR] Analysis failed: %s", e)
        import traceback
        logger.error("Full error traceback:")
        logger.error(traceback.format_exc())
        sys.exit(1)
    
    logger.info("Analysis completed successfully, exiting with code 0")
    sys.exit(0)

# ========== Environment Mesh Loading ========== #
# This function loads the environment mesh and applies scaling based on flight data.
def load_environment_mesh():
    """Load the environment mesh with smart scaling based on flight data."""
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
        settings_path_used = None
        
        for settings_path in settings_paths:
            if os.path.exists(settings_path):
                with open(settings_path, 'r') as f:
                    settings = json.load(f)
                settings_path_used = settings_path
                logger.info(f"Loaded AirSim settings from: {settings_path}")
                break
        
        # Define mesh file candidates - include your actual location
        mesh_file_candidates = [
            # Your actual mesh location
            r"H:\Documents\AirSimExperiments\Hybrid_Navigation\Map_Reactive\Map_Reactive.obj",
            
            # Try from settings.json if it exists
            settings.get("environment_mesh", "") if settings else "",
            
            # Other common locations
            "Map_Reactive.obj",
            "Map_Reactive/Map_Reactive.obj", 
            "../Map_Reactive.obj",
            "../Map_Reactive/Map_Reactive.obj",
        ]
        
        # If settings.json specifies a mesh path, add more candidates
        if settings and "environment_mesh" in settings:
            mesh_path = settings["environment_mesh"]
            logger.info(f"Settings specifies environment mesh: {mesh_path}")
            
            # Add settings-based candidates
            mesh_file_candidates.extend([
                mesh_path,
                os.path.join(os.path.dirname(settings_path_used or ""), mesh_path),
                os.path.abspath(mesh_path),
            ])
        
        # Remove empty strings and duplicates
        mesh_file_candidates = list(filter(None, set(mesh_file_candidates)))
        
        logger.info(f"Searching for mesh in {len(mesh_file_candidates)} locations...")
        
        # Load mesh
        mesh = None
        for mpath in mesh_file_candidates:
            if os.path.exists(mpath):
                logger.info(f"✅ Found mesh file at: {mpath}")
                try:
                    logger.info(f"Loading mesh from: {mpath}")
                    mesh = trimesh.load(mpath)
                    
                    # Check mesh bounds BEFORE scaling
                    if hasattr(mesh, 'bounds'):
                        original_bounds = mesh.bounds
                        mesh_size = np.linalg.norm(original_bounds[1] - original_bounds[0])
                        logger.info(f"Original mesh bounds: {original_bounds}")
                        logger.info(f"Original mesh size: {mesh_size:.2f} units")
                        
                        # For your Map_Reactive.obj: Size is ~5,041 units
                        # We want this to be ~50 meters for your small flight paths
                        # So scale factor should be: 50 / 5041 = ~0.01
                        
                        scale_factor = 0.01  # This will make your 5000-unit map into 50 meters
                        if settings and "mesh_scale_factor" in settings:
                            scale_factor = settings["mesh_scale_factor"]
                            logger.info(f"Using manual scale factor from settings: {scale_factor}")
                        else:
                            logger.info(f"Using calculated scale factor: {scale_factor}")
                        
                        # Apply scaling only (orientation and position corrections happen in add_environment_mesh_to_plot)
                        if scale_factor != 1.0:
                            mesh.apply_scale(scale_factor)
                            new_bounds = mesh.bounds
                            new_size = np.linalg.norm(new_bounds[1] - new_bounds[0])
                            logger.info(f"Scaled mesh bounds: {new_bounds}")
                            logger.info(f"Scaled mesh size: {new_size:.2f} meters")
                    
                    # Log mesh info
                    if hasattr(mesh, 'vertices'):
                        vertex_count = len(mesh.vertices)
                        face_count = len(mesh.faces) if hasattr(mesh, 'faces') else 0
                    elif hasattr(mesh, 'scene'):
                        vertex_count = sum(len(geom.vertices) for geom in mesh.scene.geometry.values())
                        face_count = sum(len(geom.faces) for geom in mesh.scene.geometry.values())
                    else:
                        vertex_count = "unknown"
                        face_count = "unknown"
                    
                    logger.info(f"✅ Successfully loaded and scaled mesh: {vertex_count} vertices, {face_count} faces")
                    logger.info("📋 Note: Orientation and position corrections will be applied during visualization")
                    break
                    
                except Exception as e:
                    logger.warning(f"Failed to load mesh from {mpath}: {e}")
                    continue
        
        if mesh is None:
            logger.warning("❌ Environment mesh file not found!")
            logger.info("Searched in:")
            for candidate in mesh_file_candidates[:5]:  # Show first 5
                exists = "✅" if os.path.exists(candidate) else "❌"
                logger.info(f"  {exists} {candidate}")
        else:
            logger.info(f"🎉 Successfully loaded and scaled environment mesh!")
        
        return mesh
        
    except Exception as e:
        logger.error(f"Error in load_environment_mesh: {e}")
        return None

# ========== 3D Trajectory Generation ========== #
# This function generates a 3D trajectory plot with the environment mesh.
def generate_3d_trajectory(path, df, output, time_col):
    """Generate 3D trajectory with environment mesh."""
    try:
        from .visualise_flight import build_plot
        
        # Fix AirSim Z-axis (negative Z = up → positive Z = up)
        corrected_path = path.copy()
        corrected_path[:, 2] = -corrected_path[:, 2]
        
        # Create 3D plot
        fig3d = build_plot(corrected_path, [], np.array([0, 0, 0]), log=df, colour_by=None)
        
        # Add environment mesh
        env_mesh = load_environment_mesh()
        if env_mesh is not None:
            add_environment_mesh_to_plot(fig3d, env_mesh, opacity=0.3)
        
        # Set visualization ranges
        if corrected_path.size > 0:
            # Calculate ranges
            flight_ranges = [
                [corrected_path[:, i].min(), corrected_path[:, i].max()] 
                for i in range(3)
            ]
            env_ranges = [[-0.5, 49.5], [-15, 15], [0, 10]]
            
            # Combine with padding
            ranges = [
                [min(flight_ranges[i][0], env_ranges[i][0]) - (2 if i != 1 else 0),
                 max(flight_ranges[i][1], env_ranges[i][1]) + (2 if i != 1 else 0)]
                for i in range(3)
            ]
            ranges[2][0] = max(ranges[2][0], 0)  # Don't go below ground
            
            fig3d.update_scenes(
                xaxis=dict(range=ranges[0], title="X Position (m)"),
                yaxis=dict(range=ranges[1], title="Y Position (m)"),
                zaxis=dict(range=ranges[2], title="Z Position (m)"),
                aspectmode='manual',
                aspectratio=dict(x=1, y=0.6, z=0.3)
            )

        # Update layout and save
        fig3d.update_layout(
            title="🚁 3D Flight Trajectory with Environment",
            width=1500, height=1100,
            scene=dict(camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)), bgcolor='white')
        )
        
        output_path = Path(output).parent / f"trajectory_{Path(output).stem}.html"
        fig3d.write_html(str(output_path))
        logger.info(f"✅ 3D Trajectory saved: {output_path}")

    except Exception as e:
        logger.warning(f"Could not generate 3D trajectory: {e}")

# ========== Plot Generation ========== #
# This function generates plots from the log file.
def generate_plots(log_path: str, outdir: str) -> None:
    """Create state histogram and distance plots from ``log_path``."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    stats = parse_log(log_path)
    plot_state_histogram(stats, str(out / "state_histogram.html"))
    plot_distance_over_time(log_path, str(out / "distance_over_time.html"))

# ========== Argument Parsing ========== #
# This function parses command line arguments for the analysis script.
def parse_args(argv: Union[List[str], None] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyse flight logs")
    parser.add_argument("logs", nargs="+", help="CSV log files")
    parser.add_argument(
        "-o",
        "--output",
        default="analysis/flight_view.html",
        help="Output HTML file or directory",
    )
    parser.add_argument(
        "--log-timestamp", 
        type=str, 
        default=None,
        help="Timestamp for log file naming"
    )
    return parser.parse_args(argv)

# ========== Main Function ========== #
# This is the main entry point for the analysis script.
def main(argv: Union[List[str], None] = None) -> None:
    try:
        logger.info("[START] Starting main analysis function")
        
        args = parse_args(argv)
        
        logger.info("[ANALYZE] Starting analysis:")
        logger.info(f"  Input logs: {args.logs}")
        logger.info(f"  Output: {args.output}")
        logger.info(f"  Working directory: {os.getcwd()}")

        # Test file existence
        for log_file in args.logs:
            if not os.path.exists(log_file):
                raise FileNotFoundError(f"Log file not found: {log_file}")
            logger.info("[OK] Found log file: %s", log_file)  

        if args.output.lower().endswith(".html"):
            logger.info("Generating HTML analysis report...")

            log_input = args.logs[0] if len(args.logs) == 1 else args.logs
            analyse_logs(log_input, args.output)

            logger.info(f"✅ Flight analysis completed successfully")
        else:
            if len(args.logs) != 1:
                raise ValueError("Provide exactly one log when output is a directory")
            logger.info("Generating plot files...")
            generate_plots(args.logs[0], args.output)
            logger.info(f"✅ Plot generation completed successfully")
            
    except Exception as e:
        logger.error("[ERROR] Analysis failed: %s", e)
        import traceback
        logger.error("Full error traceback:")
        logger.error(traceback.format_exc())
        sys.exit(1)
    
    logger.info("Analysis completed successfully, exiting with code 0")
    sys.exit(0)

# ========== Run Main Function ========== #
# This allows the script to be run directly or imported as a module.
if __name__ == "__main__":
    main()
