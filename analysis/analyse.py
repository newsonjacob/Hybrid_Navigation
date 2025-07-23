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
    logger.info("✅ trimesh available for mesh loading")
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
        logger.info("✅ Imported modules using direct imports")
    except ImportError as e:
        logger.error(f"❌ Failed to import required modules: {e}")
        logger.error("Make sure flight_review.py and visualise_flight.py are in the analysis directory")
        sys.exit(1)

def add_environment_mesh_to_plot(fig3d, mesh, opacity=0.15):
    """Add environment mesh to the 3D plot."""
    if mesh is None:
        logger.warning("No mesh provided to add_environment_mesh_to_plot")
        return
        
    try:
        logger.info(f"Processing mesh of type: {type(mesh)}")
        
        vertices = None
        faces = None
        
        # Handle Scene objects (which is what we have)
        if hasattr(mesh, 'geometry'):
            # It's a Scene object
            logger.info("Processing Scene object")
            if mesh.geometry:
                logger.info(f"Scene has {len(mesh.geometry)} geometries")
                
                # Get all geometries from the scene
                geometries = list(mesh.geometry.values())
                if geometries:
                    # Try to get the first valid geometry
                    for i, geom in enumerate(geometries):
                        if hasattr(geom, 'vertices') and hasattr(geom, 'faces'):
                            logger.info(f"Geometry {i}: {type(geom)} with {len(geom.vertices)} vertices")
                            
                            if len(geom.vertices) > 0:
                                if vertices is None:
                                    vertices = geom.vertices.copy()  # Make a copy
                                    faces = geom.faces.copy()
                                else:
                                    # Combine with existing geometry
                                    try:
                                        vertex_offset = len(vertices)
                                        vertices = np.vstack([vertices, geom.vertices])
                                        faces = np.vstack([faces, geom.faces + vertex_offset])
                                        logger.info(f"Combined geometry {i}")
                                    except Exception as e:
                                        logger.warning(f"Could not combine geometry {i}: {e}")
                        else:
                            logger.info(f"Geometry {i}: {type(geom)} - no vertices/faces")
                    
                    logger.info(f"Combined geometries: {len(vertices) if vertices is not None else 0} total vertices")
            else:
                logger.warning("Scene object has no geometry")
                return
                
        elif hasattr(mesh, 'vertices') and hasattr(mesh, 'faces'):
            # It's already a Trimesh
            logger.info("Processing direct Trimesh object")
            vertices = mesh.vertices.copy()
            faces = mesh.faces.copy()
        else:
            logger.warning(f"Mesh object type not supported: {type(mesh)}")
            # Try the Scene conversion approach
            try:
                logger.info("Attempting to convert Scene to mesh...")
                combined_mesh = mesh.dump().sum()
                vertices = combined_mesh.vertices.copy()
                faces = combined_mesh.faces.copy()
                logger.info(f"Converted scene to mesh: {len(vertices)} vertices, {len(faces)} faces")
            except Exception as e:
                logger.error(f"Failed to convert scene: {e}")
                return
        
        if vertices is None or faces is None:
            logger.warning("Could not extract vertices or faces from mesh")
            return
            
        if len(vertices) == 0 or len(faces) == 0:
            logger.warning("Mesh has no vertices or faces")
            return
        
        # Apply mesh corrections: scaling already done in load_environment_mesh()
        logger.info(f"Applying mesh orientation and position corrections...")
        
        # Get original bounds
        original_bounds = [
            [vertices[:, 0].min(), vertices[:, 1].min(), vertices[:, 2].min()],
            [vertices[:, 0].max(), vertices[:, 1].max(), vertices[:, 2].max()]
        ]
        logger.info(f"Original mesh bounds: {original_bounds}")
        
        # 1. SWAP Y and Z axes to fix orientation (mesh is on its side)
        # Current: X, Y, Z
        # Need:    X, Z, Y (swap Y and Z)
        vertices_corrected = vertices.copy()
        vertices_corrected[:, [1, 2]] = vertices[:, [2, 1]]  # Swap Y and Z columns
        logger.info("✅ Swapped Y and Z axes to correct mesh orientation")
        
        # 2. REPOSITION mesh to start at x = -0.5
        current_x_min = vertices_corrected[:, 0].min()
        target_x_min = -0.5
        x_offset = target_x_min - current_x_min
        
        vertices_corrected[:, 0] += x_offset  # Shift X coordinates
        logger.info(f"✅ Repositioned mesh: X offset = {x_offset:.2f} (from {current_x_min:.2f} to {target_x_min})")
        
        # 3. CENTER mesh along Y-axis at y=0
        current_y_min = vertices_corrected[:, 1].min()
        current_y_max = vertices_corrected[:, 1].max()
        current_y_center = (current_y_min + current_y_max) / 2
        target_y_center = 0.0
        y_offset = target_y_center - current_y_center
        
        vertices_corrected[:, 1] += y_offset  # Shift Y coordinates
        logger.info(f"✅ Centered mesh along Y-axis: Y offset = {y_offset:.2f} (center from {current_y_center:.2f} to {target_y_center})")
        
        # Get final bounds after corrections
        final_bounds = [
            [vertices_corrected[:, 0].min(), vertices_corrected[:, 1].min(), vertices_corrected[:, 2].min()],
            [vertices_corrected[:, 0].max(), vertices_corrected[:, 1].max(), vertices_corrected[:, 2].max()]
        ]
        logger.info(f"Final corrected mesh bounds: {final_bounds}")
        
        mesh_extents = [
            final_bounds[1][0] - final_bounds[0][0],
            final_bounds[1][1] - final_bounds[0][1], 
            final_bounds[1][2] - final_bounds[0][2]
        ]
        logger.info(f"Mesh extents (L×W×H): {mesh_extents[0]:.1f} × {mesh_extents[1]:.1f} × {mesh_extents[2]:.1f} meters")
        
        # Use corrected vertices
        vertices = vertices_corrected
        
        logger.info(f"Final mesh stats: {len(vertices)} vertices, {len(faces)} faces")
            
        # Limit mesh complexity for performance
        max_faces = 8000  # Reasonable for web visualization
        if len(faces) > max_faces:
            logger.info(f"Mesh has {len(faces)} faces, simplifying to {max_faces} for performance")
            step = max(1, len(faces) // max_faces)
            faces = faces[::step]
            logger.info(f"Simplified to {len(faces)} faces")
        
        # Add mesh to plot with enhanced visibility
        fig3d.add_trace(go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1], 
            z=vertices[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            color='lightgray',
            opacity=opacity,
            name="🗺️ Environment Mesh",
            showlegend=True,
            hoverinfo='skip',
            lighting=dict(ambient=0.7, diffuse=0.8, specular=0.1),
            lightposition=dict(x=100, y=200, z=0)
        ))
        
        logger.info(f"✅ Added corrected environment mesh to 3D plot ({len(vertices)} vertices, {len(faces)} faces)")
        
    except Exception as e:
        logger.error(f"Failed to add environment mesh to plot: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")

def extract_obstacles_from_mesh(mesh):
    """Extract obstacle/building data from the mesh geometry."""
    obstacles = []
    
    if mesh is None:
        logger.warning("No mesh provided to extract obstacles from")
        return obstacles
    
    try:
        logger.info("Extracting obstacles from mesh geometry...")
        
        # Handle different mesh types
        geometries = []
        if hasattr(mesh, 'geometry'):
            # Scene object with multiple geometries
            geometries = list(mesh.geometry.values())
            logger.info(f"Found {len(geometries)} geometries in scene")
        elif hasattr(mesh, 'vertices') and hasattr(mesh, 'faces'):
            # Single Trimesh object
            geometries = [mesh]
            logger.info("Processing single mesh object")
        else:
            # Try to convert Scene to mesh
            try:
                logger.info("Attempting to convert Scene to individual meshes...")
                combined_mesh = mesh.dump().sum()
                geometries = [combined_mesh]
                logger.info(f"Converted scene to single mesh with {len(combined_mesh.vertices)} vertices")
            except Exception as e:
                logger.warning(f"Could not convert scene: {e}")
                return obstacles
        
        for i, geom in enumerate(geometries):
            if not (hasattr(geom, 'vertices') and hasattr(geom, 'faces')):
                logger.debug(f"Geometry {i} has no vertices/faces")
                continue
                
            if len(geom.vertices) == 0:
                logger.debug(f"Geometry {i} has no vertices")
                continue
            
            # Calculate bounding box for this geometry
            vertices = geom.vertices
            min_bounds = vertices.min(axis=0)
            max_bounds = vertices.max(axis=0)
            
            # Calculate dimensions
            dimensions = max_bounds - min_bounds
            center = (min_bounds + max_bounds) / 2
            
            logger.info(f"Geometry {i}: center=({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f}), "
                       f"dims=({dimensions[0]:.1f}×{dimensions[1]:.1f}×{dimensions[2]:.1f})")
            
            # More permissive filtering for obstacle detection
            # Skip tiny details (< 0.5m in any dimension) - reduced threshold
            if np.any(dimensions < 0.5):
                logger.debug(f"Skipping small geometry {i}: {dimensions}")
                continue
                
            # Skip ground plane or very large terrain (> 100m in X or Y) - reduced threshold
            if dimensions[0] > 100 or dimensions[1] > 100:
                logger.debug(f"Skipping large terrain geometry {i}: {dimensions}")
                continue
            
            # Skip very flat objects (likely ground, < 0.5m height) - reduced threshold  
            if dimensions[2] < 0.5:
                logger.debug(f"Skipping flat geometry {i}: {dimensions}")
                continue
            
            # Create obstacle entry
            obstacle = {
                "name": f"Building_{i:03d}",
                "center": center.tolist(),
                "min_bounds": min_bounds.tolist(),
                "max_bounds": max_bounds.tolist(),
                "dimensions": dimensions.tolist(),
                "vertex_count": len(vertices),
                "volume": np.prod(dimensions)  # Approximate volume
            }
            
            obstacles.append(obstacle)
            logger.info(f"✅ Extracted obstacle {obstacle['name']}: "
                       f"center=({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f}), "
                       f"size=({dimensions[0]:.1f}×{dimensions[1]:.1f}×{dimensions[2]:.1f})")
        
        # Sort obstacles by volume (largest first) for better visualization
        obstacles.sort(key=lambda x: x['volume'], reverse=True)
        
        logger.info(f"✅ Extracted {len(obstacles)} obstacles from mesh geometry")
        
        # Log top 5 largest obstacles
        for i, obs in enumerate(obstacles[:5]):
            dims = obs['dimensions']
            logger.info(f"  {obs['name']}: {dims[0]:.1f}×{dims[1]:.1f}×{dims[2]:.1f}m "
                       f"at ({obs['center'][0]:.1f}, {obs['center'][1]:.1f}, {obs['center'][2]:.1f})")
        
        if len(obstacles) > 5:
            logger.info(f"  ... and {len(obstacles)-5} more obstacles")
            
    except Exception as e:
        logger.error(f"Failed to extract obstacles from mesh: {e}")
        import traceback
        logger.error(f"Obstacle extraction error: {traceback.format_exc()}")
    
    return obstacles

def load_airsim_obstacles():
    """Load obstacle data from mesh geometry (preferred) or AirSim settings (fallback)."""
    obstacles = []
    
    # First try to load from mesh
    try:
        logger.info("Attempting to extract obstacles from environment mesh...")
        
        # Load the mesh
        env_mesh = load_environment_mesh()
        if env_mesh is not None:
            obstacles = extract_obstacles_from_mesh(env_mesh)
            
            if len(obstacles) > 0:
                logger.info(f"✅ Successfully extracted {len(obstacles)} obstacles from mesh")
                return obstacles
            else:
                logger.warning("No obstacles extracted from mesh, trying settings.json fallback")
        else:
            logger.warning("No mesh available, trying settings.json fallback")
            
    except Exception as e:
        logger.warning(f"Failed to extract obstacles from mesh: {e}")
    
    # Fallback: try to load from settings.json
    try:
        import json
        
        settings_paths = [
            "settings.json",
            "../settings.json", 
            os.path.expanduser("~/Documents/AirSim/settings.json"),
            r"C:\Users\Jacob\Documents\AirSim\settings.json"
        ]
        
        settings = None
        for settings_path in settings_paths:
            if os.path.exists(settings_path):
                logger.info(f"Found AirSim settings at: {settings_path}")
                with open(settings_path, 'r') as f:
                    settings = json.load(f)
                break
        
        if settings and "obstacles" in settings:
            logger.info(f"Loading {len(settings['obstacles'])} obstacles from AirSim settings")
            
            for i, obstacle in enumerate(settings["obstacles"]):
                if "bounds" in obstacle and len(obstacle["bounds"]) >= 2:
                    # Extract min and max bounds
                    min_bounds = obstacle["bounds"][0]  # [x_min, y_min, z_min]
                    max_bounds = obstacle["bounds"][1]  # [x_max, y_max, z_max]
                    
                    # Calculate center point and dimensions
                    center_x = (min_bounds[0] + max_bounds[0]) / 2
                    center_y = (min_bounds[1] + max_bounds[1]) / 2
                    center_z = (min_bounds[2] + max_bounds[2]) / 2
                    
                    width = max_bounds[0] - min_bounds[0]
                    depth = max_bounds[1] - min_bounds[1] 
                    height = max_bounds[2] - min_bounds[2]
                    
                    obstacles.append({
                        "name": obstacle.get("name", f"Manual_Obstacle_{i}"),
                        "center": [center_x, center_y, center_z],
                        "min_bounds": min_bounds,
                        "max_bounds": max_bounds,
                        "dimensions": [width, depth, height],
                        "source": "settings.json"
                    })
                    
                    logger.debug(f"Loaded obstacle: {obstacle.get('name')} at ({center_x:.1f}, {center_y:.1f}, {center_z:.1f})")
        
        logger.info(f"Successfully loaded {len(obstacles)} obstacles from settings.json")
        
    except Exception as e:
        logger.warning(f"Could not load obstacles from settings.json: {e}")
        import traceback
        logger.debug(f"Settings obstacle loading error: {traceback.format_exc()}")
    
    return obstacles

def analyse_logs(log_paths: List[str], output: str) -> None:
    """Parse ``log_paths`` and write an interactive HTML report."""
    dfs = []
    stats = []

    for p in log_paths:
        stats.append(parse_log(p))
        dfs.append(pd.read_csv(p))
    df = pd.concat(dfs, ignore_index=True)

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
    generate_3d_trajectory(path, df, output, time_col)

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
    total_collisions = sum(s["collisions"] for s in stats)
    total_distance = sum(s["distance"] for s in stats)
    fps_vals = [s["fps_avg"] for s in stats if not np.isnan(s["fps_avg"])]
    loop_vals = [s["loop_avg"] for s in stats if not np.isnan(s["loop_avg"])]

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
        height=1000,  # Increased for 4 rows
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

def generate_3d_trajectory(path, df, output, time_col):
    """Generate 3D trajectory with properly oriented and positioned environment."""
    try:
        from visualise_flight import build_plot
        
        # Fix AirSim Z-axis inversion (AirSim: negative Z = up, Visualization: positive Z = up)
        corrected_path = path.copy()
        corrected_path[:, 2] = -corrected_path[:, 2]  # Invert Z-axis
        logger.info("✅ Corrected AirSim Z-axis inversion (negative Z up → positive Z up)")
        
        # Generate base 3D plot with corrected path
        fig3d = build_plot(corrected_path, [], np.array([0, 0, 0]), log=df, colour_by=None)
        
        # Load and add environment mesh (with corrections)
        logger.info("Loading environment mesh...")
        env_mesh = load_environment_mesh()
        if env_mesh is not None:
            add_environment_mesh_to_plot(fig3d, env_mesh, opacity=0.3)
        else:
            logger.info("No environment mesh loaded")
        
        # Remove obstacle loading since you only want the mesh
        # (keeping this commented for future reference)
        # logger.info("Loading obstacles from settings...")
        # obstacles = load_airsim_obstacles()
        
        # Set proper axis scaling for both flight path and corrected environment
        if corrected_path.size > 0:
            # Get corrected flight path bounds
            flight_x_range = [corrected_path[:, 0].min(), corrected_path[:, 0].max()]
            flight_y_range = [corrected_path[:, 1].min(), corrected_path[:, 1].max()]
            flight_z_range = [corrected_path[:, 2].min(), corrected_path[:, 2].max()]
            
            # Environment now starts at x = -0.5 and extends to about x = 49.5 (50m total)
            # After Y/Z swap and positioning corrections
            env_x_range = [-0.5, 49.5]  # X: -0.5 to 49.5
            env_y_range = [-15, 15]     # Y: roughly ±25m after swapping
            env_z_range = [0, 10]       # Z: 0 to 25m height after swapping
            
            # Use the larger of flight or environment ranges, with padding
            x_range = [min(flight_x_range[0], env_x_range[0]) - 2, max(flight_x_range[1], env_x_range[1]) + 2]
            y_range = [min(flight_y_range[0], env_y_range[0]), max(flight_y_range[1], env_y_range[1])]
            z_range = [min(flight_z_range[0], env_z_range[0]) - 2, max(flight_z_range[1], env_z_range[1]) + 2]
            
            # Ensure Z range shows positive values (drone flying above ground)
            z_range[0] = max(z_range[0], 0)  # Don't go below ground level
            
            logger.info(f"Setting visualization ranges:")
            logger.info(f"  X: {x_range[0]:.1f} to {x_range[1]:.1f}")
            logger.info(f"  Y: {y_range[0]:.1f} to {y_range[1]:.1f}")
            logger.info(f"  Z: {z_range[0]:.1f} to {z_range[1]:.1f}")
            
            fig3d.update_scenes(
                xaxis=dict(range=x_range, title="X Position (m)"),
                yaxis=dict(range=y_range, title="Y Position (m)"),
                zaxis=dict(range=z_range, title="Z Position (m)"),
                aspectmode='manual',
                aspectratio=dict(x=1, y=0.6, z=0.3)
            )

        # Update layout for better visibility
        fig3d.update_layout(
            title="🚁 3D Flight Trajectory with Corrected Map_Reactive Environment",
            width=1500,
            height=1100,
            scene=dict(
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)),
                bgcolor='white'
            )
        )
        
        # Generate trajectory file
        output_path = Path(output)
        trajectory_path = output_path.parent / f"trajectory_{output_path.stem}.html"
        fig3d.write_html(str(trajectory_path))
        
        logger.info(f"✅ 3D Trajectory with corrected Map_Reactive environment saved: {trajectory_path}")

    except Exception as e:
        logger.warning(f"⚠️ Warning: Could not generate 3D trajectory: {e}")
        import traceback
        logger.warning(f"3D trajectory error details: {traceback.format_exc()}")


def generate_plots(log_path: str, outdir: str) -> None:
    """Create state histogram and distance plots from ``log_path``."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    stats = parse_log(log_path)
    plot_state_histogram(stats, str(out / "state_histogram.html"))
    plot_distance_over_time(log_path, str(out / "distance_over_time.html"))


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


def main(argv: Union[List[str], None] = None) -> None:
    try:
        logger.info("🚀 Starting main analysis function")
        
        args = parse_args(argv)
        
        logger.info("🔍 Starting analysis:")
        logger.info(f"  Input logs: {args.logs}")
        logger.info(f"  Output: {args.output}")
        logger.info(f"  Working directory: {os.getcwd()}")

        # Test file existence
        for log_file in args.logs:
            if not os.path.exists(log_file):
                raise FileNotFoundError(f"Log file not found: {log_file}")
            logger.info(f"✅ Found log file: {log_file}")

        if args.output.lower().endswith(".html"):
            logger.info("Generating HTML analysis report...")
            analyse_logs(args.logs, args.output)
            logger.info(f"✅ Flight analysis completed successfully")
        else:
            if len(args.logs) != 1:
                raise ValueError("Provide exactly one log when output is a directory")
            logger.info("Generating plot files...")
            generate_plots(args.logs[0], args.output)
            logger.info(f"✅ Plot generation completed successfully")
            
    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")
        import traceback
        logger.error("Full error traceback:")
        logger.error(traceback.format_exc())
        sys.exit(1)
    
    logger.info("Analysis completed successfully, exiting with code 0")
    sys.exit(0)

if __name__ == "__main__":
    main()
