# Utility functions for handling environment meshes in analysis scripts.
import logging
import numpy as np
import plotly.graph_objects as go

logger = logging.getLogger(__name__)


def extract_mesh_data(mesh):
    """Extract vertices and faces from a mesh object."""
    vertices = None
    faces = None

    if hasattr(mesh, 'geometry') and mesh.geometry:
        # Scene object - combine all geometries
        for geom in mesh.geometry.values():
            if hasattr(geom, 'vertices') and len(geom.vertices) > 0:
                if vertices is None:
                    vertices = geom.vertices.copy()
                    faces = geom.faces.copy()
                else:
                    vertex_offset = len(vertices)
                    vertices = np.vstack([vertices, geom.vertices])
                    faces = np.vstack([faces, geom.faces + vertex_offset])
    elif hasattr(mesh, 'vertices') and hasattr(mesh, 'faces'):
        # Direct Trimesh object
        vertices = mesh.vertices.copy()
        faces = mesh.faces.copy()
    else:
        # Try Scene conversion
        try:
            combined_mesh = mesh.dump().sum()
            vertices = combined_mesh.vertices.copy()
            faces = combined_mesh.faces.copy()
        except Exception:
            pass

    return vertices, faces


def apply_mesh_corrections(vertices):
    """Apply coordinate system corrections to mesh vertices."""
    corrected = vertices.copy()

    # Swap Y and Z axes (mesh is rotated)
    corrected[:, [1, 2]] = vertices[:, [2, 1]]

    # Position mesh to start at X = -0.5
    x_offset = -0.5 - corrected[:, 0].min()
    corrected[:, 0] += x_offset

    # Center mesh at Y = 0
    y_center = (corrected[:, 1].min() + corrected[:, 1].max()) / 2
    corrected[:, 1] -= y_center

    logger.info(
        f"Applied mesh corrections: X offset={x_offset:.2f}, Y center={y_center:.2f}"
    )
    return corrected


def add_environment_mesh_to_plot(fig3d, mesh, opacity=0.15):
    """Add environment mesh to the 3D plot."""
    if mesh is None:
        logger.warning("No mesh provided to add_environment_mesh_to_plot")
        return

    try:
        # Extract vertices and faces
        vertices, faces = extract_mesh_data(mesh)

        if vertices is None or len(vertices) == 0:
            logger.warning("No valid mesh data found")
            return

        # Apply mesh corrections for AirSim coordinate system
        vertices = apply_mesh_corrections(vertices)

        # Simplify mesh for performance if needed
        if len(faces) > 8000:
            step = max(1, len(faces) // 8000)
            faces = faces[::step]
            logger.info(f"Simplified mesh to {len(faces)} faces for performance")

        # Add mesh to plot
        fig3d.add_trace(
            go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                color="lightgray",
                opacity=opacity,
                name="\U0001f5fa\ufe0f Environment Mesh",
                showlegend=True,
                hoverinfo="skip",
                lighting=dict(ambient=0.7, diffuse=0.8, specular=0.1),
                lightposition=dict(x=100, y=200, z=0),
            )
        )

        logger.info(
            f"\u2705 Added environment mesh ({len(vertices)} vertices, {len(faces)} faces)"
        )

    except Exception as e:
        logger.error(f"Failed to add environment mesh: {e}")

