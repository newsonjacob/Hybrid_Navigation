import numpy as np
from scipy import ndimage as ndi

# Voxel grid parameters
GRID_SIZE = (100, 100, 20)  # (x, y, z)
ORIGIN = np.array([-25.0, -25.0, -5.0])  # world-space origin of grid


def detect_frontiers(map_points: np.ndarray, voxel_size: float = 0.5) -> np.ndarray:
    """Return frontier voxel centers from SLAM map points.

    Parameters
    ----------
    map_points : np.ndarray
        Array of shape ``(N, 3)`` with 3D points in world coordinates.
    voxel_size : float, optional
        Size of each voxel in metres.

    Returns
    -------
    np.ndarray
        Array of shape ``(M, 3)`` containing world coordinates of frontier
        voxel centres.
    """
    # Start with all voxels unknown (-1)
    grid = np.full(GRID_SIZE, -1, dtype=np.int8)

    if map_points.size == 0:
        return np.empty((0, 3))

    # Convert map points to voxel indices
    indices = ((map_points - ORIGIN) / voxel_size).astype(int)
    valid = np.all((indices >= 0) & (indices < GRID_SIZE), axis=1)
    indices = indices[valid]

    # Mark occupied voxels
    grid[indices[:, 0], indices[:, 1], indices[:, 2]] = 1

    # Voxels neighbouring occupied space using 26-connectivity
    occupied = grid == 1
    dilated = ndi.binary_dilation(occupied, structure=np.ones((3, 3, 3)))

    # Frontier = unknown voxel that touches occupied space
    frontier_mask = (dilated & (grid == -1))
    frontier_indices = np.argwhere(frontier_mask)

    # Convert back to world coordinates at voxel centres
    centers = ORIGIN + (frontier_indices + 0.5) * voxel_size

    # Debug statistics
    unknown_count = np.count_nonzero(grid == -1)
    occupied_count = np.count_nonzero(occupied)
    frontier_count = len(frontier_indices)
    print(f"Occupied voxels: {occupied_count}, Unknown voxels: {unknown_count}, "
          f"Frontiers: {frontier_count}")

    return centers
