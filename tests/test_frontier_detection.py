import numpy as np
from slam_bridge.frontier_detection import detect_frontiers, ORIGIN


def test_single_point_frontiers():
    points = np.array([[0.0, 0.0, 0.0]])
    result = detect_frontiers(points, voxel_size=0.5)
    assert result.shape == (26, 3)
    # Compute expected centers around the single occupied voxel
    center_idx = ((points[0] - ORIGIN) / 0.5).astype(int)
    expected = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == dy == dz == 0:
                    continue
                idx = center_idx + np.array([dx, dy, dz])
                expected.append(ORIGIN + (idx + 0.5) * 0.5)
    expected = np.array(sorted(expected, key=lambda x: tuple(x)))
    result_sorted = np.array(sorted(result.tolist(), key=lambda x: tuple(x)))
    assert np.allclose(result_sorted, expected)


def test_empty_input_returns_empty_array():
    """detect_frontiers should return an empty array when given no points."""
    empty_points = np.empty((0, 3))
    result = detect_frontiers(empty_points)
    assert isinstance(result, np.ndarray)
    assert result.size == 0
    assert result.shape == (0, 3)


def test_custom_grid_and_origin():
    origin = np.array([0.0, 0.0, 0.0])
    grid_size = (3, 3, 3)
    points = np.array([[1.0, 1.0, 1.0]])
    result = detect_frontiers(points, voxel_size=1.0, grid_size=grid_size, origin=origin)

    center_idx = ((points[0] - origin) / 1.0).astype(int)
    expected = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == dy == dz == 0:
                    continue
                idx = center_idx + np.array([dx, dy, dz])
                if np.all((idx >= 0) & (idx < np.array(grid_size))):
                    expected.append(origin + (idx + 0.5) * 1.0)
    expected = np.array(sorted(expected, key=lambda x: tuple(x)))
    result_sorted = np.array(sorted(result.tolist(), key=lambda x: tuple(x)))
    assert np.allclose(result_sorted, expected)
