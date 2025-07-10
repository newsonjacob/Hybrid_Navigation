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
