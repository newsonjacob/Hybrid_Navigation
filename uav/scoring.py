"""Optical flow scoring helpers for region statistics."""

import numpy as np

def compute_region_stats(
    magnitudes: np.ndarray, good_old: np.ndarray, image_width: int
) -> tuple:
    """Return average flow magnitude in spatial regions."""

    h = 720  # Fixed height based on 1280x720 resolution
    good_old = good_old.reshape(-1, 2)
    x_coords = good_old[:, 0]
    y_coords = good_old[:, 1]

    # Define vertical bands
    left_mask = x_coords < image_width // 3
    center_mask = (x_coords >= image_width // 3) & (x_coords < 2 * image_width // 3)
    right_mask = x_coords >= 2 * image_width // 3

    # Horizontal bands
    top_mask = y_coords < h // 3
    mid_mask = (y_coords >= h // 3) & (y_coords < 2 * h // 3)
    bottom_mask = y_coords >= 2 * h // 3

    # Probe band uses top-center region
    probe_band = center_mask & top_mask

    # Magnitudes
    left_mag = np.mean(magnitudes[left_mask]) if np.any(left_mask) else 0
    center_mag = np.mean(magnitudes[center_mask]) if np.any(center_mask) else 0
    right_mag = np.mean(magnitudes[right_mask]) if np.any(right_mask) else 0
    probe_mag = np.mean(magnitudes[probe_band]) if np.any(probe_band) else 0
    probe_count = int(np.sum(probe_band))

    top_mag = np.mean(magnitudes[top_mask]) if np.any(top_mask) else 0
    mid_mag = np.mean(magnitudes[mid_mask]) if np.any(mid_mask) else 0
    bottom_mag = np.mean(magnitudes[bottom_mask]) if np.any(bottom_mask) else 0

    # Feature counts
    left_count = int(np.sum(left_mask))
    center_count = int(np.sum(center_mask))
    right_count = int(np.sum(right_mask))

    top_count = int(np.sum(top_mask))
    mid_count = int(np.sum(mid_mask))
    bottom_count = int(np.sum(bottom_mask))

    return (
        left_mag,
        center_mag,
        right_mag,
        probe_mag,
        probe_count,
        left_count,
        center_count,
        right_count,
        top_mag,
        mid_mag,
        bottom_mag,
        top_count,
        mid_count,
        bottom_count,
    )
