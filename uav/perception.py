# uav/perception.py
"""Perception utilities for computing optical flow and tracking history."""

from __future__ import annotations

import time
from dataclasses import dataclass
from collections import deque
from typing import Deque, Dict, Optional, Tuple

import cv2
import numpy as np

from .utils import apply_clahe
from . import config
import logging

logger = logging.getLogger("perception")


@dataclass
class PerceptionData:
    """Container for a single frame's perception results."""

    vis_img: np.ndarray
    good_old: np.ndarray
    flow_vectors: np.ndarray
    flow_std: float
    simgetimage_s: float
    decode_s: float
    processing_s: float


@dataclass
class FrameStats:
    """Per-frame metrics computed from optical flow."""

    smooth_L: float
    smooth_C: float
    smooth_R: float
    delta_L: float
    delta_C: float
    delta_R: float
    probe_mag: float
    probe_count: int
    left_count: int
    center_count: int
    right_count: int
    top_mag: float
    mid_mag: float
    bottom_mag: float
    top_count: int
    mid_count: int
    bottom_count: int
    in_grace: bool

class FlowHistory:
    """Maintain a rolling window of recent flow magnitudes."""

    def __init__(self, size: int = 3) -> None:
        """Create a buffer storing the last ``size`` flow measurements.

        Args:
            size: Maximum number of recent flow values to retain.
        """
        self.size: int = size
        self.window: Deque[np.ndarray] = deque(maxlen=size)

    def update(self, left: float, center: float, right: float) -> None:
        """Append a new triple of flow magnitudes to the history.

        Args:
            left: Mean optical flow magnitude for the left third of the image.
            center: Magnitude for the center region.
            right: Magnitude for the right third.
        """
        self.window.append(np.array([left, center, right]))

    def average(self) -> Tuple[float, float, float]:
        """Return the mean of the stored flow readings.

        Returns:
            A tuple ``(left, center, right)`` containing the average magnitudes
            of the recorded history.  ``(0.0, 0.0, 0.0)`` is returned if no
            history is stored.
        """
        if not self.window:
            return 0.0, 0.0, 0.0
        arr = np.array(self.window)
        return tuple(arr.mean(axis=0))


class OpticalFlowTracker:
    """Track sparse optical flow features between frames."""

    def __init__(self, lk_params: dict, feature_params: dict, min_flow_mag: Optional[float] = None) -> None:
        """Initialize tracker with Lucas-Kanade and feature parameters.

        Args:
            lk_params: Parameters for ``cv2.calcOpticalFlowPyrLK``.
            feature_params: Parameters for ``cv2.goodFeaturesToTrack``.
        """
        self.lk_params = lk_params
        self.feature_params = feature_params
        self.min_flow_mag = (
            config.MIN_FLOW_MAG if min_flow_mag is None else float(min_flow_mag)
        )
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_pts: Optional[np.ndarray] = None
        self.prev_time: float = time.time()

    def initialize(self, gray_frame: np.ndarray) -> None:
        """Start tracking using the provided grayscale frame.

        Args:
            gray_frame: Grayscale image used to seed the tracker.
        """
        gray_eq = apply_clahe(gray_frame)
        self.prev_gray = gray_eq
        self.prev_pts = cv2.goodFeaturesToTrack(
            gray_eq,
            mask=None,
            **self.feature_params,
        )
        self.prev_time = time.time()

    def process_frame(
        self,
        gray: np.ndarray,
        _unused_start_time: float,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Track features in ``gray`` and return motion information.

        Args:
            gray: The next grayscale frame in which to track the features.
            _unused_start_time: Timestamp supplied by callers and ignored.

        Returns:
            A tuple ``(points, vectors, std)`` where ``points`` are the source
            feature locations, ``vectors`` are the motion vectors between
            frames and ``std`` is the standard deviation of their magnitudes.
        """
        gray_eq = apply_clahe(gray)

        if self.prev_gray is None or self.prev_pts is None:
            self.initialize(gray)
            return np.array([]), np.array([]), 0.0

        next_pts, status, err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray,
            gray_eq,
            self.prev_pts,
            None,
            **self.lk_params,
        )

        if next_pts is None or status is None:
            self.initialize(gray)
            return np.array([]), np.array([]), 0.0

        good_old = self.prev_pts[status.flatten() == 1]
        good_new = next_pts[status.flatten() == 1]

        current_time = time.time()
        dt = max(current_time - self.prev_time, 1e-6)  # avoid div by zero
        self.prev_time = current_time

        self.prev_gray = gray_eq
        self.prev_pts = cv2.goodFeaturesToTrack(
            gray_eq,
            mask=None,
            **self.feature_params,
        )

        if len(good_old) == 0:
            return np.array([]), np.array([]), 0.0
        
        flow_vectors = (good_new - good_old).reshape(-1, 2)
        good_old = good_old.reshape(-1, 2)

        magnitudes = np.linalg.norm(flow_vectors, axis=1) / dt

        # Filter out low-magnitude background flow
        if self.min_flow_mag is not None:
            valid_mask = magnitudes >= self.min_flow_mag
            good_old = good_old[valid_mask]
            flow_vectors = flow_vectors[valid_mask]
            magnitudes = magnitudes[valid_mask]

        flow_std = float(np.std(magnitudes)) if len(magnitudes) > 0 else 0.0
 
        return good_old, flow_vectors, flow_std
    
def filter_flow_by_depth(
    points: np.ndarray,
    vectors: np.ndarray,
    left_gray: np.ndarray,
    right_gray: np.ndarray,
    max_depth: float = config.DEPTH_FILTER_DIST,
    matcher: Optional[cv2.StereoMatcher] = None,
    focal_length: float = 300.0,
    baseline: float = 0.075,
) -> Tuple[np.ndarray, np.ndarray]:
    """Remove flow vectors whose stereo depth exceeds max_depth with improved stereo processing."""
    
    logger.debug(f"[DEPTH_FILTER] Processing {len(points)} points with max_depth={max_depth}m")
    
    if len(points) == 0:
        return points, vectors
    
    # Ensure images are the same size
    if left_gray.shape != right_gray.shape:
        logger.warning(f"[DEPTH_FILTER] Image size mismatch: {left_gray.shape} vs {right_gray.shape}")
        return points, vectors
    
    try:
        # Preprocess images for better stereo matching
        left_processed = cv2.GaussianBlur(left_gray, (5, 5), 0)
        right_processed = cv2.GaussianBlur(right_gray, (5, 5), 0)
        
        if matcher is None:
            # Use SGBM for better quality (slower but more accurate)
            matcher = cv2.StereoSGBM_create(
                minDisparity=0,
                numDisparities=96,  # Must be divisible by 16
                blockSize=5,
                P1=8 * 3 * 5**2,   # Penalty for disparity change of +/- 1
                P2=32 * 3 * 5**2,  # Penalty for larger disparity changes
                disp12MaxDiff=1,
                uniquenessRatio=10,
                speckleWindowSize=100,
                speckleRange=32,
                preFilterCap=63,
                mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
            )
        
        # Compute disparity
        disparity = matcher.compute(left_processed, right_processed).astype(np.float32)
        
        # SGBM returns 16-bit fixed point, convert to float
        disparity = disparity / 16.0
        
        # Validate disparity map
        valid_disp = disparity[disparity > 0]
        if len(valid_disp) == 0:
            logger.warning("[DEPTH_FILTER] No valid disparities - stereo matching failed")
            return points, vectors
        
        logger.debug(f"[DEPTH_FILTER] Disparity range: {np.min(valid_disp):.1f} - {np.max(valid_disp):.1f}")
        
        # Calculate depth map
        with np.errstate(divide='ignore', invalid='ignore'):
            depth_map = (focal_length * baseline) / disparity
        
        # Clean up depth map
        depth_map[disparity <= 0] = np.inf
        depth_map[np.isnan(depth_map)] = np.inf
        depth_map[depth_map <= 0] = np.inf
        
        # Sample depths at feature locations
        y_coords = np.clip(points[:, 1].astype(int), 0, depth_map.shape[0] - 1)
        x_coords = np.clip(points[:, 0].astype(int), 0, depth_map.shape[1] - 1)
        point_depths = depth_map[y_coords, x_coords]
        
        # Apply depth filter
        valid_mask = (point_depths <= max_depth) & (point_depths > 0.1) & np.isfinite(point_depths)
        
        filtered_points = points[valid_mask]
        filtered_vectors = vectors[valid_mask]
        
        # Statistics
        valid_depths = point_depths[valid_mask]
        logger.debug(f"[DEPTH_FILTER] Filtered: {len(filtered_points)}/{len(points)} points")
        if len(valid_depths) > 0:
            logger.debug(f"[DEPTH_FILTER] Kept depths: {np.min(valid_depths):.2f}m - {np.max(valid_depths):.2f}m (mean: {np.mean(valid_depths):.2f}m)")
        
        return filtered_points, filtered_vectors
        
    except Exception as e:
        logger.error(f"[DEPTH_FILTER] Error: {e}")
        # Return original data on error
        return points, vectors
    