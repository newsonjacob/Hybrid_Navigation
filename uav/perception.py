# uav/perception.py
"""Perception utilities for computing optical flow and tracking history."""

from __future__ import annotations

import time
from collections import deque
from typing import Deque, Dict, Optional, Tuple

import cv2
import numpy as np

from .utils import apply_clahe
from . import config
import logging

logger = logging.getLogger("perception")

class FlowHistory:
    """Maintain a rolling window of recent flow magnitudes."""

    def __init__(self, size: int = 10) -> None:
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

    def __init__(self, lk_params: Dict, feature_params: Dict, min_flow_mag: float = None) -> None:
        """Initialize tracker with Lucas-Kanade and feature parameters.

        Args:
            lk_params: Parameters for ``cv2.calcOpticalFlowPyrLK``.
            feature_params: Parameters for ``cv2.goodFeaturesToTrack``.
        """
        self.lk_params: Dict = lk_params
        self.feature_params: Dict = feature_params
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
        
        flow_vectors = good_new - good_old        
        magnitudes = np.linalg.norm(flow_vectors, axis=1) / dt
        flow_std = np.std(magnitudes)
        
        return good_old, flow_vectors, flow_std
