# === Perception Loop Module ===
import logging
import time
from queue import Queue
from threading import Thread

import cv2
import numpy as np
import airsim
from airsim import ImageRequest, ImageType

from uav import config
from uav.scoring import compute_region_stats

logger = logging.getLogger(__name__)


def perception_loop(tracker, image):
    """Process a single grayscale image for optical flow."""
    gray = image
    if tracker.prev_gray is None:
        tracker.initialize(gray)
        return np.array([]), np.array([]), 0.0
    return tracker.process_frame(gray, time.time())


def start_perception_thread(ctx):
    """Launch background perception thread and attach queue to ctx."""
    exit_flag = ctx.exit_flag
    tracker = ctx.tracker
    perception_queue = Queue(maxsize=1)
    last_vis_img = np.zeros((720, 1280, 3), dtype=np.uint8)

    def perception_worker():
        nonlocal last_vis_img
        local_client = airsim.MultirotorClient()
        local_client.confirmConnection()
        request = [ImageRequest("0", ImageType.Scene, False, True)]
        while not exit_flag.is_set():
            t0 = time.time()
            responses = local_client.simGetImages(request, vehicle_name="UAV")
            t_fetch_end = time.time()
            response = responses[0]
            if (
                response.width == 0
                or response.height == 0
                or len(response.image_data_uint8) == 0
            ):
                data = (last_vis_img, np.array([]), np.array([]), 0.0, t_fetch_end - t0, 0.0, 0.0)
            else:
                img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8).copy()
                img = cv2.imdecode(img1d, cv2.IMREAD_GRAYSCALE)
                t_decode_end = time.time()
                if img is None:
                    continue
                img = cv2.resize(img, config.VIDEO_SIZE)
                vis_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                last_vis_img = vis_img
                t_proc_start = time.time()
                good_old, flow_vectors, flow_std = perception_loop(tracker, img)
                processing_s = time.time() - t_proc_start
                data = (
                    vis_img,
                    good_old,
                    flow_vectors,
                    flow_std,
                    t_fetch_end - t0,
                    t_decode_end - t_fetch_end,
                    processing_s,
                )
            try:
                perception_queue.put(data, block=False)
            except Exception:
                pass

    perception_thread = Thread(target=perception_worker, daemon=True)
    perception_thread.start()
    ctx.perception_queue = perception_queue
    ctx.perception_thread = perception_thread


def process_perception_data(
    client,
    args,
    data,
    frame_count,
    frame_queue,
    flow_history,
    navigator,
    param_refs,
    time_now,
    max_flow_mag,
):
    """Process perception output and update histories."""
    (
        vis_img,
        good_old,
        flow_vectors,
        flow_std,
        simgetimage_s,
        decode_s,
        processing_s,
    ) = data

    image_width = vis_img.shape[1]
    if frame_count == 1 and len(good_old) == 0:
        frame_queue.put(vis_img)
        return None

    if args.manual_nudge and frame_count == 5:
        logger.info("Manual nudge forward for test")
        client.moveByVelocityAsync(2, 0, 0, 2)

    if flow_vectors.size == 0:
        magnitudes = np.array([])
    else:
        if flow_vectors.ndim == 1:
            flow_vectors = flow_vectors.reshape(-1, 2)
        magnitudes = np.linalg.norm(flow_vectors, axis=1)

    num_clamped = np.sum(magnitudes > max_flow_mag)
    if num_clamped > 100:
        logger.warning("Clamped %d large flow magnitudes to %s", num_clamped, max_flow_mag)
    magnitudes = np.clip(magnitudes, 0, max_flow_mag)

    good_old = good_old.reshape(-1, 2)
    (
        left_mag,
        center_mag,
        right_mag,
        probe_mag,
        probe_count,
        left_count,
        center_count,
        right_count,
    ) = compute_region_stats(magnitudes, good_old, image_width)

    flow_history.update(left_mag, center_mag, right_mag)
    smooth_L, smooth_C, smooth_R = flow_history.average()

    delta_L = smooth_L - param_refs.prev_L[0]
    delta_C = smooth_C - param_refs.prev_C[0]
    delta_R = smooth_R - param_refs.prev_R[0]
    param_refs.prev_L[0], param_refs.prev_C[0], param_refs.prev_R[0] = (
        smooth_L,
        smooth_C,
        smooth_R,
    )
    param_refs.delta_L[0], param_refs.delta_C[0], param_refs.delta_R[0] = (
        delta_L,
        delta_C,
        delta_R,
    )
    param_refs.L[0], param_refs.C[0], param_refs.R[0] = smooth_L, smooth_C, smooth_R

    if navigator.just_resumed and time_now < navigator.resume_grace_end_time:
        cv2.putText(vis_img, "GRACE", (1100, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
    in_grace = navigator.just_resumed and time_now < navigator.resume_grace_end_time

    return (
        vis_img,
        good_old,
        flow_vectors,
        flow_std,
        simgetimage_s,
        decode_s,
        processing_s,
        smooth_L,
        smooth_C,
        smooth_R,
        delta_L,
        delta_C,
        delta_R,
        probe_mag,
        probe_count,
        left_count,
        center_count,
        right_count,
        in_grace,
    )
