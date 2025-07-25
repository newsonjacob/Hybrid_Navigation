"""Threaded perception loop handling image capture and optical flow."""

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
from uav.perception import filter_flow_by_depth, PerceptionData, FrameStats

logger = logging.getLogger("perception")


def perception_loop(tracker, image):
    """Process a single grayscale image for optical flow."""
    gray = image
    if tracker.prev_gray is None:
        tracker.initialize(gray)
        return np.array([]), np.array([]), 0.0
    return tracker.process_frame(gray, time.time())

# This function is called by the perception thread to start processing images.
def start_perception_thread(ctx):
    """Start the background perception worker.

    A daemon thread is launched that repeatedly grabs images from AirSim and
    processes them for optical flow. Each cycle fetches a grayscale flow image
    along with the stereo pair, decodes and resizes them, then runs
    :func:`perception_loop` to obtain flow vectors.  The resulting
    :class:`uav.perception.PerceptionData` object is placed into
    ``ctx.perception_queue`` with a maximum size of one so only the most recent
    result is kept.

    The worker thread runs until ``ctx.exit_flag`` is set.  ``ctx.perception_queue``
    provides the handoff point for the navigation loop, which retrieves the
    latest processed frame using ``Queue.get``. ``ctx.perception_thread`` holds a
    reference to the thread for later shutdown coordination.
    """
    exit_flag = ctx.exit_flag
    tracker = ctx.tracker
    perception_queue = Queue(maxsize=1)
    last_vis_img = np.zeros((720, 1280, 3), dtype=np.uint8)

    # Define a worker function to run in the thread
    def perception_worker(): 
        nonlocal last_vis_img 
        local_client = airsim.MultirotorClient()
        local_client.confirmConnection()
        request = [
            ImageRequest(config.FLOW_CAMERA, ImageType.Scene, False, True),
            ImageRequest(config.STEREO_LEFT_CAMERA, ImageType.Scene, False, True),
            ImageRequest(config.STEREO_RIGHT_CAMERA, ImageType.Scene, False, True),
        ]
        while not exit_flag.is_set():
            t0 = time.time()
            responses = local_client.simGetImages(request, vehicle_name="UAV")
            t_fetch_end = time.time()
            flow_resp, left_resp, right_resp = responses[0], responses[1], responses[2]
            # Check if the response is valid
            if (
                flow_resp.width == 0
                or flow_resp.height == 0
                or len(flow_resp.image_data_uint8) == 0
                or left_resp.width == 0
                or left_resp.height == 0
                or len(left_resp.image_data_uint8) == 0
                or right_resp.width == 0
                or right_resp.height == 0
                or len(right_resp.image_data_uint8) == 0
            ):
                # If no image data, use the last valid image
                data = PerceptionData(
                    vis_img=last_vis_img,
                    good_old=np.array([]),
                    flow_vectors=np.array([]),
                    flow_std=0.0,
                    simgetimage_s=t_fetch_end - t0,
                    decode_s=0.0,
                    processing_s=0.0,
                )
            else:
                # Decode the image data
                flow1d = np.frombuffer(flow_resp.image_data_uint8, dtype=np.uint8).copy()
                left1d = np.frombuffer(left_resp.image_data_uint8, dtype=np.uint8).copy()
                right1d = np.frombuffer(right_resp.image_data_uint8, dtype=np.uint8).copy()

                img = cv2.imdecode(flow1d, cv2.IMREAD_GRAYSCALE)
                left_img = cv2.imdecode(left1d, cv2.IMREAD_GRAYSCALE)
                right_img = cv2.imdecode(right1d, cv2.IMREAD_GRAYSCALE)

                t_decode_end = time.time()

                # Resize the image to the configured size
                if img is None or left_img is None or right_img is None:
                    continue
           
                #Resize images to the configured video size
                target_size = config.VIDEO_SIZE # (width, height)
                img = cv2.resize(img, target_size)
                left_img = cv2.resize(left_img, target_size)
                right_img = cv2.resize(right_img, target_size)
                vis_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                
                # Update the last valid image
                last_vis_img = vis_img
                t_proc_start = time.time()

                # Process the image for optical flow
                good_old, flow_vectors, flow_std = perception_loop(tracker, img)
                if good_old.size > 0:
                    # Filter flow vectors by depth if stereo images are available
                    good_old, flow_vectors = filter_flow_by_depth(
                        good_old,  # points
                        flow_vectors, # vectors
                        left_img, # (stereo left)
                        right_img, # (stereo right)
                        max_depth=config.DEPTH_FILTER_DIST,
                    )
                processing_s = time.time() - t_proc_start
                data = PerceptionData(
                    vis_img=vis_img,
                    good_old=good_old,
                    flow_vectors=flow_vectors,
                    flow_std=flow_std,
                    simgetimage_s=t_fetch_end - t0,
                    decode_s=t_decode_end - t_fetch_end,
                    processing_s=processing_s,
                )
            try:
                # Put the processed data into the queue
                perception_queue.put(data, block=False)
            except Exception:
                pass
    # Start the worker thread
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
    """Process perception output and update histories.

    Parameters
    ----------
    client : airsim.MultirotorClient
        Client used for issuing control commands when testing manual nudges.
    args : argparse.Namespace
        Parsed command-line arguments controlling runtime options.
    data : PerceptionData
        Perception results for the current frame.
    frame_count : int
        Index of the current processed frame starting at 1.
    frame_queue : Queue
        Queue used for storing visualisation frames.
    flow_history : FlowHistory
        History buffer for computing smoothed left/centre/right magnitudes.
    navigator : Navigation
        Navigator instance used to check grace-period state.
    param_refs : Namespace
        Namespace containing arrays for previous and delta flow values.
    time_now : float
        Timestamp for the current frame in seconds.
    max_flow_mag : float
        Maximum allowed optical flow magnitude when clamping.

    Returns
    -------
    Tuple[PerceptionData, FrameStats]
        The original :class:`PerceptionData` and computed frame statistics.
    """
    vis_img = data.vis_img
    good_old = data.good_old
    flow_vectors = data.flow_vectors
    flow_std = data.flow_std
    simgetimage_s = data.simgetimage_s
    decode_s = data.decode_s
    processing_s = data.processing_s

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

    # Clamp abnormally large flow magnitudes to avoid spikes
    num_clamped = np.sum(magnitudes > max_flow_mag)
    if num_clamped > 100:
        logger.warning("Clamped %d large flow magnitudes to %s", num_clamped, max_flow_mag)
    magnitudes = np.clip(magnitudes, 0, max_flow_mag)

    # Note: Minimum flow filtering is now handled in OpticalFlowTracker.process_frame()
    # The flow_vectors and good_old arrays are already filtered

    good_old = good_old.reshape(-1, 2)
    # Compute average magnitude and feature counts for spatial regions
    (
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
    ) = compute_region_stats(magnitudes, good_old, image_width)

    # Update history with region magnitudes and get smoothed values
    flow_history.update(left_mag, center_mag, right_mag)
    smooth_L, smooth_C, smooth_R = flow_history.average()

    # Track change since previous frame
    delta_L = smooth_L - param_refs.prev_L[0]
    delta_C = smooth_C - param_refs.prev_C[0]
    delta_R = smooth_R - param_refs.prev_R[0]
    # Store smoothed values and deltas for external modules
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

    # Log flow filtering statistics
    logger.debug(f"[FLOW_STATS] L:{smooth_L:.2f} C:{smooth_C:.2f} R:{smooth_R:.2f} | "
               f"Features: L:{left_count} C:{center_count} R:{right_count} | "
               f"Total filtered features: {len(good_old)}")

    stats = FrameStats(
        smooth_L=smooth_L,
        smooth_C=smooth_C,
        smooth_R=smooth_R,
        delta_L=delta_L,
        delta_C=delta_C,
        delta_R=delta_R,
        probe_mag=probe_mag,
        probe_count=probe_count,
        left_count=left_count,
        center_count=center_count,
        right_count=right_count,
        top_mag=top_mag,
        mid_mag=mid_mag,
        bottom_mag=bottom_mag,
        top_count=top_count,
        mid_count=mid_count,
        bottom_count=bottom_count,
        in_grace=in_grace,
    )
    return data, stats
