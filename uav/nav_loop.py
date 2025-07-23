"""Main navigation loop responsible for moving the UAV."""

import os
import cv2
import time
import subprocess
import numpy as np
import logging
import sys
from datetime import datetime
from queue import Queue
from threading import Thread
from pathlib import Path

# === AirSim Imports ===
import airsim
try:
    from airsim import ImageRequest, ImageType, LandedState
except Exception:  # LandedState may not exist in stubbed environments
    from airsim import ImageRequest, ImageType
    LandedState = None

# === Internal Module Imports ===
from uav.overlay import draw_overlay
from uav.navigation_rules import compute_thresholds
from uav.video_utils import start_video_writer_thread
from uav.logging_utils import format_log_line
from uav.perception import OpticalFlowTracker, FlowHistory
from uav.navigation import Navigator
from uav.state_checks import in_grace_period
from uav.scoring import compute_region_stats
from uav.utils import (get_drone_state, retain_recent_logs, init_client)
from uav.utils import retain_recent_files, retain_recent_views
from uav import config
from uav.logging_helpers import log_frame_data, write_video_frame, write_frame_output, handle_reset
from uav.context import ParamRefs, NavContext
from uav.perception_loop import perception_loop, start_perception_thread, process_perception_data
from uav.navigation_core import detect_obstacle, determine_side_safety, handle_obstacle, navigation_step
from uav.navigation_slam_boot import run_slam_bootstrap
from uav.paths import STOP_FLAG_PATH
from uav.slam_utils import (is_slam_stable, generate_pose_comparison_plot,)

logger = logging.getLogger("nav_loop")

# === Perception Processing ===

def setup_environment(args, client):
    """Initialize the navigation environment and return a context dict."""
    from uav.interface import exit_flag
    from uav.utils import retain_recent_logs
    param_refs = ParamRefs()
    logger.info("Available vehicles: %s", client.listVehicles())
    init_client(client)
    client.takeoffAsync().join(); client.moveToPositionAsync(0, 0, -2, 2).join()

    feature_params = dict(
        maxCorners=150, 
        qualityLevel=0.01, 
        minDistance=8, 
        blockSize=7, 
        useHarrisDetector=True, 
        k=0.04)
    lk_params = dict(
        winSize=(21, 21), 
        maxLevel=2, 
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        flags=cv2.OPTFLOW_LK_GET_MIN_EIGENVALS,
        minEigThreshold=1e-3
    )

    # Add minimum flow magnitude filter to tracker initialization
    tracker = OpticalFlowTracker(
        lk_params,
        feature_params,
        config.MIN_FLOW_MAG,
    )

    flow_history, navigator = FlowHistory(size=config.FLOW_HISTORY_SIZE), Navigator(client)

    from collections import deque
    state_history, pos_history = deque(maxlen=3), deque(maxlen=3)
    start_time = time.time()
    GOAL_X, MAX_SIM_DURATION = args.goal_x, args.max_duration
    logger.info("Config:\n  Goal X: %sm\n  Max Duration: %ss", GOAL_X, MAX_SIM_DURATION)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs("flow_logs", exist_ok=True)
    log_file = open(f"flow_logs/full_log_{timestamp}.csv", 'w')
    log_file.write(
        "frame,flow_left,flow_center,flow_right,"
        "delta_left,delta_center,delta_right,flow_std,"
        "left_count,center_count,right_count,"
        "brake_thres,dodge_thres,probe_req,fps,"
        "state,collided,obstacle,side_safe,"
        "pos_x,pos_y,pos_z,yaw,speed,"
        "time,features,simgetimage_s,decode_s,processing_s,loop_s,cpu_percent,memory_rss,"
        "sudden_rise,center_blocked,combination_flow,minimum_flow\n"
    )
    retain_recent_logs("flow_logs")
    retain_recent_logs("logs")
    retain_recent_files("analysis", "slam_traj_*.html", keep=5)
    retain_recent_files("analysis", "slam_output_*.mp4", keep=5)
    try: fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    except AttributeError: fourcc = cv2.FOURCC(*'MJPG')
    out = cv2.VideoWriter(config.VIDEO_OUTPUT, fourcc, config.VIDEO_FPS, config.VIDEO_SIZE)
    frame_queue = Queue(maxsize=20)
    video_thread = start_video_writer_thread(frame_queue, out, exit_flag)
    ctx = NavContext(
        exit_flag=exit_flag,
        param_refs=param_refs,
        tracker=tracker,
        flow_history=flow_history,
        navigator=navigator,
        state_history=state_history,
        pos_history=pos_history,
        frame_queue=frame_queue,
        video_thread=video_thread,
        out=out,
        log_file=log_file,
        log_buffer=[],
        timestamp=timestamp,
        start_time=start_time,
        fps_list=[],
        fourcc=fourcc,
    )
    return ctx

# === Helper Functions ===

def check_startup_grace(ctx, time_now):
    """Return True when startup grace period has elapsed."""
    if ctx.startup_grace_over:
        return True
    if time_now - ctx.start_time < config.GRACE_PERIOD_SEC:
        ctx.param_refs.state[0] = "startup_grace"
        if not ctx.grace_logged:
            logger.info(
                "Startup grace period active — waiting to start perception and nav"
            )
            ctx.grace_logged = True
        time.sleep(0.05)
        return False
    ctx.startup_grace_over = True
    logger.info("Startup grace period complete — beginning full nav logic")
    return True


def check_exit_conditions(client, ctx, time_now, max_duration, goal_x, goal_y):
    """Return True if navigation loop should terminate."""
    if os.path.exists(STOP_FLAG_PATH):
        logger.info("Stop flag detected. Landing and shutting down.")
        ctx.exit_flag.set()
        return True
    if time_now - ctx.start_time >= max_duration:
        logger.info("Time limit reached — landing and stopping.")
        return True
    pos_goal, _, _ = get_drone_state(client)
    if abs(pos_goal.x_val - goal_x) < 0.5 and abs(pos_goal.y_val - goal_y) < 0.5:
        logger.info("Goal reached — landing.")
        return True
    return False


def get_perception_data(ctx):
    """Retrieve the latest perception result or None."""
    try:
        return ctx.perception_queue.get(timeout=1.0)
    except Exception:
        return None


def update_navigation_state(client, args, ctx, data, frame_count, time_now, max_flow_mag):
    """Process perception data and decide the next navigation action."""
    processed = process_perception_data(
        client,
        args,
        data,
        frame_count,
        ctx.frame_queue,
        ctx.flow_history,
        ctx.navigator,
        ctx.param_refs,
        time_now,
        max_flow_mag,
    )
    if processed is None:
        return None
    (
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
        top_mag,
        mid_mag,
        bottom_mag,
        top_count,
        mid_count,
        bottom_count,
        in_grace,
    ) = processed
    prev_state = ctx.param_refs.state[0]
    nav_decision = navigation_step(
        client,
        ctx.navigator,
        ctx.flow_history,
        good_old,
        flow_vectors,
        flow_std,
        smooth_L,
        smooth_C,
        smooth_R,
        delta_L,
        delta_C,
        delta_R,
        left_count,
        center_count,
        right_count,
        ctx.frame_queue,
        vis_img,
        time_now,
        frame_count,
        prev_state,
        ctx.state_history,
        ctx.pos_history,
        ctx.param_refs,
        probe_mag=probe_mag,
        probe_count=probe_count,
    )
    return processed, nav_decision

# === Logging and Frame Recording ===
# This function logs the processed frame data, overlays telemetry information,
# and writes the frame to the video output queue.
def log_and_record_frame(
    client,
    ctx,
    loop_start,
    frame_duration,
    processed,
    nav_decision,
    frame_count,
    time_now,
):
    """Overlay telemetry, log, and queue video frames."""
    (
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
        top_mag,
        mid_mag,
        bottom_mag,
        top_count,
        mid_count,
        bottom_count,
        in_grace,
    ) = processed
    (
        state_str,
        obstacle_detected,
        side_safe,
        brake_thres,
        dodge_thres,
        probe_req,
        sudden_rise,
        center_blocked,
        combination_flow,
        minimum_flow,
    ) = nav_decision
    return write_frame_output(
        client,
        vis_img,
        ctx.frame_queue,
        loop_start,
        frame_duration,
        ctx.fps_list,
        ctx.start_time,
        smooth_L,
        smooth_C,
        smooth_R,
        delta_L,
        delta_C,
        delta_R,
        left_count,
        center_count,
        right_count,
        good_old,
        flow_vectors,
        in_grace,
        frame_count,
        time_now,
        ctx.param_refs,
        ctx.log_file,
        ctx.log_buffer,
        state_str,
        obstacle_detected,
        side_safe,
        brake_thres,
        dodge_thres,
        probe_req,
        simgetimage_s,
        decode_s,
        processing_s,
        flow_std,
        sudden_rise,
        center_blocked,
        combination_flow,
        minimum_flow,
    )

# === Main Navigation Loop ===
# This function runs the main navigation loop for Reactive Navigation, processing perception data
# and making navigation decisions based on the UAV's state and environment.
def navigation_loop(args, client, ctx):
    """Run the reactive navigation cycle.

    Each iteration retrieves perception data, checks exit conditions and
    delegates decision making to :func:`update_navigation_state` before
    logging the frame.
    """
    exit_flag = ctx.exit_flag

    max_flow_mag = config.MAX_FLOW_MAG
    max_duration = args.max_duration
    goal_x, goal_y = args.goal_x, config.GOAL_Y
    frame_count = 0
    frame_duration = 1.0 / config.TARGET_FPS

    logger.info("[NavLoop] Starting navigation loop with args: %s", args)
    loop_start = time.time()
    try:
        while not exit_flag.is_set():
            frame_count += 1
            time_now = time.time()

            if not check_startup_grace(ctx, time_now):
                continue

            data = get_perception_data(ctx)
            if data is None:
                continue

            if check_exit_conditions(client, ctx, time_now, max_duration, goal_x, goal_y):
                break

            result = update_navigation_state(
                client,
                args,
                ctx,
                data,
                frame_count,
                time_now,
                max_flow_mag,
            )
            if result is None:
                continue
            processed, nav_decision = result

            if ctx.param_refs.reset_flag[0]:
                frame_count = handle_reset(client, ctx, frame_count)
                continue

            loop_start = log_and_record_frame(
                client,
                ctx,
                loop_start,
                frame_duration,
                processed,
                nav_decision,
                frame_count,
                time_now,
            )
    except KeyboardInterrupt:
        logger.info("Interrupted.")
    finally:
        logger.info("Navigation loop complete.")


def slam_navigation_loop(args, client, ctx, config=None, pose_source="slam"):
    """SLAM-based navigation loop with basic obstacle avoidance.

    ``config`` or ``args`` may provide custom SLAM stability thresholds which
    are forwarded to :func:`is_slam_stable`.
    """
    # After drone takeoff and camera ready, perform an initial calibration
    # sequence so SLAM has diverse motion before waypoint navigation.

    # logger.info("[SLAMNav] Starting SLAM navigation loop.")

    from slam_bridge.slam_receiver import get_latest_pose_matrix, get_pose_history
    from slam_bridge.frontier_detection import detect_frontiers  

    # --- Incorporate exit_flag from ctx for GUI stop button ---
    exit_flag = None
    navigator = None
    last_action = "none"
    if ctx is not None:
        exit_flag = getattr(ctx, "exit_flag", None)
        navigator = getattr(ctx, "navigator", None)

    # --- Initialize SLAM navigation parameters ---
    start_time = time.time()
    max_duration = getattr(args, "max_duration", 60)
    goal_x = getattr(args, "goal_x", 29)
    goal_y = getattr(args, "goal_y", 0) if hasattr(args, "goal_y") else 0
    goal_z = getattr(args, "goal_z", -2) if hasattr(args, "goal_z") else -2
    threshold = 0.5  # meters

    cov_thres = getattr(args, "slam_covariance_threshold", None)
    inlier_thres = getattr(args, "slam_inlier_threshold", None)
    if cov_thres is None and config is not None:
        try:
            cov_thres = config.getfloat("slam", "covariance_threshold")
        except Exception:
            cov_thres = None
    if inlier_thres is None and config is not None:
        try:
            inlier_thres = config.getint("slam", "inlier_threshold")
        except Exception:
            inlier_thres = None

    # Perform an initial SLAM calibration manoeuvre before navigating.
    if max_duration != 0:
        if ctx is not None and getattr(ctx, "param_refs", None):
            ctx.param_refs.state[0] = "bootstrap"
        run_slam_bootstrap(client, duration=6.0)
        time.sleep(1.0)  # Allow SLAM to settle after calibration
        if ctx is not None and getattr(ctx, "param_refs", None):
            ctx.param_refs.state[0] = "waypoint_nav"

    # Simplified execution path used by tests
    if max_duration == 0 and navigator is not None:
        if pose_source == "airsim" and hasattr(client, "simGetVehiclePose"):
            air_pose = client.simGetVehiclePose("UAV")
            pos = air_pose.position
            yaw = airsim.to_eularian_angles(air_pose.orientation)[2]
            cy, sy = np.cos(yaw), np.sin(yaw)
            pose_mat = np.array([
                [cy, -sy, 0, pos.x_val],
                [sy,  cy, 0, pos.y_val],
                [0,   0,  1, pos.z_val],
            ])
        else:
            pose_mat = get_latest_pose_matrix()
        return navigator.slam_to_goal(pose_mat, (goal_x, goal_y, goal_z))

    # --- Define waypoints for SLAM navigation ---
    waypoints = [
        (20, 0, -2),  # (x, y, z)
        (20, -2.5, -2),
        (22.5, -2.5, -2),
        (23.5, -0.5, -2),
        (45, 0, -2)
        
    ]
    current_waypoint_index = 0
    logger.info("[DEBUG] Entered slam_navigation_loop")
    try:
        while True:
            # --- Check for exit_flag to allow GUI stop button to interrupt navigation ---
            if (exit_flag is not None and exit_flag.is_set()) or os.path.exists(STOP_FLAG_PATH):
                logger.info("[SLAMNav] Stop flag detected. Landing and exiting navigation loop.")
                break

            # --- Get the latest SLAM pose ---
            # If SLAM is unstable, reinitialise it
            # This is a basic stability check that can be improved.
            # Need to ensure that this check is continuously performed during the waypoint navigation. 
            if pose_source == "airsim" and hasattr(client, "simGetVehiclePose"):
                air_pose = client.simGetVehiclePose("UAV")
                pos = air_pose.position
                yaw = airsim.to_eularian_angles(air_pose.orientation)[2]
                cy, sy = np.cos(yaw), np.sin(yaw)
                transformed_pose = np.array(
                    [
                        [cy, -sy, 0, pos.x_val],
                        [sy, cy, 0, pos.y_val],
                        [0, 0, 1, pos.z_val],
                    ]
                )
                x, y, z = pos.x_val, pos.y_val, pos.z_val
            else:
                pose = get_latest_pose_matrix()

                # Check if the pose is None or SLAM is unstable
                if pose is None or not is_slam_stable(cov_thres, inlier_thres):
                    logger.warning(
                        "[SLAMNav] SLAM tracking lost. Attempting reinitialisation."
                    )
                    while True:

                        # Run SLAM bootstrap to reinitialise the SLAM system
                        if ctx is not None and getattr(ctx, "param_refs", None):
                            ctx.param_refs.state[0] = "bootstrap"
                        run_slam_bootstrap(client, duration=4.0)
                        time.sleep(1.0)
                        pose = get_latest_pose_matrix()

                        # Check if SLAM is stable after reinitialisation
                        if pose is not None and is_slam_stable(cov_thres, inlier_thres):
                            logger.info(
                                "[SLAMNav] SLAM reinitialised. Resuming navigation."
                            )

                            # Reset the waypoint index to start from the first waypoint
                            if ctx is not None and getattr(ctx, "param_refs", None):
                                ctx.param_refs.state[0] = "waypoint_nav"
                            break

                        # Check for exit conditions during reinitialisation
                        if (exit_flag is not None and exit_flag.is_set()) or os.path.exists(
                            STOP_FLAG_PATH
                        ):
                            logger.info(
                                "[SLAMNav] Exit signal during reinitialisation."
                            )
                            return last_action
                    continue

                # --- Transform SLAM pose to AirSim coordinates ---
                transformed_pose, (x, y, z) = transform_slam_to_airsim(pose)

                # Original position for debugging
                x_slam, y_slam, z_slam = pose[0][3], pose[1][3], pose[2][3]

                if hasattr(client, "simGetVehiclePose"):
                    airsim_pose = client.simGetVehiclePose("UAV")
                    airsim_pos = airsim_pose.position

                    # Debug coordinate alignment
                    logger.debug("SLAM pose: (%.2f, %.2f, %.2f)", x_slam, y_slam, z_slam)
                    logger.debug(
                        "AirSim pose: (%.2f, %.2f, %.2f)",
                        airsim_pos.x_val,
                        airsim_pos.y_val,
                        airsim_pos.z_val,
                    )
                    logger.debug("Transformed: (%.2f, %.2f, %.2f)", x, y, z)

                    # Calculate differences to see alignment
                    diff_x = x - airsim_pos.x_val
                    diff_y = y - airsim_pos.y_val
                    diff_z = z - airsim_pos.z_val
                    logger.debug(
                        "Differences: (%.2f, %.2f, %.2f)", diff_x, diff_y, diff_z
                    )
            
            # Detect exploration frontiers from accumulated SLAM poses
            history = get_pose_history()
            map_pts = np.array(
                [[m[0][3], m[1][3], m[2][3]] for _, m in history], dtype=float
            )
            frontiers = detect_frontiers(map_pts)
            # if frontiers.size:
            #     logger.debug("[SLAMNav] Frontier voxels detected: %d", len(frontiers))
            #     logger.debug("[SLAMNav] Sample frontier: x=%.2f y=%.2f z=%.2f",
            #         frontiers[0][0],
            #         frontiers[0][1],
            #         frontiers[0][2],
            #     )

            # Check for collision/obstacle
            # collision = client.simGetCollisionInfo()
            # if getattr(collision, "has_collided", False):
            #     logger.warning("[SLAMNav] Obstacle detected! Executing avoidance maneuver.")
            #     client.moveByVelocityAsync(-1.0, 0, 0, 1).join()  # Back up
            #     continue

            # --- Get the current waypoint (goal) ---
            goal_x, goal_y, goal_z = waypoints[current_waypoint_index]
            if ctx is not None and getattr(ctx, "param_refs", None):
                ctx.param_refs.state[0] = f"waypoint_{current_waypoint_index + 1}"
            logger.info(f"[SLAMNav] Current waypoint: {current_waypoint_index + 1} at ({goal_x}, {goal_y}, {goal_z})")

            # --- Calculate the distance to the current waypoint ---
            distance_to_goal = np.sqrt((x - goal_x)**2 + (y - goal_y)**2)
            
            logger.info(f"Distance to waypoint: {distance_to_goal:.2f} meters")

            # --- If the drone is within the threshold of the waypoint, move to the next waypoint ---
            if distance_to_goal < threshold:  # Threshold for reaching waypoint
                logger.info(f"Reached waypoint {current_waypoint_index + 1}, moving to next waypoint.")
                current_waypoint_index = (current_waypoint_index + 1) % len(waypoints)  # Move to next waypoint

            # --- Use transformed pose for navigation ---
            # Pass the transformed pose instead of the original
            last_action = navigator.slam_to_goal(transformed_pose, (goal_x, goal_y, goal_z))
            # logger.info("[SLAMNav] Action: %s", last_action)
        
            # --- Check if the stop flag is set ---
            if os.path.exists(STOP_FLAG_PATH):
                logger.info("Stop flag detected. Landing and shutting down.")
                if ctx is not None and getattr(ctx, "param_refs", None):
                    ctx.param_refs.state[0] = "landing"
                break

            # End condition
            if time.time() - start_time > max_duration:
                logger.info("[SLAMNav] Max duration reached, ending navigation.")
                if ctx is not None and getattr(ctx, "param_refs", None):
                    ctx.param_refs.state[0] = "landing"
                break

            # # Depth-based obstacle check before moving toward the goal # Depth check optional
            # ahead, depth = is_obstacle_ahead(client)
            # if ahead:
            #     msg = "[SLAMNav] Depth obstacle detected"
            #     if depth is not None:
            #         msg += f" at {depth:.2f}m"
            #     logger.warning(msg)
            #     if navigator is not None:
            #         navigator.dodge(0, 0, 0, direction="right")
            #     else:
            #         client.hoverAsync().join()
            #     continue

            time.sleep(0.1)  # Allow for periodic updates
    except KeyboardInterrupt:
        logger.info("[SLAMNav] Interrupted by user.")
    finally:
        logger.info("[SLAMNav] SLAM navigation loop finished.")
        generate_pose_comparison_plot()
    return last_action # Return the last action taken

# === SLAM to AirSim Coordinate Transformation ===
# This function transforms a SLAM pose matrix to the AirSim coordinate system.
def transform_slam_to_airsim(slam_pose_matrix):
    """Transform SLAM pose matrix to AirSim coordinate system."""
    
    # Convert to numpy array if it's a list
    if isinstance(slam_pose_matrix, list):
        slam_pose_matrix = np.array(slam_pose_matrix)
    
    # Extract position
    x_slam, y_slam, z_slam = slam_pose_matrix[0, 3], slam_pose_matrix[1, 3], slam_pose_matrix[2, 3]
    
    # Transform position (adjust as needed for your setup)
    x_airsim = z_slam
    y_airsim = x_slam  
    z_airsim = -y_slam
    
    # Extract rotation matrix (3x3 upper-left block)
    R_slam = slam_pose_matrix[:3, :3]
    
    # Define transformation matrix for coordinate system conversion
    T_slam_to_airsim = np.array([
        [0,  0,  1],   # SLAM Z-axis → AirSim X-axis
        [1,  0,  0],   # SLAM X-axis → AirSim Y-axis  
        [0, -1,  0]    # SLAM Y-axis → AirSim -Z-axis
    ])
    
    # Transform rotation
    R_airsim = T_slam_to_airsim @ R_slam @ T_slam_to_airsim.T
    
    # Create transformed pose matrix
    transformed_pose = np.eye(4)
    transformed_pose[:3, :3] = R_airsim
    transformed_pose[0, 3] = x_airsim
    transformed_pose[1, 3] = y_airsim
    transformed_pose[2, 3] = z_airsim
    
    return transformed_pose, (x_airsim, y_airsim, z_airsim)

# === Thread Management ===
# This section handles the shutdown of worker threads and ensures they exit cleanly.
def shutdown_threads(ctx):
    """Stop worker threads and wait for them to exit."""
    if ctx is None:
        return

    exit_flag = getattr(ctx, "exit_flag", None)
    if exit_flag is not None:
        exit_flag.set()

    frame_queue = getattr(ctx, "frame_queue", None)
    if frame_queue is not None:
        try:
            frame_queue.put(None, block=False)
        except Exception:
            pass

    for attr in ("video_thread", "perception_thread"):
        thread = getattr(ctx, attr, None)
        if thread is not None:
            try:
                thread.join()
            except Exception:
                pass

# === Logging ===
# This section handles the logging of flight data, video frames, and cleanup of log files.
def close_logging(ctx):
    """Flush buffered log/video data and close file handles."""
    if ctx is None:
        return

    out = getattr(ctx, "out", None)
    if out is not None:
        try:
            out.release()
        except Exception:
            pass

    log_file = getattr(ctx, "log_file", None)
    log_buffer = getattr(ctx, "log_buffer", None)
    if log_file is not None:
        try:
            if log_buffer:
                log_file.writelines(log_buffer)
                log_buffer.clear()
            log_file.close()
        except Exception as exc:
            logger.warning("⚠️ Log file already closed or error writing: %s", exc)


def shutdown_airsim(client):
    """Land the drone and disable API control."""
    if client is None:
        return
    try:
        fut = client.landAsync()
        fut.join()
        client.armDisarm(False)
        client.enableApiControl(False)
    except Exception as exc:
        logger.error("Landing error: %s", exc)

# === Finalization and Cleanup ===
# This section handles the finalization of flight analysis, cleanup of generated files, 
# and removal of stop flags.
def finalise_files(ctx):
    """Generate flight analysis and clean up generated files."""
    if ctx is None:
        logger.warning("No context provided for finalization.")
        return

    timestamp = getattr(ctx, "timestamp", None)
    if not timestamp:
        logger.warning("No timestamp found - skipping file finalization")
        return

    logger.info(f"🎯 Starting post-flight analysis for timestamp: {timestamp}")
    
    try:
        log_csv = f"flow_logs/full_log_{timestamp}.csv"
        
        # Check if log file exists and has data
        if not os.path.exists(log_csv):
            logger.error(f"Log file not found: {log_csv}")
            return
            
        # Check log file size
        file_size = os.path.getsize(log_csv)
        if file_size < 100:  # Less than 100 bytes probably means empty/corrupt
            logger.warning(f"Log file appears empty or corrupt: {log_csv} ({file_size} bytes)")
            return
        
        logger.info(f"Processing log file: {log_csv} ({file_size} bytes)")
        
        # Ensure analysis directory exists
        os.makedirs("analysis", exist_ok=True)
        
        # Generate flight visualization
        try:
            html_output = f"analysis/flight_view_{timestamp}.html"
            logger.info(f"Generating flight visualization: {html_output}")
            
            # Method 1: Direct function call (avoids subprocess issues)
            try:
                # Add analysis directory to path temporarily
                import sys
                analysis_path = os.path.abspath('analysis')
                if analysis_path not in sys.path:
                    sys.path.insert(0, analysis_path)
                
                # Import and call visualization function
                from visualise_flight import main as visualize_main
                visualize_main([html_output, "--log", log_csv])
                logger.info(f"✅ Flight visualization saved: {html_output}")
                
                # Remove from path
                if analysis_path in sys.path:
                    sys.path.remove(analysis_path)
                    
            except Exception as direct_error:
                logger.warning(f"Direct call failed: {direct_error}")
                
                # Method 2: Subprocess fallback
                try:
                    visualization_script = os.path.abspath("analysis/visualise_flight.py")
                    result = subprocess.run(
                        [
                            sys.executable,
                            visualization_script,
                            html_output,
                            "--log", 
                            log_csv
                        ],
                        check=True,
                        capture_output=True,
                        text=True,
                        cwd=os.getcwd(),
                        timeout=60  # 60 second timeout
                    )
                    
                    if result.stdout.strip():
                        logger.info(f"Visualization output: {result.stdout.strip()}")
                    
                    logger.info(f"✅ Flight visualization saved via subprocess: {html_output}")
                    
                except subprocess.CalledProcessError as proc_error:
                    logger.error(f"Subprocess visualization failed: {proc_error.stderr}")
                except subprocess.TimeoutExpired:
                    logger.error("Visualization generation timed out")
                except Exception as subprocess_error:
                    logger.error(f"Subprocess method failed: {subprocess_error}")

        except Exception as viz_error:
            logger.error(f"Flight visualization generation failed: {viz_error}")

        # Generate performance plots
        try:
            perf_output = f"analysis/performance_{timestamp}.html"
            logger.info(f"Generating performance plots: {perf_output}")
            
            # Direct function call for performance plots
            try:
                analysis_path = os.path.abspath('analysis')
                if analysis_path not in sys.path:
                    sys.path.insert(0, analysis_path)
                    
                from performance_plots import main as perf_main
                perf_main([log_csv, "--output", perf_output])
                logger.info(f"✅ Performance plots saved: {perf_output}")
                
                if analysis_path in sys.path:
                    sys.path.remove(analysis_path)
                    
            except Exception as perf_error:
                logger.warning(f"Performance plots generation failed: {perf_error}")

        except Exception as perf_outer_error:
            logger.error(f"Performance analysis failed: {perf_outer_error}")

        # Generate flight report (if analyse module exists)
        try:
            report_path = f"analysis/flight_report_{timestamp}.html"
            analyse_script = os.path.abspath("analysis/analyse.py")
            
            if os.path.exists(analyse_script):
                logger.info(f"Generating flight report: {report_path}")
                
                result = subprocess.run(
                    [
                        sys.executable,
                        analyse_script,
                        log_csv,
                        "-o",
                        report_path,
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    cwd=os.getcwd(),
                    timeout=30
                )
                logger.info(f"✅ Flight report saved: {report_path}")
            else:
                logger.info("analyse.py not found - skipping flight report")
                
        except subprocess.CalledProcessError as report_error:
            logger.warning(f"Flight report generation failed: {report_error.stderr}")
        except Exception as report_outer_error:
            logger.info(f"Flight report generation skipped: {report_outer_error}")

        # Generate SLAM comparison plot (if available)
        try:
            from uav import slam_utils
            slam_utils.generate_pose_comparison_plot()
            logger.info("✅ SLAM pose comparison plot generated")
        except Exception as slam_error:
            logger.info(f"SLAM plot generation skipped: {slam_error}")

        # Summary of generated files
        generated_files = []
        for file_pattern in [
            f"analysis/flight_view_{timestamp}.html",
            f"analysis/performance_{timestamp}.html", 
            f"analysis/flight_report_{timestamp}.html"
        ]:
            if os.path.exists(file_pattern):
                generated_files.append(file_pattern)
        
        if generated_files:
            logger.info(f"🎯 Analysis complete! Generated files:")
            for file_path in generated_files:
                file_size = os.path.getsize(file_path)
                logger.info(f"  📊 {file_path} ({file_size} bytes)")
        else:
            logger.warning("No analysis files were successfully generated")

    except Exception as outer_error:
        logger.error(f"Unexpected error in finalise_files: {outer_error}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")

    # Clean up files
    try:
        from uav.utils import retain_recent_views
        retain_recent_views("analysis", 5)
        logger.info("✅ Old analysis files cleaned up")
    except Exception as cleanup_error:
        logger.error(f"Error retaining recent views: {cleanup_error}")

    # Remove stop flag
    try:
        from uav.paths import STOP_FLAG_PATH
        if os.path.exists(STOP_FLAG_PATH):
            os.remove(STOP_FLAG_PATH)
            logger.info("✅ Stop flag file removed")
    except Exception as flag_error:
        logger.error(f"Error removing stop flag file: {flag_error}")

    logger.info("🏁 Post-flight analysis finalization complete")

def cleanup(client, sim_process, ctx):
    """Clean up resources and land the drone."""
    logger.info("Landing...")

    # Step 1: Shutdown threads and close logging
    shutdown_threads(ctx)
    close_logging(ctx)

    # Step 2: Generate analysis files after log file is closed
    logger.info("Finalizing flight analysis and cleanup.")
    finalise_files(ctx)

    # Step 3: Shutdown AirSim client
    shutdown_airsim(client)
    
    # Step 4: Close simulation
    if sim_process:
        sim_process.terminate()
        try:
            sim_process.wait(timeout=5)
        except Exception:
            sim_process.kill()
        logger.info("UE4 simulation closed.")
    
    logger.info("Cleanup complete. Exiting navigation loop.")

