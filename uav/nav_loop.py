# === Standard Library Imports ===
import os
import cv2
import time
import subprocess
import numpy as np
import logging
from datetime import datetime
from queue import Queue
from threading import Thread

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
from uav.utils import (get_drone_state, retain_recent_logs)
from uav.utils import retain_recent_files, retain_recent_views
from uav import config
from uav.logging_helpers import log_frame_data, write_video_frame, write_frame_output, handle_reset
from uav.perception_loop import perception_loop, start_perception_thread, process_perception_data
from uav.navigation_core import detect_obstacle, determine_side_safety, handle_obstacle, navigation_step, apply_navigation_decision
from uav.navigation_slam_boot import run_slam_bootstrap
from uav.slam_utils import is_slam_stable

logger = logging.getLogger("nav_loop")
logger.warning("[TEST] __name__ = %s | handlers = %s", __name__, logger.handlers)

frame_counter = 0
MIN_INLIERS_THRESHOLD = 100  # Minimum inliers to consider SLAM stable

# Grace period duration (seconds) after dodge/brake actions
NAV_GRACE_PERIOD_SEC = 0.5

# Flag file to stop the drone
STOP_FLAG_PATH = "flags/stop.flag"

# === Perception Processing ===

def setup_environment(args, client):
    """Initialize the navigation environment and return a context dict."""
    from uav.interface import exit_flag
    from uav.utils import retain_recent_logs
    param_refs = {
        'L': [0.0], 'C': [0.0], 'R': [0.0],
        'prev_L': [0.0], 'prev_C': [0.0], 'prev_R': [0.0],
        'delta_L': [0.0], 'delta_C': [0.0], 'delta_R': [0.0],
        'state': [''], 'reset_flag': [False]
    }
    logger.info("Available vehicles: %s", client.listVehicles())
    client.enableApiControl(True); client.armDisarm(True)
    client.takeoffAsync().join(); client.moveToPositionAsync(0, 0, -2, 2).join()
    feature_params = dict(maxCorners=150, qualityLevel=0.05, minDistance=5, blockSize=5)
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    tracker, flow_history, navigator = OpticalFlowTracker(lk_params, feature_params), FlowHistory(), Navigator(client)
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
        "time,features,simgetimage_s,decode_s,processing_s,loop_s\n"
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
    ctx = {
        'exit_flag': exit_flag, 'param_refs': param_refs, 'tracker': tracker, 'flow_history': flow_history,
        'navigator': navigator, 'state_history': state_history, 'pos_history': pos_history,
        'frame_queue': frame_queue, 'video_thread': video_thread, 'out': out, 'log_file': log_file,
        'log_buffer': [], 'timestamp': timestamp, 'start_time': start_time, 'fps_list': [], 'fourcc': fourcc,
    }
    return ctx

def navigation_loop(args, client, ctx):
    """Main navigation loop processing perception results."""
    exit_flag, flow_history, navigator = ctx['exit_flag'], ctx['flow_history'], ctx['navigator']
    param_refs, state_history, pos_history = ctx['param_refs'], ctx['state_history'], ctx['pos_history']
    perception_queue, frame_queue, video_thread = ctx['perception_queue'], ctx['frame_queue'], ctx['video_thread']

    out, log_file, log_buffer = ctx['out'], ctx['log_file'], ctx['log_buffer']
    start_time, timestamp, fps_list, fourcc = ctx['start_time'], ctx['timestamp'], ctx['fps_list'], ctx['fourcc']

    MAX_FLOW_MAG, MAX_VECTOR_COMPONENT = config.MAX_FLOW_MAG, config.MAX_VECTOR_COMPONENT
    GRACE_PERIOD_SEC, MAX_SIM_DURATION = config.GRACE_PERIOD_SEC, args.max_duration
    GOAL_X, GOAL_Y = args.goal_x, config.GOAL_Y
    frame_count, target_fps, frame_duration = 0, config.TARGET_FPS, 1.0 / config.TARGET_FPS
    grace_logged, startup_grace_over = False, False
    logger.info("[NavLoop] Starting navigation loop with args: %s", args)
    try:
        loop_start = time.time()
        while not exit_flag.is_set():
            if os.path.exists(STOP_FLAG_PATH):
                logger.info("Stop flag detected. Landing and shutting down.")
                exit_flag.set()
                break
            frame_count += 1
            time_now = time.time()

            # Print real drone position from AirSim
            pose = client.simGetVehiclePose("UAV")
            pos = pose.position
            logger.debug("[UAV Pose] x=%.2f, y=%.2f, z=%.2f", pos.x_val, pos.y_val, pos.z_val)

            if not startup_grace_over:
                if time_now - start_time < GRACE_PERIOD_SEC:
                    param_refs['state'][0] = "startup_grace"
                    if not grace_logged:
                        logger.info("Startup grace period active — waiting to start perception and nav")
                        grace_logged = True
                    time.sleep(0.05)
                    continue
                else:
                    startup_grace_over = True
                    logger.info("Startup grace period complete — beginning full nav logic")
            try: data = perception_queue.get(timeout=1.0)
            except Exception: continue
            prev_state = param_refs['state'][0]
            # if navigator.settling and time_now >= navigator.settle_end_time:
            #     logger.info("Settle period over — resuming evaluation")
            #     navigator.settling = False
            if time_now - start_time >= MAX_SIM_DURATION:
                logger.info("Time limit reached — landing and stopping."); break
            pos_goal, _, _ = get_drone_state(client)
            threshold = 0.5  # Define a threshold for goal position proximity
            if abs(pos_goal.x_val - GOAL_X) < threshold and abs(pos_goal.y_val - GOAL_Y) < threshold:
                logger.info("Goal reached — landing.")
                break
            processed = process_perception_data(
                client, args, data, frame_count, frame_queue, flow_history, navigator, param_refs, time_now, MAX_FLOW_MAG,
            )
            if processed is None: continue
            (   vis_img, good_old, flow_vectors, flow_std, simgetimage_s, decode_s, processing_s,
                smooth_L, smooth_C, smooth_R, delta_L, delta_C, delta_R,
                probe_mag, probe_count, left_count, center_count, right_count, in_grace,
            ) = processed
            (
                state_str, obstacle_detected, side_safe, brake_thres, dodge_thres, probe_req,
            ) = apply_navigation_decision(
                client, navigator, flow_history, good_old, flow_vectors, flow_std,
                smooth_L, smooth_C, smooth_R, delta_L, delta_C, delta_R, probe_mag, probe_count,
                left_count, center_count, right_count, frame_queue, vis_img,
                time_now, frame_count, prev_state, state_history, pos_history, param_refs,
            )
            if param_refs['reset_flag'][0]:
                frame_count = handle_reset(client, ctx, frame_count)
                flow_history, navigator, log_file, video_thread, out = ctx['flow_history'], ctx['navigator'], ctx['log_file'], ctx['video_thread'], ctx['out']
                continue
            loop_start = write_frame_output(
                client, vis_img, frame_queue, loop_start, frame_duration, fps_list, start_time,
                smooth_L, smooth_C, smooth_R, delta_L, delta_C, delta_R,
                left_count, center_count, right_count,
                good_old, flow_vectors, in_grace, frame_count, time_now, param_refs,
                log_file, log_buffer, state_str, obstacle_detected, side_safe,
                brake_thres, dodge_thres, probe_req, simgetimage_s, decode_s, processing_s, flow_std,
            )
    except KeyboardInterrupt:
        logger.info("Interrupted.")

def is_obstacle_ahead(client, depth_threshold=2.0, vehicle_name="UAV"):
    from airsim import ImageRequest, ImageType
    logger.info("[Obstacle Check] Checking for obstacles ahead.")
    try:
        responses = client.simGetImages([
            ImageRequest("oakd_camera", ImageType.DepthPlanar, True)
        ], vehicle_name=vehicle_name)
        if not responses or responses[0].height == 0:
            logger.error("[Obstacle Check] No depth image received or image height is zero.")
            return False, None
        depth_image = airsim.get_pfm_array(responses[0])
        h, w = depth_image.shape
        cx, cy = w // 2, h // 2
        roi = depth_image[cy-20:cy+20, cx-20:cx+20]
        mean_depth = np.nanmean(roi)
        return mean_depth < depth_threshold, mean_depth
    except Exception as e:
        logger.error("[Obstacle Check] Depth read failed: %s", e)
        return False, None

import subprocess

def generate_pose_comparison_plot():
    logger.info("[Plotting] Generating pose comparison plot.")
    try:
        logger.info("[Plotting] Running pose_comparison_plotter.py script.")
        result = subprocess.run(
            ["python", "slam_bridge/pose_comparison_plotter.py"],
            check=True,
            capture_output=True,
            text=True
        )
        print("[Plotting] Pose comparison plot generated.")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        logger.error("[Plotting] Failed to generate pose comparison plot.")
        print("[Plotting] Failed to generate plot:")
        print(e.stderr)

def slam_navigation_loop(args, client, ctx):
    """
    Main navigation loop for SLAM-based navigation with basic obstacle avoidance.
    """
    # After drone takeoff and camera ready
    run_slam_bootstrap(client, duration=6.0)  # you can tune this
    time.sleep(1.0)  # Let SLAM settle after bootstrap

    # logger.info("[SLAMNav] Starting SLAM navigation loop.")

    from slam_bridge.slam_receiver import get_latest_pose, get_pose_history
    from slam_bridge.frontier_detection import detect_frontiers  

    # --- Incorporate exit_flag from ctx for GUI stop button ---
    exit_flag = None
    navigator = None
    last_action = "none"
    if ctx is not None:
        exit_flag = ctx.get("exit_flag", None)
        navigator = ctx.get("navigator", None)

    # --- Initialize SLAM navigation parameters ---
    start_time = time.time()
    max_duration = getattr(args, "max_duration", 60)
    goal_x = getattr(args, "goal_x", 29)
    goal_y = getattr(args, "goal_y", 0) if hasattr(args, "goal_y") else 0
    goal_z = getattr(args, "goal_z", -2) if hasattr(args, "goal_z") else -2
    threshold = 0.5  # meters

    # --- Define waypoints for SLAM navigation ---
    waypoints = [
        (5, 0, -2),  # (x, y, z)
        (10, 0, -2),
        (29, 0, -2)
    ]
    current_waypoint_index = 0
    logger.info("[DEBUG] Entered slam_navigation_loop")
    try:
        while True:
            # --- Check for exit_flag to allow GUI stop button to interrupt navigation ---
            if (exit_flag is not None and exit_flag.is_set()) or os.path.exists(STOP_FLAG_PATH):
                logger.info("[SLAMNav] Stop flag detected. Landing and exiting navigation loop.")
                client.landAsync().join()
                break

            # --- Get the latest SLAM pose ---
            pose = get_latest_pose()
            if pose is None: # No pose received, hover to recover
                logger.warning("[SLAMNav] No pose received – hovering to recover.")
                client.hoverAsync().join()
                time.sleep(1.0)  # allow SLAM to reinitialize
                continue

            # --- Check if SLAM is stable ---
            if not is_slam_stable():  # Check SLAM stability
                logger.warning("[SLAMNav] SLAM is unstable. Pausing navigation.")
                client.hoverAsync().join()  # Pause the drone (hover)
                break  # Exit the loop or you can reset/restart SLAM if necessary
            # else:
            #     logger.info("[SLAMNav] SLAM is stable. Continuing navigation.")

            x, y, z = pose # Unpack the pose
            # logger.info(f"[SLAMNav] Received pose: x={x:.2f}, y={y:.2f}, z={z:.2f}")

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
            collision = client.simGetCollisionInfo()
            if getattr(collision, "has_collided", False):
                logger.warning("[SLAMNav] Obstacle detected! Executing avoidance maneuver.")
                client.moveByVelocityAsync(-1.0, 0, 0, 1).join()  # Back up
                continue

            # --- Get the current waypoint (goal) ---
            goal_x, goal_y, goal_z = waypoints[current_waypoint_index]
            logger.info(f"[SLAMNav] Current waypoint: {current_waypoint_index + 1} at ({goal_x}, {goal_y}, {goal_z})")

            # --- Calculate the distance to the current waypoint ---
            distance_to_goal = np.sqrt((x - goal_x)**2 + (y - goal_y)**2)
            logger.info(f"Distance to waypoint: {distance_to_goal:.2f} meters")

            # --- If the drone is within the threshold of the waypoint, move to the next waypoint ---
            if distance_to_goal < threshold:  # Threshold for reaching waypoint
                logger.info(f"Reached waypoint {current_waypoint_index + 1}, moving to next waypoint.")
                current_waypoint_index = (current_waypoint_index + 1) % len(waypoints)  # Move to next waypoint

            # --- Use SLAM for deliberative navigation ---
            if navigator is None:
                navigator = Navigator(client)
            last_action = navigator.slam_to_goal(pose, (goal_x, goal_y, goal_z))
            logger.info("[SLAMNav] Action: %s", last_action)
        
            # --- Check if the stop flag is set ---
            if os.path.exists(STOP_FLAG_PATH):
                logger.info("Stop flag detected. Landing and shutting down.")
                client.landAsync().join()
                break

            # End condition
            if time.time() - start_time > max_duration:
                logger.info("[SLAMNav] Max duration reached, ending navigation.")
                client.landAsync().join()
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
        client.landAsync().join()
    finally:
        logger.info("[SLAMNav] SLAM navigation loop finished.")
        # generate_pose_comparison_plot()
    return last_action

def cleanup(client, sim_process, ctx):
    """Clean up resources and land the drone."""
    logger.info("Landing...")

    if ctx is not None:
        exit_flag = ctx.get('exit_flag')
        frame_queue = ctx.get('frame_queue')
        video_thread = ctx.get('video_thread')
        perception_thread = ctx.get('perception_thread')
        out = ctx.get('out')
        log_file = ctx.get('log_file')
        log_buffer = ctx.get('log_buffer')
        timestamp = ctx.get('timestamp')
        fps_list = ctx.get('fps_list')

        if exit_flag:
            exit_flag.set()

        if frame_queue:
            try: frame_queue.put(None)
            except Exception: pass

        if video_thread:
            try: video_thread.join()
            except Exception: pass

        if perception_thread:
            try: perception_thread.join()
            except Exception: pass

        if out:
            try: out.release()
            except Exception: pass

        if log_file:
            try:
                if log_buffer:
                    log_file.writelines(log_buffer)
                    log_buffer.clear()
                log_file.close()
            except Exception as e:
                logger.warning("⚠️ Log file already closed or error writing: %s", e)

        try:
            html_output = f"analysis/flight_view_{timestamp}.html"
            subprocess.run(["python3", "-m", "analysis.visualize_flight", html_output])
            logger.info("Flight path analysis saved to %s", html_output)
        except Exception as e:
            logger.error("Error generating flight path analysis: %s", e)

        try:
            retain_recent_views("analysis", 5)
        except Exception as e:
            logger.error("Error retaining recent views: %s", e)

    try:
        if client:
            client.landAsync().join()
            client.armDisarm(False)
            client.enableApiControl(False)
    except Exception as e:
        logger.error("Landing error: %s", e)

    # Wait after landing for graceful shutdown if early exit
    if client and ctx is not None and ctx.get("exit_flag", None) and ctx["exit_flag"].is_set():
        logger.info("🕒 Pausing briefly after landing for graceful shutdown...")
        for _ in range(30):  # Wait up to 3 seconds
            try:
                pos = client.getMultirotorState().kinematics_estimated.position
                if pos.z_val > -0.2:
                    logger.info("Drone appears to be landed.")
                    break
                time.sleep(0.1)
            except Exception:
                break

    try:
        # Wait until the drone is landed or until a timeout (e.g., 15 seconds)
        if client and ctx is not None and ctx.get("exit_flag", None) and ctx["exit_flag"].is_set():
            logger.info("🕒 Waiting for drone to land for graceful shutdown...")
            max_wait = 7  # seconds
            start_wait = time.time()
            while True:
                if time.time() - start_wait > max_wait:
                    logger.warning("Timeout waiting for drone to land.")
                    break
                time.sleep(0.1)
    except Exception as e:
        logger.error("Error during graceful shutdown wait: %s", e)

    if sim_process:
        sim_process.terminate()
        try:
            sim_process.wait(timeout=5)
        except Exception:
            sim_process.kill()
        logger.info("UE4 simulation closed.")

    # --- New cleanup code ---
    try:
        # Ensure the stop flag file is removed on cleanup
        if os.path.exists(STOP_FLAG_PATH):
            os.remove(STOP_FLAG_PATH)
            logger.info("Removed stop flag file.")
    except Exception as e:
        logger.error("Error removing stop flag file: %s", e)


