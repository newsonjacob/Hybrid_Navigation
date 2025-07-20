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
from uav.utils import (get_drone_state, retain_recent_logs, init_client)
from uav.utils import retain_recent_files, retain_recent_views
from uav import config
from uav.logging_helpers import log_frame_data, write_video_frame, write_frame_output, handle_reset
from uav.context import ParamRefs, NavContext
from uav.perception_loop import perception_loop, start_perception_thread, process_perception_data
from uav.navigation_core import detect_obstacle, determine_side_safety, handle_obstacle, navigation_step, apply_navigation_decision
from uav.navigation_slam_boot import run_slam_bootstrap
from uav.paths import STOP_FLAG_PATH
from uav.slam_utils import (
    is_slam_stable,
    is_obstacle_ahead,
    generate_pose_comparison_plot,
)

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
        in_grace,
    ) = processed
    prev_state = ctx.param_refs.state[0]
    nav_decision = apply_navigation_decision(
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
        probe_mag,
        probe_count,
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
    )
    return processed, nav_decision


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
        in_grace,
    ) = processed
    (
        state_str,
        obstacle_detected,
        side_safe,
        brake_thres,
        dodge_thres,
        probe_req,
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
    )

def navigation_loop(args, client, ctx):
    """Main navigation loop processing perception results."""
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
            if os.path.exists(STOP_FLAG_PATH):
                logger.info("Stop flag detected. Landing and shutting down.")
                exit_flag.set()
                break

            frame_count += 1
            time_now = time.time()

            if not check_startup_grace(ctx, time_now):
                continue

            data = get_perception_data(ctx)
            if data is None:
                continue

            if time_now - ctx.start_time >= max_duration:
                logger.info("Time limit reached — landing and stopping.")
                break

            pos_goal, _, _ = get_drone_state(client)
            if abs(pos_goal.x_val - goal_x) < 0.5 and abs(pos_goal.y_val - goal_y) < 0.5:
                logger.info("Goal reached — landing.")
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

def slam_navigation_loop(args, client, ctx):
    """
    Main navigation loop for SLAM-based navigation with basic obstacle avoidance.
    """
    # After drone takeoff and camera ready, perform an initial calibration
    # sequence so SLAM has diverse motion before waypoint navigation.

    # logger.info("[SLAMNav] Starting SLAM navigation loop.")

    from slam_bridge.slam_receiver import get_latest_pose, get_pose_history
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

    # Perform an initial SLAM calibration manoeuvre before navigating.
    if max_duration != 0:
        run_slam_bootstrap(client, duration=6.0)
        time.sleep(1.0)  # Allow SLAM to settle after calibration

    # Simplified execution path used by tests
    if max_duration == 0 and navigator is not None:
        pose = get_latest_pose()
        return navigator.slam_to_goal(pose, (goal_x, goal_y, goal_z))

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
                client.landAsync().join()
                break

            # --- Get the latest SLAM pose ---
            # If SLAM is unstable, reinitialise it
            # This is a basic stability check that can be improved.
            # Need to ensure that this check is continuously performed during the waypoint navigation. 
            pose = get_latest_pose()
            if pose is None or not is_slam_stable():
                logger.warning(
                    "[SLAMNav] SLAM tracking lost. Attempting reinitialisation."
                )
                while True:
                    run_slam_bootstrap(client, duration=4.0)
                    time.sleep(1.0)
                    pose = get_latest_pose()
                    if pose is not None and is_slam_stable():
                        logger.info(
                            "[SLAMNav] SLAM reinitialised. Resuming navigation."
                        )
                        break
                    if (exit_flag is not None and exit_flag.is_set()) or os.path.exists(
                        STOP_FLAG_PATH
                    ):
                        logger.info(
                            "[SLAMNav] Exit signal during reinitialisation."
                        )
                        client.landAsync().join()
                        return last_action
                continue

            x_slam, y_slam, z_slam = pose
            x, y, z = x_slam, y_slam, -z_slam  # Adjust z for AirSim
            # logger.info(f"[SLAMNav] Received pose: x={x:.2f}, y={y:.2f}, z={z:.2f}")
            # logger.debug("[SLAM Pose] x=%.2f, y=%.2f, z=%.2f", x, y, z)

            # Print real drone position from AirSim
            # Air_pose = client.simGetVehiclePose("UAV")
            # pos = Air_pose.position
            # airsim_x, airsim_y, airsim_z = pos.x_val, pos.y_val, pos.z_val
            if hasattr(client, "simGetVehiclePose"):
                Air_pose = client.simGetVehiclePose("UAV")
                pos = getattr(Air_pose, "position", None)
                if pos is not None:
                    airsim_x, airsim_y, airsim_z = pos.x_val, pos.y_val, pos.z_val
                else:
                    airsim_x, airsim_y, airsim_z = x, y, z
            else:
                airsim_x, airsim_y, airsim_z = x_slam, y_slam, z_slam
            # logger.debug("[UAV Pose] x=%.2f, y=%.2f, z=%.2f", pos.x_val, pos.y_val, pos.z_val)

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
            # last_action = client.moveToPositionAsync(2,0,-2, 2)
            last_action = navigator.slam_to_goal(pose, (goal_x, goal_y, goal_z))
            # logger.info("[SLAMNav] Action: %s", last_action)
        
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


def finalize_files(ctx):
    """Generate flight analysis and clean up generated files."""
    if ctx is None:
        return

    timestamp = getattr(ctx, "timestamp", None)
    if timestamp:
        try:
            html_output = f"analysis/flight_view_{timestamp}.html"
            subprocess.run(["python3", "-m", "analysis.visualise_flight", html_output])
            logger.info("Flight path analysis saved to %s", html_output)
        except Exception as exc:
            logger.error("Error generating flight path analysis: %s", exc)

    try:
        retain_recent_views("analysis", 5)
    except Exception as exc:
        logger.error("Error retaining recent views: %s", exc)

    try:
        if os.path.exists(STOP_FLAG_PATH):
            os.remove(STOP_FLAG_PATH)
            logger.info("Removed stop flag file.")
    except Exception as exc:
        logger.error("Error removing stop flag file: %s", exc)


def cleanup(client, sim_process, ctx):
    """Clean up resources and land the drone."""
    logger.info("Landing...")

    shutdown_threads(ctx)
    close_logging(ctx)
    shutdown_airsim(client)
    finalize_files(ctx)

    if sim_process:
        sim_process.terminate()
        try:
            sim_process.wait(timeout=5)
        except Exception:
            sim_process.kill()
        logger.info("UE4 simulation closed.")

