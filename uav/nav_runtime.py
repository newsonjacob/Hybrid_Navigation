"""Runtime utilities used by the navigation loops."""

import logging
import os
import time
from datetime import datetime
from pathlib import Path
from queue import Queue

import cv2
import numpy as np
import airsim
try:
    from airsim import LandedState
except Exception:  # pragma: no cover - stubbed environments
    LandedState = None

from uav.video_utils import start_video_writer_thread
from uav.perception import OpticalFlowTracker, FlowHistory
from uav.navigation import Navigator
from uav.utils import get_drone_state, retain_recent_logs, init_client
from uav.utils import retain_recent_files
from uav import config
from uav.logging_helpers import write_frame_output
from uav.context import ParamRefs, NavContext
from uav.perception_loop import (
    perception_loop,
    start_perception_thread,
    process_perception_data,
)
from uav.navigation_core import (
    navigation_step,
    NavigationInput,
)
from uav.navigation_slam_boot import run_slam_bootstrap
from uav.paths import STOP_FLAG_PATH
from uav.slam_utils import is_slam_stable

logger = logging.getLogger("nav_loop")

__all__ = [
    "setup_environment",
    "check_startup_grace",
    "_initiate_landing",
    "has_landed",
    "check_exit_conditions",
    "get_perception_data",
    "update_navigation_state",
    "log_and_record_frame",
    "transform_slam_to_airsim",
    "check_slam_stop",
    "ensure_stable_slam_pose",
    "handle_waypoint_progress",
    "shutdown_threads",
    "close_logging",
    "shutdown_airsim",
    "ThreadManager",
    "LoggingContext",
    "SimulationProcess",
    "cleanup",
]


# === Perception Processing ===

def setup_environment(args, client):
    """Initialize the navigation environment and return a context dict."""
    from uav.interface import exit_flag

    param_refs = ParamRefs()
    logger.info("Available vehicles: %s", client.listVehicles())
    init_client(client)
    client.takeoffAsync().join()
    client.moveToPositionAsync(0, 0, -2, 2).join()

    feature_params = dict(
        maxCorners=150,
        qualityLevel=0.01,
        minDistance=8,
        blockSize=7,
        useHarrisDetector=True,
        k=0.04,
    )
    lk_params = dict(
        winSize=(21, 21),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        flags=cv2.OPTFLOW_LK_GET_MIN_EIGENVALS,
        minEigThreshold=1e-3,
    )

    tracker = OpticalFlowTracker(lk_params, feature_params, config.MIN_FLOW_MAG)
    flow_history, navigator = FlowHistory(size=config.FLOW_HISTORY_SIZE), Navigator(client)

    from collections import deque

    state_history, pos_history = deque(maxlen=3), deque(maxlen=3)
    start_time = time.time()
    GOAL_X, MAX_SIM_DURATION = args.goal_x, args.max_duration
    logger.info("Config:\n  Goal X: %sm\n  Max Duration: %ss", GOAL_X, MAX_SIM_DURATION)
    output_base = Path(getattr(args, "output_dir", "."))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    flow_dir = output_base / "flow_logs"
    flow_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(flow_dir / f"full_log_{timestamp}.csv", "w")
    log_file.write(
        "frame,flow_left,flow_center,flow_right,"
        "delta_left,delta_center,delta_right,flow_std,"
        "left_count,center_count,right_count,"
        "brake_thres,dodge_thres,fps,"
        "state,collided,obstacle,side_safe,"
        "pos_x,pos_y,pos_z,slam_x,slam_y,slam_z,yaw,speed,"
        "time,features,simgetimage_s,decode_s,processing_s,loop_s,cpu_percent,memory_rss,"
        "sudden_rise,center_blocked,combination_flow,minimum_flow\n"
    )
    retain_recent_logs(str(flow_dir))
    retain_recent_logs(str(output_base / "logs"))
    retain_recent_files(str(output_base / "analysis"), "slam_traj_*.html", keep=5)
    retain_recent_files(str(output_base / "analysis"), "slam_output_*.mp4", keep=5)
    try:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    except AttributeError:  # pragma: no cover
        fourcc = cv2.FOURCC(*"MJPG")
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
        output_dir=str(output_base),
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
            logger.info("Startup grace period active — waiting to start perception and nav")
            ctx.grace_logged = True
        time.sleep(0.05)
        return False
    ctx.startup_grace_over = True
    logger.info("Startup grace period complete — beginning full nav logic")
    return True


def _initiate_landing(client, ctx):
    """Start asynchronous landing and update state."""
    if getattr(ctx, "landing_future", None) is not None:
        logger.debug("Landing already in progress, skipping new landing command.")
        return

    logger.info("Initiating landing sequence.")

    try:
        ctx.navigator.brake()
    except Exception as exc:
        logger.warning("Brake before landing failed: %s", exc)

    ctx.landing_future = client.landAsync()
    if getattr(ctx, "param_refs", None):
        ctx.param_refs.state[0] = "landing"


def has_landed(client):
    """Return True when the vehicle reports a landed state."""
    try:
        state = client.getMultirotorState()
        landed = getattr(state, "landed_state", None)
        if landed is not None:
            if LandedState is not None:
                return landed == LandedState.Landed
            return landed in (0, 1) and landed == 1
    except Exception:
        pass
    return False


def check_exit_conditions(client, ctx, time_now, max_duration, goal_x, goal_y):
    """Check for termination triggers and initiate landing when required."""
    if os.path.exists(STOP_FLAG_PATH):
        logger.info("Stop flag detected. Landing and shutting down.")
        try:
            os.remove(STOP_FLAG_PATH)
            logger.debug("Stop flag file removed")
        except Exception as remove_error:
            logger.error(f"Error removing stop flag file: {remove_error}")
        _initiate_landing(client, ctx)
        return False

    if time_now - ctx.start_time >= max_duration:
        logger.info("Time limit reached — landing and stopping.")
        _initiate_landing(client, ctx)
        return False

    pos_goal, _, _ = get_drone_state(client)
    if abs(pos_goal.x_val - goal_x) < config.GOAL_THRESHOLD and abs(pos_goal.y_val - goal_y) < config.GOAL_THRESHOLD:
        logger.info("Goal reached — landing.")
        _initiate_landing(client, ctx)
        return False

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
    perception_data, stats = processed
    vis_img = perception_data.vis_img
    good_old = perception_data.good_old
    flow_vectors = perception_data.flow_vectors
    flow_std = perception_data.flow_std
    simgetimage_s = perception_data.simgetimage_s
    decode_s = perception_data.decode_s
    processing_s = perception_data.processing_s
    smooth_L = stats.smooth_L
    smooth_C = stats.smooth_C
    smooth_R = stats.smooth_R
    delta_L = stats.delta_L
    delta_C = stats.delta_C
    delta_R = stats.delta_R
    probe_mag = stats.probe_mag
    probe_count = stats.probe_count
    left_count = stats.left_count
    center_count = stats.center_count
    right_count = stats.right_count
    top_mag = stats.top_mag
    mid_mag = stats.mid_mag
    bottom_mag = stats.bottom_mag
    top_count = stats.top_count
    mid_count = stats.mid_count
    bottom_count = stats.bottom_count
    in_grace = stats.in_grace
    nav_input = NavigationInput(
        good_old=good_old,
        flow_vectors=flow_vectors,
        flow_std=flow_std,
        smooth_L=smooth_L,
        smooth_C=smooth_C,
        smooth_R=smooth_R,
        delta_L=delta_L,
        delta_C=delta_C,
        delta_R=delta_R,
        left_count=left_count,
        center_count=center_count,
        right_count=right_count,
        frame_queue=ctx.frame_queue,
        vis_img=vis_img,
        time_now=time_now,
        frame_count=frame_count,
        state_history=ctx.state_history,
        pos_history=ctx.pos_history,
        param_refs=ctx.param_refs,
        probe_mag=probe_mag,
        probe_count=probe_count,
    )
    nav_decision = navigation_step(
        client,
        ctx.navigator,
        ctx.flow_history,
        nav_input,
    )
    return processed, nav_decision


# === Logging and Frame Recording ===

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
    perception_data, stats = processed
    vis_img = perception_data.vis_img
    good_old = perception_data.good_old
    flow_vectors = perception_data.flow_vectors
    flow_std = perception_data.flow_std
    simgetimage_s = perception_data.simgetimage_s
    decode_s = perception_data.decode_s
    processing_s = perception_data.processing_s
    smooth_L = stats.smooth_L
    smooth_C = stats.smooth_C
    smooth_R = stats.smooth_R
    delta_L = stats.delta_L
    delta_C = stats.delta_C
    delta_R = stats.delta_R
    probe_mag = stats.probe_mag
    probe_count = stats.probe_count
    left_count = stats.left_count
    center_count = stats.center_count
    right_count = stats.right_count
    top_mag = stats.top_mag
    mid_mag = stats.mid_mag
    bottom_mag = stats.bottom_mag
    top_count = stats.top_count
    mid_count = stats.mid_count
    bottom_count = stats.bottom_count
    in_grace = stats.in_grace
    (
        state_str,
        obstacle_detected,
        side_safe,
        brake_thres,
        dodge_thres,
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
        simgetimage_s,
        decode_s,
        processing_s,
        flow_std,
        sudden_rise,
        center_blocked,
        combination_flow,
        minimum_flow,
    )


# === SLAM Navigation Helpers ===

def transform_slam_to_airsim(slam_pose_matrix):
    """Transform SLAM pose matrix to AirSim coordinate system."""
    if isinstance(slam_pose_matrix, list):
        slam_pose_matrix = np.array(slam_pose_matrix)

    x_slam, y_slam, z_slam = slam_pose_matrix[0, 3], slam_pose_matrix[1, 3], slam_pose_matrix[2, 3]

    x_airsim = z_slam
    y_airsim = x_slam
    z_airsim = -y_slam

    R_slam = slam_pose_matrix[:3, :3]
    T_slam_to_airsim = np.array([
        [0, 0, 1],
        [1, 0, 0],
        [0, -1, 0],
    ])
    R_airsim = T_slam_to_airsim @ R_slam @ T_slam_to_airsim.T

    transformed_pose = np.eye(4)
    transformed_pose[:3, :3] = R_airsim
    transformed_pose[0, 3] = x_airsim
    transformed_pose[1, 3] = y_airsim
    transformed_pose[2, 3] = z_airsim

    return transformed_pose, (x_airsim, y_airsim, z_airsim)


def check_slam_stop(exit_flag, start_time, max_duration):
    """Check if SLAM navigation should stop."""
    if exit_flag.is_set():
        return True

    if max_duration is None:
        logger.warning("max_duration is None in check_slam_stop")
        return False

    if time.time() - start_time > max_duration:
        logger.info(f"SLAM navigation time limit reached ({max_duration}s)")
        return True

    return False


def ensure_stable_slam_pose(
    client,
    pose_source,
    cov_thres,
    inlier_thres,
    exit_flag,
    start_time,
    max_duration,
    ctx=None,
):
    """Return a stable SLAM pose transformed to AirSim coordinates."""
    from slam_bridge.slam_receiver import get_latest_pose_matrix

    if pose_source == "airsim" and hasattr(client, "simGetVehiclePose"):
        air_pose = client.simGetVehiclePose("UAV")
        pos = air_pose.position
        yaw = airsim.to_eularian_angles(air_pose.orientation)[2]
        cy, sy = np.cos(yaw), np.sin(yaw)
        transformed_pose = np.array(
            [[cy, -sy, 0, pos.x_val], [sy, cy, 0, pos.y_val], [0, 0, 1, pos.z_val]]
        )
        return transformed_pose, (pos.x_val, pos.y_val, pos.z_val)

    pose = get_latest_pose_matrix()
    if pose is None or not is_slam_stable(cov_thres, inlier_thres):
        logger.warning("[SLAMNav] SLAM tracking lost. Attempting reinitialisation.")
    while pose is None or not is_slam_stable(cov_thres, inlier_thres):
        if config.ENABLE_SLAM_BOOTSTRAP:
            if ctx is not None and getattr(ctx, "param_refs", None):
                ctx.param_refs.state[0] = "bootstrap"
            run_slam_bootstrap(client, duration=4.0)
        time.sleep(1.0)
        if check_slam_stop(exit_flag, start_time, max_duration):
            logger.info("[SLAMNav] Exit signal during reinitialisation.")
            return None, (None, None, None)
        pose = get_latest_pose_matrix()

    if ctx is not None and getattr(ctx, "param_refs", None):
        ctx.param_refs.state[0] = "waypoint_nav"

    return transform_slam_to_airsim(pose)


def handle_waypoint_progress(x, y, waypoints, current_index, threshold=0.5):
    """Return updated waypoint index and goal after checking progress."""
    waypoint_x, waypoint_y, waypoint_z = waypoints[current_index]
    distance = np.sqrt((x - waypoint_x) ** 2 + (y - waypoint_y) ** 2)
    if distance < threshold:
        logger.info(f"Reached waypoint {current_index + 1}, moving to next waypoint.")
        current_index = (current_index + 1) % len(waypoints)
        waypoint_x, waypoint_y, waypoint_z = waypoints[current_index]
    return (waypoint_x, waypoint_y, waypoint_z), current_index, distance


# === Thread Management ===

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

    # Use the new finalize_logging function
    log_file = getattr(ctx, "log_file", None)
    log_buffer = getattr(ctx, "log_buffer", None)
    
    if log_file is not None:
        try:
            from uav.logging_helpers import finalize_logging
            finalize_logging(log_file, log_buffer)
        except Exception as exc:
            logger.warning("Log finalization failed: %s", exc)


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


class ThreadManager:
    """Context manager for shutting down worker threads."""

    def __init__(self, ctx):
        self.ctx = ctx

    def __enter__(self):
        return self.ctx

    def __exit__(self, exc_type, exc, tb):
        shutdown_threads(self.ctx)


class LoggingContext:
    """Context manager for flushing and closing log resources."""

    def __init__(self, ctx):
        self.ctx = ctx

    def __enter__(self):
        return self.ctx

    def __exit__(self, exc_type, exc, tb):
        close_logging(self.ctx)


class SimulationProcess:
    """Context manager for gracefully terminating the UE4 simulation."""

    def __init__(self, proc):
        self.proc = proc

    def __enter__(self):
        return self.proc

    def __exit__(self, exc_type, exc, tb):
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except Exception:
                self.proc.kill()
            logger.info("UE4 simulation closed.")


def cleanup(client, sim_process, ctx):
    """Clean up resources and land the drone."""
    logger.info("Landing...")

    with ThreadManager(ctx):
        pass

    with LoggingContext(ctx):
        pass

    logger.info("Finalizing flight analysis and cleanup.")
    from uav.nav_analysis import finalise_files

    finalise_files(ctx)

    shutdown_airsim(client)

    with SimulationProcess(sim_process):
        pass

    logger.info("Cleanup complete. Exiting navigation loop.")
