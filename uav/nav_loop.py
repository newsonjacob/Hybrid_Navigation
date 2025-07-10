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
from uav.utils import retain_recent_files

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
from uav.utils import (
    FLOW_STD_MAX, get_drone_state, retain_recent_logs, should_flat_wall_dodge
)
from uav.utils import retain_recent_files, retain_recent_views
from uav import config

logger = logging.getLogger(__name__)

# Grace period duration (seconds) after dodge/brake actions
NAV_GRACE_PERIOD_SEC = 0.5

# Flag file to stop the drone
STOP_FLAG_PATH = "flags/stop.flag"

# === Perception Processing ===

def perception_loop(tracker, image):
    """Process a single image for optical flow."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if tracker.prev_gray is None:
        tracker.initialize(gray)
        return np.array([]), np.array([]), 0.0
    return tracker.process_frame(gray, time.time())

# === Navigation Step ===

def navigation_step(
    client, navigator, flow_history, good_old, flow_vectors, flow_std,
    smooth_L, smooth_C, smooth_R, delta_L, delta_C, delta_R,
    left_count, center_count, right_count, frame_queue, vis_img,
    time_now, frame_count, prev_state, state_history, pos_history, param_refs,
    probe_mag=0.0, probe_count=0,
):
    """
    Decide and execute navigation action based on perception and state.
    Returns: state_str, obstacle_detected, side_safe, brake_thres, dodge_thres, probe_req
    """
    state_str = "none"
    brake_thres = 0.0
    dodge_thres = 0.0
    probe_req = 0.0
    side_safe = False
    left_safe = False
    right_safe = False
    obstacle_detected = 0

    # --- Feature Validity ---
    valid_L = left_count >= config.MIN_FEATURES_PER_ZONE
    valid_C = center_count >= config.MIN_FEATURES_PER_ZONE
    valid_R = right_count >= config.MIN_FEATURES_PER_ZONE

    logger.debug("Flow Magnitudes — L: %.2f, C: %.2f, R: %.2f", smooth_L, smooth_C, smooth_R,)

    # --- Grace Period Handling ---
    if in_grace_period(time_now, navigator):
        param_refs['state'][0] = "🕒 grace"
        obstacle_detected = 0
        try:
            frame_queue.get_nowait() # check this!
        except Exception:
            pass
        frame_queue.put(vis_img)
        return state_str, obstacle_detected, side_safe, brake_thres, dodge_thres, probe_req
        
    navigator.just_resumed = False

    # --- Navigation Logic ---
    # First check if we have enough features to make a decision
    if len(good_old) < 10: 
        if smooth_L > 1.5 and smooth_R > 1.5 and smooth_C < 0.2: 
            state_str = navigator.brake()
        else:
            state_str = navigator.blind_forward()
    else: # Enough features to make a decision
        pos, yaw, speed = get_drone_state(client)
        brake_thres, dodge_thres = compute_thresholds(speed)

        # Define certain navigation conditions
        sudden_center_flow_rise = delta_C > 1 and center_count >= 20 # Sudden rise in center flow magnitude
        center_blocked = smooth_C > brake_thres and center_count >= 20 # Center flow is high enough to indicate an obstacle
        left_clearing = delta_L < -0.3 # Sudden drop in left flow magnitude
        right_clearing = delta_R < -0.3 # Sudden drop in right flow magnitude       
        probe_reliable = probe_count > config.MIN_PROBE_FEATURES and probe_mag > 0.05 # Probe data is reliable
        
        # --- Side safety checks ---
        # Check left side safety
        if valid_L and smooth_L < brake_thres:
            left_safe = True # Left side is safe
        elif left_count < 10 and center_count >= left_count * 5:
            left_safe = True # Left side has very few features, indicating it may be clear
        # Check right side safety
        if valid_R and smooth_R < brake_thres:
            right_safe = True # Right side is safe
        elif right_count < 10 and center_count >= right_count * 5:
            right_safe = True # Right side has very few features, indicating it may be clear
        # Determine if at least one side is safe
        if left_safe == True and right_safe == True:
            side_safe = True
        
        # --- Obstacle Detection ---
        if sudden_center_flow_rise or center_blocked or (center_count > 100 and (smooth_C > brake_thres * 0.5 or delta_C > 0.5)):
            obstacle_detected = 1
        elif delta_C > 0.5 and smooth_C > brake_thres * 0.5 and center_count > 50:
            obstacle_detected = 1  
        else:
            obstacle_detected = 0

        # --- Obstacle handling logic ---
        # Sides are safe
        if obstacle_detected and side_safe and not navigator.dodging:
            if left_safe and right_safe:
                if left_count < right_count:
                    logger.info("\U0001F500 Both sides safe — Dodging left")
                    state_str = navigator.dodge(smooth_L, smooth_C, smooth_R, direction='left')
                else:
                    logger.info("\U0001F500 Both sides safe — Dodging right")
                    state_str = navigator.dodge(smooth_L, smooth_C, smooth_R, direction='right')
            elif left_safe:
                    logger.info("\U0001F500 Left safe — Dodging left")
                    state_str = navigator.dodge(smooth_L, smooth_C, smooth_R, direction='left')   
            else:
                logger.info("\U0001F500 Right safe — Dodging right")
                state_str = navigator.dodge(smooth_L, smooth_C, smooth_R, direction='right')            

        # Sides are clearing
        elif obstacle_detected and (left_clearing or right_clearing) and not navigator.dodging: # Sides are clearing, dodge to the side with lower flow magnitude
            logger.info("\U0001F500 Sides clearing, Dodging")
            if delta_L < delta_R:
                state_str = navigator.dodge(smooth_L, smooth_C, smooth_R, direction='left')
            else:
                state_str = navigator.dodge(smooth_L, smooth_C, smooth_R, direction='right')   
        
        # Sides are not safe
        elif obstacle_detected and not (navigator.braked or navigator.dodging): # Sudden rise in Center flow but sides are not safe, just brake
            logger.info("\U0001F6D1 Sides not safe — Braking")
            state_str = navigator.brake()
        
        # Dodge maintenance
        if navigator.dodging and obstacle_detected == 1:
            navigator.maintain_dodge()
        if (navigator.dodging or navigator.braked) and obstacle_detected == 0:
            logger.info("\u2705 Obstacle cleared — resuming forward")
            state_str = navigator.resume_forward()

    # --- Recovery/State Maintenance ---
    if state_str == "none": # 
        if (navigator.braked
            and smooth_C < brake_thres * 0.8
            and smooth_L < brake_thres * 0.8
            and smooth_R < brake_thres * 0.8
            and time_now >= navigator.grace_period_end_time):
            logger.info("Brake released — resuming forward at frame %s", frame_count)
            state_str = navigator.resume_forward() # Resume forward after brake
        elif not navigator.braked and not navigator.dodging and time_now - navigator.last_movement_time > 2:
            state_str = navigator.reinforce() # Reinforce forward motion if no movement for a while
        elif (navigator.braked or navigator.dodging) and speed < 0.2 and smooth_C < 5 and smooth_L < 5 and smooth_R < 5:
            state_str = navigator.nudge_forward() # Nudge forward if braked/dodging and very low speed
        elif time_now - navigator.last_movement_time > 4:
            state_str = navigator.timeout_recover() # Timeout recovery if no movement for too long

    # --- Update State and History ---
    param_refs['state'][0] = state_str
    # obstacle_detected = int('dodge' in state_str or state_str == 'brake')

    pos_hist, _, _ = get_drone_state(client) 
    state_history.append(state_str)
    pos_history.append((pos_hist.x_val, pos_hist.y_val))
    if len(state_history) == state_history.maxlen: 
        if all(s == state_history[-1] for s in state_history) and state_history[-1].startswith("dodge"):
            dx = pos_history[-1][0] - pos_history[0][0]
            dy = pos_history[-1][1] - pos_history[0][1]
            if abs(dx) < 0.5 and abs(dy) < 1.0:
                logger.warning("Repeated dodges detected — extending dodge")
                state_str = navigator.dodge(smooth_L, smooth_C, smooth_R, duration=3.0)
                state_history[-1] = state_str
                param_refs['state'][0] = state_str

    pos, yaw, speed = get_drone_state(client)
    brake_thres, dodge_thres = compute_thresholds(speed)
    return state_str, obstacle_detected, side_safe, brake_thres, dodge_thres, probe_req

def log_frame_data(log_file, log_buffer, line):
    """Buffer log lines and periodically flush to disk."""
    log_buffer.append(line)
    if len(log_buffer) >= config.LOG_INTERVAL:
        log_file.writelines(log_buffer)
        log_buffer.clear()

def write_video_frame(queue, frame):
    """Queue a video frame for asynchronous writing."""
    try: queue.put_nowait(frame)
    except Exception: pass

def process_perception_data(
    client, args, data, frame_count, frame_queue, flow_history, navigator, param_refs, time_now, max_flow_mag
):
    """
    Process perception output and update histories.
    Returns processed perception data and region statistics.
    """
    vis_img, good_old, flow_vectors, flow_std, simgetimage_s, decode_s, processing_s = data
    gray = cv2.cvtColor(vis_img, cv2.COLOR_BGR2GRAY)
    if frame_count == 1 and len(good_old) == 0:
        frame_queue.put(vis_img)
        return None
    if args.manual_nudge and frame_count == 5:
        logger.info("Manual nudge forward for test")
        client.moveByVelocityAsync(2, 0, 0, 2)
    if flow_vectors.size == 0:
        magnitudes = np.array([])
    else:
        if flow_vectors.ndim == 1: flow_vectors = flow_vectors.reshape(-1, 2)
        magnitudes = np.linalg.norm(flow_vectors, axis=1)
    num_clamped = np.sum(magnitudes > max_flow_mag)
    if num_clamped > 100:
        logger.warning("Clamped %d large flow magnitudes to %s", num_clamped, max_flow_mag)
    magnitudes = np.clip(magnitudes, 0, max_flow_mag)
    good_old = good_old.reshape(-1, 2)
    left_mag, center_mag, right_mag, probe_mag, probe_count, left_count, center_count, right_count = compute_region_stats(magnitudes, good_old, gray.shape[1])
    flow_history.update(left_mag, center_mag, right_mag)
    smooth_L, smooth_C, smooth_R = flow_history.average()
    delta_L = smooth_L - param_refs['prev_L'][0]
    delta_C = smooth_C - param_refs['prev_C'][0]
    delta_R = smooth_R - param_refs['prev_R'][0]
    param_refs['prev_L'][0], param_refs['prev_C'][0], param_refs['prev_R'][0] = (
        smooth_L, smooth_C, smooth_R
    )
    param_refs['delta_L'][0], param_refs['delta_C'][0], param_refs['delta_R'][0] = (
        delta_L, delta_C, delta_R
    )
    param_refs['L'][0], param_refs['C'][0], param_refs['R'][0] = smooth_L, smooth_C, smooth_R
    if navigator.just_resumed and time_now < navigator.resume_grace_end_time:
        cv2.putText(vis_img, "GRACE", (1100, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
    in_grace = navigator.just_resumed and time_now < navigator.resume_grace_end_time
    return (
        vis_img, good_old, flow_vectors, flow_std, simgetimage_s, decode_s, processing_s,
        smooth_L, smooth_C, smooth_R, delta_L, delta_C, delta_R,
        probe_mag, probe_count, left_count, center_count, right_count, in_grace,
    )

def apply_navigation_decision(
    client, navigator, flow_history, good_old, flow_vectors, flow_std,
    smooth_L, smooth_C, smooth_R, delta_L, delta_C, delta_R, probe_mag, probe_count,
    left_count, center_count, right_count, frame_queue, vis_img,
    time_now, frame_count, prev_state, state_history, pos_history, param_refs,
):
    """Wrapper around navigation_step for clarity."""
    return navigation_step(
        client, navigator, flow_history, good_old, flow_vectors, flow_std,
        smooth_L, smooth_C, smooth_R, delta_L, delta_C, delta_R,
        left_count, center_count, right_count, frame_queue, vis_img,
        time_now, frame_count, prev_state, state_history, pos_history, param_refs,
        probe_mag=probe_mag, probe_count=probe_count,
    )

def write_frame_output(
    client, vis_img, frame_queue, loop_start, frame_duration, fps_list, start_time,
    smooth_L, smooth_C, smooth_R, delta_L, delta_C, delta_R,
    left_count, center_count, right_count,
    good_old, flow_vectors, in_grace, frame_count, time_now, param_refs,
    log_file, log_buffer, state_str, obstacle_detected, side_safe,
    brake_thres, dodge_thres, probe_req, simgetimage_s, decode_s, processing_s, flow_std,
):
    """
    Write video frame, overlay, and log data for the current step.
    Returns updated loop_start time.
    """
    pos, yaw, speed = get_drone_state(client)
    collision = client.simGetCollisionInfo()
    collided = int(getattr(collision, "has_collided", False))
    vis_img = draw_overlay(
        vis_img, frame_count, speed, param_refs['state'][0], time_now - start_time,
        smooth_L, smooth_C, smooth_R,
        delta_L, delta_C, delta_R,
        left_count, center_count, right_count,
        good_old, flow_vectors, in_grace=in_grace,
    )
    write_video_frame(frame_queue, vis_img)
    elapsed = time.time() - loop_start
    if elapsed < frame_duration: time.sleep(frame_duration - elapsed)
    loop_elapsed = time.time() - loop_start
    actual_fps = 1 / max(loop_elapsed, 1e-6)
    loop_start = time.time()
    fps_list.append(actual_fps)
    log_line = format_log_line(
        frame_count, smooth_L, smooth_C, smooth_R, 
        delta_L, delta_C, delta_R, flow_std,
        left_count, center_count, right_count,
        brake_thres, dodge_thres, probe_req, actual_fps,
        state_str, collided, obstacle_detected, side_safe,
        pos, yaw, speed,
        time_now, good_old,
        simgetimage_s, decode_s, processing_s, loop_elapsed,
    )
    log_frame_data(log_file, log_buffer, log_line)
    logger.debug("Actual FPS: %.2f", actual_fps)
    logger.debug("Features detected: %d", len(good_old))
    return loop_start

def handle_reset(client, ctx, frame_count):
    """
    Reset simulation and restart logging/video.
    Returns reset frame_count.
    """
    param_refs, flow_history, navigator = ctx['param_refs'], ctx['flow_history'], ctx['navigator']
    frame_queue, video_thread, out = ctx['frame_queue'], ctx['video_thread'], ctx['out']
    log_file, log_buffer, fourcc = ctx['log_file'], ctx['log_buffer'], ctx['fourcc']
    logger.info("Resetting simulation...")
    try:
        client.landAsync().join(); client.reset(); client.enableApiControl(True)
        client.armDisarm(True); client.takeoffAsync().join(); client.moveToPositionAsync(0, 0, -2, 2).join()
    except Exception as e: logger.error("Reset error: %s", e)
    ctx['flow_history'], ctx['navigator'], frame_count = FlowHistory(), Navigator(client), 0
    param_refs['reset_flag'][0] = False
    if log_buffer: log_file.writelines(log_buffer); log_buffer.clear()
    log_file.close()
    ctx['timestamp'] = datetime.now().strftime('%Y%m%d_%H%M%S')
    timestamp = ctx['timestamp']
    log_file = open(f"flow_logs/full_log_{timestamp}.csv", 'w')
    ctx['log_file'] = log_file
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

    frame_queue.put(None)
    video_thread.join()
    out.release()
    out = cv2.VideoWriter(config.VIDEO_OUTPUT, fourcc, config.VIDEO_FPS, config.VIDEO_SIZE)
    ctx['out'] = out
    video_thread = start_video_writer_thread(frame_queue, out, ctx['exit_flag'])
    ctx['video_thread'] = video_thread
    return frame_count

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

def start_perception_thread(ctx):
    """Launch background perception thread."""
    exit_flag, tracker = ctx['exit_flag'], ctx['tracker']
    perception_queue = Queue(maxsize=1)
    last_vis_img = np.zeros((720, 1280, 3), dtype=np.uint8)
    def perception_worker():
        nonlocal last_vis_img
        local_client = airsim.MultirotorClient()
        local_client.confirmConnection()
        request = [ImageRequest("oakd_camera", ImageType.Scene, False, True)]
        while not exit_flag.is_set():
            t0 = time.time()
            responses = local_client.simGetImages(request, vehicle_name="UAV")
            t_fetch_end = time.time()
            response = responses[0]
            if (response.width == 0 or response.height == 0 or len(response.image_data_uint8) == 0):
                data = (last_vis_img, np.array([]), np.array([]), 0.0, t_fetch_end - t0, 0.0, 0.0)
            else:
                img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8).copy()
                img = cv2.imdecode(img1d, cv2.IMREAD_COLOR)
                t_decode_end = time.time()
                if img is None: continue
                img = cv2.resize(img, config.VIDEO_SIZE)
                vis_img = img.copy()
                last_vis_img = vis_img
                t_proc_start = time.time()
                good_old, flow_vectors, flow_std = perception_loop(tracker, img)
                processing_s = time.time() - t_proc_start
                data = (
                    vis_img, good_old, flow_vectors, flow_std,
                    t_fetch_end - t0, t_decode_end - t_fetch_end, processing_s,
                )
            try: perception_queue.put(data, block=False)
            except Exception: pass
    perception_thread = Thread(target=perception_worker, daemon=True)
    perception_thread.start()
    ctx['perception_queue'] = perception_queue
    ctx['perception_thread'] = perception_thread

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
    try:
        loop_start = time.time()
        while not exit_flag.is_set():
            if os.path.exists(STOP_FLAG_PATH):
                logger.info("Stop flag detected. Landing and shutting down.")
                exit_flag.set()
                break
            frame_count += 1
            time_now = time.time()

            # ✅ Print real drone position from AirSim
            pose = client.simGetVehiclePose("UAV")
            pos = pose.position
            logger.info("[UAV Pose] x=%.2f, y=%.2f, z=%.2f", pos.x_val, pos.y_val, pos.z_val)

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
    try:
        responses = client.simGetImages([
            ImageRequest("oakd_camera", ImageType.DepthPlanar, True)
        ], vehicle_name=vehicle_name)
        if not responses or responses[0].height == 0:
            return False, None
        depth_image = airsim.get_pfm_array(responses[0])
        h, w = depth_image.shape
        cx, cy = w // 2, h // 2
        roi = depth_image[cy-20:cy+20, cx-20:cx+20]
        mean_depth = np.nanmean(roi)
        return mean_depth < depth_threshold, mean_depth
    except Exception as e:
        print(f"[Obstacle Check] Depth read failed: {e}")
        return False, None

def slam_navigation_loop(args, client, ctx):
    """
    Main navigation loop for SLAM-based navigation with basic obstacle avoidance.
    """
    import logging
    logger = logging.getLogger(__name__)
    logger.info("[SLAMNav] Starting SLAM navigation loop with obstacle avoidance.")

    from slam_bridge.slam_receiver import get_latest_pose, get_pose_history
    from slam_bridge.frontier_detection import detect_frontiers

    # --- Incorporate exit_flag from ctx for GUI stop button ---
    exit_flag = None
    navigator = None
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
            else:
                x, y, z = pose # Unpack the pose
                logger.info(f"[SLAMNav] Received pose: x={x:.2f}, y={y:.2f}, z={z:.2f}")

                # Detect exploration frontiers from accumulated SLAM poses
                history = get_pose_history()
                map_pts = np.array(
                    [[m[0][3], m[1][3], m[2][3]] for _, m in history], dtype=float
                )
                frontiers = detect_frontiers(map_pts)
                if frontiers.size:
                    logger.debug(
                        "[SLAMNav] Frontier voxels detected: %d", len(frontiers)
                    )
                    logger.debug(
                        "[SLAMNav] Sample frontier: x=%.2f y=%.2f z=%.2f",
                        frontiers[0][0],
                        frontiers[0][1],
                        frontiers[0][2],
                    )

                # Check for collision/obstacle
                collision = client.simGetCollisionInfo()
                if getattr(collision, "has_collided", False):
                    logger.warning("[SLAMNav] Obstacle detected! Executing avoidance maneuver.")
                    # Back up and try to move sideways
                    client.moveByVelocityAsync(-1.0, 0, 0, 1).join()  # Back up
                    client.moveByVelocityAsync(0, 1.0, 0, 1).join()   # Move right
                    continue

                # Depth-based obstacle check before moving toward the goal
                ahead, depth = is_obstacle_ahead(client)
                if ahead:
                    msg = "[SLAMNav] Depth obstacle detected"
                    if depth is not None:
                        msg += f" at {depth:.2f}m"
                    logger.warning(msg)
                    if navigator is not None:
                        navigator.dodge(0, 0, 0, direction="right")
                    else:
                        client.hoverAsync().join()
                    continue

                # Check if goal reached
                if abs(x - goal_x) < threshold and abs(y - goal_y) < threshold:
                    if frontiers.size:
                        goal_x, goal_y, _ = frontiers[0]
                        logger.info(
                            "[SLAMNav] Goal reached — switching to frontier at x=%.2f y=%.2f",
                            goal_x,
                            goal_y,
                        )
                    else:
                        logger.info("[SLAMNav] Goal reached — landing.")
                        client.moveToPositionAsync(x, y, goal_z, 1).join()
                        client.landAsync().join()
                        break

                # Move toward goal (simple proportional controller)
                vx = 1.0 if x < goal_x - threshold else 0.0
                vy = 1.0 if y < goal_y - threshold else 0.0
                vz = 0.0  # Maintain altitude
                if vx != 0.0 or vy != 0.0:
                    logger.info(f"[SLAMNav] Moving toward goal with vx={vx}, vy={vy}")
                    client.moveByVelocityAsync(vx, vy, vz, 1, drivetrain=airsim.DrivetrainType.ForwardOnly, yaw_mode=airsim.YawMode(False, 0))
                else:
                    logger.info("[SLAMNav] Holding position.")

            # End condition
            if time.time() - start_time > max_duration:
                logger.info("[SLAMNav] Max duration reached, ending navigation.")
                client.landAsync().join()
                break

            time.sleep(0.1)
    except KeyboardInterrupt:
        logger.info("[SLAMNav] Interrupted by user.")
        client.landAsync().join()
    finally:
        logger.info("[SLAMNav] SLAM navigation loop finished.")

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
            subprocess.run(["python3", "-m", "analysis.flight_path_viewer", html_output])
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

    # ✅ Wait after landing for graceful shutdown if early exit
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
            max_wait = 15  # seconds
            start_wait = time.time()
            while True:
                try:
                    state = client.getMultirotorState()
                    if state.landed_state == LandedState.Landed:
                        logger.info("Drone is landed (LandedState.Landed).")
                        break
                except Exception:
                    break
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


