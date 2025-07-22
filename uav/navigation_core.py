"""Core navigation helpers and step logic."""

import logging

from uav.navigation_rules import compute_thresholds
from uav.state_checks import in_grace_period
from uav.utils import get_drone_state
from uav import config

logger = logging.getLogger(__name__)


def detect_obstacle_with_hysteresis(
    navigator,
    smooth_C=None,
    delta_C=None,
    center_count=None,
    brake_thres=None,
    mode="reactive",
    depth=None,
    depth_threshold=2.0,
    pose=None,
    pose_goal=None,
    pose_threshold=0.5,
):
    """Detect an obstacle using optical flow or SLAM data with hysteresis.

    Parameters
    ----------
    navigator : Navigator
        Navigator instance to track detection state.
    ... (other parameters same as detect_obstacle)

    Returns
    -------
    tuple
        (confirmed_detection, sudden_rise, center_blocked, combination_flow, minimum_flow)
    """
    # Get raw detection result (without hysteresis)
    raw_result = detect_obstacle(
        smooth_C, delta_C, center_count, brake_thres, mode,
        depth, depth_threshold, pose, pose_goal, pose_threshold
    )
    
    if isinstance(raw_result, tuple):
        raw_detection, sudden_rise, center_blocked, combination_flow, minimum_flow = raw_result
    else:
        # For SLAM mode or older detection function
        raw_detection = raw_result
        sudden_rise = center_blocked = combination_flow = minimum_flow = False
    
    # Apply hysteresis logic
    if raw_detection:
        # Obstacle detected - increment detection counter
        navigator.obstacle_detection_count += 1
        navigator.obstacle_clear_count = 0  # Reset clear counter
        
        # Confirm obstacle only after required number of detections
        if navigator.obstacle_detection_count >= navigator.DETECTION_THRESHOLD:
            if not navigator.obstacle_confirmed:
                logger.info(f"[HYSTERESIS] Obstacle CONFIRMED after {navigator.obstacle_detection_count} frames")
            navigator.obstacle_confirmed = True
    else:
        # No obstacle detected - increment clear counter
        navigator.obstacle_clear_count += 1
        navigator.obstacle_detection_count = 0  # Reset detection counter
        
        # Clear obstacle only after required number of clear frames
        if navigator.obstacle_clear_count >= navigator.DETECTION_THRESHOLD and navigator.obstacle_confirmed:
            logger.info(f"[HYSTERESIS] Obstacle CLEARED after {navigator.obstacle_clear_count} frames")
            navigator.obstacle_confirmed = False
    
    # Store condition states for logging
    navigator.last_sudden_rise = sudden_rise
    navigator.last_center_blocked = center_blocked
    navigator.last_combination_flow = combination_flow
    navigator.last_minimum_flow = minimum_flow
    
    logger.debug(f"[HYSTERESIS] Raw: {raw_detection}, Detection count: {navigator.obstacle_detection_count}, "
                f"Clear count: {navigator.obstacle_clear_count}, Confirmed: {navigator.obstacle_confirmed}")
    
    return navigator.obstacle_confirmed, sudden_rise, center_blocked, combination_flow, minimum_flow


def detect_obstacle(
    smooth_C=None,
    delta_C=None,
    center_count=None,
    brake_thres=None,
    mode="reactive",
    depth=None,
    depth_threshold=2.0,
    pose=None,
    pose_goal=None,
    pose_threshold=0.5,
):
    """Detect an obstacle using optical flow or SLAM data.

    Parameters
    ----------
    smooth_C : float, optional
        Smoothed center flow magnitude.
    delta_C : float, optional
        Change in center flow magnitude between frames.
    center_count : int, optional
        Number of tracked features in the center region.
    brake_thres : float, optional
        Threshold above which braking should occur in reactive mode.
    mode : {"reactive", "slam"}, optional
        Detection mode. ``"reactive"`` uses optical flow, ``"slam"`` uses SLAM
        depth or pose information.
    depth : float, optional
        Measured depth ahead of the drone when using SLAM depth.
    depth_threshold : float, optional
        Depth threshold used to trigger an obstacle in SLAM mode.
    pose : sequence, optional
        Current SLAM pose ``(x, y, z)``.
    pose_goal : sequence, optional
        Target pose to compare against in SLAM mode.
    pose_threshold : float, optional
        Distance threshold for pose based detection.

    Returns
    -------
    bool
        ``True`` if an obstacle is considered detected.
    """
    if mode == "reactive":
        if smooth_C is None or delta_C is None or center_count is None or brake_thres is None:
            raise ValueError("Reactive mode requires smooth_C, delta_C, center_count, brake_thres")
        
        # Calculate individual conditions
        sudden_rise = delta_C > 1 and center_count >= 5
        center_blocked = smooth_C > brake_thres and center_count >= 5 and delta_C > 0
        combination_flow = center_count > 75 and (smooth_C > brake_thres * 0.75 or delta_C > 0.75)
        minimum_flow = delta_C > 0.6 and smooth_C > brake_thres * 0.25 and center_count > 50
        test_flow = delta_C > 0.6 and smooth_C > brake_thres * 0.5 and center_count >= 5

        # Overall obstacle detection logic
        obstacle_detected = sudden_rise or center_blocked or combination_flow or minimum_flow or test_flow

        # Log which conditions triggered the detection
        if obstacle_detected:
            active_conditions = []
            if sudden_rise: active_conditions.append("SUDDEN_RISE")
            if center_blocked: active_conditions.append("CENTER_BLOCKED")
            if combination_flow: active_conditions.append("COMBINATION_FLOW")
            if minimum_flow: active_conditions.append("MINIMUM_FLOW")
            logger.debug(f"[Obstacle Detection] Active: {' + '.join(active_conditions)}")
        
        # Return tuple with individual condition states
        return obstacle_detected, sudden_rise, center_blocked, combination_flow, minimum_flow
    
    elif mode == "slam":
        if depth is not None:
            return depth < depth_threshold
        if pose is not None and pose_goal is not None:
            dx = pose[0] - pose_goal[0]
            dy = pose[1] - pose_goal[1]
            dz = pose[2] - pose_goal[2]
            dist = (dx ** 2 + dy ** 2 + dz ** 2) ** 0.5
            return dist < pose_threshold
        return False
    else:
        raise ValueError(f"Unknown mode for detect_obstacle: {mode}")


def determine_side_safety(
    smooth_L,
    smooth_R,
    brake_thres,
    left_count,
    center_count,
    right_count,
):
    """Determine if the left and right sides are safe for dodging.

    Parameters
    ----------
    smooth_L, smooth_R : float
        Smoothed flow magnitudes for the left and right regions.
    brake_thres : float
        Threshold above which a side is considered blocked.
    left_count, center_count, right_count : int
        Number of tracked features in each region.

    Returns
    -------
    Tuple[bool, bool, bool]
        Flags ``(left_safe, right_safe, side_safe)``.
    """
    valid_L = left_count >= config.MIN_FEATURES_PER_ZONE
    valid_R = right_count >= config.MIN_FEATURES_PER_ZONE

    left_safe = False
    right_safe = False

    if valid_L and smooth_L < brake_thres:
        left_safe = True
    elif left_count < 10 and center_count >= left_count * 5:
        left_safe = True

    if valid_R and smooth_R < brake_thres:
        right_safe = True
    elif right_count < 10 and center_count >= right_count * 5:
        right_safe = True

    side_safe = left_safe and right_safe
    return left_safe, right_safe, side_safe


def handle_obstacle(
    navigator,
    obstacle_detected,
    side_safe,
    left_safe,
    right_safe,
    left_clearing,
    right_clearing,
    smooth_L,
    smooth_C,
    smooth_R,
    left_count,
    right_count,
):
    """Handle obstacle avoidance manoeuvres.

    Parameters
    ----------
    navigator : Navigator
        Navigator instance controlling the drone.
    obstacle_detected : bool
        Whether an obstacle is currently detected ahead.
    side_safe : bool
        ``True`` if both sides are considered safe for dodging.
    left_safe, right_safe : bool
        Individual side safety flags.
    left_clearing, right_clearing : bool
        Flags indicating if a previously blocked side is clearing.
    smooth_L, smooth_C, smooth_R : float
        Smoothed flow magnitudes for logging.
    left_count, right_count : int
        Number of features on each side.

    Returns
    -------
    str
        Action string executed by the navigator.
    """
    state_str = "none"

    if obstacle_detected and side_safe and not navigator.dodging:
        if left_safe and right_safe:
            if left_count < right_count:
                logger.info("\U0001F500 Both sides safe — Dodging left")
                state_str = navigator.dodge(smooth_L, smooth_C, smooth_R, direction="left")
            else:
                logger.info("\U0001F500 Both sides safe — Dodging right")
                state_str = navigator.dodge(smooth_L, smooth_C, smooth_R, direction="right")
        elif left_safe:
            logger.info("\U0001F500 Left safe — Dodging left")
            state_str = navigator.dodge(smooth_L, smooth_C, smooth_R, direction="left")
        else:
            logger.info("\U0001F500 Right safe — Dodging right")
            state_str = navigator.dodge(smooth_L, smooth_C, smooth_R, direction="right")

    elif obstacle_detected and (left_clearing or right_clearing) and not navigator.dodging:
        logger.info("\U0001F500 Sides clearing, Dodging")
        if left_clearing and not right_clearing:
            state_str = navigator.dodge(smooth_L, smooth_C, smooth_R, direction="left")
        elif right_clearing and not left_clearing:
            state_str = navigator.dodge(smooth_L, smooth_C, smooth_R, direction="right")
        else:
            if left_count < right_count:
                state_str = navigator.dodge(smooth_L, smooth_C, smooth_R, direction="left")
            else:
                state_str = navigator.dodge(smooth_L, smooth_C, smooth_R, direction="right")

    elif obstacle_detected and not (navigator.braked or navigator.dodging):
        logger.info("\U0001F6D1 Sides not safe — Braking")
        state_str = navigator.brake()

    if navigator.dodging and obstacle_detected:
        navigator.maintain_dodge()
    if (navigator.dodging or navigator.braked) and not obstacle_detected:
        logger.info("\u2705 Obstacle cleared — resuming forward")
        state_str = navigator.resume_forward()

    return state_str


def handle_grace_period(time_now, navigator, frame_queue, vis_img, param_refs):
    """Return ``True`` if the start-up grace period is active.

    Frames are queued without navigation actions until the grace period
    expires.
    """
    if in_grace_period(time_now, navigator):
        param_refs.state[0] = "\U0001F552 grace"
        try:
            frame_queue.get_nowait()
        except Exception:
            pass
        frame_queue.put(vis_img)
        return True
    return False


def decide_low_feature_action(navigator, smooth_L, smooth_C, smooth_R):
    """Choose an action when few optical-flow features are tracked.

    Returns
    -------
    str
        Action chosen by the navigator.
    """
    if smooth_L > 1.5 and smooth_R > 1.5 and smooth_C < 0.2:
        return navigator.brake()
    return navigator.blind_forward()


def update_dodge_history(client, state_history, pos_history, state_str, navigator,
                         smooth_L, smooth_C, smooth_R, param_refs):
    """Extend dodge time if the UAV oscillates without progress.

    Parameters
    ----------
    client : airsim.MultirotorClient
        AirSim client for retrieving position.
    state_history : deque
        Recent navigation actions.
    pos_history : deque
        Recent position history.
    state_str : str
        Current action string.
    navigator : Navigator
        Navigator controlling the drone.
    smooth_L, smooth_C, smooth_R : float
        Current smoothed flow magnitudes.
    param_refs : ParamRefs
        Parameter reference object for state updates.

    Returns
    -------
    str
        Possibly updated action string.
    """
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
                param_refs.state[0] = state_str
    return state_str


def recovery_actions(navigator, speed, smooth_C, smooth_L, smooth_R, brake_thres,
                     time_now, frame_count):
    """Return a recovery action if no obstacle manoeuvre was triggered.

    Returns
    -------
    str
        Action string chosen to recover progress.
    """
    if (
        navigator.braked
        and smooth_C < brake_thres * 0.8
        and smooth_L < brake_thres * 0.8
        and smooth_R < brake_thres * 0.8
        and time_now >= navigator.grace_period_end_time
    ):
        logger.info("Brake released — resuming forward at frame %s", frame_count)
        return navigator.resume_forward()
    if not navigator.braked and not navigator.dodging and time_now - navigator.last_movement_time > 2:
        return navigator.reinforce()
    if (
        navigator.braked or navigator.dodging
    ) and speed < 0.2 and smooth_C < 5 and smooth_L < 5 and smooth_R < 5:
        return navigator.nudge_forward()
    if time_now - navigator.last_movement_time > 4:
        return navigator.timeout_recover()
    return "none"


def navigation_step(
    client,
    navigator,
    flow_history,
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
    frame_queue,
    vis_img,
    time_now,
    frame_count,
    prev_state,
    state_history,
    pos_history,
    param_refs,
    probe_mag=0.0,
    probe_count=0,
):
    """Evaluate flow data and choose the next UAV action.

    Parameters
    ----------
    client : airsim.MultirotorClient
        AirSim client used for telemetry queries.
    navigator : Navigator
        Navigator controlling the drone.
    flow_history : FlowHistory
        Rolling buffer of recent flow magnitudes.
    good_old : list
        Previous frame feature points.
    flow_vectors : np.ndarray
        Optical flow vectors for the current frame.
    flow_std : float
        Standard deviation of flow magnitudes.
    smooth_L, smooth_C, smooth_R : float
        Smoothed flow magnitudes.
    delta_L, delta_C, delta_R : float
        Change in flow magnitudes between frames.
    left_count, center_count, right_count : int
        Feature counts per region.
    frame_queue : Queue
        Queue for video frames to be written.
    vis_img : np.ndarray
        Visualisation image for this frame.
    time_now : float
        Current timestamp.
    frame_count : int
        Current frame index.
    prev_state : str
        Previous navigation state.
    state_history, pos_history : deque
        Recent navigation states and positions.
    param_refs : ParamRefs
        Shared parameter references.
    probe_mag : float, optional
        Magnitude of probe flow, defaults to ``0.0``.
    probe_count : int, optional
        Number of probe features, defaults to ``0``.

    Returns
    -------
    Tuple[str, int, bool, float, float, float, bool, bool, bool, bool]
        Tuple containing the selected state string, obstacle flag, side safety
        flag, dynamic thresholds and detailed obstacle condition flags.
    """
    state_str = "none"
    brake_thres = 0.0
    dodge_thres = 0.0
    probe_req = 0.0
    side_safe = False
    left_safe = False
    right_safe = False
    obstacle_detected = 0

    # Initialise condition flags
    sudden_rise = False
    center_blocked = False
    combination_flow = False
    minimum_flow = False

    valid_L = left_count >= config.MIN_FEATURES_PER_ZONE
    valid_C = center_count >= config.MIN_FEATURES_PER_ZONE
    valid_R = right_count >= config.MIN_FEATURES_PER_ZONE

    logger.debug("Flow Magnitudes — L: %.2f, C: %.2f, R: %.2f", smooth_L, smooth_C, smooth_R)

    if handle_grace_period(time_now, navigator, frame_queue, vis_img, param_refs):
        return (
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
        )

    navigator.just_resumed = False

    if len(good_old) < 10:
        state_str = decide_low_feature_action(navigator, smooth_L, smooth_C, smooth_R)
    else:
        pos, yaw, speed = get_drone_state(client)
        brake_thres, dodge_thres = compute_thresholds(speed)
        left_clearing = delta_L < -0.3
        right_clearing = delta_R < -0.3

        probe_reliable = probe_count > config.MIN_PROBE_FEATURES and probe_mag > 0.05

        left_safe, right_safe, side_safe = determine_side_safety(
            smooth_L, smooth_R, brake_thres, left_count, center_count, right_count
        )

        # Use hysteresis-based obstacle detection instead of raw detection
        detection_result = detect_obstacle_with_hysteresis(
            navigator, smooth_C, delta_C, center_count, brake_thres
        )

        if isinstance(detection_result, tuple):
            obstacle_detected, sudden_rise, center_blocked, combination_flow, minimum_flow = detection_result
            obstacle_detected = int(obstacle_detected)
        else:
            obstacle_detected = int(detection_result)

        action = handle_obstacle(
            navigator,
            obstacle_detected,
            side_safe,
            left_safe,
            right_safe,
            left_clearing,
            right_clearing,
            smooth_L,
            smooth_C,
            smooth_R,
            left_count,
            right_count,
        )
        if action != "none":
            state_str = action

    if state_str == "none":
        state_str = recovery_actions(
            navigator,
            speed,
            smooth_C,
            smooth_L,
            smooth_R,
            brake_thres,
            time_now,
            frame_count,
        )

    param_refs.state[0] = state_str

    state_str = update_dodge_history(
        client,
        state_history,
        pos_history,
        state_str,
        navigator,
        smooth_L,
        smooth_C,
        smooth_R,
        param_refs,
    )

    pos, yaw, speed = get_drone_state(client)
    brake_thres, dodge_thres = compute_thresholds(speed)
    return (
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
    )


def apply_navigation_decision(
    client,
    navigator,
    flow_history,
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
    frame_queue,
    vis_img,
    time_now,
    frame_count,
    prev_state,
    state_history,
    pos_history,
    param_refs,
):
    """Wrapper around :func:`navigation_step` for clarity.

    Returns
    -------
    Tuple[str, int, bool, float, float, float, bool, bool, bool, bool]
        Output of :func:`navigation_step`.
    """
    return navigation_step(
        client,
        navigator,
        flow_history,
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
        frame_queue,
        vis_img,
        time_now,
        frame_count,
        prev_state,
        state_history,
        pos_history,
        param_refs,
        probe_mag=probe_mag,
        probe_count=probe_count,
    )
