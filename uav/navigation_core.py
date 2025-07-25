"""Core navigation helpers and step logic."""

import logging
from dataclasses import dataclass
from typing import Any, Deque

from uav.navigation_rules import compute_thresholds
from uav.state_checks import in_grace_period
from uav.utils import get_drone_state
from uav import config
from uav.navigation_state import NavigationState

logger = logging.getLogger(__name__)


@dataclass
class NavigationInput:
    """Container for parameters required by :func:`navigation_step`."""

    good_old: list
    flow_vectors: Any
    flow_std: float
    smooth_L: float
    smooth_C: float
    smooth_R: float
    delta_L: float
    delta_C: float
    delta_R: float
    left_count: int
    center_count: int
    right_count: int
    frame_queue: Any
    vis_img: Any
    time_now: float
    frame_count: int
    state_history: Deque
    pos_history: Deque
    param_refs: Any
    probe_mag: float = 0.0
    probe_count: int = 0


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
    # Retrieve hysteresis counters with sensible defaults for tests
    detection_count = getattr(navigator, "obstacle_detection_count", 0)
    clear_count = getattr(navigator, "obstacle_clear_count", 0)
    confirmed = getattr(navigator, "obstacle_confirmed", False)
    DETECTION_THRESHOLD = getattr(navigator, "DETECTION_THRESHOLD", 1)
    CLEAR_THRESHOLD = getattr(navigator, "CLEAR_THRESHOLD", 1)

    if raw_detection:
        # Obstacle detected - increment detection counter
        detection_count += 1
        clear_count = 0  # Reset clear counter

        # Confirm obstacle only after required number of detections
        if detection_count >= DETECTION_THRESHOLD:
            if not confirmed:
                logger.info(
                    f"[HYSTERESIS] Obstacle CONFIRMED after {detection_count} frames"
                )
            confirmed = True
    else:
        # No obstacle detected - increment clear counter
        clear_count += 1
        detection_count = 0  # Reset detection counter

        # Clear obstacle only after required number of clear frames
        if clear_count >= CLEAR_THRESHOLD and confirmed:
            logger.info(
                f"[HYSTERESIS] Obstacle CLEARED after {clear_count} frames"
            )
            confirmed = False

    # Persist counters back to navigator for stateful Navigator objects
    setattr(navigator, "obstacle_detection_count", detection_count)
    setattr(navigator, "obstacle_clear_count", clear_count)
    setattr(navigator, "obstacle_confirmed", confirmed)
    
    # Store condition states for logging
    setattr(navigator, "last_sudden_rise", sudden_rise)
    setattr(navigator, "last_center_blocked", center_blocked)
    setattr(navigator, "last_combination_flow", combination_flow)
    setattr(navigator, "last_minimum_flow", minimum_flow)
    
    logger.debug(
        f"[HYSTERESIS] Raw: {raw_detection}, Detection count: {detection_count}, "
        f"Clear count: {clear_count}, Confirmed: {confirmed}"
    )
    return confirmed, sudden_rise, center_blocked, combination_flow, minimum_flow


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
    navigator=None,
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

        DELTA_C_SUDDEN = 1.5 # Sudden rise threshold
        CENTER_COUNT_MIN = 5 # Minimum center count
        COMBO_COUNT = 40 # Combination high flow count
        DELTA_C_MIN = 0.75 # Minimum delta C
        SMOOTH_C_MIN = 0.5 # Minimum smooth C
        MIN_COUNT = 30 # Minimum count

        # Calculate individual conditions
        sudden_rise = delta_C > DELTA_C_SUDDEN and center_count >= CENTER_COUNT_MIN
        center_blocked = smooth_C > brake_thres and center_count >= CENTER_COUNT_MIN # and delta_C > 0

        # Apply post-dodge grace period to combination_flow
        combination_flow_raw = center_count > COMBO_COUNT and (smooth_C > brake_thres * 0.5 or delta_C > 1)
        
        # Check if we're in post-dodge grace period
        in_grace = False
        if navigator is not None:
            in_grace = navigator.check_post_dodge_grace()
        
        # Suppress combination_flow during grace period
        if in_grace and combination_flow_raw:
            logger.debug("[GRACE] Suppressing combination_flow detection during post-dodge grace period")
            combination_flow = False
        else:
            combination_flow = combination_flow_raw
            
        minimum_flow = delta_C > DELTA_C_MIN and smooth_C > brake_thres * SMOOTH_C_MIN and center_count > MIN_COUNT

        # Confidence-based
        confidence = sum([sudden_rise, center_blocked, combination_flow, minimum_flow])
        obstacle_detected = confidence >= 1  # require at least two triggers

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
    NavigationState
        Action executed by the navigator.
    """
    state_str = NavigationState.NONE

    if obstacle_detected and not navigator.dodging:
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
        elif right_safe:
            logger.info("\U0001F500 Right safe — Dodging right")
            state_str = navigator.dodge(smooth_L, smooth_C, smooth_R, direction="right")
        elif left_clearing or right_clearing:
            if left_clearing and not right_clearing:
                logger.info("\U0001F6D1 Left clearing — Dodging left")
                state_str = navigator.dodge(smooth_L, smooth_C, smooth_R, direction="left")
            elif right_clearing and not left_clearing:
                logger.info("\U0001F6D1 Right clearing — Dodging right")
                state_str = navigator.dodge(smooth_L, smooth_C, smooth_R, direction="right")
            else:
                if left_count < right_count:
                    logger.info("\U0001F6D1 Both sides clearing — Dodging left")
                    state_str = navigator.dodge(smooth_L, smooth_C, smooth_R, direction="left")
                else:
                    logger.info("\U0001F6D1 Both sides clearing — Dodging right")
                    state_str = navigator.dodge(smooth_L, smooth_C, smooth_R, direction="right")
        else:
            logger.info("\U0001F6D1 No sides safe — Braking")
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
    state_str : NavigationState
        Current action state.
    navigator : Navigator
        Navigator controlling the drone.
    smooth_L, smooth_C, smooth_R : float
        Current smoothed flow magnitudes.
    param_refs : ParamRefs
        Parameter reference object for state updates.

    Returns
    -------
    NavigationState
        Possibly updated action state.
    """
    pos_hist, _, _ = get_drone_state(client)
    state_history.append(state_str)
    pos_history.append((pos_hist.x_val, pos_hist.y_val))
    if len(state_history) == state_history.maxlen:
        if all(s == state_history[-1] for s in state_history) and state_history[-1] in (NavigationState.DODGE_LEFT, NavigationState.DODGE_RIGHT):
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
    NavigationState
        Action chosen to recover progress.
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
    return NavigationState.NONE


def navigation_step(
    client,
    navigator,
    flow_history,
    nav_input: NavigationInput,
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
    nav_input : NavigationInput
        Aggregated inputs for the navigation decision.

    Returns
    -------
    Tuple[NavigationState, int, bool, float, float, bool, bool, bool, bool]
        Tuple containing the selected navigation state, obstacle flag, side
        safety flag, dynamic thresholds and detailed obstacle condition flags.
    """
    good_old = nav_input.good_old
    flow_vectors = nav_input.flow_vectors
    flow_std = nav_input.flow_std
    smooth_L = nav_input.smooth_L
    smooth_C = nav_input.smooth_C
    smooth_R = nav_input.smooth_R
    delta_L = nav_input.delta_L
    delta_C = nav_input.delta_C
    delta_R = nav_input.delta_R
    left_count = nav_input.left_count
    center_count = nav_input.center_count
    right_count = nav_input.right_count
    frame_queue = nav_input.frame_queue
    vis_img = nav_input.vis_img
    time_now = nav_input.time_now
    frame_count = nav_input.frame_count
    state_history = nav_input.state_history
    pos_history = nav_input.pos_history
    param_refs = nav_input.param_refs
    probe_mag = nav_input.probe_mag
    probe_count = nav_input.probe_count
    state_str = NavigationState.NONE
    brake_thres = 0.0
    dodge_thres = 0.0
    side_safe = False
    left_safe = False
    right_safe = False
    obstacle_detected = 0

    # Initialise condition flags
    sudden_rise = False
    center_blocked = False
    combination_flow = False
    minimum_flow = False


    logger.debug("Flow Magnitudes — L: %.2f, C: %.2f, R: %.2f", smooth_L, smooth_C, smooth_R)

    if handle_grace_period(time_now, navigator, frame_queue, vis_img, param_refs):
        return (
            state_str,
            obstacle_detected,
            side_safe,
            brake_thres,
            dodge_thres,
            sudden_rise,
            center_blocked,
            combination_flow,
            minimum_flow,
        )

    navigator.just_resumed = False

    
    pos, yaw, speed = get_drone_state(client)
    brake_thres, dodge_thres = compute_thresholds(speed)
    left_clearing = delta_L < -0.3
    right_clearing = delta_R < -0.3


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
    if action != NavigationState.NONE:
        state_str = action

    if state_str == NavigationState.NONE:
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
        sudden_rise,
        center_blocked,
        combination_flow,
        minimum_flow,
    )


