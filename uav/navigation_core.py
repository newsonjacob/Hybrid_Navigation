"""Core navigation helpers and step logic."""

import logging

from uav.navigation_rules import compute_thresholds
from uav.state_checks import in_grace_period
from uav.utils import get_drone_state
from uav import config

logger = logging.getLogger(__name__)


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
    """Detect obstacle ahead using optical flow or SLAM/depth."""
    if mode == "reactive":
        if smooth_C is None or delta_C is None or center_count is None or brake_thres is None:
            raise ValueError("Reactive mode requires smooth_C, delta_C, center_count, brake_thres")
        sudden_rise = delta_C > 1 and center_count >= 20
        center_blocked = smooth_C > brake_thres and center_count >= 20
        if sudden_rise or center_blocked or (
            center_count > 100 and (smooth_C > brake_thres * 0.5 or delta_C > 0.5)
        ):
            return True
        if delta_C > 0.5 and smooth_C > brake_thres * 0.5 and center_count > 50:
            return True
        return False
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
    """Determine if the left and right sides are safe for dodging."""
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
    """Handle obstacle avoidance manoeuvres and return an action string."""
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
    """Return True if still in start-up grace period."""
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
    """Choose an action when few optical-flow features are tracked."""
    if smooth_L > 1.5 and smooth_R > 1.5 and smooth_C < 0.2:
        return navigator.brake()
    return navigator.blind_forward()


def update_dodge_history(client, state_history, pos_history, state_str, navigator,
                         smooth_L, smooth_C, smooth_R, param_refs):
    """Extend dodge time if the UAV oscillates without progress."""
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
    """Return a recovery action if no obstacle manoeuvre was triggered."""
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

    The step runs after perception processing and is responsible for:
    1. Handling the initial grace period.
    2. Deciding motion when few features are detected.
    3. Performing obstacle avoidance manoeuvres.
    4. Triggering recovery behaviours if stuck.
    """
    state_str = "none"
    brake_thres = 0.0
    dodge_thres = 0.0
    probe_req = 0.0
    side_safe = False
    left_safe = False
    right_safe = False
    obstacle_detected = 0

    valid_L = left_count >= config.MIN_FEATURES_PER_ZONE
    valid_C = center_count >= config.MIN_FEATURES_PER_ZONE
    valid_R = right_count >= config.MIN_FEATURES_PER_ZONE

    logger.debug("Flow Magnitudes — L: %.2f, C: %.2f, R: %.2f", smooth_L, smooth_C, smooth_R)

    if handle_grace_period(time_now, navigator, frame_queue, vis_img, param_refs):
        return state_str, obstacle_detected, side_safe, brake_thres, dodge_thres, probe_req

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

        obstacle_detected = int(detect_obstacle(smooth_C, delta_C, center_count, brake_thres))

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
    return state_str, obstacle_detected, side_safe, brake_thres, dodge_thres, probe_req


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
    """Wrapper around :func:`navigation_step` for clarity."""
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
