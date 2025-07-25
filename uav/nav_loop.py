"""Main navigation loop responsible for moving the UAV."""

import logging
import time
import numpy as np
import airsim

import uav.config as uav_config
from uav.nav_runtime import (
    setup_environment,
    check_startup_grace,
    has_landed,
    check_exit_conditions,
    get_perception_data,
    update_navigation_state,
    log_and_record_frame,
    check_slam_stop,
    ensure_stable_slam_pose,
    handle_waypoint_progress,
    ThreadManager,
    LoggingContext,
    SimulationProcess,
    shutdown_threads,
    close_logging,
    shutdown_airsim,
    cleanup,
)
from uav.navigation_core import (
    detect_obstacle,
    determine_side_safety,
    handle_obstacle,
)
from uav.navigation_slam_boot import run_slam_bootstrap
from slam_bridge.slam_receiver import get_latest_pose_matrix, get_pose_history
from uav.slam_utils import COVARIANCE_THRESHOLD, MIN_INLIERS_THRESHOLD
from uav.perception_loop import process_perception_data

logger = logging.getLogger("nav_loop")


def _resolve(cli_val, default_val):
    """Return CLI override if provided, otherwise default."""
    return default_val if cli_val is None else cli_val


def navigation_loop(args, client, ctx):
    """Run the reactive navigation cycle."""
    max_flow_mag = uav_config.MAX_FLOW_MAG
    frame_count = 0
    frame_duration = 1.0 / uav_config.TARGET_FPS
    max_duration = _resolve(args.max_duration, uav_config.MAX_SIM_DURATION)
    goal_x = _resolve(args.goal_x, uav_config.GOAL_X)
    goal_y = _resolve(args.goal_y, uav_config.GOAL_Y)

    logger.info("[NavLoop] Starting navigation loop with args: %s", args)
    loop_start = time.time()

    try:
        while True:
            frame_count += 1
            time_now = time.time()
            if not check_startup_grace(ctx, time_now):
                continue
            data = get_perception_data(ctx)
            if data is None:
                continue
            if not check_exit_conditions(client, ctx, time_now, max_duration, goal_x, goal_y):
                break
            if getattr(ctx, "landing_future", None) is not None:
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
                    continue
                nav_decision = (
                    "landing",
                    0,
                    False,
                    0.0,
                    0.0,
                    False,
                    False,
                    False,
                    False,
                )
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
                if has_landed(client):
                    logger.info("Landing complete.")
                    break
                continue

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

        try:
            from uav.logging_helpers import finalize_logging
            finalize_logging(ctx.log_file, ctx.log_buffer)
            logger.info("Final log data flushed successfully")
        except Exception as e:
            logger.error(f"Failed to finalize logging: {e}")


def slam_navigation_loop(args, client, ctx, config=None, pose_source="slam"):
    """SLAM-based navigation loop with basic obstacle avoidance."""
    exit_flag = getattr(ctx, "exit_flag", None)
    navigator = getattr(ctx, "navigator", None)
    last_action = "none"

    max_duration = _resolve(args.max_duration, uav_config.MAX_SIM_DURATION)
    goal_x = _resolve(args.goal_x, uav_config.GOAL_X)
    goal_y = _resolve(args.goal_y, uav_config.GOAL_Y)
    goal_z = _resolve(getattr(args, "goal_z", None), -2.0)

    start_time = time.time()
    frame_count = 0
    frame_duration = 1.0 / uav_config.TARGET_FPS
    cov_thres = getattr(config, "covariance_threshold", COVARIANCE_THRESHOLD) if config else COVARIANCE_THRESHOLD
    inlier_thres = getattr(config, "inlier_threshold", MIN_INLIERS_THRESHOLD) if config else MIN_INLIERS_THRESHOLD
    threshold = 0.5

    # Store client reference in context for logging
    ctx.client = client

    if max_duration != 0:
        if ctx is not None and getattr(ctx, "param_refs", None):
            ctx.param_refs.state[0] = "bootstrap"
        run_slam_bootstrap(client, duration=6.0)
        time.sleep(1.0)
        if ctx is not None and getattr(ctx, "param_refs", None):
            ctx.param_refs.state[0] = "waypoint_nav"

    if max_duration == 0 and navigator is not None:
        if pose_source == "airsim" and hasattr(client, "simGetVehiclePose"):
            air_pose = client.simGetVehiclePose("UAV")
            pos = air_pose.position
            yaw = airsim.to_eularian_angles(air_pose.orientation)[2]
            cy, sy = np.cos(yaw), np.sin(yaw)
            pose_mat = np.array([
                [cy, -sy, 0, pos.x_val],
                [sy, cy, 0, pos.y_val],
                [0, 0, 1, pos.z_val],
            ])
        else:
            pose_mat = get_latest_pose_matrix()
        return navigator.slam_to_goal(pose_mat, (goal_x, goal_y, goal_z))

    waypoints = [
        (20, 0, -2),
        (20, -2.5, -2),
        (22.5, -2.5, -2),
        (23.5, -0.5, -2),
        (45, 0, -2),
    ]
    current_waypoint_index = 0
    landing_future = None

    try:
        while True:
            frame_count += 1  # Increment frame counter
            time_now = time.time()
            loop_start = time_now  # Track loop timing

            if check_slam_stop(exit_flag, start_time, max_duration):
                if ctx is not None and getattr(ctx, "param_refs", None):
                    ctx.param_refs.state[0] = "landing"
                break

            pose_data = ensure_stable_slam_pose(
                client,
                pose_source,
                cov_thres,
                inlier_thres,
                exit_flag,
                start_time,
                max_duration,
                ctx,
            )
            if pose_data[0] is None:
                logger.info("[SLAMNav] SLAM pose not stable yet, retrying...")
                # Log unstable state
                log_slam_frame(ctx, frame_count, time_now, 0, 0, 0, 
                              current_waypoint_index, 999.0, "unstable", "UNSTABLE")
                continue

            transformed_pose, (x, y, z) = pose_data
            history = get_pose_history()
            map_pts = np.array([[m[0][3], m[1][3], m[2][3]] for _, m in history], dtype=float)
            
            if not check_exit_conditions(client, ctx, time_now, max_duration, goal_x, goal_y):
                break

            # Get the current goal
            curr_goal = waypoints[current_waypoint_index] # (waypoint_x, waypoint_y, waypoint_z)

            # Calculate distance to the current goal
            dist_to_goal = np.sqrt((x - curr_goal[0]) ** 2 + (y - curr_goal[1]) ** 2) 

            # Check if the final waypoint has been reached
            if current_waypoint_index == len(waypoints) - 1 and dist_to_goal < threshold:
                logger.info("[SLAMNav] Final goal reached — landing.")
                # Log final goal reached
                log_slam_frame(ctx, frame_count, time_now, x, y, z, 
                              current_waypoint_index, dist_to_goal, "final_goal", "COMPLETE")
                break

            (waypoint_x, waypoint_y, waypoint_z), current_waypoint_index, dist = handle_waypoint_progress(
                x, y, waypoints, current_waypoint_index, threshold
            )

            if ctx is not None and getattr(ctx, "param_refs", None):
                ctx.param_refs.state[0] = f"waypoint_{current_waypoint_index + 1}"

            last_action = navigator.slam_to_goal(transformed_pose, (waypoint_x, waypoint_y, waypoint_z))
            
            # Log SLAM navigation data
            slam_state = "TRACKING"
            if hasattr(ctx, "param_refs") and ctx.param_refs.state:
                slam_state = ctx.param_refs.state[0].upper()
            
            log_slam_frame(ctx, frame_count, time_now, x, y, z, 
                          current_waypoint_index, dist_to_goal, last_action, slam_state)
            
            # Log waypoint progress every 10 frames
            if frame_count % 10 == 0:
                logger.info(f"[SLAMNav] Frame {frame_count}: Pos({x:.2f},{y:.2f},{z:.2f}) "
                           f"-> WP{current_waypoint_index + 1}({waypoint_x:.1f},{waypoint_y:.1f}) "
                           f"Dist: {dist_to_goal:.2f}m Action: {last_action}")
            
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        logger.info("[SLAMNav] Interrupted by user.")
    finally:
        logger.info("[SLAMNav] SLAM navigation loop finished.")
        
        # Final flush of SLAM data
        try:
            from uav.logging_helpers import finalize_logging
            finalize_logging(ctx.log_file, ctx.log_buffer)
            logger.info("Final SLAM log data flushed successfully")
        except Exception as e:
            logger.error(f"Failed to finalize SLAM logging: {e}")
            
    return last_action


def log_slam_frame(ctx, frame_count, time_now, x, y, z, waypoint_index, dist_to_goal, action, slam_state="OK"):
    """Log SLAM-specific navigation data in a format compatible with analysis tools."""
    if not ctx.log_file:
        return
        
    try:
        # Get drone state for additional context
        client = getattr(ctx, 'client', None)
        speed = 0.0
        yaw = 0.0
        if client:
            try:
                pose = client.simGetVehiclePose("UAV")
                speed = np.linalg.norm([pose.linear_velocity.x_val, 
                                     pose.linear_velocity.y_val, 
                                     pose.linear_velocity.z_val])
                yaw = airsim.to_eularian_angles(pose.orientation)[2]
            except:
                pass
        
        # Create log line compatible with existing analysis format
        # Format: frame,flow_left,flow_center,flow_right,delta_left,delta_center,delta_right,flow_std,
        #         left_count,center_count,right_count,brake_thres,dodge_thres,fps,state,collided,obstacle,side_safe,
        #         pos_x,pos_y,pos_z,yaw,speed,time,features,simgetimage_s,decode_s,processing_s,loop_s,cpu_percent,memory_rss,
        #         sudden_rise,center_blocked,combination_flow,minimum_flow
        
        log_line = (f"{frame_count},"         # frame
                   f"0.0,0.0,0.0,"           # flow_left,flow_center,flow_right (N/A for SLAM)
                   f"0.0,0.0,0.0,"           # delta_left,delta_center,delta_right (N/A)
                   f"0.0,"                   # flow_std (N/A)
                   f"0,0,0,"                 # left_count,center_count,right_count (N/A)
                   f"0.0,0.0,"               # brake_thres,dodge_thres (N/A)
                   f"10.0,"                  # fps (approximate SLAM rate)
                   f"{slam_state}_WP{waypoint_index + 1}," # state (includes waypoint info)
                   f"0,"                     # collided (assume no collision)
                   f"0,"                     # obstacle (N/A for SLAM)
                   f"1,"                     # side_safe (assume safe)
                   f"{x:.6f},{y:.6f},{z:.6f}," # pos_x,pos_y,pos_z (actual SLAM position)
                   f"{yaw:.6f},"             # yaw
                   f"{speed:.6f},"           # speed
                   f"{time_now:.6f},"        # time
                   f"0,"                     # features (could add SLAM feature count later)
                   f"0.0,0.0,0.0,"          # simgetimage_s,decode_s,processing_s (N/A)
                   f"0.1,"                   # loop_s (SLAM loop time ~0.1s)
                   f"0.0,0,"                 # cpu_percent,memory_rss (could add later)
                   f"0,0,"                   # sudden_rise,center_blocked (N/A)
                   f"{dist_to_goal:.6f},"    # combination_flow (repurpose for distance to goal)
                   f"{action}\n")            # minimum_flow (repurpose for action)
        
        # Write to log file
        ctx.log_file.write(log_line)
        
        # Flush every 5 frames for SLAM (more frequent than reactive)
        if frame_count % 5 == 0:
            ctx.log_file.flush()
            logger.debug(f"SLAM frame {frame_count} logged and flushed")
            
    except Exception as e:
        logger.error(f"Failed to log SLAM frame {frame_count}: {e}")
