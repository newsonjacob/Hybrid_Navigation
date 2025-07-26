"""Utilities for logging frames and managing retention."""

import logging
import time
from datetime import datetime
from pathlib import Path

import cv2

from uav import config
from uav.video_utils import start_video_writer_thread
from uav.logging_utils import format_log_line
from uav.utils import retain_recent_logs, retain_recent_files, get_drone_state
from uav.performance import get_cpu_percent, get_memory_info
from uav.overlay import draw_overlay
from uav.perception import FlowHistory
from uav.navigation import Navigator

logger = logging.getLogger(__name__)


def log_frame_data(log_file, log_buffer, line, force_flush=False):
    """Buffer log lines and periodically flush to disk."""
    log_buffer.append(line)

    # Write immediately if forced or buffer is full
    if force_flush or len(log_buffer) >= config.LOG_INTERVAL:
        try:
            log_file.writelines(log_buffer)
            log_file.flush()  # Force immediate write to disk
            log_buffer.clear()
            if force_flush:
                logger.debug(f"Forced flush of {len(log_buffer)} log lines")
        except Exception as e:
            logger.error(f"Failed to write log data: {e}")


def write_video_frame(queue, frame):
    """Queue a video frame for asynchronous writing."""
    try:
        queue.put_nowait(frame)
    except Exception:
        pass


def write_frame_output(
    client,
    vis_img,
    frame_queue,
    loop_start,
    frame_duration,
    fps_list,
    start_time,
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
    param_refs,
    log_file,
    log_buffer,
    state_str,
    obstacle_detected,
    side_safe,
    brake_thres,
    simgetimage_s,
    decode_s,
    processing_s,
    flow_std,
    sudden_rise,
    center_blocked,
    combination_flow,
    minimum_flow,
    slam_pos=None,
):
    """Overlay telemetry, write video, and log the frame."""
    pos, yaw, speed = get_drone_state(client)
    collision = client.simGetCollisionInfo()
    collided = int(getattr(collision, "has_collided", False))
    
    vis_img = draw_overlay(
        vis_img,
        frame_count,
        speed,
        param_refs.state[0],
        time_now - start_time,
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
        in_grace=in_grace,
    )
    
    write_video_frame(frame_queue, vis_img)
    elapsed = time.time() - loop_start
    if elapsed < frame_duration:
        time.sleep(frame_duration - elapsed)
    
    loop_elapsed = time.time() - loop_start
    actual_fps = 1 / max(loop_elapsed, 1e-6)
    loop_start = time.time()
    cpu_percent = get_cpu_percent()
    mem_rss = get_memory_info().rss
    fps_list.append(actual_fps)
    
    state_str_name = state_str.name if hasattr(state_str, "name") else str(state_str)
    
    log_line = format_log_line(
        frame_count,
        smooth_L,
        smooth_C,
        smooth_R,
        delta_L,
        delta_C,
        delta_R,
        flow_std,
        left_count,
        center_count,
        right_count,
        brake_thres,
        actual_fps,
        state_str_name,
        collided,
        obstacle_detected,
        side_safe,
        pos,
        yaw,
        speed,
        time_now,
        good_old,
        simgetimage_s,
        decode_s,
        processing_s,
        loop_elapsed,
        cpu_percent,
        mem_rss,
        sudden_rise,
        center_blocked,
        combination_flow,
        minimum_flow,
    )
    
    # Force flush every 25 frames to ensure data is saved
    force_flush = (frame_count % 25 == 0)
    log_frame_data(log_file, log_buffer, log_line, force_flush)
    
    # Add debug logging for critical frames
    if frame_count % 50 == 0:
        logger.info(f"Logged frame {frame_count} - Buffer size: {len(log_buffer)}")
    
    return loop_start


def handle_reset(client, ctx, frame_count):
    """Reset simulation and restart logging/video."""
    param_refs = ctx.param_refs
    flow_history = ctx.flow_history
    navigator = ctx.navigator
    frame_queue = ctx.frame_queue
    video_thread = ctx.video_thread
    out = ctx.out
    log_file = ctx.log_file
    log_buffer = ctx.log_buffer
    fourcc = ctx.fourcc

    logger.info("Resetting simulation...")
    try:
        client.landAsync().join()
        client.reset()
        client.enableApiControl(True)
        client.armDisarm(True)
        client.takeoffAsync().join()
        client.moveToPositionAsync(0, 0, -2, 2).join()
    except Exception as e:
        logger.error("Reset error: %s", e)
    ctx.flow_history, ctx.navigator, frame_count = FlowHistory(), Navigator(client), 0
    param_refs.reset_flag[0] = False
    finalize_logging(log_file, log_buffer)
    ctx.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    timestamp = ctx.timestamp
    base_dir = Path(getattr(ctx, "output_dir", "."))
    flow_dir = base_dir / "flow_logs"
    flow_dir.mkdir(parents=True, exist_ok=True)
    log_path = flow_dir / f"reactive_log_{timestamp}.csv"
    with open(log_path, 'w') as new_log:
        new_log.write(
            "frame,flow_left,flow_center,flow_right,"
            "delta_left,delta_center,delta_right,flow_std,"
            "left_count,center_count,right_count,"
            "brake_thres,fps,"
            "state,collided,obstacle,side_safe,"
            "pos_x,pos_y,pos_z,yaw,speed,"
            "time,features,simgetimage_s,decode_s,processing_s,loop_s,cpu_percent,memory_rss,"
            "sudden_rise,center_blocked,combination_flow,minimum_flow\n"
        )
    log_file = open(log_path, 'a')
    ctx.log_file = log_file
    retain_recent_logs(str(flow_dir))
    retain_recent_logs(str(base_dir / "logs"))
    retain_recent_files(str(base_dir / "analysis"), "slam_traj_*.html", keep=5)
    retain_recent_files(str(base_dir / "analysis"), "slam_output_*.mp4", keep=5)

    frame_queue.put(None)
    video_thread.join()
    out.release()
    out = cv2.VideoWriter(config.VIDEO_OUTPUT, fourcc, config.VIDEO_FPS, config.VIDEO_SIZE)
    ctx.out = out
    video_thread = start_video_writer_thread(frame_queue, out, ctx.exit_flag)
    ctx.video_thread = video_thread
    return frame_count


def finalize_logging(log_file, log_buffer):
    """Flush any remaining log data and close file properly."""
    if log_buffer and log_file:
        try:
            logger.info(f"Final flush of {len(log_buffer)} log lines")
            log_file.writelines(log_buffer)
            log_file.flush()
            log_buffer.clear()
        except Exception as e:
            logger.error(f"Failed to flush final log data: {e}")
    
    if log_file:
        try:
            log_file.close()
            logger.info("Log file closed successfully")
        except Exception as e:
            logger.error(f"Failed to close log file: {e}")

class ThreadManager:
    """Context manager for shutting down worker threads."""

    def __init__(self, ctx):
        self.ctx = ctx

    def __enter__(self):
        return self.ctx

    def __exit__(self, exc_type, exc, tb):
        from uav.nav_runtime import shutdown_threads
        shutdown_threads(self.ctx)


class LoggingContext:
    """Context manager for flushing and closing log resources."""

    def __init__(self, ctx):
        self.ctx = ctx

    def __enter__(self):
        return self.ctx

    def __exit__(self, exc_type, exc, tb):
        from uav.nav_runtime import close_logging
        close_logging(self.ctx)
