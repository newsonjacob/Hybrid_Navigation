"""Utilities for logging frames and managing retention."""

import logging
import time
from datetime import datetime

import cv2

from uav import config
from uav.video_utils import start_video_writer_thread
from uav.logging_utils import format_log_line
from uav.utils import retain_recent_logs, retain_recent_files, get_drone_state
from uav.overlay import draw_overlay
from uav.perception import FlowHistory
from uav.navigation import Navigator

logger = logging.getLogger(__name__)


def log_frame_data(log_file, log_buffer, line):
    """Buffer log lines and periodically flush to disk."""
    log_buffer.append(line)
    if len(log_buffer) >= config.LOG_INTERVAL:
        log_file.writelines(log_buffer)
        log_buffer.clear()


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
    dodge_thres,
    probe_req,
    simgetimage_s,
    decode_s,
    processing_s,
    flow_std,
):
    """Overlay telemetry, write video, and log the frame."""
    pos, yaw, speed = get_drone_state(client)
    collision = client.simGetCollisionInfo()
    collided = int(getattr(collision, "has_collided", False))
    vis_img = draw_overlay(
        vis_img,
        frame_count,
        speed,
        param_refs['state'][0],
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
    fps_list.append(actual_fps)
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
        dodge_thres,
        probe_req,
        actual_fps,
        state_str,
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
    )
    log_frame_data(log_file, log_buffer, log_line)
    logger.debug("Actual FPS: %.2f", actual_fps)
    logger.debug("Features detected: %d", len(good_old))
    return loop_start


def handle_reset(client, ctx, frame_count):
    """Reset simulation and restart logging/video."""
    param_refs = ctx['param_refs']
    flow_history = ctx['flow_history']
    navigator = ctx['navigator']
    frame_queue = ctx['frame_queue']
    video_thread = ctx['video_thread']
    out = ctx['out']
    log_file = ctx['log_file']
    log_buffer = ctx['log_buffer']
    fourcc = ctx['fourcc']

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
    ctx['flow_history'], ctx['navigator'], frame_count = FlowHistory(), Navigator(client), 0
    param_refs['reset_flag'][0] = False
    if log_buffer:
        log_file.writelines(log_buffer)
        log_buffer.clear()
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
