import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import os
import argparse
import logging

from datetime import datetime

import socket
import struct
import time

from typing import List

import airsim
import numpy as np

# ---Separate Process so requires logging setup---
from uav.logging_config import setup_logging

# Parse the timestamp early for logging
parser = argparse.ArgumentParser()
parser.add_argument("--log-timestamp", type=str, default=None)
args, remaining_argv = parser.parse_known_args()
timestamp = args.log_timestamp or datetime.now().strftime('%Y%m%d_%H%M%S')

# Fix sys.argv so parse_args still works
sys.argv = [sys.argv[0]] + remaining_argv

# Set up module logging
module_logs = {"stream_airsim_image": f"stream_airsim_image_{timestamp}.log"}
setup_logging(log_file=f"launch_{timestamp}.log", module_logs=module_logs, level=logging.DEBUG)
print(f"[stream_airsim_image.py] Logging configured. Writing to logs/stream_airsim_image_{timestamp}.log")

# Then import the rest
logger = logging.getLogger("stream_airsim_image")

class ImageStreamer:
    """Stream RGB + Depth or Stereo RGB images from AirSim to a TCP server."""

    def __init__(self, host: str, port: int, mode: str, retries: int = 10) -> None:
        self.host = host
        self.port = port
        self.mode = mode  # ✅ fixed typo
        self.retries = retries
        self.sock: socket.socket | None = None
        self.client = airsim.MultirotorClient()
        self.frame_index = 0

    def connect(self) -> None:
        logger.info(f"[CONNECT] Attempting to connect to SLAM server at {self.host}:{self.port}")
        for attempt in range(1, self.retries + 1):
            try:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                logger.debug(f"[CONNECT] Socket created (attempt {attempt})")
                self.sock.connect((self.host, self.port))
                logger.info(f"[CONNECT] Connected to SLAM server at {self.host}:{self.port}")
                return
            except socket.error as exc:
                logger.warning(f"[CONNECT] Connection attempt {attempt} failed: {exc}")
                time.sleep(1)
        logger.error(f"[CONNECT] Could not connect to {self.host}:{self.port} after {self.retries} attempts")
        raise ConnectionRefusedError(f"Could not connect to {self.host}:{self.port}")

    def _send_all(self, data: bytes) -> None:
        assert self.sock is not None
        total_sent = 0
        logger.debug(f"[SEND] Sending {len(data)} bytes")
        while total_sent < len(data):
            sent = self.sock.send(data[total_sent:])
            if sent == 0:
                logger.error("[SEND] Socket connection broken during send")
                raise RuntimeError("Socket connection broken")
            total_sent += sent
        logger.debug(f"[SEND] Sent {total_sent} bytes successfully")

    def _send_frame(self, responses: List[airsim.ImageResponse]) -> None:
        logger.debug(f"[FRAME] Preparing to send frame {self.frame_index}")
        if not responses or len(responses) != 2:
            logger.warning("[FRAME] Invalid stereo response count.")
            return

        if len(responses[0].image_data_uint8) == 0 or len(responses[1].image_data_uint8) == 0:
            logger.warning("[FRAME] Empty stereo frame detected. Skipping this frame.")
            return

        for i, resp in enumerate(responses):
            expected_bytes = resp.height * resp.width * 3
            actual_bytes = len(resp.image_data_uint8)
            logger.debug(f"[FRAME] Image {i}: expected {expected_bytes} bytes, got {actual_bytes} bytes")
            if actual_bytes != expected_bytes:
                logger.warning(f"[FRAME] Frame {i} has unexpected size: got {actual_bytes} bytes, expected {expected_bytes}")
                return

        try:
            left = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8).reshape(
                responses[0].height, responses[0].width, 3
            )
            right = np.frombuffer(responses[1].image_data_uint8, dtype=np.uint8).reshape(
                responses[1].height, responses[1].width, 3
            )
            logger.debug(f"[FRAME] Left shape: {left.shape}, Right shape: {right.shape}")
        except ValueError as e:
            logger.exception(f"[FRAME] Image reshape failed — skipping frame: {e}")
            return

        synced = responses[0].time_stamp == responses[1].time_stamp
        logger.info(f"[FRAME] Frame {self.frame_index} ts={responses[0].time_stamp} sync={synced} left={left.shape} right={right.shape}")

        for img_idx, img in enumerate((left, right)):
            header = struct.pack("!III", img.shape[0], img.shape[1], img.nbytes)
            logger.debug(f"[SEND] Sending header for image {img_idx}: height={img.shape[0]}, width={img.shape[1]}, bytes={img.nbytes}")
            self._send_all(header)
            logger.debug(f"[SEND] Sending image {img_idx} data ({img.nbytes} bytes)")
            self._send_all(img.tobytes())

        logger.info(f"[FRAME] Frame {self.frame_index} sent successfully")
        self.frame_index += 1


    def init_first_frame(self) -> None:
        logger.info("[INIT] Waiting for valid image from AirSim...")
        for attempt in range(10):
            responses = self.get_image_pair()
            logger.debug(f"[INIT] Attempt {attempt}: response lengths = {[len(r.image_data_uint8) for r in responses]}")
            if responses and responses[0].height > 0 and responses[1].height > 0:
                logger.info("[INIT] Valid images received, sending first frame")
                self._send_frame(responses)
                logger.info("[INIT] First frame sent")
                return
            logger.warning("[INIT] No valid image yet...")
            time.sleep(1)
        logger.error("[INIT] No valid image received from AirSim after 10 attempts")
        raise RuntimeError("No valid image received from AirSim")

    def get_image_pair(self) -> List[airsim.ImageResponse]:
        logger.debug("[GET] Requesting stereo images from AirSim")
        try:
            requests = [
                airsim.ImageRequest("front_left", airsim.ImageType.Scene, False, False),
                airsim.ImageRequest("front_right", airsim.ImageType.Scene, False, False)
            ]
            responses = self.client.simGetImages(requests)
            logger.debug(f"[GET] Received {len(responses)} responses")
            if any(len(r.image_data_uint8) == 0 for r in responses):
                logger.warning("[GET] One or more images have zero-length data.")
            return responses
        except Exception as e:
            logger.error(f"[GET] Error getting stereo images: {e}")
            return []

    def stream_loop(self) -> None:
        logger.info("[STREAM] Starting main streaming loop")
        while True:
            responses = self.get_image_pair()
            if not responses or responses[0].height == 0 or responses[1].height == 0:
                logger.warning("[STREAM] Received empty image frame, skipping")
                time.sleep(0.1)
                continue
            try:
                logger.debug(f"[STREAM] Sending frame {self.frame_index}")
                self._send_frame(responses)
            except Exception as exc:
                logger.exception(f"[STREAM] Failed to send frame: {exc}")
                break
            time.sleep(0.05)

    def run(self) -> None:
        logger.info("[RUN] Starting streamer")
        self.connect()
        logger.info("[RUN] Connected to SLAM server")
        self.client.confirmConnection()
        logger.info("[RUN] AirSim client connection confirmed")
        self.init_first_frame()
        logger.info("[RUN] First frame initialized")
        self.stream_loop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AirSim image streamer")
    parser.add_argument("--host", default=os.environ.get("SLAM_SERVER_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("SLAM_SERVER_PORT", "6000")))
    parser.add_argument("--retries", type=int, default=int(os.environ.get("CONNECT_RETRIES", "10")),
                        help="Number of connection retries")
    parser.add_argument("--mode", choices=["rgbd", "stereo"], default="rgbd", 
                        help="Streaming mode: 'rgbd' for RGB + Depth or 'stereo' for Left + Right RGB")
        
    return parser.parse_args()


def main() -> None:
    log_name = f"airsim_stream_{datetime.now():%Y%m%d_%H%M%S}.log"
    Path("flags").mkdir(exist_ok=True)
    args = parse_args()
    streamer = ImageStreamer(args.host, args.port, args.mode, args.retries)
    try:
        streamer.run()
    except KeyboardInterrupt:
        logging.info("Stopping image streaming...")
    finally:
        if streamer.sock:
            streamer.sock.close()


if __name__ == "__main__":
    main()
