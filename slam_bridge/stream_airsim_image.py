import sys
import os
import argparse
import logging
from pathlib import Path
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
logger.warning("[TEST] __name__ = %s | handlers = %s", __name__, logger.handlers)


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
        for attempt in range(1, self.retries + 1):
            try:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.connect((self.host, self.port))
                logger.info("Connected to SLAM server at %s:%s", self.host, self.port)
                return
            except socket.error as exc:
                logger.warning("Connection attempt %s failed: %s", attempt, exc)
                time.sleep(1)
        raise ConnectionRefusedError(f"Could not connect to {self.host}:{self.port}")

    def _send_all(self, data: bytes) -> None:
        assert self.sock is not None
        total_sent = 0
        while total_sent < len(data):
            sent = self.sock.send(data[total_sent:])
            if sent == 0:
                raise RuntimeError("Socket connection broken")
            total_sent += sent

    def _send_frame(self, responses: List[airsim.ImageResponse]) -> None:
        if not responses or len(responses) != 2:
            logger.warning("Invalid stereo response count.")
            return

        if len(responses[0].image_data_uint8) == 0 or len(responses[1].image_data_uint8) == 0:
            logger.warning("Empty stereo frame detected. Skipping this frame.")
            return

        for i, resp in enumerate(responses):
            expected_bytes = resp.height * resp.width * 3
            actual_bytes = len(resp.image_data_uint8)
            if actual_bytes != expected_bytes:
                logger.warning("Frame %d has unexpected size: got %d bytes, expected %d", i, actual_bytes, expected_bytes)
                return

        try:
            left = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8).reshape(
                responses[0].height, responses[0].width, 3
            )
            right = np.frombuffer(responses[1].image_data_uint8, dtype=np.uint8).reshape(
                responses[1].height, responses[1].width, 3
            )
        except ValueError as e:
            logger.exception("Image reshape failed — skipping frame: %s", e)
            return

        synced = responses[0].time_stamp == responses[1].time_stamp
        logger.info("Frame %d ts=%d sync=%s left=%s right=%s",
                    self.frame_index, responses[0].time_stamp, synced, left.shape, right.shape)

        for img in (left, right):
            header = struct.pack("!III", img.shape[0], img.shape[1], img.nbytes)
            self._send_all(header)
            self._send_all(img.tobytes())

        self.frame_index += 1


    def init_first_frame(self) -> None:
        logger.info("Waiting for valid image from AirSim...")
        for _ in range(10):
            responses = self.get_image_pair()
            print(f"[DEBUG] init_first_frame response lengths: {[len(r.image_data_uint8) for r in responses]}")
            if responses and responses[0].height > 0 and responses[1].height > 0:
                print("LEFT:", responses[0].image_data_uint8[:10])
                print("RIGHT:", responses[1].image_data_uint8[:10])
                self._send_frame(responses)
                logger.info("First frame sent")
                return
            logger.warning("No valid image yet...")
            time.sleep(1)
        raise RuntimeError("No valid image received from AirSim")

    def get_image_pair(self) -> List[airsim.ImageResponse]:
        try:
            requests = [
                airsim.ImageRequest("front_left", airsim.ImageType.Scene, False, False),
                airsim.ImageRequest("front_right", airsim.ImageType.Scene, False, False)
            ]
            responses = self.client.simGetImages(requests)
            
            if any(len(r.image_data_uint8) == 0 for r in responses):
                print("[WARNING] One or more images have zero-length data.")
            
            return responses
        except Exception as e:
            print(f"[ERROR] Error getting stereo images: {e}")
            return []




    def stream_loop(self) -> None:
        while True:
            responses = self.get_image_pair()
            if not responses or responses[0].height == 0 or responses[1].height == 0:
                logger.warning("Received empty image frame, skipping")
                time.sleep(0.1)
                continue
            try:
                self._send_frame(responses)
            except Exception as exc:
                logger.exception("Failed to send frame: %s", exc)
                break
            time.sleep(0.05)

    def run(self) -> None:
        self.connect()
        self.client.confirmConnection()
        self.init_first_frame()
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
