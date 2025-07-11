import sys
import os
import argparse
import logging
from pathlib import Path
from datetime import datetime

from uav.logging_config import setup_logging

import socket
import struct
import time

from typing import List

import airsim
import numpy as np


class ImageStreamer:
    """Stream RGB and depth frames from AirSim to a TCP server."""

    def __init__(self, host: str, port: int, retries: int = 10) -> None:
        self.host = host
        self.port = port
        self.retries = retries
        self.sock: socket.socket | None = None
        self.client = airsim.MultirotorClient()
        self.frame_index = 0

    def connect(self) -> None:
        for attempt in range(1, self.retries + 1):
            try:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.connect((self.host, self.port))
                logging.info("Connected to SLAM server at %s:%s", self.host, self.port)
                return
            except socket.error as exc:
                logging.warning("Connection attempt %s failed: %s", attempt, exc)
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
        rgb = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8).reshape(
            responses[0].height, responses[0].width, 3
        )
        depth = np.array(responses[1].image_data_float, dtype=np.float32).reshape(
            responses[1].height, responses[1].width
        )

        synced = responses[0].time_stamp == responses[1].time_stamp
        logging.info(
            "Frame %d ts=%d sync=%s rgb=%s depth=%s",
            self.frame_index,
            responses[0].time_stamp,
            synced,
            rgb.shape,
            depth.shape,
        )

        header = struct.pack("!III", responses[0].height, responses[0].width, rgb.nbytes)
        self._send_all(header)
        self._send_all(rgb.tobytes())
        header = struct.pack("!III", responses[1].height, responses[1].width, depth.nbytes)
        self._send_all(header)
        self._send_all(depth.tobytes())
        self.frame_index += 1

    def init_first_frame(self) -> None:
        logging.info("Waiting for valid image from AirSim...")
        for _ in range(10):
            responses = self.client.simGetImages(
                [
                    airsim.ImageRequest("oakd_camera", airsim.ImageType.Scene, False, False),
                    airsim.ImageRequest("oakd_camera", airsim.ImageType.DepthPerspective, True),
                ]
            )
            if responses and responses[0].height > 0 and responses[1].height > 0:
                self._send_frame(responses)
                logging.info("First frame sent")
                return
            logging.warning("No valid image yet...")
            time.sleep(1)
        raise RuntimeError("No valid image received from AirSim")

    def stream_loop(self) -> None:
        while True:
            responses = self.client.simGetImages(
                [
                    airsim.ImageRequest("oakd_camera", airsim.ImageType.Scene, False, False),
                    airsim.ImageRequest("oakd_camera", airsim.ImageType.DepthPerspective, True),
                ]
            )
            if not responses or responses[0].height == 0 or responses[1].height == 0:
                logging.warning("Received empty image frame, skipping")
                time.sleep(0.1)
                continue
            logging.debug(
                "Streaming frame %d ts=%d", self.frame_index, responses[0].time_stamp
            )
            try:
                self._send_frame(responses)
            except Exception as exc:
                logging.exception("Failed to send frame: %s", exc)
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
    parser.add_argument(
        "--retries", type=int, default=int(os.environ.get("CONNECT_RETRIES", "10")),
        help="Number of connection retries"
    )
    return parser.parse_args()


def main() -> None:
    log_name = f"airsim_stream_{datetime.now():%Y%m%d_%H%M%S}.log"
    setup_logging(log_name)
    Path("flags").mkdir(exist_ok=True)
    args = parse_args()
    streamer = ImageStreamer(args.host, args.port, args.retries)
    try:
        streamer.run()
    except KeyboardInterrupt:
        logging.info("Stopping image streaming...")
    finally:
        if streamer.sock:
            streamer.sock.close()


if __name__ == "__main__":
    main()
