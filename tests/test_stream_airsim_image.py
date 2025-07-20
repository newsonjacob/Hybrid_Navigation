import socket
import struct
import threading
import sys
from types import SimpleNamespace
import cv2

import numpy as np



def _recvall(conn, n):
    data = b''
    while len(data) < n:
        packet = conn.recv(n - len(data))
        if not packet:
            break
        data += packet
    return data


def test_image_streamer_sends_headers_and_data(monkeypatch):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    host, port = server.getsockname()
    server.listen(1)
    results = {}

    def handle_client():
        conn, _ = server.accept()
        with conn:
            left_header = _recvall(conn, 12)
            h, w, size = struct.unpack("!III", left_header)
            left_data = _recvall(conn, size)
            right_header = _recvall(conn, 12)
            rh, rw, rsize = struct.unpack("!III", right_header)
            right_data = _recvall(conn, rsize)
            results["left"] = (h, w, left_data)
            results["right"] = (rh, rw, right_data)

    t = threading.Thread(target=handle_client)
    t.start()

    # Provide an AirSim stub before importing the module
    airsim_stub = SimpleNamespace(ImageResponse=object, MultirotorClient=lambda: SimpleNamespace())
    monkeypatch.setitem(sys.modules, "airsim", airsim_stub)
    from slam_bridge.stream_airsim_image import ImageStreamer
    streamer = ImageStreamer(host, port, mode="stereo")
    streamer.connect()

    left_rgb = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    right_rgb = np.arange(12, 24, dtype=np.uint8).reshape(2, 2, 3)
    responses = [
        SimpleNamespace(
            image_data_uint8=left_rgb.tobytes(),
            height=2,
            width=2,
            time_stamp=1,
        ),
        SimpleNamespace(
            image_data_uint8=right_rgb.tobytes(),
            height=2,
            width=2,
            time_stamp=1,
        ),
    ]

    streamer._send_frame(responses)
    streamer.sock.close()
    t.join(timeout=1)
    server.close()

    expected_gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY).tobytes()
    left_gray = np.dot(left_rgb[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
    right_gray = np.dot(right_rgb[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)

    assert results["left"] == (2, 2, left_gray.tobytes())
    assert results["right"] == (2, 2, right_gray.tobytes())
