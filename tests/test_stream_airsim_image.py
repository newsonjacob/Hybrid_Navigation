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
            rgb_header = _recvall(conn, 12)
            h, w, size = struct.unpack("!III", rgb_header)
            rgb_data = _recvall(conn, size)
            depth_header = _recvall(conn, 12)
            dh, dw, dsize = struct.unpack("!III", depth_header)
            depth_data = _recvall(conn, dsize)
            results["rgb"] = (h, w, rgb_data)
            results["depth"] = (dh, dw, depth_data)

    t = threading.Thread(target=handle_client)
    t.start()

    # Provide an AirSim stub before importing the module
    airsim_stub = SimpleNamespace(ImageResponse=object, MultirotorClient=lambda: SimpleNamespace())
    monkeypatch.setitem(sys.modules, "airsim", airsim_stub)
    from slam_bridge.stream_airsim_image import ImageStreamer
    streamer = ImageStreamer(host, port)
    streamer.connect()

    rgb = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    depth = np.arange(4, dtype=np.float32).reshape(2, 2)
    responses = [
        SimpleNamespace(
            image_data_uint8=rgb.tobytes(),
            height=2,
            width=2,
            time_stamp=1,
        ),
        SimpleNamespace(
            image_data_float=depth.flatten().tolist(),
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
    assert results["rgb"] == (2, 2, expected_gray)
    assert results["depth"] == (2, 2, depth.tobytes())
