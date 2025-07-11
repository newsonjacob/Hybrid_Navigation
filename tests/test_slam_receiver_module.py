import socket
import struct
import time

from slam_bridge.slam_receiver import SlamReceiver


def _send_pose(port, matrix):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect(("127.0.0.1", port))
        sock.sendall(struct.pack('<12f', *matrix))


def test_slam_receiver_parses_and_handles_disconnects():
    receiver = SlamReceiver(host="127.0.0.1", port=0)
    receiver.start()
    port = receiver.port
    time.sleep(0.1)

    _send_pose(port, list(range(12)))
    time.sleep(0.1)
    assert receiver.get_latest_pose() == (3.0, 7.0, 11.0)

    _send_pose(port, list(range(12, 24)))
    time.sleep(0.1)
    assert receiver.get_latest_pose() == (15.0, 19.0, 23.0)

    receiver.stop()
