import socket
import struct
import time

from slam_bridge.pose_receiver import PoseReceiver


def _send_pose(host, port, matrix):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((host, port))
        sock.sendall(struct.pack('<12f', *matrix))


def test_start_and_stop():
    with PoseReceiver(host="127.0.0.1", port=0) as receiver:
        assert receiver._thread is not None and receiver._thread.is_alive()
        assert receiver.port != 0
    assert receiver._thread is None or not receiver._thread.is_alive()


def test_receives_pose():
    with PoseReceiver(host="127.0.0.1", port=0) as receiver:
        port = receiver.port
        time.sleep(0.1)
        _send_pose("127.0.0.1", port, list(range(12)))
        time.sleep(0.1)
        pose = receiver.get_latest_pose()
    assert pose == (3.0, 7.0, 11.0)


def test_receives_pose_matrix():
    with PoseReceiver(host="127.0.0.1", port=0) as receiver:
        port = receiver.port
        time.sleep(0.1)
        matrix_values = list(range(12))
        _send_pose("127.0.0.1", port, matrix_values)
        time.sleep(0.1)
        matrix = receiver.get_latest_pose_matrix()
    expected = [matrix_values[i * 4 : (i + 1) * 4] for i in range(3)]
    assert matrix == expected


def test_context_manager_cleans_up():
    with PoseReceiver(host="127.0.0.1", port=0) as receiver:
        assert receiver._thread is not None and receiver._thread.is_alive()
        time.sleep(0.1)
    assert receiver._sock is None
    assert receiver._thread is None or not receiver._thread.is_alive()

