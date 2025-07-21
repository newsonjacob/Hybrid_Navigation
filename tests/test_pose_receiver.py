import socket
import struct
import time

from slam_bridge.pose_receiver import PoseReceiver


def _send_pose(host, port, matrix):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((host, port))
        sock.sendall(struct.pack('<12f', *matrix))


def test_start_and_stop():
    receiver = PoseReceiver(host="127.0.0.1", port=0)
    receiver.start()
    assert receiver._thread is not None and receiver._thread.is_alive()
    assert receiver.port != 0
    receiver.stop()
    assert not receiver._thread.is_alive()


def test_receives_pose():
    receiver = PoseReceiver(host="127.0.0.1", port=0)
    receiver.start()
    port = receiver.port
    time.sleep(0.1)
    _send_pose("127.0.0.1", port, list(range(12)))
    time.sleep(0.1)
    pose = receiver.get_latest_pose()
    receiver.stop()
    assert pose == (3.0, 7.0, 11.0)


def test_receives_pose_matrix():
    receiver = PoseReceiver(host="127.0.0.1", port=0)
    receiver.start()
    port = receiver.port
    time.sleep(0.1)
    matrix_values = list(range(12))
    _send_pose("127.0.0.1", port, matrix_values)
    time.sleep(0.1)
    matrix = receiver.get_latest_pose_matrix()
    receiver.stop()
    expected = [matrix_values[i * 4 : (i + 1) * 4] for i in range(3)]
    assert matrix == expected

