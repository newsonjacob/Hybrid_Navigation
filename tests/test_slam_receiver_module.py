import socket
import struct
import time
import sys

from slam_bridge.slam_receiver import SlamReceiver
import types


def _send_pose(port, matrix, cov=0.1, inliers=50):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect(("127.0.0.1", port))
        sock.sendall(struct.pack('<12f', *matrix))
        sock.sendall(struct.pack('<f', cov))
        sock.sendall(struct.pack('<i', inliers))


def test_slam_receiver_parses_and_handles_disconnects(monkeypatch, tmp_path):
    class DummyClient:
        def confirmConnection(self):
            pass

        def simGetVehiclePose(self):
            pos = types.SimpleNamespace(x_val=0.0, y_val=0.0, z_val=0.0)
            return types.SimpleNamespace(position=pos)

    monkeypatch.setitem(
        sys.modules,
        "airsim",
        types.SimpleNamespace(MultirotorClient=lambda: DummyClient()),
    )

    monkeypatch.chdir(tmp_path)
    (tmp_path / "flags").mkdir()
    state = ["bootstrap"]
    receiver = SlamReceiver(host="127.0.0.1", port=0, state_ref=state)
    receiver.start()
    port = receiver.port
    time.sleep(0.1)

    _send_pose(port, list(range(12)), cov=0.2, inliers=80)
    time.sleep(0.1)
    assert receiver.get_latest_pose() == (3.0, 7.0, 11.0)

    receiver.stop()
    csv_path = tmp_path / "analysis" / "pose_comparison.csv"
    assert csv_path.exists()
    with open(csv_path) as f:
        header = f.readline().strip().split(",")
        row = f.readline().strip().split(",")

    assert "covariance" in header and "inliers" in header and "slam_confidence" in header
    assert len(row) == len(header)
    assert row[-1] == "bootstrap"

