import subprocess
from datetime import datetime
import importlib, sys, types

if 'pygetwindow' not in sys.modules:
    sys.modules['pygetwindow'] = types.SimpleNamespace(getAllTitles=lambda: [])

import launch_all

class DummyProc:
    pass


def test_start_streamer_invokes_subprocess(monkeypatch):
    calls = []
    monkeypatch.setattr(subprocess, "Popen", lambda cmd: calls.append(cmd) or DummyProc())
    proc = launch_all.start_streamer("127.0.0.1", 5555)
    assert isinstance(proc, DummyProc)
    assert calls and calls[0][0] == "python"
    assert "--host" in calls[0] and "--port" in calls[0]


def test_launch_slam_backend_invokes_subprocess(monkeypatch):
    calls = []
    monkeypatch.setattr(subprocess, "Popen", lambda cmd: calls.append(cmd) or DummyProc())
    proc = launch_all.launch_slam_backend("1.2.3.4", 6001)
    assert isinstance(proc, DummyProc)
    assert calls and "POSE_RECEIVER_IP=1.2.3.4" in calls[0][-1]


def test_record_slam_video_returns_path(monkeypatch):
    calls = []
    monkeypatch.setattr(subprocess, "Popen", lambda cmd: calls.append(cmd) or DummyProc())
    monkeypatch.setattr(launch_all.gw, "getAllTitles", lambda: ["ORB-SLAM2"])

    class DummyDT:
        @staticmethod
        def now():
            return datetime(2020, 1, 1, 0, 0, 0)
    monkeypatch.setattr(launch_all, "datetime", DummyDT)

    proc, path = launch_all.record_slam_video("ORB-SLAM2", duration=5)
    assert isinstance(proc, DummyProc)
    assert path.endswith("20200101_000000.mp4")
    assert calls and calls[0][0] == "ffmpeg"

def test_wait_helpers_cancel(tmp_path, monkeypatch):
    flags = tmp_path / "flags"
    flags.mkdir()
    stop = flags / "stop.flag"
    monkeypatch.setattr(launch_all, "STOP_FLAG", stop, raising=False)
    stop.touch()

    assert launch_all.wait_for_flag(flags / "dummy.flag", timeout=0.1) is False

    monkeypatch.setattr(launch_all.gw, "getAllTitles", lambda: [])
    assert launch_all.wait_for_window("dummy", timeout=0.1) is False

    monkeypatch.setattr(launch_all.socket, "create_connection", lambda *a, **k: (_ for _ in ()).throw(OSError()))
    assert launch_all.wait_for_port("127.0.0.1", 1234, timeout=0.1) is False
    