import subprocess
from datetime import datetime
import subprocess
import sys
import types
from pathlib import Path

if 'pygetwindow' not in sys.modules:
    sys.modules['pygetwindow'] = types.SimpleNamespace(getAllTitles=lambda: [], getAllWindows=lambda: [])

from uav import launch_utils as lutils

class DummyProc:
    pass


def test_start_streamer_invokes_subprocess(monkeypatch):
    calls = []
    monkeypatch.setattr(subprocess, "Popen", lambda cmd, *a, **k: calls.append(cmd) or DummyProc())
    proc = lutils.start_streamer("127.0.0.1", 5555)
    assert isinstance(proc, DummyProc)
    assert calls and calls[0][0] == "python"
    assert "--host" in calls[0] and "--port" in calls[0]


def test_launch_slam_backend_invokes_subprocess(monkeypatch):
    calls = []
    monkeypatch.setattr(subprocess, "Popen", lambda cmd, *a, **k: calls.append(cmd) or DummyProc())
    proc = lutils.launch_slam_backend("1.2.3.4", 6001)
    assert isinstance(proc, DummyProc)
    assert calls
    bash_cmd = calls[0][-1]
    assert "POSE_RECEIVER_IP=1.2.3.4" in bash_cmd
    assert "SLAM_FLAG_DIR=" in bash_cmd
    assert "SLAM_LOG_DIR=" in bash_cmd
    assert "SLAM_IMAGE_DIR=" in bash_cmd


def test_wait_helpers_cancel(tmp_path, monkeypatch):
    flags = tmp_path / "flags"
    flags.mkdir()
    stop = flags / "stop.flag"
    monkeypatch.setattr(lutils, "STOP_FLAG", stop, raising=False)
    stop.touch()

    assert lutils.wait_for_flag(flags / "dummy.flag", timeout=0.1) is False

    monkeypatch.setattr(lutils, "gw", types.SimpleNamespace(getAllTitles=lambda: []))
    assert lutils.wait_for_window("dummy", timeout=0.1) is False

    monkeypatch.setattr(lutils.socket, "create_connection", lambda *a, **k: (_ for _ in ()).throw(OSError()))
    assert lutils.wait_for_port("127.0.0.1", 1234, timeout=0.1) is False
    stop.unlink()


def test_resize_window(monkeypatch):
    called = {}

    class DummyWin:
        title = "Blocks"

        def resizeTo(self, w, h):
            called["size"] = (w, h)

    monkeypatch.setattr(lutils, "gw", types.SimpleNamespace(getAllWindows=lambda: [DummyWin()]))
    monkeypatch.setattr(lutils, "STOP_FLAG", Path("does_not_exist.flag"), raising=False)
    assert lutils.resize_window("Blocks", 800, 600) is True
    assert called["size"] == (800, 600)
