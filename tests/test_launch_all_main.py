import types
import subprocess
import configparser
import sys

if 'pygetwindow' not in sys.modules:
    sys.modules['pygetwindow'] = types.SimpleNamespace(getAllTitles=lambda: [], getAllWindows=lambda: [])

import launch_all

class DummyProc:
    def __init__(self, cmd=None, *args, **kwargs):
        self.is_main = isinstance(cmd, list) and "main.py" in cmd
    def terminate(self):
        pass
    def kill(self):
        pass
    def wait(self, timeout=None):
        if self.is_main:
            for f in [launch_all.AIRSIM_READY_FLAG, launch_all.START_NAV_FLAG]:
                f.unlink(missing_ok=True)
        return 0

    # Context manager support for subprocess.call
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_launch_all_main_flag_flow(tmp_path, monkeypatch):
    # temporary flags directory
    flags = tmp_path / "flags"
    flags.mkdir()

    monkeypatch.setattr(launch_all, "flags_dir", flags)
    monkeypatch.setattr(launch_all, "AIRSIM_READY_FLAG", flags / "airsim_ready.flag", raising=False)
    monkeypatch.setattr(launch_all, "SLAM_READY_FLAG", flags / "slam_ready.flag", raising=False)
    monkeypatch.setattr(launch_all, "SLAM_FAILED_FLAG", flags / "slam_failed.flag", raising=False)
    monkeypatch.setattr(launch_all, "START_NAV_FLAG", flags / "start_nav.flag", raising=False)
    monkeypatch.setattr(launch_all, "STOP_FLAG", flags / "stop.flag", raising=False)

    created = []

    def fake_wait_for_flag(path, timeout=15):
        path.touch()
        created.append(path.name)
        return True
    monkeypatch.setattr(launch_all, "wait_for_flag", fake_wait_for_flag)

    def fake_wait_for_start_flag():
        launch_all.START_NAV_FLAG.touch()
        created.append(launch_all.START_NAV_FLAG.name)
        return True
    monkeypatch.setattr(launch_all, "wait_for_start_flag", fake_wait_for_start_flag)

    monkeypatch.setattr(launch_all, "wait_for_window", lambda *a, **k: True)
    monkeypatch.setattr(launch_all, "record_slam_video", lambda *a, **k: (DummyProc(), str(tmp_path/"video.mp4")))
    monkeypatch.setattr(launch_all, "start_streamer", lambda *a, **k: DummyProc())
    monkeypatch.setattr(launch_all, "launch_slam_backend", lambda *a, **k: DummyProc())
    monkeypatch.setattr(launch_all.time, "sleep", lambda *_: None)

    monkeypatch.setattr(launch_all.subprocess, "Popen", DummyProc)

    args = types.SimpleNamespace(
        nav_mode="slam",
        slam_server_host="127.0.0.1",
        slam_server_port=6000,
        slam_receiver_host="127.0.0.1",
        slam_receiver_port=6001,
        config="none",
        goal_y=0,
    )
    monkeypatch.setattr(launch_all, "parse_args", lambda: args)
    monkeypatch.setattr(launch_all, "load_app_config", lambda p: configparser.ConfigParser())

    launch_all.main("20240101_000000")

    assert "airsim_ready.flag" in created
    assert "slam_ready.flag" in created
    assert "start_nav.flag" in created

    assert not launch_all.AIRSIM_READY_FLAG.exists()
    assert launch_all.SLAM_READY_FLAG.exists()
    assert not launch_all.START_NAV_FLAG.exists()


class WaitProc(DummyProc):
    def __init__(self, cmd=None, *args, **kwargs):
        super().__init__(cmd, *args, **kwargs)
        self.wait_called = 0
        self.terminated = False
        self.killed = False
        self._alive = True

    def poll(self):
        return None if self._alive else 0

    def wait(self, timeout=None):
        self.wait_called += 1
        self._alive = False
        return 0

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True


def test_stop_flag_waits_for_main(tmp_path, monkeypatch):
    flags = tmp_path / "flags"
    flags.mkdir()

    monkeypatch.setattr(launch_all, "flags_dir", flags)
    monkeypatch.setattr(launch_all, "AIRSIM_READY_FLAG", flags / "airsim_ready.flag", raising=False)
    monkeypatch.setattr(launch_all, "SLAM_READY_FLAG", flags / "slam_ready.flag", raising=False)
    monkeypatch.setattr(launch_all, "SLAM_FAILED_FLAG", flags / "slam_failed.flag", raising=False)
    monkeypatch.setattr(launch_all, "START_NAV_FLAG", flags / "start_nav.flag", raising=False)
    monkeypatch.setattr(launch_all, "STOP_FLAG", flags / "stop.flag", raising=False)

    monkeypatch.setattr(launch_all, "wait_for_flag", lambda *a, **k: True)
    def fake_start():
        launch_all.START_NAV_FLAG.touch()
        launch_all.STOP_FLAG.touch()
        return True
    monkeypatch.setattr(launch_all, "wait_for_start_flag", fake_start)
    monkeypatch.setattr(launch_all, "wait_for_port", lambda *a, **k: True)
    monkeypatch.setattr(launch_all, "wait_for_window", lambda *a, **k: True)
    monkeypatch.setattr(launch_all, "record_slam_video", lambda *a, **k: (DummyProc(), str(tmp_path/"v.mp4")))
    monkeypatch.setattr(launch_all, "start_streamer", lambda *a, **k: DummyProc())
    monkeypatch.setattr(launch_all, "launch_slam_backend", lambda *a, **k: DummyProc())
    monkeypatch.setattr(launch_all.time, "sleep", lambda *_: None)

    proc_holder = {}
    class TrackingWaitProc(WaitProc):
        def __init__(self, cmd=None, *a, **k):
            super().__init__(cmd, *a, **k)
            if self.is_main:
                proc_holder['proc'] = self
    monkeypatch.setattr(launch_all.subprocess, "Popen", TrackingWaitProc)

    args = types.SimpleNamespace(
        nav_mode="slam",
        slam_server_host="127.0.0.1",
        slam_server_port=6000,
        slam_receiver_host="127.0.0.1",
        slam_receiver_port=6001,
        config="none",
        goal_y=0,
    )
    monkeypatch.setattr(launch_all, "parse_args", lambda: args)
    monkeypatch.setattr(launch_all, "load_app_config", lambda p: configparser.ConfigParser())

    launch_all.main("20240101_000001")

    proc = proc_holder['proc']
    assert proc.wait_called == 1
    assert not proc.terminated
    assert not proc.killed


class HangProc(WaitProc):
    """Process that never exits until killed."""
    def wait(self, timeout=None):
        self.wait_called += 1
        raise subprocess.TimeoutExpired(cmd="main.py", timeout=timeout)


def test_shutdown_force_kills_after_timeout(tmp_path, monkeypatch):
    flags = tmp_path / "flags"
    flags.mkdir()

    monkeypatch.setattr(launch_all, "flags_dir", flags)
    monkeypatch.setattr(launch_all, "STOP_FLAG", flags / "stop.flag", raising=False)

    proc = HangProc(["python", "main.py"])
    launcher = launch_all.Launcher(logger=launch_all.logger, timestamp="ts", main_proc=proc)

    launcher.shutdown()

    assert proc.wait_called == 2  # initial grace wait + forced terminate wait
    assert proc.terminated
    assert proc.killed
