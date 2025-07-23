import importlib
import sys
import types
import queue
import io
import types

import pytest


def _reload_nav_loop(monkeypatch):
    airsim_stub = types.SimpleNamespace(ImageRequest=object, ImageType=object)
    monkeypatch.setitem(sys.modules, 'airsim', airsim_stub)
    nl = importlib.import_module('uav.nav_loop')
    importlib.reload(nl)
    return nl


class DummyFlag:
    def __init__(self):
        self.set_called = False
    def set(self):
        self.set_called = True


class DummyThread:
    def __init__(self):
        self.join_called = False
    def join(self):
        self.join_called = True


class DummyWriter:
    def __init__(self):
        self.released = False
    def release(self):
        self.released = True


class DummyFuture:
    def __init__(self):
        self.join_called = False
    def join(self):
        self.join_called = True


class DummyClient:
    def __init__(self):
        self.landed = False
        self.api_disabled = False
    def landAsync(self):
        self.landed = True
        return DummyFuture()
    def armDisarm(self, val):
        self.armed = val
    def enableApiControl(self, val):
        self.api_disabled = not val


def test_shutdown_threads(monkeypatch):
    nl = _reload_nav_loop(monkeypatch)
    ctx = types.SimpleNamespace(
        exit_flag=DummyFlag(),
        frame_queue=queue.Queue(),
        video_thread=DummyThread(),
        perception_thread=DummyThread(),
    )
    nl.shutdown_threads(ctx)
    assert ctx.exit_flag.set_called
    assert ctx.frame_queue.get_nowait() is None
    assert ctx.video_thread.join_called
    assert ctx.perception_thread.join_called


def test_close_logging(monkeypatch):
    nl = _reload_nav_loop(monkeypatch)
    log = io.StringIO()
    buffer = ['x']
    ctx = types.SimpleNamespace(out=DummyWriter(), log_file=log, log_buffer=buffer)
    nl.close_logging(ctx)
    assert ctx.out.released
    assert buffer == []
    assert log.closed


def test_shutdown_airsim(monkeypatch):
    nl = _reload_nav_loop(monkeypatch)
    client = DummyClient()
    nl.shutdown_airsim(client)
    assert client.landed
    assert client.api_disabled


def test_finalise_files(monkeypatch, tmp_path):
    nl = _reload_nav_loop(monkeypatch)
    calls = []
    monkeypatch.setattr(nl.subprocess, 'run', lambda cmd, **kw: calls.append(cmd))
    monkeypatch.setattr(nl, 'retain_recent_views', lambda *a, **k: calls.append(('retain', a, k)))
    monkeypatch.setattr('uav.slam_utils.generate_pose_comparison_plot', lambda: calls.append('pose_plot'))
    nl.STOP_FLAG_PATH = tmp_path/'stop.flag'
    nl.STOP_FLAG_PATH.write_text('1')
    ctx = types.SimpleNamespace(timestamp='1234')
    nl.finalise_files(ctx)
    assert any('analysis.visualise_flight' in ' '.join(c) for c in calls)
    assert any('analysis.analyse' in ' '.join(c) for c in calls)
    assert 'pose_plot' in calls
    assert not nl.STOP_FLAG_PATH.exists()

def test_finalise_files_calledprocesserror(monkeypatch, tmp_path, caplog):
    nl = _reload_nav_loop(monkeypatch)

    def raise_error(cmd, **kwargs):
        raise nl.subprocess.CalledProcessError(1, cmd, stderr="fail")

    monkeypatch.setattr(nl.subprocess, 'run', raise_error)
    monkeypatch.setattr(nl, 'retain_recent_views', lambda *a, **k: None)
    monkeypatch.setattr('uav.slam_utils.generate_pose_comparison_plot', lambda: None)

    nl.STOP_FLAG_PATH = tmp_path / 'stop.flag'
    ctx = types.SimpleNamespace(timestamp='ts')

    with caplog.at_level(nl.logging.ERROR):
        nl.finalise_files(ctx)

    assert any('fail' in record.message for record in caplog.records)
