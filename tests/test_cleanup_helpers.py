import importlib
import sys
import types
import queue
import io
from pathlib import Path

import pytest


def _reload_nav_loop(monkeypatch):
    airsim_stub = types.SimpleNamespace(ImageRequest=object, ImageType=object)
    monkeypatch.setitem(sys.modules, 'airsim', airsim_stub)
    runtime = importlib.import_module('uav.nav_runtime')
    importlib.reload(runtime)
    analysis = importlib.import_module('uav.nav_analysis')
    importlib.reload(analysis)
    ns = types.SimpleNamespace(**{k: getattr(runtime, k) for k in dir(runtime) if not k.startswith('_')})
    for k in dir(analysis):
        if not k.startswith('_'):
            setattr(ns, k, getattr(analysis, k))
    return ns


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
    monkeypatch.setattr('uav.nav_analysis.retain_recent_views', lambda *a, **k: calls.append(('retain', a, k)))
    monkeypatch.setattr('uav.slam_utils.generate_pose_comparison_plot', lambda: calls.append('pose_plot'))
    nl.STOP_FLAG_PATH = tmp_path/'stop.flag'
    nl.STOP_FLAG_PATH.write_text('1')
    monkeypatch.setattr('uav.paths.STOP_FLAG_PATH', nl.STOP_FLAG_PATH, raising=False)
    log_dir = Path('flow_logs')
    log_dir.mkdir(exist_ok=True)
    (log_dir / 'reactive_log_1234.csv').write_text('x' * 200)
    ctx = types.SimpleNamespace(timestamp='1234')
    nl.finalise_files(ctx)
    assert any('analysis/performance_plots.py' in ' '.join(c) for c in calls)
    assert any('analysis/analyse.py' in ' '.join(c) for c in calls)
    assert 'pose_plot' in calls
    assert not nl.STOP_FLAG_PATH.exists()


def test_finalise_files_slam(monkeypatch, tmp_path):
    nl = _reload_nav_loop(monkeypatch)
    calls = []
    monkeypatch.setattr(nl.subprocess, 'run', lambda cmd, **kw: calls.append(cmd))
    monkeypatch.setattr('uav.nav_analysis.retain_recent_views', lambda *a, **k: calls.append(('retain', a, k)))
    monkeypatch.setattr('uav.slam_utils.generate_pose_comparison_plot', lambda: calls.append('pose_plot'))
    nl.STOP_FLAG_PATH = tmp_path / 'stop.flag'
    nl.STOP_FLAG_PATH.write_text('1')
    monkeypatch.setattr('uav.paths.STOP_FLAG_PATH', nl.STOP_FLAG_PATH, raising=False)
    log_dir = Path('flow_logs')
    log_dir.mkdir(exist_ok=True)
    (log_dir / 'slam_log_5678.csv').write_text('x' * 200)
    ctx = types.SimpleNamespace(timestamp='5678')
    nl.finalise_files(ctx)
    assert any('analysis/performance_plots.py' in ' '.join(c) for c in calls)
    assert any('analysis/analyse.py' in ' '.join(c) for c in calls)
    assert 'pose_plot' in calls
    assert not nl.STOP_FLAG_PATH.exists()

def test_finalise_files_calledprocesserror(monkeypatch, tmp_path, caplog):
    nl = _reload_nav_loop(monkeypatch)

    def raise_error(cmd, **kwargs):
        raise nl.subprocess.CalledProcessError(1, cmd, stderr="fail")

    monkeypatch.setattr(nl.subprocess, 'run', raise_error)
    monkeypatch.setattr('uav.nav_analysis.retain_recent_views', lambda *a, **k: None)
    monkeypatch.setattr('uav.slam_utils.generate_pose_comparison_plot', lambda: None)

    nl.STOP_FLAG_PATH = tmp_path / 'stop.flag'
    monkeypatch.setattr('uav.paths.STOP_FLAG_PATH', nl.STOP_FLAG_PATH, raising=False)
    log_dir = Path('flow_logs')
    log_dir.mkdir(exist_ok=True)
    (log_dir / 'reactive_log_ts.csv').write_text('x' * 200)
    ctx = types.SimpleNamespace(timestamp='ts')

    with caplog.at_level(nl.logging.WARNING):
        nl.finalise_files(ctx)

    assert any('fail' in record.message for record in caplog.records)


def test_context_managers(monkeypatch):
    nl = _reload_nav_loop(monkeypatch)
    calls = []

    monkeypatch.setattr('uav.nav_runtime.shutdown_threads', lambda ctx: calls.append('threads'))
    monkeypatch.setattr('uav.nav_runtime.close_logging', lambda ctx: calls.append('logging'))
    monkeypatch.setattr('uav.nav_analysis.finalise_files', lambda ctx: calls.append('finalise'))
    monkeypatch.setattr('uav.nav_runtime.shutdown_airsim', lambda client: calls.append('airsim'))

    proc = types.SimpleNamespace(
        terminate=lambda: calls.append('terminate'),
        wait=lambda timeout=5: calls.append(('wait', timeout)),
    )

    nl.cleanup('client', proc, 'ctx')

    assert calls == [
        'threads',
        'logging',
        'finalise',
        'airsim',
        'terminate',
        ('wait', 5),
    ]


def test_logging_context(monkeypatch):
    nl = _reload_nav_loop(monkeypatch)
    calls = []
    monkeypatch.setattr('uav.nav_runtime.close_logging', lambda ctx: calls.append('closed'))
    with nl.LoggingContext('ctx'):
        calls.append('inside')
    assert calls == ['inside', 'closed']


def test_simulation_process(monkeypatch):
    nl = _reload_nav_loop(monkeypatch)
    calls = []
    proc = types.SimpleNamespace(
        terminate=lambda: calls.append('term'),
        wait=lambda timeout=5: calls.append(('wait', timeout)),
        kill=lambda: calls.append('kill'),
    )
    with nl.SimulationProcess(proc):
        calls.append('run')
    assert calls[:2] == ['run', 'term']
    assert ('wait', 5) in calls
