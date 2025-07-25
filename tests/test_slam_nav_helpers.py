import importlib
import sys
import types
import unittest.mock as mock

import tests.conftest  # ensure stubs loaded


def _load_nav_loop(monkeypatch):
    airsim_stub = types.SimpleNamespace(
        ImageRequest=object,
        ImageType=object,
        DrivetrainType=types.SimpleNamespace(ForwardOnly=1),
        YawMode=lambda *a, **k: None,
        to_eularian_angles=lambda o: (0, 0, 0),
    )
    monkeypatch.setitem(sys.modules, "airsim", airsim_stub)
    nl = importlib.import_module("uav.nav_runtime")
    importlib.reload(nl)
    return nl


def test_handle_waypoint_progress_advances(monkeypatch):
    nl = _load_nav_loop(monkeypatch)
    wps = [(0, 0, -2), (1, 0, -2)]
    goal, idx, dist = nl.handle_waypoint_progress(0.0, 0.0, wps, 0, threshold=0.5)
    assert goal == wps[1]
    assert idx == 1
    assert dist == 0.0


def test_handle_waypoint_progress_stays(monkeypatch):
    nl = _load_nav_loop(monkeypatch)
    wps = [(0, 0, -2), (1, 0, -2)]
    goal, idx, dist = nl.handle_waypoint_progress(0.6, 0.0, wps, 0, threshold=0.5)
    assert goal == wps[0]
    assert idx == 0
    assert dist > 0.0


def test_check_slam_stop(monkeypatch, tmp_path):
    nl = _load_nav_loop(monkeypatch)
    flag = types.SimpleNamespace(is_set=lambda: True)
    assert nl.check_slam_stop(flag, 0, 10) is True

    flag = types.SimpleNamespace(is_set=lambda: False)
    nl.STOP_FLAG_PATH = tmp_path / 'stop.flag'
    nl.STOP_FLAG_PATH.write_text('1')
    assert nl.check_slam_stop(flag, 0, 10) is True

    nl.STOP_FLAG_PATH.unlink()
    assert nl.check_slam_stop(flag, -2, 1) is True


def test_ensure_stable_pose_uses_airsim(monkeypatch):
    nl = _load_nav_loop(monkeypatch)
    pos = types.SimpleNamespace(x_val=1.0, y_val=2.0, z_val=-2.0)
    orientation = object()
    client = types.SimpleNamespace(simGetVehiclePose=lambda name: types.SimpleNamespace(position=pos, orientation=orientation))
    pose, coords = nl.ensure_stable_slam_pose(client, 'airsim', None, None, None, 0, 1)
    assert coords == (1.0, 2.0, -2.0)
    assert nl.np.allclose(pose, [[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 2.0], [0.0, 0.0, 1.0, -2.0]])


def test_ensure_stable_pose_reinitialises(monkeypatch):
    nl = _load_nav_loop(monkeypatch)
    seq = [None, nl.np.eye(4), nl.np.eye(4)]
    monkeypatch.setattr('slam_bridge.slam_receiver.get_latest_pose_matrix', lambda: seq.pop(0))
    stable = [False, True]
    monkeypatch.setattr(nl, 'is_slam_stable', lambda *a, **k: stable.pop(0))
    boot_calls = []
    monkeypatch.setattr(nl, 'run_slam_bootstrap', lambda *a, **k: boot_calls.append(1))
    monkeypatch.setattr(nl.time, 'sleep', lambda *a, **k: None)
    monkeypatch.setattr(nl, 'check_slam_stop', lambda *a, **k: False)
    pose, coords = nl.ensure_stable_slam_pose(types.SimpleNamespace(), 'slam', None, None, None, 0, 10)
    assert boot_calls
    assert pose is not None
