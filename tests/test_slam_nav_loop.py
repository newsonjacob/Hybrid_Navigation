import importlib
import sys
import types
import unittest.mock as mock

import tests.conftest  # ensure stubs loaded
from collections import deque
from uav.context import NavContext

def test_slam_navigation_calls_navigator(monkeypatch):
    airsim_stub = types.SimpleNamespace(
        ImageRequest=object,
        ImageType=object,
        DrivetrainType=types.SimpleNamespace(ForwardOnly=1),
        YawMode=lambda *a, **k: None,
    )
    monkeypatch.setitem(sys.modules, 'airsim', airsim_stub)
    nl = importlib.import_module('uav.nav_loop')
    importlib.reload(nl)

    import slam_bridge.slam_receiver as sr
    import slam_bridge.frontier_detection as fd
    monkeypatch.setattr(sr, 'get_latest_pose', lambda: (0.0, 0.0, -2.0))
    monkeypatch.setattr(sr, 'get_pose_history', lambda: [])
    monkeypatch.setattr(fd, 'detect_frontiers', lambda m: nl.np.empty((0, 3)))
    monkeypatch.setattr(nl, 'is_obstacle_ahead', lambda *a, **k: (False, None))

    dummy_future = types.SimpleNamespace(join=lambda *a, **k: None)
    client = types.SimpleNamespace(
        simGetCollisionInfo=lambda: types.SimpleNamespace(has_collided=False),
        moveByVelocityAsync=lambda *a, **k: dummy_future,
        moveToPositionAsync=lambda *a, **k: dummy_future,
        hoverAsync=lambda *a, **k: dummy_future,
        landAsync=lambda *a, **k: dummy_future,
    )

    navigator = nl.Navigator(client)
    slam_mock = mock.MagicMock(return_value='slam_nav')
    monkeypatch.setattr(navigator, 'slam_to_goal', slam_mock)

    ctx = NavContext(
        exit_flag=None,
        param_refs=None,
        tracker=None,
        flow_history=None,
        navigator=navigator,
        state_history=deque(),
        pos_history=deque(),
        frame_queue=None,
        video_thread=None,
        out=None,
        log_file=None,
        log_buffer=[],
        timestamp="",
        start_time=0.0,
        fps_list=[],
        fourcc=None,
    )
    args = types.SimpleNamespace(max_duration=0, goal_x=1.0, goal_y=2.0, goal_z=-2.0)

    result = nl.slam_navigation_loop(args, client, ctx)

    assert result == 'slam_nav'
    slam_mock.assert_called_once_with((0.0, 0.0, -2.0), (1.0, 2.0, -2.0))


def test_slam_bootstrap_runs_when_tracking_lost(monkeypatch):
    airsim_stub = types.SimpleNamespace(
        ImageRequest=object,
        ImageType=object,
        DrivetrainType=types.SimpleNamespace(ForwardOnly=1),
        YawMode=lambda *a, **k: None,
    )
    monkeypatch.setitem(sys.modules, "airsim", airsim_stub)
    nl = importlib.import_module("uav.nav_loop")
    importlib.reload(nl)

    import slam_bridge.slam_receiver as sr
    import slam_bridge.frontier_detection as fd
    monkeypatch.setattr(sr, "get_latest_pose", lambda: (0.0, 0.0, -2.0))
    monkeypatch.setattr(sr, "get_pose_history", lambda: [])
    monkeypatch.setattr(fd, "detect_frontiers", lambda m: nl.np.empty((0, 3)))
    monkeypatch.setattr(nl, "is_obstacle_ahead", lambda *a, **k: (False, None))

    monkeypatch.setattr(nl, "is_slam_stable", lambda: False)
    called = {}

    def fake_boot(*a, **k):
        called["run"] = True

    monkeypatch.setattr(nl, "run_slam_bootstrap", fake_boot)
    monkeypatch.setattr(nl.time, "sleep", lambda *_: None)

    class Flag:
        def __init__(self):
            self.calls = 0

        def is_set(self):
            self.calls += 1
            return self.calls > 1

    dummy_future = types.SimpleNamespace(join=lambda *a, **k: None)
    client = types.SimpleNamespace(
        simGetCollisionInfo=lambda: types.SimpleNamespace(has_collided=False),
        moveByVelocityAsync=lambda *a, **k: dummy_future,
        moveToPositionAsync=lambda *a, **k: dummy_future,
        hoverAsync=lambda *a, **k: dummy_future,
        landAsync=lambda *a, **k: dummy_future,
        simGetVehiclePose=lambda *a, **k: types.SimpleNamespace(
            position=types.SimpleNamespace(x_val=0, y_val=0, z_val=-2)
        ),
    )

    navigator = nl.Navigator(client)
    monkeypatch.setattr(navigator, "slam_to_goal", lambda *a, **k: "slam_nav")

    ctx = NavContext(
        exit_flag=Flag(),
        param_refs=None,
        tracker=None,
        flow_history=None,
        navigator=navigator,
        state_history=deque(),
        pos_history=deque(),
        frame_queue=None,
        video_thread=None,
        out=None,
        log_file=None,
        log_buffer=[],
        timestamp="",
        start_time=0.0,
        fps_list=[],
        fourcc=None,
    )

    args = types.SimpleNamespace(max_duration=1, goal_x=1.0, goal_y=2.0, goal_z=-2.0)

    nl.slam_navigation_loop(args, client, ctx)

    assert called.get("run") is True
