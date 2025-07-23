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
    sample_matrix = [[1, 0, 0, 0.0], [0, 1, 0, 0.0], [0, 0, 1, -2.0]]
    monkeypatch.setattr(sr, 'get_latest_pose_matrix', lambda: sample_matrix)
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
    args = types.SimpleNamespace(max_duration=0, goal_x=1.0, goal_y=2.0, goal_z=-2.0, slam_pose_source='slam')

    result = nl.slam_navigation_loop(args, client, ctx, pose_source=args.slam_pose_source)

    assert result == 'slam_nav'
    slam_mock.assert_called_once_with(sample_matrix, (1.0, 2.0, -2.0))

def test_slam_nav_uses_airsim_pose(monkeypatch):
    """Ground-truth AirSim pose should be forwarded to Navigator."""
    dummy_future = types.SimpleNamespace(join=lambda *a, **k: None)
    pos = types.SimpleNamespace(x_val=1.0, y_val=2.0, z_val=-2.0)
    orientation = object()

    airsim_stub = types.SimpleNamespace(
        ImageRequest=object,
        ImageType=object,
        DrivetrainType=types.SimpleNamespace(ForwardOnly=1),
        YawMode=lambda *a, **k: None,
        to_eularian_angles=lambda ori: (0.0, 0.0, 0.0),
    )
    monkeypatch.setitem(sys.modules, "airsim", airsim_stub)
    nl = importlib.import_module("uav.nav_loop")
    importlib.reload(nl)

    monkeypatch.setattr(nl, "transform_slam_to_airsim", lambda m: (m, (m[0][3], m[1][3], m[2][3])))
    import slam_bridge.slam_receiver as sr
    import slam_bridge.frontier_detection as fd
    # Should not be called when pose_source="airsim"
    monkeypatch.setattr(sr, "get_latest_pose_matrix", lambda: None)
    monkeypatch.setattr(sr, "get_pose_history", lambda: [])
    monkeypatch.setattr(fd, "detect_frontiers", lambda m: nl.np.empty((0, 3)))
    monkeypatch.setattr(nl, "is_obstacle_ahead", lambda *a, **k: (False, None))

    client = types.SimpleNamespace(
        simGetCollisionInfo=lambda: types.SimpleNamespace(has_collided=False),
        simGetVehiclePose=lambda name: types.SimpleNamespace(position=pos, orientation=orientation),
        moveByVelocityAsync=lambda *a, **k: dummy_future,
        moveToPositionAsync=lambda *a, **k: dummy_future,
        hoverAsync=lambda *a, **k: dummy_future,
        landAsync=lambda *a, **k: dummy_future,
    )

    navigator = nl.Navigator(client)
    slam_mock = mock.MagicMock(return_value="slam_nav")
    monkeypatch.setattr(navigator, "slam_to_goal", slam_mock)

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

    args = types.SimpleNamespace(
        max_duration=0,
        goal_x=1.0,
        goal_y=2.0,
        goal_z=-2.0,
        slam_pose_source="airsim",
    )

    expected = [[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 2.0], [0.0, 0.0, 1.0, -2.0]]

    result = nl.slam_navigation_loop(args, client, ctx, pose_source=args.slam_pose_source)

    assert result == "slam_nav"
    slam_mock.assert_called_once_with(expected, (1.0, 2.0, -2.0))


def test_slam_navigation_performs_bootstrap(monkeypatch):
    """SLAM loop should run an initial calibration when duration > 0."""
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
    sample_matrix = [[1, 0, 0, 0.0], [0, 1, 0, 0.0], [0, 0, 1, -2.0]]
    monkeypatch.setattr(sr, "get_latest_pose_matrix", lambda: sample_matrix)
    monkeypatch.setattr(sr, "get_pose_history", lambda: [])
    monkeypatch.setattr(fd, "detect_frontiers", lambda m: nl.np.empty((0, 3)))
    monkeypatch.setattr(nl, "is_obstacle_ahead", lambda *a, **k: (False, None))

    boot_mock = mock.MagicMock()
    monkeypatch.setattr(nl, "run_slam_bootstrap", boot_mock)
    monkeypatch.setattr(nl.os.path, "exists", lambda p: True)

    dummy_future = types.SimpleNamespace(join=lambda *a, **k: None)
    client = types.SimpleNamespace(
        simGetCollisionInfo=lambda: types.SimpleNamespace(has_collided=False),
        moveByVelocityAsync=lambda *a, **k: dummy_future,
        moveToPositionAsync=lambda *a, **k: dummy_future,
        hoverAsync=lambda *a, **k: dummy_future,
        landAsync=lambda *a, **k: dummy_future,
    )

    navigator = nl.Navigator(client)
    monkeypatch.setattr(navigator, "slam_to_goal", lambda *a, **k: "slam_nav")

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
    args = types.SimpleNamespace(max_duration=1, goal_x=1.0, goal_y=2.0, goal_z=-2.0, slam_pose_source='slam')

    nl.slam_navigation_loop(args, client, ctx, pose_source=args.slam_pose_source)

    boot_mock.assert_called_once()

def test_slam_bootstrap_runs_when_tracking_lost(monkeypatch):
    """If SLAM loses tracking, the loop should rerun the bootstrap motion."""
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
    poses = [None, [[1,0,0,0.0],[0,1,0,0.0],[0,0,1,-2.0]]]
    monkeypatch.setattr(sr, "get_latest_pose_matrix", lambda: poses.pop(0) if poses else [[1,0,0,0.0],[0,1,0,0.0],[0,0,1,-2.0]])
    monkeypatch.setattr(sr, "get_pose_history", lambda: [])
    monkeypatch.setattr(fd, "detect_frontiers", lambda m: nl.np.empty((0, 3)))

    stable = [False, True]
    monkeypatch.setattr(nl, "is_slam_stable", lambda: stable.pop(0) if stable else True)
    monkeypatch.setattr(nl, "is_obstacle_ahead", lambda *a, **k: (False, None))

    boot_mock = mock.MagicMock()
    monkeypatch.setattr(nl, "run_slam_bootstrap", boot_mock)
    monkeypatch.setattr(nl.os.path, "exists", lambda p: False)
    monkeypatch.setattr(nl.time, "sleep", lambda *a, **k: None)

    dummy_future = types.SimpleNamespace(join=lambda *a, **k: None)
    client = types.SimpleNamespace(
        simGetCollisionInfo=lambda: types.SimpleNamespace(has_collided=False),
        moveByVelocityAsync=lambda *a, **k: dummy_future,
        moveToPositionAsync=lambda *a, **k: dummy_future,
        hoverAsync=lambda *a, **k: dummy_future,
        landAsync=lambda *a, **k: dummy_future,
    )

    navigator = nl.Navigator(client)
    monkeypatch.setattr(navigator, "slam_to_goal", lambda *a, **k: "slam_nav")

    class Flag:
        def __init__(self):
            self.calls = 0

        def is_set(self):
            self.calls += 1
            return self.calls > 1

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

    args = types.SimpleNamespace(max_duration=1, goal_x=1.0, goal_y=2.0, goal_z=-2.0, slam_pose_source='slam')

    nl.slam_navigation_loop(args, client, ctx, pose_source=args.slam_pose_source)

    assert boot_mock.call_count >= 2
