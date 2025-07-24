import unittest.mock as mock
import time
import types
from collections import deque
from queue import Queue
from tests.conftest import airsim_stub

from uav.navigation import Navigator


class DummyFuture:
    def __init__(self):
        self.join_called = False

    def join(self):
        self.join_called = True


class DummyState:
    def __init__(self):
        self.kinematics_estimated = types.SimpleNamespace(
            linear_velocity=types.SimpleNamespace(x_val=0, y_val=0, z_val=0),
            position=types.SimpleNamespace(z_val=0),
            orientation=types.SimpleNamespace(),
        )


class DummyClient:
    def __init__(self):
        self.moveByVelocityAsync = mock.MagicMock(
            side_effect=self._moveByVelocityAsync
        )
        self.moveByVelocityZAsync = mock.MagicMock(
            side_effect=self._moveByVelocityZAsync
        )
        self.moveByVelocityBodyFrameAsync = mock.MagicMock(
            side_effect=self._moveByVelocityBodyFrameAsync
        )
        self.calls = []

    def _record(self, name, args, kwargs):
        fut = DummyFuture()
        self.calls.append((name, args, kwargs, fut))
        return fut

    def _moveByVelocityAsync(self, *args, **kwargs):
        return self._record('moveByVelocityAsync', args, kwargs)

    def _moveByVelocityZAsync(self, *args, **kwargs):
        return self._record('moveByVelocityZAsync', args, kwargs)

    def _moveByVelocityBodyFrameAsync(self, *args, **kwargs):
        return self._record('moveByVelocityBodyFrameAsync', args, kwargs)

    def getMultirotorState(self):
        return DummyState()


def test_brake_updates_flags_and_calls():
    client = DummyClient()
    nav = Navigator(client)
    prev = nav.last_movement_time
    result = nav.brake()
    assert result == 'brake'
    assert nav.braked is True
    assert nav.dodging is False
    assert nav.last_movement_time == prev
    name, args, kwargs, fut = client.calls[-1]
    assert name == 'moveByVelocityAsync'
    assert args == (0, 0, 0, 0.5)
    assert fut.join_called is False


def test_dodge_left_sets_flags_and_calls():
    client = DummyClient()
    nav = Navigator(client)
    prev = nav.last_movement_time
    result = nav.dodge(0, 0, 20)
    assert result == 'dodge_None'
    assert nav.braked is False
    assert nav.dodging is True
    assert nav.last_movement_time > prev
    client.moveByVelocityAsync.assert_called_once_with(0, 0, 0, 0.5)
    client.moveByVelocityBodyFrameAsync.assert_called_once()
    call = client.moveByVelocityBodyFrameAsync.call_args
    assert call.args == (0.0, -1.0, 0, 2.0)
    fut_brake = client.calls[0][3]
    fut_lateral = client.calls[1][3]
    assert fut_brake.join_called is False
    assert fut_lateral.join_called is False


def test_ambiguous_dodge_forces_lower_flow_side():
    client = DummyClient()
    nav = Navigator(client)
    result = nav.dodge(10, 10.5, 11)
    assert result == 'dodge_None'
    client.moveByVelocityAsync.assert_called_once_with(0, 0, 0, 0.5)
    client.moveByVelocityBodyFrameAsync.assert_called_once()
    call = client.moveByVelocityBodyFrameAsync.call_args
    assert call.args == (0.0, -1.0, 0, 2.0)


def test_resume_forward_clears_flags_and_calls():
    client = DummyClient()
    nav = Navigator(client)
    prev = nav.last_movement_time
    result = nav.resume_forward()
    assert result == 'resume'
    assert nav.braked is False
    assert nav.dodging is False
    assert nav.last_movement_time > prev
    client.moveByVelocityAsync.assert_called_once_with(0, 0, 0, 0)
    client.moveByVelocityZAsync.assert_called_once()
    args, kwargs = client.moveByVelocityZAsync.call_args
    assert args[:3] == (2, 0, 0)
    assert kwargs.get('duration') == 3
    assert kwargs.get('drivetrain') == airsim_stub.DrivetrainType.ForwardOnly


def test_nudge_updates_time_and_calls():
    client = DummyClient()
    nav = Navigator(client)
    prev = nav.last_movement_time
    result = nav.nudge_forward()
    assert result == 'nudge'
    assert nav.braked is False
    assert nav.dodging is False
    assert nav.last_movement_time > prev
    name, args, kwargs, fut = client.calls[-1]
    assert name == 'moveByVelocityZAsync'
    assert args == (0.5, 0, 0, 1)
    assert fut.join_called is False


def test_reinforce_updates_time_and_calls():
    client = DummyClient()
    nav = Navigator(client)
    prev = nav.last_movement_time
    result = nav.reinforce()
    assert result == 'resume_reinforce'
    assert nav.braked is False
    assert nav.dodging is False
    assert nav.last_movement_time > prev
    client.moveByVelocityZAsync.assert_called_once()
    args, kwargs = client.moveByVelocityZAsync.call_args
    assert args[:3] == (2, 0, 0)
    assert kwargs.get('duration') == 3
    assert kwargs.get('drivetrain') == airsim_stub.DrivetrainType.ForwardOnly


def test_dodge_settle_duration_short():
    client = DummyClient()
    nav = Navigator(client)
    before = time.time()
    nav.dodge(0, 0, 20)
    assert nav.settling is False


def test_resume_forward_not_called_during_grace():
    client = DummyClient()
    nav = Navigator(client)
    nav.dodging = True
    nav.grace_period_end_time = time.time() + 1.0
    nav.resume_forward = mock.MagicMock()
    time_now = time.time()
    smooth_L = smooth_C = smooth_R = 0
    if (
        (nav.braked or nav.dodging)
        and smooth_C < 10
        and smooth_L < 10
        and smooth_R < 10
        and time_now >= nav.grace_period_end_time
    ):
        nav.resume_forward()
    nav.resume_forward.assert_not_called()


def test_blind_forward_starts_grace_once():
    client = DummyClient()
    nav = Navigator(client)
    before = time.time()
    nav.grace_used = False

    nav.blind_forward()

    assert nav.just_resumed is True
    assert nav.grace_used is True
    assert nav.resume_grace_end_time >= before + 0.9
    first_end = nav.resume_grace_end_time

    nav.blind_forward()

    assert nav.resume_grace_end_time == first_end


def test_navigation_skips_actions_during_grace_after_blind_forward(monkeypatch):
    import importlib
    import types
    import sys

    airsim_stub = types.SimpleNamespace(ImageRequest=object, ImageType=object)
    monkeypatch.setitem(sys.modules, 'airsim', airsim_stub)
    nl = importlib.import_module('uav.nav_loop')
    importlib.reload(nl)
    navigation_step = nl.navigation_step

    client = DummyClient()
    nav = Navigator(client)
    nav.blind_forward()
    frame_q = Queue()
    params = {'state': [None]}
    result = navigation_step(
        client,
        nav,
        None,
        [],
        None,
        0.0,
        2.0,
        0.1,
        2.0,
        0.0,
        0.0,
        0,
        0,
        0,
        0,
        frame_q,
        object(),
        time.time(),
        1,
        deque(maxlen=10),
        deque(maxlen=10),
        params,
    )

    assert result[0] == "none"
    assert client.moveByVelocityZAsync.call_count == 1

def test_slam_to_goal_accepts_pose_matrix(monkeypatch):
    client = DummyClient()
    nav = Navigator(client)

    from uav import config
    monkeypatch.setattr(config, "SLAM_YAW_OFFSET", 10.0, raising=False)

    pose = [
        [0, -1, 0, 5.0],
        [1, 0, 0, 3.0],
        [0, 0, 1, -2.0],
    ]
    nav.slam_to_goal(pose, (6.0, 3.0, -2.0), max_speed=1.0)

    name, args, kwargs, fut = client.calls[-1]
    assert name == "moveByVelocityAsync"
    assert abs(kwargs["yaw_mode"].yaw_or_rate - 100.0) < 1e-3

