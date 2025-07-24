import importlib
import sys
import types
import unittest.mock as mock
from collections import deque
from queue import Queue

import tests.conftest  # ensure stubs loaded


def _load_nav_loop(monkeypatch):
    airsim_stub = types.SimpleNamespace(
        ImageRequest=object,
        ImageType=object,
        DrivetrainType=types.SimpleNamespace(ForwardOnly=1),
        YawMode=lambda *a, **k: None,
        to_eularian_angles=lambda o: (0, 0, 0),
        Vector3r=lambda x=0.0, y=0.0, z=0.0: types.SimpleNamespace(
            x_val=x, y_val=y, z_val=z
        ),
    )
    monkeypatch.setitem(sys.modules, "airsim", airsim_stub)
    nl = importlib.import_module("uav.nav_loop")
    importlib.reload(nl)
    return nl, airsim_stub


class DummyClient:
    def __init__(self, airsim_stub):
        self._airsim = airsim_stub

    def getMultirotorState(self):
        vec = self._airsim.Vector3r()
        kin = types.SimpleNamespace(
            linear_velocity=vec,
            position=vec,
            orientation=types.SimpleNamespace(),
        )
        return types.SimpleNamespace(kinematics_estimated=kin)


def _make_nav():
    nav = types.SimpleNamespace(
        braked=False,
        dodging=False,
        just_resumed=False,
        resume_grace_end_time=0.0,
        grace_period_end_time=0.0,
        last_movement_time=0.0,
    )
    nav.brake = mock.MagicMock(return_value=NavigationState.BRAKE)
    nav.blind_forward = mock.MagicMock(return_value=NavigationState.BLIND_FORWARD)
    nav.dodge = mock.MagicMock(return_value=NavigationState.DODGE_LEFT)
    nav.maintain_dodge = mock.MagicMock()
    nav.resume_forward = mock.MagicMock(return_value=NavigationState.RESUME)
    nav.nudge_forward = mock.MagicMock(return_value=NavigationState.NUDGE)
    nav.reinforce = mock.MagicMock(return_value=NavigationState.RESUME_REINFORCE)
    nav.timeout_recover = mock.MagicMock(return_value=NavigationState.TIMEOUT_NUDGE)
    return nav


from uav.context import ParamRefs
from uav.navigation_state import NavigationState


def _default_params():
    return ParamRefs(
        state=[NavigationState.NONE],
        prev_L=[0.0],
        prev_C=[0.0],
        prev_R=[0.0],
        delta_L=[0.0],
        delta_C=[0.0],
        delta_R=[0.0],
    )


def test_brake_when_side_flow_high(monkeypatch):
    nl, airsim_stub = _load_nav_loop(monkeypatch)
    client = DummyClient(airsim_stub)
    nav = _make_nav()
    frame_q = Queue()
    params = _default_params()

    result = nl.navigation_step(
        client,
        nav,
        None,
        [],
        None,
        0.0,
        2.0,
        0.1,
        2.5,
        0.0,
        0.0,
        0.0,
        0,
        0,
        0,
        frame_q,
        object(),
        0.0,
        1,
        deque(maxlen=3),
        deque(maxlen=3),
        params,
    )

    assert result[0] is NavigationState.BRAKE
    nav.brake.assert_called_once()
    nav.blind_forward.assert_not_called()


def test_blind_forward_with_low_flow(monkeypatch):
    nl, airsim_stub = _load_nav_loop(monkeypatch)
    client = DummyClient(airsim_stub)
    nav = _make_nav()
    frame_q = Queue()
    params = _default_params()

    result = nl.navigation_step(
        client,
        nav,
        None,
        [],
        None,
        0.0,
        0.5,
        0.1,
        0.4,
        0.0,
        0.0,
        0.0,
        0,
        0,
        0,
        frame_q,
        object(),
        0.0,
        1,
        deque(maxlen=3),
        deque(maxlen=3),
        params,
    )

    assert result[0] is NavigationState.BLIND_FORWARD
    nav.blind_forward.assert_called_once()
    nav.brake.assert_not_called()


def test_dodge_when_obstacle_and_sides_clear(monkeypatch):
    nl, airsim_stub = _load_nav_loop(monkeypatch)
    client = DummyClient(airsim_stub)
    nav = _make_nav()
    params = _default_params()

    monkeypatch.setattr(
        nl,
        "get_drone_state",
        lambda c: (airsim_stub.Vector3r(), 0.0, 0.0),
    )

    frame_q = Queue()
    good_old = [0] * 20
    state_hist = deque(maxlen=3)
    pos_hist = deque(maxlen=3)

    result = nl.navigation_step(
        client,
        nav,
        None,
        good_old,
        None,
        0.0,
        1.0,
        7.0,
        1.0,
        0.0,
        0.0,
        0.0,
        15,
        20,
        20,
        frame_q,
        object(),
        0.0,
        1,
        state_hist,
        pos_hist,
        params,
    )

    assert result[0] in (NavigationState.DODGE_LEFT, NavigationState.DODGE_RIGHT)
    nav.dodge.assert_called_once()
    assert nav.dodge.call_args.kwargs.get("direction") == "left"
