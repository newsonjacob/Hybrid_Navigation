import importlib
import sys
import types
import unittest.mock as mock


def test_helper_functions_exist(monkeypatch):
    airsim_stub = types.SimpleNamespace(ImageRequest=object, ImageType=object)
    monkeypatch.setitem(sys.modules, 'airsim', airsim_stub)
    nl = importlib.import_module('uav.nav_loop')
    importlib.reload(nl)
    for name in (
        'setup_environment',
        'start_perception_thread',
        'navigation_loop',
        'check_startup_grace',
        'get_perception_data',
        'update_navigation_state',
        'log_and_record_frame',
        'process_perception_data',
        'apply_navigation_decision',
        'write_frame_output',
        'handle_reset',
        'cleanup',
        'detect_obstacle',
        'determine_side_safety',
        'handle_obstacle',
    ):
        assert hasattr(nl, name)


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
        self.moveByVelocityAsync = mock.MagicMock(side_effect=self._record)
        self.moveByVelocityZAsync = mock.MagicMock(side_effect=self._record)
        self.moveByVelocityBodyFrameAsync = mock.MagicMock(side_effect=self._record)
        self.calls = []

    def _record(self, *args, **kwargs):
        fut = DummyFuture()
        self.calls.append((args, kwargs, fut))
        return fut

    def getMultirotorState(self):
        return DummyState()


def test_detect_obstacle(monkeypatch):
    airsim_stub = types.SimpleNamespace(ImageRequest=object, ImageType=object)
    monkeypatch.setitem(sys.modules, 'airsim', airsim_stub)
    nl = importlib.import_module('uav.nav_loop')
    importlib.reload(nl)

    assert nl.detect_obstacle(2.0, 1.2, 30, 1.0) is True
    assert nl.detect_obstacle(0.2, 0.1, 5, 1.0) is False


def test_determine_side_safety(monkeypatch):
    airsim_stub = types.SimpleNamespace(ImageRequest=object, ImageType=object)
    monkeypatch.setitem(sys.modules, 'airsim', airsim_stub)
    nl = importlib.import_module('uav.nav_loop')
    importlib.reload(nl)

    left_safe, right_safe, side_safe = nl.determine_side_safety(
        0.3, 0.2, 1.0, 12, 50, 5
    )
    assert left_safe is True
    assert right_safe is True
    assert side_safe is True


def test_handle_obstacle_dodge_and_resume(monkeypatch):
    airsim_stub = types.SimpleNamespace(ImageRequest=object, ImageType=object)
    monkeypatch.setitem(sys.modules, 'airsim', airsim_stub)
    nl = importlib.import_module('uav.nav_loop')
    importlib.reload(nl)
    from uav.navigation import Navigator

    client = DummyClient()
    nav = Navigator(client)
    # Dodge left
    state = nl.handle_obstacle(
        nav, 1, True, True, False, False, False,
        0.0, 0.0, 0.0, 15, 20
    )
    assert state.startswith('dodge')
    assert nav.dodging is True

    # Resume when obstacle cleared
    state = nl.handle_obstacle(
        nav, 0, False, False, False, False, False,
        0.0, 0.0, 0.0, 15, 20
    )
    assert state == 'resume'
