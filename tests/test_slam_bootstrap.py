import importlib
import sys
import types
import unittest.mock as mock


def test_bootstrap_ends_facing_forward(monkeypatch):
    """run_slam_bootstrap should rotate drone to yaw 0 at completion."""
    airsim_stub = types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, 'airsim', airsim_stub)
    boot = importlib.import_module('uav.navigation_slam_boot')
    importlib.reload(boot)

    client = types.SimpleNamespace(
        moveByVelocityAsync=lambda *a, **k: types.SimpleNamespace(join=lambda: None),
        rotateToYawAsync=mock.MagicMock(),
    )

    boot.run_slam_bootstrap(client, duration=0.0)

    client.rotateToYawAsync.assert_called_with(0, vehicle_name='UAV')

