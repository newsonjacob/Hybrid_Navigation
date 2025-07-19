import importlib
import sys
import types
import unittest.mock as mock

import tests.conftest  # ensure stubs loaded


def _load_module(monkeypatch, depth_value=None, height=40):
    arr = None
    if depth_value is not None:
        import numpy as np
        arr = np.full((height, height), depth_value, dtype=float)
    airsim_stub = types.SimpleNamespace(
        ImageRequest=lambda *a, **k: None,
        ImageType=types.SimpleNamespace(DepthPlanar=0),
        get_pfm_array=lambda r: arr,
    )
    monkeypatch.setitem(sys.modules, "airsim", airsim_stub)
    su = importlib.import_module("uav.slam_utils")
    importlib.reload(su)
    return su


def test_is_obstacle_ahead_detects_obstacle(monkeypatch):
    su = _load_module(monkeypatch, depth_value=1.0)
    resp = [types.SimpleNamespace(height=40)]
    client = types.SimpleNamespace(simGetImages=lambda *a, **k: resp)
    ahead, depth = su.is_obstacle_ahead(client, depth_threshold=2.0)
    assert ahead is True
    assert depth == 1.0


def test_is_obstacle_ahead_handles_missing_image(monkeypatch):
    su = _load_module(monkeypatch, depth_value=3.0)
    client = types.SimpleNamespace(simGetImages=lambda *a, **k: [])
    ahead, depth = su.is_obstacle_ahead(client)
    assert ahead is False
    assert depth is None


def test_generate_pose_comparison_plot_invokes_subprocess(monkeypatch):
    su = _load_module(monkeypatch)
    run_mock = mock.MagicMock(return_value=types.SimpleNamespace(stdout="ok"))
    monkeypatch.setattr(su.subprocess, "run", run_mock)
    su.generate_pose_comparison_plot()
    run_mock.assert_called_once_with(
        ["python", "slam_bridge/pose_comparison_plotter.py"],
        check=True,
        capture_output=True,
        text=True,
    )

