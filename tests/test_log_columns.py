import importlib
import sys
import types

import tests.conftest  # ensure stubs loaded


def test_setup_environment_header_includes_perf(monkeypatch, tmp_path):
    airsim_stub = types.SimpleNamespace(ImageRequest=object, ImageType=object)
    monkeypatch.setitem(sys.modules, "airsim", airsim_stub)
    nl = importlib.import_module("uav.nav_loop")
    importlib.reload(nl)

    monkeypatch.setattr(nl, "start_video_writer_thread", lambda *a, **k: types.SimpleNamespace(join=lambda: None))
    monkeypatch.setattr(nl, "retain_recent_logs", lambda *a, **k: None)
    monkeypatch.setattr(nl, "retain_recent_files", lambda *a, **k: None)
    monkeypatch.setattr(nl, "retain_recent_views", lambda *a, **k: None)
    monkeypatch.setattr(nl, "init_client", lambda *a, **k: None)
    monkeypatch.setattr(nl, "OpticalFlowTracker", lambda *a, **k: object())
    monkeypatch.setattr(nl, "FlowHistory", lambda *a, **k: object())
    monkeypatch.setattr(nl, "Navigator", lambda *a, **k: object())
    monkeypatch.setattr(nl.cv2, "VideoWriter_fourcc", lambda *a: 0)
    monkeypatch.setattr(nl.cv2, "VideoWriter", lambda *a, **k: types.SimpleNamespace(release=lambda: None))

    args = types.SimpleNamespace(goal_x=0.0, goal_y=0.0, max_duration=1)
    dummy_future = types.SimpleNamespace(join=lambda *a, **k: None)
    client = types.SimpleNamespace(
        listVehicles=lambda: [],
        takeoffAsync=lambda: dummy_future,
        moveToPositionAsync=lambda *a, **k: dummy_future,
    )
    monkeypatch.chdir(tmp_path)
    ctx = nl.setup_environment(args, client)
    ctx.log_file.close()
    log_file = next((tmp_path / "flow_logs").glob("full_log_*.csv"))
    header = log_file.read_text().splitlines()[0]
    assert "cpu_percent" in header
    assert "memory_rss" in header
    assert "sudden_rise" in header
    assert "center_blocked" in header
    assert "combination_flow" in header
    assert "minimum_flow" in header

