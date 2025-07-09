import os
import time
from uav.utils import retain_recent_logs, should_flat_wall_dodge, retain_recent_views, retain_recent_files


def test_retain_recent_logs_keeps_latest(tmp_path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()

    for i in range(6):
        ts = f"20240101_00000{i}"
        p = log_dir / f"full_log_{ts}.csv"
        p.write_text("data")

    retain_recent_logs(str(log_dir), keep=3)
    remaining = sorted(f.name for f in log_dir.iterdir())
    assert remaining == [
        f"full_log_20240101_00000{i}.csv" for i in range(3, 6)
    ]


def test_retain_recent_logs_missing_dir(tmp_path):
    missing = tmp_path / "missing"
    retain_recent_logs(str(missing), keep=3)
    assert not missing.exists()


def test_should_flat_wall_dodge_threshold():
    assert should_flat_wall_dodge(1.0, 0.2, 5, 5) is True
    # Not enough probe features -> should be False
    assert should_flat_wall_dodge(1.0, 0.2, 3, 5) is False


def test_should_flat_wall_dodge_flow_std_limit():
    # Excessive variance should disable the fallback dodge
    assert should_flat_wall_dodge(1.0, 0.2, 5, 5, flow_std=50.0) is False


def test_retain_recent_views_keeps_latest(tmp_path):
    view_dir = tmp_path / "views"
    view_dir.mkdir()

    now = time.time()
    for i in range(6):
        p = view_dir / f"flight_view_{i}.html"
        p.write_text("data")
        mod_time = now - i
        os.utime(p, (mod_time, mod_time))

    retain_recent_views(str(view_dir), keep=5)
    remaining = sorted(f.name for f in view_dir.iterdir())
    assert remaining == [f"flight_view_{i}.html" for i in range(5)]


def test_retain_recent_views_missing_dir(tmp_path):
    missing = tmp_path / "missing"
    retain_recent_views(str(missing), keep=5)
    assert not missing.exists()


def test_retain_recent_files_keeps_latest(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    now = time.time()
    for i in range(4):
        p = data_dir / f"file_{i}.txt"
        p.write_text("data")
        mod_time = now - i
        os.utime(p, (mod_time, mod_time))

    retain_recent_files(str(data_dir), "*.txt", keep=2)
    remaining = sorted(f.name for f in data_dir.iterdir())
    assert remaining == ["file_0.txt", "file_1.txt"]


def test_retain_recent_files_missing_dir(tmp_path):
    missing = tmp_path / "none"
    retain_recent_files(str(missing), "*.txt", keep=2)
    assert not missing.exists()

