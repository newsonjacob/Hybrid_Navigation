import subprocess
import sys
import pandas as pd
from analysis.flight_review import plot_state_histogram, plot_distance_over_time


def test_state_histogram_creates_html(tmp_path):
    stats = {"states": {"resume": 2, "brake": 1}}
    out = tmp_path / "hist.html"
    plot_state_histogram(stats, str(out))
    assert out.exists()
    assert "<html" in out.read_text().lower()


def test_distance_plot_creates_html(tmp_path):
    df = pd.DataFrame({
        "pos_x": [0, 1, 2],
        "pos_y": [0, 0, 0],
        "pos_z": [0, 0, 0],
        "time": [0, 1, 2],
    })
    csv_path = tmp_path / "log.csv"
    df.to_csv(csv_path, index=False)
    out = tmp_path / "dist.html"
    plot_distance_over_time(str(csv_path), str(out))
    assert out.exists()
    assert "<html" in out.read_text().lower()


def test_analyze_cli(tmp_path):
    df = pd.DataFrame({
        "pos_x": [0, 1],
        "pos_y": [0, 0],
        "pos_z": [0, 0],
        "time": [0, 1],
        "state": ["resume", "brake"],
    })
    log_path = tmp_path / "log.csv"
    df.to_csv(log_path, index=False)
    outdir = tmp_path / "out"

    result = subprocess.run([
        sys.executable,
        "-m",
        "analysis.analyze",
        str(log_path),
        "-o",
        str(outdir),
    ], capture_output=True, text=True)
    assert result.returncode == 0
    assert (outdir / "state_histogram.html").exists()
    assert (outdir / "distance_over_time.html").exists()

