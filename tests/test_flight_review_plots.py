import pandas as pd
from analysis.flight_review import (
    plot_state_histogram,
    plot_distance_over_time,
)


def test_plot_state_histogram_creates_html(tmp_path):
    out = tmp_path / "states.html"
    stats = {"states": {"resume": 2, "brake": 1}}
    plot_state_histogram(stats, str(out))
    assert out.exists()
    assert "<html" in out.read_text().lower()


def test_plot_distance_over_time_creates_html(tmp_path):
    df = pd.DataFrame({
        "pos_x": [0, 1, 2],
        "pos_y": [0, 0, 0],
        "pos_z": [0, 0, 0],
        "time": [0, 1, 2],
    })
    csv = tmp_path / "log.csv"
    df.to_csv(csv, index=False)
    out = tmp_path / "dist.html"
    plot_distance_over_time(str(csv), str(out))
    assert out.exists()
    assert "<html" in out.read_text().lower()

