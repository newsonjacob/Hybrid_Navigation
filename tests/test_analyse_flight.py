import subprocess
import sys
import pandas as pd


def test_analyse_cli_produces_html(tmp_path):
    df = pd.DataFrame({
        "pos_x": [0, 1, 2],
        "pos_y": [0, 0, 0],
        "pos_z": [0, 0, 0],
        "time": [0, 1, 2],
        "flow_left": [0.1, 0.2, 0.3],
        "flow_center": [0.2, 0.3, 0.4],
        "flow_right": [0.3, 0.4, 0.5],
        "speed": [1.0, 1.0, 1.0],
        "fps": [10, 10, 10],
        "loop_s": [0.1, 0.1, 0.1],
        "state": ["resume", "resume", "resume"],
    })
    log_path = tmp_path / "log.csv"
    df.to_csv(log_path, index=False)
    out_path = tmp_path / "view.html"

    result = subprocess.run(
        [sys.executable, "-m", "analysis.analyse", str(log_path), "-o", str(out_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert out_path.exists()
    content = out_path.read_text().lower()
    assert "<html" in content
