import subprocess
import sys
import pandas as pd


def test_performance_cli_generates_html(tmp_path):
    df = pd.DataFrame({
        'time': [0, 1, 2],
        'cpu_percent': [10, 20, 30],
        'memory_rss': [1_000_000, 2_000_000, 3_000_000],
    })
    log_path = tmp_path / 'perf.csv'
    df.to_csv(log_path, index=False)
    out_path = tmp_path / 'perf.html'

    result = subprocess.run([
        sys.executable,
        '-m',
        'analysis.performance_plots',
        str(log_path),
        '-o',
        str(out_path),
    ], capture_output=True, text=True)
    assert result.returncode == 0
    assert out_path.exists()
    assert '<html' in out_path.read_text().lower()
