import subprocess, sys


def test_main_help_shows_usage():
    result = subprocess.run(
        [sys.executable, 'main.py', '--help'], capture_output=True, text=True
    )
    assert 'usage: main.py' in result.stdout
    assert '-h, --help' in result.stdout
