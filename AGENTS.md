# Contributor Guidelines

- Install required packages with `pip install -r requirements.txt` **before** running tests.
- Ensure the system package `libgl1` is installed (required by OpenCV). On Debian/Ubuntu run:
  `sudo apt-get update && sudo apt-get install -y libgl1`.
- Run `pytest` before committing to ensure all tests pass.
- Keep documentation in sync with code changes, especially `README.md`.

The dependency list pins **NumPy** to `<1.27` so it remains compatible with SciPy
and pandas. Make sure your environment reflects this pin when installing
packages.
