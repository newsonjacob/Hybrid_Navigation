# Reactive Optical Flow UAV Navigation
[![CI](https://github.com/newsonjacob/ReactiveOptical_Flow/actions/workflows/tests.yml/badge.svg)](https://github.com/newsonjacob/ReactiveOptical_Flow/actions/workflows/tests.yml)

This repository contains the implementation of a reactive obstacle avoidance system for a simulated UAV using optical flow. The UAV operates in a GPS-denied environment simulated via AirSim and Unreal Engine 4.

---

## 📦 Features

- Real-time optical flow tracking using Lucas-Kanade method
- Zone-based obstacle avoidance using vector summation
- Modular helpers for obstacle detection and side safety
- Logs flight data to CSV for post-flight analysis
- Visualises 3D flight paths with interactive HTML output
- Compatible with a custom stereo camera setup (`oakd_camera`)
- Works with auto-launched AirSim simulation
- Receives SLAM poses via a `PoseReceiver` that can be started and stopped programmatically
- Integrated frontier-based exploration using SLAM map points
- SLAM loop checks depth ahead and dodges obstacles before advancing

---

## Installation

Install the pinned dependencies and the package:

```bash
pip install -r requirements.txt
pip install -e .
```

## 🚁 How It Works

- A drone is spawned in a UE4 + AirSim environment
- The system captures images from a front-facing virtual camera
- Optical flow vectors are extracted and used to detect motion in different zones
- The UAV avoids obstacles by adjusting yaw based on zone flow magnitudes
- After the first forward command a brief grace period suppresses additional navigation actions

---

## 🔧 Setup Instructions

1. Clone this repository:
   ```bash
   git clone https://github.com/newsonjacob/ReactiveOptical_Flow.git
   cd ReactiveOptical_Flow
   ```

2. Install required Python packages using the pinned versions:
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

3. Place a valid AirSim `settings.json` at:
   ```
   C:\Users\<YourUsername>\Documents\AirSim\settings.json
   ```

   Example config to enable oakd-style stereo camera:
   ```json
   {
     "SettingsVersion": 1.2,
     "SimMode": "Multirotor",
     "Vehicles": {
       "UAV": {
         "VehicleType": "SimpleFlight",
         "AutoCreate": true,
         "Cameras": {
           "oakd_camera": {
             "CaptureSettings": [
               {
                 "ImageType": 0,
                 "Width": 1280,
                 "Height": 720,
                 "FOV_Degrees": 90
               }
             ],
             "X": 0.0,
             "Y": 0.0,
             "Z": -0.1,
             "Pitch": 0.0,
             "Roll": 0.0,
             "Yaw": 0.0
           }
         }
       }
     }
   }
   ```

4. Build your Unreal map and generate a packaged `.exe` (e.g. `Blocks.exe`).

5. Edit `config.ini` to point to your AirSim `settings.json`, UE4 executables and the SLAM networking details. The `[network]` section controls the IP and port used by the image streamer and pose receiver. The streamer also reads `SLAM_SERVER_HOST`, `SLAM_SERVER_PORT` and `CONNECT_RETRIES` from the environment (or command line) so these values can be overridden without editing the config file.

6. Run the system:
   ```bash
   hybrid-nav
   ```
   You can override paths at runtime:
   ```bash
   python hybrid-nav --settings-path C:\path\to\settings.json --ue4-path C:\path\to\Blocks.exe
   ```
   To load a different configuration file:
   ```bash
  hybrid-nav --config custom.ini
   ```
   Network addresses can also be overridden:
   ```bash
  hybrid-nav --slam-server-host 10.0.0.2 --slam-server-port 6000 \
             --slam-receiver-host 10.0.0.3 --slam-receiver-port 6001
   ```

## Running Tests

Install the required dependencies and run the test suite:

```bash
pip install -r requirements.txt
pip install -e .
pytest
```

### Command Line Interface

The `hybrid-nav` entry point exposes several options:

| Option | Description |
| ------ | ----------- |
| `--manual-nudge` | Enable manual nudge at frame 5 for testing |
| `--map {reactive, deliberative, hybrid}` | Which map to load |
| `--ue4-path PATH` | Override the path to the Unreal Engine executable |
| `--settings-path PATH` | Path to the AirSim `settings.json` file |
| `--config FILE` | Path to configuration file (default: `config.ini`) |
| `--goal-x INT` | Distance from start to goal on the X axis |
| `--max-duration INT` | Maximum simulation duration in seconds |
| `--nav-mode {slam, reactive}` | Navigation mode to run |
| `--slam-server-host HOST` | SLAM server IP or hostname |
| `--slam-server-port PORT` | SLAM server TCP port |
| `--slam-receiver-host HOST` | Pose receiver IP address |
| `--slam-receiver-port PORT` | Pose receiver TCP port |
| `--log-timestamp STR` | Timestamp used to sync logging across modules |

Example quick start:

```bash
hybrid-nav --ue4-path /path/to/Blocks.exe --settings-path ~/Documents/AirSim/settings.json --nav-mode slam
```

### GUI and Flag Files

The Tkinter GUI writes small flag files under `flags/` which the main loop
monitors to coordinate the run. Selecting a mode and pressing *Launch
Simulation* writes `nav_mode.flag`. Once all systems report ready, clicking
*Start Navigation* creates `start_nav.flag` which unblocks the navigation loop.
Pressing the stop button touches `stop.flag` so the running process can safely
land and exit.

### SLAM Utilities

Two helper applications live under `linux_slam/app`:

- `offline_slam_evaluation` now accepts `--data-dir=DIR` to specify where RGB and depth images are loaded from.
- `tcp_slam_server` reads log and flag locations from the command line or the environment variables `SLAM_LOG_DIR`, `SLAM_FLAG_DIR` and `SLAM_IMAGE_DIR`.
- `launch_slam_backend` automatically exports these variables, pointing to the
  repository's `flags/` and `logs/` folders, before starting `tcp_slam_server`.


---

## 📊 Logs and Visualisation

### Enabling Logging

Call `setup_logging()` to configure console output. Provide a file name to also
store logs under `logs/`:

```python
from uav.logging_config import setup_logging
setup_logging("run.log")  # also prints to stdout
```

- Flight logs are stored in `flow_logs/` as `.csv`
- 3D trajectory plots are saved in `analysis/` as interactive `.html` files
- Runtime messages are configured via `uav.logging_config.setup_logging` using the standard `logging` module
- SLAM pose and feature debugging is printed to stdout and stored in `logs/`

---

## 📁 Folder Structure

```
ReactiveOptical_Flow/
├── main.py                   # Main control loop
├── airsim_test.py           # Test AirSim connection
├── test_oakd_camera.py      # Debug camera stream
├── uav/
│   ├── perception.py        # Optical flow tracking
│   ├── navigation.py        # Control logic
│   ├── utils.py             # Helper functions
│   └── __init__.py
├── flow_logs/               # CSV logs of motion + control
├── analysis/                # 3D path visualisations
```

---

## 🚧 Coming Soon

The current system is purely reactive. The following upgrades are planned:

- ✅ **SLAM Integration:** Add stereo ORB-SLAM2 to enable global localisation
- ✅ **FSM-Based Switching:** Use a Finite State Machine to toggle between optical flow and SLAM
- ✅ **Hybrid Navigation Mode:** Leverage both reactive and deliberative control based on context
- ⏳ **Trajectory Comparison:** Benchmark reactive vs SLAM vs hybrid strategies
- ⏳ **ROS Support (optional)**

---

## 🧠 Acknowledgements

- Built using [AirSim](https://github.com/microsoft/AirSim)
- Inspired by biologically inspired reactive navigation strategies

---

## 📬 Contact

For questions, issues, or contributions, reach out via [GitHub Issues](https://github.com/newsonjacob/ReactiveOptical_Flow/issues)
