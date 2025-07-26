# Script Dependencies

This document lists the main third-party Python packages used by each script or module in this repository. The versions pinned in `requirements.txt` apply unless noted otherwise.

| Location | Key Dependencies | Notes |
|---------|-----------------|------|
| `main.py` | `airsim`, `opencv-python`, `numpy`, `cv2`, `msgpack-rpc-python`, `psutil` | controls overall UAV navigation |
| `launch_all.py` | `airsim`, `psutil`, `subprocess`, `tkinter` | launches AirSim, SLAM and GUI |
| `batch_runs.py` | standard library `subprocess` | batch run helper |
| `airsim_test.py` | `airsim` | tests AirSim connection |
| `data_capture/capture_airsim_data.py` | `airsim`, `opencv-python` | records stereo images and IMU data |
| `data_capture/format_captured_data.py` | standard library only | converts images for ORB‑SLAM |
| `extract_map_bounds.py` | `trimesh` | reads FBX/OBJ files |
| `slam_bridge/stream_airsim_image.py` | `airsim`, `numpy`, `opencv-python` | sends frames to SLAM backend |
| `slam_bridge/*` | `numpy`, `plotly`, `socket`, `csv` | pose and SLAM utilities |
| `analysis/*` | `numpy`, `pandas`, `plotly`, `trimesh` (optional) | flight analysis and visualisation |
| `uav/*` | `airsim`, `opencv-python`, `numpy`, `psutil` | navigation core and helpers |

If additional system or Python dependencies are introduced by new scripts, please update this table to keep the documentation current.
