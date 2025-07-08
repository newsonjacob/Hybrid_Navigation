Data Capture for SLAM Input
This folder contains scripts to capture sensor data from AirSim needed for SLAM integration.

capture_airsim_data.py
- Connects to AirSim via RPC
- Captures synchronized stereo images from cameras "0" and "1"
- Records IMU data with timestamps
- Saves images and IMU logs to the output/ folder

Usage
1. Ensure AirSim simulation is running with your drone and cameras set up.
2. Activate your Python virtual environment with required packages (airsim, opencv-python).
3. Run the capture script:
python capture_airsim_data.py
4. Data will be saved in the output/ directory inside this folder.

