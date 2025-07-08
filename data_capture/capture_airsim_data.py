import airsim
import os
import cv2
import time
import csv

# Create folder to save images and imu data
output_dir = "airsim_data"
os.makedirs(output_dir, exist_ok=True)

# Connect to AirSim
client = airsim.MultirotorClient()
client.confirmConnection()
print("Connected to AirSim!")

# CSV file to save IMU data: timestamp, accel(x,y,z), gyro(x,y,z)
imu_file = open(os.path.join(output_dir, "imu_data.csv"), mode='w', newline='')
imu_writer = csv.writer(imu_file)
imu_writer.writerow(["timestamp", "accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"])

# Number of frames to capture
num_frames = 100
frame_delay = 0.1  # seconds between frames (~10 FPS)

for i in range(num_frames):
    timestamp = time.time()

    # Get stereo images from cameras 0 and 1 (adjust names if needed)
    responses = client.simGetImages([
        airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),
        airsim.ImageRequest("1", airsim.ImageType.Scene, False, False)
    ])

    if len(responses) == 2:
        # Left image
        left_img = cv2.imdecode(responses[0].image_data_uint8, cv2.IMREAD_COLOR)
        left_filename = os.path.join(output_dir, f"left_{i:05d}.png")
        cv2.imwrite(left_filename, left_img)

        # Right image
        right_img = cv2.imdecode(responses[1].image_data_uint8, cv2.IMREAD_COLOR)
        right_filename = os.path.join(output_dir, f"right_{i:05d}.png")
        cv2.imwrite(right_filename, right_img)
        print(f"Saved stereo pair {i}")

    else:
        print("Failed to get stereo images")

    # Get IMU data
    imu_data = client.getImuData()
    imu_writer.writerow([
        timestamp,
        imu_data.linear_acceleration.x_val,
        imu_data.linear_acceleration.y_val,
        imu_data.linear_acceleration.z_val,
        imu_data.angular_velocity.x_val,
        imu_data.angular_velocity.y_val,
        imu_data.angular_velocity.z_val
    ])

    time.sleep(frame_delay)

imu_file.close()
print(f"Saved {num_frames} frames and IMU data to '{output_dir}'")
