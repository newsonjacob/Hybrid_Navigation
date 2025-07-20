"""Format captured AirSim data into ORB-SLAM compatible dataset."""

import os
import shutil

src_dir = "data_capture/output"
dst_dir = "data_capture/orbslam_dataset"

rgb_dir = os.path.join(dst_dir, "rgb")
depth_dir = os.path.join(dst_dir, "depth")

os.makedirs(rgb_dir, exist_ok=True)
os.makedirs(depth_dir, exist_ok=True)

# Copy and rename left images to rgb/
for filename in sorted(os.listdir(src_dir)):
    if filename.startswith("left_") and filename.endswith(".png"):
        src_path = os.path.join(src_dir, filename)
        # ORB-SLAM2 filenames are timestamps, so just rename with a running number or timestamp
        dst_path = os.path.join(rgb_dir, filename.replace("left_", ""))
        shutil.copyfile(src_path, dst_path)

# Copy and rename right images to depth/
for filename in sorted(os.listdir(src_dir)):
    if filename.startswith("right_") and filename.endswith(".png"):
        src_path = os.path.join(src_dir, filename)
        dst_path = os.path.join(depth_dir, filename.replace("right_", ""))
        shutil.copyfile(src_path, dst_path)

# Create associate.txt with paired timestamps (use filenames for timestamps here)
with open(os.path.join(dst_dir, "associate.txt"), "w") as f:
    left_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith(".png")])
    right_files = sorted([f for f in os.listdir(depth_dir) if f.endswith(".png")])

    for left, right in zip(left_files, right_files):
        # Using filename (e.g. 00000.png) as timestamp for simplicity
        ts_left = os.path.splitext(left)[0]
        ts_right = os.path.splitext(right)[0]
        f.write(f"{ts_left} rgb/{left} {ts_right} depth/{right}\n")

print(f"ORB-SLAM2 dataset prepared in {dst_dir}")
