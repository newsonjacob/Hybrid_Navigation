import os
import sys
import numpy as np

def load_timestamps_and_files(folder):
    files = sorted(os.listdir(folder))
    timestamps = [float(os.path.splitext(f)[0]) for f in files]
    return timestamps, files

def associate(rgb_ts, depth_ts, max_diff=0.02):
    matches = []
    depth_idx = 0
    for rgb_time in rgb_ts:
        while depth_idx + 1 < len(depth_ts) and abs(depth_ts[depth_idx + 1] - rgb_time) < abs(depth_ts[depth_idx] - rgb_time):
            depth_idx += 1
        if abs(depth_ts[depth_idx] - rgb_time) < max_diff:
            matches.append((rgb_time, depth_idx))
    return matches

if __name__ == "__main__":
    rgb_folder = "rgb"
    depth_folder = "depth"

    rgb_ts, rgb_files = load_timestamps_and_files(rgb_folder)
    depth_ts, depth_files = load_timestamps_and_files(depth_folder)

    matches = associate(rgb_ts, depth_ts)

    with open("rgbd_dataset_freiburg1_xyz.association", "w") as f:
        for rgb_time, depth_idx in matches:
            rgb_file = rgb_files[rgb_ts.index(rgb_time)]
            depth_file = depth_files[depth_idx]
            f.write(f"{rgb_time} {rgb_folder}/{rgb_file} {depth_ts[depth_idx]} {depth_folder}/{depth_file}\n")

    print(f"Association file generated with {len(matches)} matches.")
