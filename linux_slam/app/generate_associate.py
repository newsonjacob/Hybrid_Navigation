import os
import re
from datetime import datetime

def extract_timestamp(filename):
    # Extract timestamp float from filename, e.g. 1305031102.175304.png
    match = re.search(r'(\d+\.\d+)', filename)
    return float(match.group(1)) if match else None

def get_sorted_images(folder):
    files = [f for f in os.listdir(folder) if f.endswith('.png')]
    # Create list of tuples (timestamp, filename)
    ts_files = [(extract_timestamp(f), f) for f in files]
    # Filter out files where timestamp could not be extracted
    ts_files = [t for t in ts_files if t[0] is not None]
    # Sort by timestamp
    ts_files.sort(key=lambda x: x[0])
    return ts_files

def associate(rgb_list, depth_list, max_diff=0.02):
    # Associate rgb and depth images by closest timestamp within max_diff seconds
    associations = []
    depth_idx = 0
    for rgb_ts, rgb_file in rgb_list:
        best_match = None
        best_diff = max_diff + 1
        while depth_idx < len(depth_list):
            depth_ts, depth_file = depth_list[depth_idx]
            diff = abs(rgb_ts - depth_ts)
            if diff < best_diff:
                best_diff = diff
                best_match = (depth_ts, depth_file)
                if depth_ts > rgb_ts:
                    break
            else:
                if depth_ts > rgb_ts:
                    break
            depth_idx += 1
        if best_diff <= max_diff:
            associations.append((rgb_ts, rgb_file, best_match[0], best_match[1]))
    return associations

def write_associations(associations, filename='associate_fixed.txt'):
    with open(filename, 'w') as f:
        for rgb_ts, rgb_file, depth_ts, depth_file in associations:
            line = f"{rgb_ts:.6f} {rgb_file} {depth_ts:.6f} {depth_file}\n"
            f.write(line)
    print(f"Written {len(associations)} associations to {filename}")

if __name__ == "__main__":
    rgb_list = get_sorted_images('rgb')
    depth_list = get_sorted_images('depth')
    associations = associate(rgb_list, depth_list)
    write_associations(associations)
