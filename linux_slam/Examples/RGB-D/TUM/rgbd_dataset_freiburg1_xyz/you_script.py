import cv2
import os

rgb_path = '/home/jacob/slam_ws/ORB_SLAM2/Examples/RGB-D/TUM/rgbd_dataset_freiburg1_xyz/rgb/1305031102.175304.png'
depth_path = '/home/jacob/slam_ws/ORB_SLAM2/Examples/RGB-D/TUM/rgbd_dataset_freiburg1_xyz/depth/1305031102.160407.png'

if os.path.exists(rgb_path):
    rgb_img = cv2.imread(rgb_path)
    print('RGB Image loaded:', rgb_img is not None, 'Shape:', rgb_img.shape if rgb_img is not None else 'N/A')
else:
    print('RGB file not found:', rgb_path)

if os.path.exists(depth_path):
    depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    print('Depth Image loaded:', depth_img is not None, 'Shape:', depth_img.shape if depth_img is not None else 'N/A')
else:
    print('Depth file not found:', depth_path)
