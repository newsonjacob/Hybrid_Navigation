import cv2
rgb_img = cv2.imread("/home/jacob/slam_ws/ORB_SLAM2/Examples/RGB-D/TUM/rgbd_dataset_freiburg1_xyz/rgb/1305031102.175304.png")
depth_img = cv2.imread("/home/jacob/slam_ws/ORB_SLAM2/Examples/RGB-D/TUM/rgbd_dataset_freiburg1_xyz/depth/1305031102.175304.png", cv2.IMREAD_UNCHANGED)
print(rgb_img.shape, depth_img.shape)
