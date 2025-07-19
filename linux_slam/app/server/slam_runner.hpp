#pragma once
#include <System.h>
#include <opencv2/core/core.hpp>

namespace slam_server {
cv::Mat get_pose_covariance_with_inliers(const cv::Mat& current_pose, const cv::Mat& previous_pose);
int get_feature_inliers(ORB_SLAM2::System &SLAM);
}
