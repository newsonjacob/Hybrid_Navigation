#include "slam_runner.hpp"
#include "logging.hpp"
#include <opencv2/core/core.hpp>

namespace slam_server {

cv::Mat get_pose_covariance_with_inliers(const cv::Mat& current_pose, const cv::Mat& previous_pose) {
    if (current_pose.empty() || previous_pose.empty() || current_pose.rows != 4 || current_pose.cols != 4 ||
        previous_pose.rows != 4 || previous_pose.cols != 4) {
        log_event("[ERROR] Invalid pose matrices.");
        return cv::Mat();
    }
    cv::Mat pose_diff = current_pose - previous_pose;
    double norm = cv::norm(pose_diff, cv::NORM_L2);
    double uncertainty = std::max(norm, 0.1);
    cv::Mat covariance = cv::Mat::eye(4, 4, CV_32F) * uncertainty;
    return covariance;
}

int get_feature_inliers(ORB_SLAM2::System &SLAM) {
    auto tracker = SLAM.GetTracker();
    if (!tracker) {
        log_event("[ERROR] Tracker not available in SLAM system.");
        return -1;
    }
    int inliers = tracker->mCurrentFrame.N;
    log_event("[DEBUG] Inliers tracked: " + std::to_string(inliers));
    return inliers;
}

} // namespace slam_server
