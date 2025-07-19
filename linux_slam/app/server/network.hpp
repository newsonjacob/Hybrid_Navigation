#pragma once
#include <opencv2/core/core.hpp>
namespace slam_server {
extern const char* POSE_RECEIVER_IP;
extern const int POSE_RECEIVER_PORT;

int create_server_socket(int port);
int accept_client(int server_fd);
int connect_pose_sender(const char* ip, int port);

bool recv_all(int sock, char* buffer, int len);
bool send_pose(int pose_sock, const cv::Mat& Tcw);
void cleanup_resources(int sock, int server_fd, int pose_sock);
}
