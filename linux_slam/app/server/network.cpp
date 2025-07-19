#include "network.hpp"
#include "logging.hpp"
#include <arpa/inet.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <cstring>
#include <iostream>

namespace slam_server {

const char* POSE_RECEIVER_IP = getenv("POSE_RECEIVER_IP") ? getenv("POSE_RECEIVER_IP") : "192.168.1.102";
const int POSE_RECEIVER_PORT = getenv("POSE_RECEIVER_PORT") ? atoi(getenv("POSE_RECEIVER_PORT")) : 6001;

int create_server_socket(int port) {
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd == -1) {
        log_event("[ERROR] Socket creation failed for image stream.");
        return -1;
    }
    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt));

    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(port);

    if (bind(server_fd, (struct sockaddr*)&address, sizeof(address)) < 0) {
        log_event("[ERROR] Bind failed for server socket.");
        close(server_fd);
        return -1;
    }
    if (listen(server_fd, 1) < 0) {
        log_event("[ERROR] Listen failed for server socket.");
        close(server_fd);
        return -1;
    }
    return server_fd;
}

int accept_client(int server_fd) {
    sockaddr_in address{};
    socklen_t addrlen = sizeof(address);
    int sock = accept(server_fd, (struct sockaddr*)&address, &addrlen);
    if (sock < 0) {
        log_event("[ERROR] Failed to accept client connection.");
    }
    return sock;
}

int connect_pose_sender(const char* ip, int port) {
    int pose_sock = socket(AF_INET, SOCK_STREAM, 0);
    if (pose_sock < 0) {
        log_event("[ERROR] Could not create pose sender socket.");
        return -1;
    }
    sockaddr_in pose_addr{};
    pose_addr.sin_family = AF_INET;
    pose_addr.sin_port = htons(port);
    inet_pton(AF_INET, ip, &pose_addr.sin_addr);

    for (int attempt = 0; attempt < 10; ++attempt) {
        if (connect(pose_sock, (struct sockaddr*)&pose_addr, sizeof(pose_addr)) >= 0) {
            log_event("[INFO] Connected to Python pose receiver.");
            return pose_sock;
        }
        log_event("[WARN] Failed to connect to pose receiver, retrying...");
        sleep(1);
    }
    log_event("[ERROR] Failed to connect to Python pose receiver after retries.");
    close(pose_sock);
    return -1;
}

bool recv_all(int sock, char* buffer, int len) {
    int total = 0;
    int retries = 0;
    while (total < len) {
        int received = recv(sock, buffer + total, len - total, 0);
        if (received == 0 && total == 0 && retries < 1) {
            log_event("recv() returned 0 on first attempt — retrying once...");
            retries++;
            sleep(1);
            continue;
        }
        if (received <= 0) {
            log_event("recv_all failed");
            return false;
        }
        total += received;
    }
    return true;
}

bool send_pose(int pose_sock, const cv::Mat& Tcw) {
    if (Tcw.empty() || Tcw.rows != 4 || Tcw.cols != 4)
        return false;
    cv::Mat Tcw_f;
    if (Tcw.type() != CV_32F)
        Tcw.convertTo(Tcw_f, CV_32F);
    else
        Tcw_f = Tcw;
    float data[12];
    for (int r=0;r<3;++r)
        for (int c=0;c<4;++c)
            data[r*4+c]=Tcw_f.at<float>(r,c);
    int bytes = 12*sizeof(float);
    int sent = send(pose_sock, reinterpret_cast<char*>(data), bytes, 0);
    return sent==bytes;
}

void cleanup_resources(int sock, int server_fd, int pose_sock) {
    if (sock >= 0) close(sock);
    if (server_fd >= 0) close(server_fd);
    if (pose_sock >= 0) close(pose_sock);
}

} // namespace slam_server
