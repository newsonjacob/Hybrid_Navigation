// Minimal test function for pose sender
#include <arpa/inet.h>
#include <unistd.h>
#include <iostream>
#include <cstring>

int main() {
    const char* POSE_RECEIVER_IP = "192.168.1.102"; // or your receiver's IP
    const int POSE_RECEIVER_PORT = 6001;

    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
        std::cerr << "Socket creation failed\n";
        return 1;
    }

    sockaddr_in pose_addr;
    pose_addr.sin_family = AF_INET;
    pose_addr.sin_port = htons(POSE_RECEIVER_PORT);
    inet_pton(AF_INET, POSE_RECEIVER_IP, &pose_addr.sin_addr);

    std::cout << "Connecting to Python pose receiver...\n";
    if (connect(sock, (struct sockaddr*)&pose_addr, sizeof(pose_addr)) < 0) {
        std::cerr << "Connect failed\n";
        close(sock);
        return 1;
    }
    std::cout << "Connected!\n";

    // Send dummy 3x4 pose matrix (12 floats)
    float data[12] = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f};
    int bytes = 12 * sizeof(float);
    int sent = send(sock, reinterpret_cast<char*>(data), bytes, 0);
    std::cout << "Sent " << sent << " bytes\n";

    close(sock);
    return 0;
}