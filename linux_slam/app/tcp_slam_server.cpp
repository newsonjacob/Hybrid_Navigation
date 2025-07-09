#include <System.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <vector>
#include <arpa/inet.h>
#include <unistd.h>
#include <fstream>
#include <ctime>
#include <iomanip>
#include <thread>
#include <chrono>
#include <sstream>
#include <mutex>

using namespace std;

// Helper function to receive exactly n bytes
bool recv_all(int sock, char* buffer, int len) {
    int total = 0;
    int retries = 0;

    while (total < len) {
        int received = recv(sock, buffer + total, len - total, 0);

        if (received == 0 && total == 0 && retries < 1) {
            std::cerr << "[WARN] recv() returned 0 on first attempt — retrying once..." << std::endl;
            retries++;
            sleep(1); // brief pause to wait for client
            continue;
        }

        if (received <= 0) {
            std::cerr << "[ERROR] recv() returned " << received << " at byte " << total << " of " << len << std::endl;
            return false;
        }

        total += received;
    }
    return true;
}

#include <mutex> // for thread safety
static std::string g_log_file_path; // Global variable to hold the log file path
static std::mutex g_log_mutex;
// Thread-safe logging function.
// Writes the given message to the log file specified by g_log_file_path, prefixing each entry with a timestamp.
static void log_event(const std::string& msg) {
    static bool warned = false;
    static std::unique_ptr<std::ofstream> log_stream;

    if (g_log_file_path.empty()) {
        if (!warned) {
            std::cerr << "[WARN] Logging is disabled because g_log_file_path is empty." << std::endl;
            warned = true;
        }
        return;
    }

    std::lock_guard<std::mutex> lock(g_log_mutex); // Ensure thread safety when accessing the log file

    // Open log file only once
    if (!log_stream) {
        log_stream = std::make_unique<std::ofstream>(g_log_file_path, std::ios::app);
        if (!log_stream->is_open() && !warned) {
            std::cerr << "[WARN] Could not open log file: " << g_log_file_path << std::endl;
            warned = true;
        }
    }

    if (log_stream && log_stream->is_open()) { // Check if the log stream is valid
        std::time_t t = std::time(nullptr);
        // Timestamp format: "%F %T" means "YYYY-MM-DD HH:MM:SS"
        (*log_stream) << "[" << std::put_time(std::localtime(&t), "%F %T") << "] " << msg << std::endl;
        log_stream->flush(); // Ensure log is written immediately
    }
}

const char* POSE_RECEIVER_IP = "192.168.1.102"; // Python receiver IP
const int   POSE_RECEIVER_PORT = 6001;      // Python receiver port

// Helper to send pose as 12 floats (row-major 3x4 matrix)
bool send_pose(int pose_sock, const cv::Mat& Tcw) {
    if (Tcw.empty() || Tcw.rows != 4 || Tcw.cols != 4)
        return false;
    float data[12];
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 4; ++c)
            data[r * 4 + c] = Tcw.at<float>(r, c);
    int bytes = 12 * sizeof(float);
    int sent = send(pose_sock, reinterpret_cast<char*>(data), bytes, 0);
    return sent == bytes;
}

// ------- Main function to set up the TCP server, receive images, and process them with ORB-SLAM2 -------
int main(int argc, char **argv) {
    if (argc < 3) {
        cerr << "Usage: ./tcp_slam_server path_to_vocabulary path_to_settings [log_file_path]" << endl;
        return 1;
    }

    std::string vocab = argv[1];
    std::string settings = argv[2];

    // Set log file path with timestamp if not provided
    if (argc >= 4) {
        g_log_file_path = argv[3];
    } else {
        std::ostringstream oss;
        std::time_t t = std::time(nullptr);
        char timebuf[32];
        std::strftime(timebuf, sizeof(timebuf), "%Y%m%d_%H%M%S", std::localtime(&t));
        oss << "/mnt/h/Documents/AirSimExperiments/Hybrid_Navigation/logs/slam_server_debug_" << timebuf << ".log";
        g_log_file_path = oss.str();
    }

    // --- Setup pose sent log file ---
    std::ostringstream pose_log_oss;
    std::time_t pose_log_t = std::time(nullptr);
    char pose_log_timebuf[32];
    std::strftime(pose_log_timebuf, sizeof(pose_log_timebuf), "%Y%m%d_%H%M%S", std::localtime(&pose_log_t));
    pose_log_oss << "/mnt/h/Documents/AirSimExperiments/Hybrid_Navigation/logs/pose_sent_" << pose_log_timebuf << ".log";
    std::string pose_log_file_path = pose_log_oss.str();
    std::ofstream pose_log_stream(pose_log_file_path, std::ios::app);

    log_event("=== SLAM TCP Server launched ===");
    std::cout << "[BOOT] SLAM TCP Server launched" << std::endl;

    // Initialize ORB-SLAM2
    ORB_SLAM2::System SLAM(vocab, settings, ORB_SLAM2::System::RGBD, true);
    cout << "[INFO] SLAM system initialized." << endl;

    // -- Setup TCP server --
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd == -1) {
        cerr << "Socket creation failed" << endl;
        return 1;
    }
    // Set socket options to allow reuse of the address and port
    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt));

    sockaddr_in address;
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;  // binds to 0.0.0.0 — all interfaces
    address.sin_port = htons(6000);

    if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0) {
        cerr << "Bind failed" << endl;
        close(server_fd);
        return 1;
    }

    if (listen(server_fd, 1) < 0) {
        cerr << "Listen failed" << endl;
        close(server_fd);
        return 1;
    }

    cout << "[INFO] Waiting for Python streamer on port 6000..." << endl;

    int addrlen = sizeof(address);

    log_event("Calling accept() and waiting for streamer...");
    std::cout << "[INFO] Calling accept() and waiting for streamer..." << std::endl;
    std::cout.flush();

    int sock = accept(server_fd, (struct sockaddr *)&address, (socklen_t*)&addrlen);
    if (sock < 0) {
        log_event("[ERROR] Failed to accept() client connection.");
        cerr << "Accept failed" << endl;
        close(server_fd);
        return 1;
    }
    log_event("✅ Client connection accepted");
    std::cout << "[BOOT] Client connection accepted" << std::endl;
    std::cout.flush();

    log_event("Client connection accepted, entering image receive loop.");

    if (sock < 0) {
        cerr << "Accept failed" << endl;
        close(server_fd);
        return 1;
    }
    cout << "[INFO] Connected to Python streamer!" << endl;

    bool slam_ready_flag_written = false;
    bool clean_exit = true;
    int frame_counter = 0;

    // --- Setup pose sender socket ---
    int pose_sock = socket(AF_INET, SOCK_STREAM, 0);
    if (pose_sock < 0) {
        log_event("[ERROR] Could not create pose sender socket.");
        pose_sock = -1;
    } else {
        sockaddr_in pose_addr;
        pose_addr.sin_family = AF_INET;
        pose_addr.sin_port = htons(POSE_RECEIVER_PORT);
        inet_pton(AF_INET, POSE_RECEIVER_IP, &pose_addr.sin_addr);
        log_event("Connecting to Python pose receiver...");
        int pose_conn = connect(pose_sock, (struct sockaddr*)&pose_addr, sizeof(pose_addr));
        if (pose_conn < 0) {
            log_event("[ERROR] Could not connect to Python pose receiver.");
            close(pose_sock);
            pose_sock = -1;
        } else {
            log_event("Connected to Python pose receiver.");
        }
    }

    while (true) {
        log_event("----- Begin image receive loop -----");

        double loop_timestamp = (double)cv::getTickCount() / cv::getTickFrequency();
        {
            std::ostringstream oss;
            oss << "Frame #" << frame_counter << " | Loop timestamp: " << std::fixed << std::setprecision(6) << loop_timestamp;
            log_event(oss.str());
        }

        // --- Receive RGB image ---
        char rgb_header[12];
        uint32_t net_height, net_width, net_bytes;
        uint32_t rgb_height, rgb_width, rgb_bytes;

        log_event("Waiting to receive 12-byte RGB header...");
        bool got_header = recv_all(sock, rgb_header, 12);
        if (!got_header) {
            std::cerr << "[ERROR] Failed to receive full 12-byte RGB header." << std::endl;
            log_event("Failed to receive full 12-byte RGB header. Closing connection (likely client disconnect or crash).");
            std::ofstream fail_flag("/mnt/h/Documents/AirSimExperiments/Hybrid_Navigation/flags/slam_failed.flag");
            if (fail_flag.is_open()) fail_flag.close();
            clean_exit = false;
            break;
        }
        std::ostringstream oss;
        oss << "Raw RGB header bytes: ";
        for (int i = 0; i < 12; ++i) {
            oss << std::hex << std::uppercase << std::setw(2) << std::setfill('0')
                << (static_cast<unsigned int>(static_cast<unsigned char>(rgb_header[i]))) << " ";
        }
        log_event(oss.str());
        log_event("Received 12-byte RGB header.");

        memcpy(&net_height, rgb_header, 4);
        memcpy(&net_width,  rgb_header + 4, 4);
        memcpy(&net_bytes, rgb_header + 8, 4);
        rgb_height = ntohl(net_height);
        rgb_width  = ntohl(net_width);
        rgb_bytes  = ntohl(net_bytes);

        if (rgb_bytes != rgb_height * rgb_width * 3) {
            std::cerr << "[ERROR] RGB byte count mismatch. Expected: "
                      << (rgb_height * rgb_width * 3) << ", Got: " << rgb_bytes << std::endl;
            log_event("Disconnect: RGB byte count mismatch.");
            clean_exit = false;
            break;
        }

        vector<uchar> rgb_buffer(rgb_bytes);
        if (!recv_all(sock, (char*)rgb_buffer.data(), rgb_bytes)) {
            log_event("Disconnect: Failed to receive full RGB image data.");
            clean_exit = false;
            break;
        }

        log_event("About to create slam_ready.flag (pre-check passed)");
        if (!slam_ready_flag_written) {
            std::ofstream flag_file("/mnt/h/Documents/AirSimExperiments/Hybrid_Navigation/flags/slam_ready.flag");
            if (flag_file.is_open()) {
                log_event("slam_ready.flag opened successfully for writing.");
                flag_file.close();
                std::cout << "[INFO] Created slam_ready.flag — SLAM server ready." << std::endl;
                log_event("Created slam_ready.flag — SLAM server ready.");
            } else {
                std::cerr << "[WARN] Could not create slam_ready.flag." << std::endl;
                log_event("Could not create slam_ready.flag.");
            }
            slam_ready_flag_written = true;
        }

        cv::Mat imRGB(rgb_height, rgb_width, CV_8UC3, rgb_buffer.data());
        imRGB = imRGB.clone();

        // --- Receive Depth image ---
        uint32_t d_height, d_width, d_bytes;
        if (!recv_all(sock, (char*)&d_height, 4)) {
            log_event("Disconnect: Failed to receive depth height.");
            clean_exit = false;
            break;
        }
        if (!recv_all(sock, (char*)&d_width, 4)) {
            log_event("Disconnect: Failed to receive depth width.");
            clean_exit = false;
            break;
        }
        if (!recv_all(sock, (char*)&d_bytes, 4)) {
            log_event("Disconnect: Failed to receive depth bytes.");
            clean_exit = false;
            break;
        }

        d_height = ntohl(d_height);
        d_width  = ntohl(d_width);
        d_bytes  = ntohl(d_bytes);

        {
            std::ostringstream oss;
            oss << "Depth header: height=" << d_height << ", width=" << d_width << ", bytes=" << d_bytes;
            log_event(oss.str());
        }

        if (d_bytes != d_height * d_width * 4) {
            std::ostringstream oss;
            oss << "Disconnect: Depth byte count mismatch. Expected: "
                << (d_height * d_width * 4) << ", Got: " << d_bytes;
            log_event(oss.str());
            clean_exit = false;
            break;
        }

        std::vector<char> raw_depth(d_bytes);
        if (!recv_all(sock, raw_depth.data(), d_bytes)) {
            log_event("Disconnect: Failed to receive full depth image data.");
            clean_exit = false;
            break;
        }

        float* float_ptr = reinterpret_cast<float*>(raw_depth.data());
        cv::Mat imD(d_height, d_width, CV_32F, float_ptr);
        imD = imD.clone();

        double timestamp = (double)cv::getTickCount() / cv::getTickFrequency();

        // --- Process with SLAM ---
        {
            std::ostringstream oss;
            oss << "Calling SLAM.TrackRGBD at timestamp=" << std::fixed << std::setprecision(6) << timestamp
                << " | Frame #" << frame_counter;
            log_event(oss.str());
        }
        try {
            SLAM.TrackRGBD(imRGB, imD, timestamp);
        } catch (const std::exception& ex) {
            std::ostringstream oss;
            oss << "Exception in SLAM.TrackRGBD: " << ex.what();
            log_event(oss.str());
            std::cerr << "[ERROR] Exception in SLAM.TrackRGBD: " << ex.what() << std::endl;
        } catch (...) {
            log_event("Unknown exception in SLAM.TrackRGBD.");
            std::cerr << "[ERROR] Unknown exception in SLAM.TrackRGBD." << std::endl;
        }

        // --- After SLAM.TrackRGBD ---
        cv::Mat Tcw = SLAM.GetTracker()->mCurrentFrame.mTcw;
        if (pose_sock >= 0 && !Tcw.empty()) {
            if (send_pose(pose_sock, Tcw)) {
                log_event("Pose sent to Python receiver.");
                // Log the pose data to the dedicated pose log file
                if (pose_log_stream.is_open()) {
                    pose_log_stream << std::fixed << std::setprecision(6);
                    pose_log_stream << "Frame #" << frame_counter << " | ";
                    pose_log_stream << "Tcw: ";
                    for (int r = 0; r < 3; ++r)
                        for (int c = 0; c < 4; ++c)
                            pose_log_stream << Tcw.at<float>(r, c) << (r == 2 && c == 3 ? "" : ", ");
                    pose_log_stream << std::endl;
                    pose_log_stream.flush();
                }
            } else {
                log_event("[WARN] Failed to send pose to Python receiver.");
            }
        }

        frame_counter++;
    }

    if (clean_exit) {
        log_event("Image receive loop exited cleanly (no error).");
        std::cout << "[INFO] Image receive loop exited cleanly (no error)." << std::endl;
    } else {
        log_event("Image receive loop exited due to error or disconnect.");
        std::cout << "[INFO] Image receive loop exited due to error or disconnect." << std::endl;
    }
    log_event("Connection closed or error occurred. Shutting down main loop.");
    std::cout << "[INFO] Connection closed or error occurred. Shutting down." << std::endl;
    close(sock);
    close(server_fd);
    if (pose_sock >= 0) close(pose_sock);
    if (pose_log_stream.is_open()) pose_log_stream.close();
    SLAM.Shutdown();
    return 0;
}
