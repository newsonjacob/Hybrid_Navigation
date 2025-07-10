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
#include <sys/stat.h>   // Add this


using namespace std;

// Forward declaration for log_event
static void log_event(const std::string& msg);

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
            std::ostringstream oss;
            oss << "recv_all failed: received=" << received << ", total=" << total << ", expected=" << len;
            log_event(oss.str());
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
    freopen("/mnt/h/Documents/AirSimExperiments/Hybrid_Navigation/logs/slam_console.txt", "w", stdout);
    freopen("/mnt/h/Documents/AirSimExperiments/Hybrid_Navigation/logs/slam_console_err.txt", "w", stderr);

    if (argc < 3) {
        cerr << "Usage: ./tcp_slam_server path_to_vocabulary path_to_settings [log_file_path]" << endl;
        return 1;
    }

    #ifdef _WIN32
        _mkdir("H:\\Documents\\AirSimExperiments\\Hybrid_Navigation\\logs");
    #else
        mkdir("/mnt/h/Documents/AirSimExperiments/Hybrid_Navigation/logs", 0777);
    #endif

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
        int snapshot_limit = 5;         // Declare at the top of the loop
        float max_depth_m = 10.0f;      // Declare at the top of the loop

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
        
        {
            std::ostringstream oss;
            oss << "Decoded RGB image resolution: " << rgb_width << " x " << rgb_height << " (" << rgb_bytes << " bytes)";
            log_event(oss.str());
        }

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

        // --- Check if images are changing ---
        static cv::Mat prev_rgb;
        if (!prev_rgb.empty()) {
            cv::Mat diff;
            cv::absdiff(imRGB, prev_rgb, diff);
            double frame_diff = cv::sum(diff)[0] / (imRGB.rows * imRGB.cols * 3);  // Normalize by total pixels * channels
            std::ostringstream motion_msg;
            motion_msg << "[DEBUG] RGB frame difference (mean): " << frame_diff;
            log_event(motion_msg.str());
        }
        prev_rgb = imRGB.clone();

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

        // Convert network byte order to host byte order
        d_height = ntohl(d_height);
        d_width  = ntohl(d_width);
        d_bytes  = ntohl(d_bytes);

        {
            std::ostringstream oss;
            oss << "Depth header: height=" << d_height << ", width=" << d_width << ", bytes=" << d_bytes;
            log_event(oss.str());
        }

        // Check if depth bytes match expected size
        if (d_bytes != d_height * d_width * 4) {
            std::ostringstream oss;
            oss << "Disconnect: Depth byte count mismatch. Expected: "
                << (d_height * d_width * 4) << ", Got: " << d_bytes;
            log_event(oss.str());
            clean_exit = false;
            break;
        }

        // --- Receive raw depth data ---
        std::vector<char> raw_depth(d_bytes);
        if (!recv_all(sock, raw_depth.data(), d_bytes)) {
            log_event("Disconnect: Failed to receive full depth image data.");
            clean_exit = false;
            break;
        }

        // --- Convert raw depth data to OpenCV Matrix (cv::Mat) ---
        float* float_ptr = reinterpret_cast<float*>(raw_depth.data());
        cv::Mat imD_raw(d_height, d_width, CV_32F, float_ptr);
        cv::Mat imD = imD_raw.clone();  // clone from raw buffer
        
        // --- Add a histogram log to check depth spread ---
        double hist_min, hist_max;
        cv::minMaxLoc(imD_raw, &hist_min, &hist_max);
        std::ostringstream hist_msg;
        hist_msg << "[DEBUG] Raw depth stats — min: " << hist_min << ", max: " << hist_max;
        log_event(hist_msg.str());

        // --- Save raw depth image for debugging ---
        if (frame_counter % 25 == 0) {
            std::ostringstream raw_depth_filename;
            raw_depth_filename << "/mnt/h/Documents/AirSimExperiments/Hybrid_Navigation/logs/frame_depth_raw_" << frame_counter << ".png";

            cv::Mat raw_depth_vis;
            imD_raw.convertTo(raw_depth_vis, CV_8U, 255.0 / max_depth_m); // Now max_depth_m is in scope
            cv::imwrite(raw_depth_filename.str(), raw_depth_vis);
        }

        // Clamp depth to max range (e.g. 5m or 10m)
        cv::Mat imD_clipped;
        cv::threshold(imD, imD_clipped, max_depth_m, max_depth_m, cv::THRESH_TRUNC);

        // Save visualization for debug
        if (frame_counter % 25 == 0) {
            std::ostringstream depth_filename;
            depth_filename << "/mnt/h/Documents/AirSimExperiments/Hybrid_Navigation/logs/frame_depth_" << frame_counter << ".png";

            cv::Mat depth_vis;
            imD_clipped.convertTo(depth_vis, CV_8U, 255.0 / max_depth_m);
            cv::imwrite(depth_filename.str(), depth_vis);
        }

        // --- Apply depth filtering to reduce tracking issues from flat or far geometry ---

        // Replace invalid or unhelpful depth (e.g. very near or very far)
        cv::Mat imD_filtered = imD_clipped.clone();
        const float min_depth = 0.1f;
        const float max_depth = 8.0f;
        for (int y = 0; y < imD_filtered.rows; ++y) {
            float* row = imD_filtered.ptr<float>(y);
            for (int x = 0; x < imD_filtered.cols; ++x) {
                float d = row[x];
                if (d < min_depth || d > max_depth || std::isnan(d))
                    row[x] = 0.0f; // ORB-SLAM2 treats 0 as invalid
            }
        }

        // Optional: filter out flat areas (low gradient)
        cv::Mat grad_x, grad_y;
        cv::Sobel(imD_filtered, grad_x, CV_32F, 1, 0, 3);
        cv::Sobel(imD_filtered, grad_y, CV_32F, 0, 1, 3);
        cv::Mat grad_mag;
        cv::magnitude(grad_x, grad_y, grad_mag);

        // Zero out pixels with almost no depth change (flat walls/floors)
        float flat_thresh = 0.001f;
        for (int y = 0; y < grad_mag.rows; ++y) {
            float* d_row = imD_filtered.ptr<float>(y);
            const float* g_row = grad_mag.ptr<float>(y);
            for (int x = 0; x < grad_mag.cols; ++x) {
                if (g_row[x] < flat_thresh)
                    d_row[x] = 0.0f;
            }
        }

        imD = imD_filtered;
        if (frame_counter % 25 == 0) {
            cv::Mat debug_vis;
            imD.convertTo(debug_vis, CV_8U, 255.0 / max_depth);  // Scale for viewing
            cv::imwrite("/mnt/h/Documents/AirSimExperiments/Hybrid_Navigation/logs/frame_depth_filtered_" + std::to_string(frame_counter) + ".png", debug_vis);
        }

        // Log depth value at center pixel (optional, but not used for logic anymore)
        int cx = d_width / 2;
        int cy = d_height / 2;
        float center_depth = imD.at<float>(cy, cx);
        {
            std::ostringstream oss;
            oss << "Depth at center (" << cx << ", " << cy << ") = "
                << std::fixed << std::setprecision(3) << center_depth << " meters";
            log_event(oss.str());
        }

        // ✅ NEW: count valid depth pixels instead of using center only
        int valid_depth_pixels = cv::countNonZero(imD > 0.0f);
        log_event("[DEBUG] Valid depth pixels after filtering: " + std::to_string(valid_depth_pixels));

        if (valid_depth_pixels < 300) {
            log_event("[WARN] Too few valid depth pixels — skipping SLAM update.");
            frame_counter++;
            continue;
        }


        double timestamp = (double)cv::getTickCount() / cv::getTickFrequency();
        // Log timestamp delta between frames
        static double last_ts = -1.0;
        if (last_ts >= 0.0) {
            double dt = timestamp - last_ts;
            std::ostringstream delta_log;
            delta_log << "[DEBUG] Timestamp delta: " << std::fixed << std::setprecision(6) << dt << " seconds";
            log_event(delta_log.str());
        }
        last_ts = timestamp;

        {
            std::ostringstream ts_msg;
            ts_msg << "[DEBUG] Timestamp at Frame #" << frame_counter << ": " << std::fixed << std::setprecision(6) << timestamp;
            log_event(ts_msg.str());
        }


        // --- Process with SLAM ---
        {
            std::ostringstream oss;
            oss << "Calling SLAM.TrackRGBD at timestamp=" << std::fixed << std::setprecision(6) << timestamp
                << " | Frame #" << frame_counter;
            log_event(oss.str());
        }
        try {
            double min_val, max_val;
            cv::minMaxLoc(imD, &min_val, &max_val);
            std::ostringstream d_stats;
            d_stats << "Depth image stats — min: " << min_val << ", max: " << max_val;
            log_event(d_stats.str());

            // Log the depth image size
            SLAM.TrackRGBD(imRGB, imD, timestamp);
            
            if (SLAM.GetTracker()->mState == ORB_SLAM2::Tracking::LOST) {
                log_event("[WARN] SLAM tracking lost — resetting system.");
                SLAM.Reset();
            }
            
            // Log the current frame's timestamp
            cv::Mat Tcw_copy = SLAM.GetTracker()->mCurrentFrame.mTcw.clone();
            static cv::Mat prev_Tcw;
            if (!prev_Tcw.empty() && prev_Tcw.size() == Tcw_copy.size()) {
                cv::Mat diff = Tcw_copy - prev_Tcw;
                double delta = cv::norm(diff, cv::NORM_L2);
                std::ostringstream pose_msg;
                pose_msg << "[DEBUG] Pose matrix difference from previous frame: " << delta;
                log_event(pose_msg.str());
            }
            prev_Tcw = Tcw_copy.clone();

            log_event("Tcw_copy rows: " + std::to_string(Tcw_copy.rows) +
                    ", cols: " + std::to_string(Tcw_copy.cols) +
                    ", type: " + std::to_string(Tcw_copy.type()));
            cv::Mat identity = cv::Mat::eye(4, 4, Tcw_copy.type());

            // Check if Tcw_copy is close to identity matrix
            double frobenius_norm = cv::norm(Tcw_copy - identity, cv::NORM_L2);
            if (frobenius_norm < 1e-3) {
                static int identity_frame_count = 0;
                identity_frame_count++;
                log_event("[WARN] Tcw appears to be an identity matrix — no motion detected. Count: " + std::to_string(identity_frame_count));
            }


            // Defensive checks
            bool tcw_valid = true;
            if (Tcw_copy.empty() || Tcw_copy.rows != 4 || Tcw_copy.cols != 4) {
                log_event("[WARN] Tcw_copy is empty or not 4x4.");
                tcw_valid = false;
            }
            if (tcw_valid && (Tcw_copy.type() != CV_32F && Tcw_copy.type() != CV_64F)) {
                log_event("[WARN] Tcw_copy is not CV_32F or CV_64F.");
                tcw_valid = false;
            }
            if (tcw_valid && cv::countNonZero(Tcw_copy != Tcw_copy) > 0) {
                log_event("[ERROR] Tcw_copy contains NaNs.");
                tcw_valid = false;
            }
            if (tcw_valid && !cv::checkRange(Tcw_copy)) {
                log_event("[ERROR] Tcw_copy contains Infs or out-of-range values.");
                tcw_valid = false;
            }

            if (tcw_valid) {
                try {
                    cv::Mat Twc = Tcw_copy.inv();
                    if (!Twc.empty() && Twc.rows >= 3 && Twc.cols >= 4) {
                        float x = Twc.at<float>(0, 3);
                        float y = Twc.at<float>(1, 3);
                        float z = Twc.at<float>(2, 3);
                        std::ostringstream twc_log;
                        twc_log << "Camera center (twc): " << x << ", " << y << ", " << z;
                        log_event(twc_log.str());
                    } else {
                        log_event("[WARN] Twc is invalid after inversion — skipping camera center log.");
                    }
                } catch (const cv::Exception& e) {
                    log_event(std::string("[WARN] Exception during Twc inversion: ") + e.what());
                }
            } else {
                log_event("[WARN] Tcw_copy invalid — skipping pose log.");
            }


        } catch (const std::exception& ex) {
            std::ostringstream oss;
            oss << "Exception in SLAM.TrackRGBD: " << ex.what();
            log_event(oss.str());
            std::cerr << "[ERROR] Exception in SLAM.TrackRGBD: " << ex.what() << std::endl;
        } catch (...) {
            log_event("Unknown exception in SLAM.TrackRGBD.");
            std::cerr << "[ERROR] Unknown exception in SLAM.TrackRGBD." << std::endl;
        }

        // Save first N frames for visual inspection
        if (frame_counter % 25 == 0) {
            std::ostringstream rgb_filename, depth_filename;
            rgb_filename << "/mnt/h/Documents/AirSimExperiments/Hybrid_Navigation/logs/frame_rgb_" << frame_counter << ".png";
            depth_filename << "/mnt/h/Documents/AirSimExperiments/Hybrid_Navigation/logs/frame_depth_" << frame_counter << ".png";

            cv::imwrite(rgb_filename.str(), imRGB);

            // Save depth visualization that was already clamped earlier (imD is already imD_clipped)
            cv::Mat depth_vis;
            imD.convertTo(depth_vis, CV_8U, 255.0 / max_depth_m);
            cv::imwrite(depth_filename.str(), depth_vis);
        }

        // Add map tracking state logs
        auto tracker = SLAM.GetTracker();
        if (tracker) {
            int state = tracker->mState;
            int n_kfs = tracker->GetNumLocalKeyFrames();  // number of local keyframes
            int n_mappoints = tracker->mCurrentFrame.mvpMapPoints.size();  // number of tracked map points

            std::ostringstream oss;
            oss << "TrackingState=" << state
                << ", KeyFrames=" << n_kfs
                << ", MapPoints=" << n_mappoints
                << ", TrackedFramePoints=" << n_mappoints
                << ", ORBFeatures=" << tracker->mCurrentFrame.N;
            log_event(oss.str());
        }

        // Optional: save ORB keypoints on RGB image
        if (frame_counter % 25 == 0) {
            std::vector<cv::KeyPoint> orb_kps = tracker->mCurrentFrame.mvKeys;
            cv::Mat rgb_kp;
            cv::drawKeypoints(imRGB, orb_kps, rgb_kp);
            std::ostringstream kp_filename;
            kp_filename << "/mnt/h/Documents/AirSimExperiments/Hybrid_Navigation/logs/frame_rgb_kp_" << frame_counter << ".png";
            cv::imwrite(kp_filename.str(), rgb_kp);
        }


        // --- After SLAM.TrackRGBD ---
        cv::Mat Tcw = SLAM.GetTracker()->mCurrentFrame.mTcw;
        int track_state = SLAM.GetTracker()->mState;

        if (pose_sock >= 0 && !Tcw.empty() &&
            Tcw.rows == 4 && Tcw.cols == 4 &&
            Tcw.type() == CV_32F &&
            track_state >= ORB_SLAM2::Tracking::OK &&
            cv::checkRange(Tcw)) {

            // Ensure Tcw is a 4x4 matrix
            std::ostringstream pose_stream;
            pose_stream << "Raw Tcw values: ";
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 4; ++c)
                    pose_stream << Tcw.at<float>(r, c) << ", ";
            log_event(pose_stream.str());
            // Prepare the Tcw matrix for sending
            std::ostringstream oss;
            oss << "Sending Tcw | State: " << track_state << " | Matrix: ";
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 4; ++c) {
                    float val = Tcw.at<float>(r, c);
                    if (std::isnan(val) || std::abs(val) < 1e-5)
                        val = 0.0;
                    oss << val << (r == 2 && c == 3 ? "" : ", ");
                }
            }

            // ✅ Now append translation norm after the loop:
            float tx_raw = Tcw.at<float>(0, 3);
            float ty_raw = Tcw.at<float>(1, 3);
            float tz_raw = Tcw.at<float>(2, 3);

            // Clamp tiny noise to 0.0
            float tx = std::abs(tx_raw) < 1e-5 ? 0.0f : tx_raw;
            float ty = std::abs(ty_raw) < 1e-5 ? 0.0f : ty_raw;
            float tz = std::abs(tz_raw) < 1e-5 ? 0.0f : tz_raw;

            float motion = std::sqrt(tx*tx + ty*ty + tz*tz);

            // Log the translation vector components
            std::ostringstream vec_log;
            vec_log << "[DEBUG] Translation vector components — ΔX: " << tx
                    << ", ΔY: " << ty
                    << ", ΔZ: " << tz;
            log_event(vec_log.str());



            oss << " | TranslationNorm=" << std::fixed << std::setprecision(6) << motion;
            log_event(oss.str());

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
        // Log translation norm even if Tcw is invalid or tracking lost
        if (!Tcw.empty() && Tcw.rows == 4 && Tcw.cols == 4 && Tcw.type() == CV_32F) {
            float tx = Tcw.at<float>(0, 3);
            float ty = Tcw.at<float>(1, 3);
            float tz = Tcw.at<float>(2, 3);
            float motion = std::sqrt(tx * tx + ty * ty + tz * tz);

            std::ostringstream fallback_log;
            fallback_log << "[INFO] Pose (regardless of track state): tx=" << tx
                        << ", ty=" << ty << ", tz=" << tz
                        << " | Norm=" << std::fixed << std::setprecision(6) << motion;
            log_event(fallback_log.str());
        }

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