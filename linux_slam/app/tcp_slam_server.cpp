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
            log_event("recv() returned 0 on first attempt — retrying once...");
            std::cerr << "[WARN] recv() returned 0 on first attempt — retrying once..." << std::endl;
            retries++;
            sleep(1); // brief pause to wait for client
            continue;
        }

        if (received <= 0) {
            log_event("recv() returned " + std::to_string(received) + " at byte " + std::to_string(total) + " of " + std::to_string(len));
            std::cerr << "[ERROR] recv() returned " << received << " at byte " << total << " of " << len << std::endl;
            std::ostringstream oss; // Create a string stream to format the error message
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
            log_event("[WARN] Logging is disabled because g_log_file_path is empty.");
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
            log_event("[WARN] Could not open log file: " + g_log_file_path);
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

const char* POSE_RECEIVER_IP = getenv("POSE_RECEIVER_IP") ? getenv("POSE_RECEIVER_IP") : "192.168.1.103"; // Python receiver IP
const int POSE_RECEIVER_PORT = getenv("POSE_RECEIVER_PORT") ? atoi(getenv("POSE_RECEIVER_PORT")) : 6001; // Python receiver port

// Helper to send pose as 12 floats (row-major 3x4 matrix)
bool send_pose(int pose_sock, const cv::Mat& Tcw) {
    std::cout << "[DEBUG] send_pose() CALLED" << std::endl;
    log_event("[DEBUG] Attempting to send pose to Python receiver.");

    // Validate the pose matrix dimensions
    if (Tcw.empty() || Tcw.rows != 4 || Tcw.cols != 4) {
        log_event("[ERROR] Pose matrix is invalid — empty or wrong size.");
        return false;
    }

    // Convert the pose matrix to a 12-element float array (3x4 matrix)
    float data[12];
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 4; ++c)
            data[r * 4 + c] = Tcw.at<float>(r, c);

    // Log the matrix being sent
    std::ostringstream log_msg;
    log_msg << "[POSE] Sending 3x4 pose matrix: ";
    for (int i = 0; i < 12; ++i) {
        log_msg << data[i];
        if (i < 11) log_msg << ", ";
    }
    log_event(log_msg.str());

    int bytes = 12 * sizeof(float);
    int sent = send(pose_sock, reinterpret_cast<char*>(data), bytes, 0);

    std::ostringstream dbg;
    dbg << "[DEBUG] send() returned " << sent << " of " << bytes << " bytes.";
    log_event(dbg.str());

    if (sent != bytes) {
        std::ostringstream oss;
        oss << "[ERROR] send_pose failed: sent " << sent << " of " << bytes << " bytes.";
        log_event(oss.str());
    } else {
        log_event("[DEBUG] send_pose succeeded: 48 bytes sent to Python receiver.");
    }

    return sent == bytes;
}



// ------- Main function to set up the TCP server, receive images, and process them with ORB-SLAM2 -------
int main(int argc, char **argv) {
    // Redirect stdout and stderr to log files
    freopen("/mnt/h/Documents/AirSimExperiments/Hybrid_Navigation/logs/slam_console.txt", "w", stdout);
    freopen("/mnt/h/Documents/AirSimExperiments/Hybrid_Navigation/logs/slam_console_err.txt", "w", stderr);
    // Set the locale to C for consistent number formatting
    if (argc < 3) {
        cerr << "Usage: ./tcp_slam_server path_to_vocabulary path_to_settings [log_file_path]" << endl;
        return 1;
    }
    // Create logs directory if it doesn't exist
    #ifdef _WIN32
        _mkdir("H:\\Documents\\AirSimExperiments\\Hybrid_Navigation\\logs");
    #else
        mkdir("/mnt/h/Documents/AirSimExperiments/Hybrid_Navigation/logs", 0777);
    #endif
    // Get vocabulary and settings file paths from command line arguments
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

    // Initialize ORB-SLAM2
    ORB_SLAM2::System SLAM(vocab, settings, ORB_SLAM2::System::STEREO, true);
    log_event("[INFO] SLAM system initialized.");

    // -- Setup TCP server for receiving AirSim images --
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd == -1) {
        log_event("Socket creation failed");
        return 1;
    }
    // Set socket options to allow reuse of the address and port
    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt)); // Allow reuse of address and port

    sockaddr_in address; // Define the address structure
    address.sin_family = AF_INET; // Use IPv4
    address.sin_addr.s_addr = INADDR_ANY;  // binds to 0.0.0.0 — all interfaces
    address.sin_port = htons(6000);
    
    // Bind the socket to the address and port
    if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0) {
        cerr << "Bind failed" << endl;
        close(server_fd);
        return 1;
    }
    // Set the socket to listen for incoming connections
    if (listen(server_fd, 1) < 0) {
        cerr << "Listen failed" << endl;
        close(server_fd);
        return 1;
    }
    cout << "[INFO] Waiting for Python streamer on port 6000..." << endl;

    // Log the event
    int addrlen = sizeof(address);
    log_event("Calling accept() and waiting for streamer...");
    log_event("[INFO] Calling accept() and waiting for streamer...");
    std::cout.flush();

    // Accept a client connection
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

    // Log the connection details
    if (sock < 0) {
        cerr << "Accept failed" << endl;
        close(server_fd);
        return 1;
    }
    cout << "[INFO] Connected to Python streamer!" << endl;

    // --- Initialize SLAM ---
    bool slam_ready_flag_written = false;
    bool clean_exit = true;
    int frame_counter = 0;

    // --- Setup pose sender socket ---
    // Initialize pose sender socket
    int pose_sock = socket(AF_INET, SOCK_STREAM, 0);
    if (pose_sock < 0) {
        log_event("[ERROR] Could not create pose sender socket.");
        pose_sock = -1;
    } else {
        log_event("[DEBUG] Pose sender socket created successfully.");

        sockaddr_in pose_addr;
        pose_addr.sin_family = AF_INET;
        pose_addr.sin_port = htons(POSE_RECEIVER_PORT);
        inet_pton(AF_INET, POSE_RECEIVER_IP, &pose_addr.sin_addr);

        // Log IP and port information
        char ip_str[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &pose_addr.sin_addr, ip_str, INET_ADDRSTRLEN);
        std::ostringstream sockinfo;
        sockinfo << "[DEBUG] Attempting connection to pose receiver at "
                << ip_str << ":" << ntohs(pose_addr.sin_port);
        log_event(sockinfo.str());

        log_event("Connecting to Python pose receiver...");
        bool connected = false;
        for (int attempt = 0; attempt < 10; ++attempt) {
            int pose_conn = connect(pose_sock, (struct sockaddr*)&pose_addr, sizeof(pose_addr));
            if (pose_conn >= 0) {
                connected = true;
                log_event("✅ Connected to Python pose receiver.");
                break;
            } else {
                std::ostringstream retry_msg;
                retry_msg << "[WARN] Attempt " << (attempt + 1)
                        << " failed to connect to Python pose receiver — retrying in 1s...";
                log_event(retry_msg.str());
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }
        }

        if (!connected) {
            log_event("[ERROR] Failed to connect to Python pose receiver after 10 attempts.");
            close(pose_sock);
            pose_sock = -1;
        }
    }



    // --- Main loop to receive images and process them with SLAM ---
    cv::Mat imLeft, imRight;
    cv::Mat imLeftGray, imRightGray;

    while (true) {
        int snapshot_limit = 5;             
        log_event("----- Begin image receive loop -----");

        // Log the loop start time
        double loop_timestamp = (double)cv::getTickCount() / cv::getTickFrequency();
        {
            std::ostringstream oss;
            oss << "Frame #" << frame_counter << " | Loop timestamp: " << std::fixed << std::setprecision(6) << loop_timestamp;
            log_event(oss.str());
        }
        uint32_t net_height, net_width, net_bytes;
        uint32_t rgb_height, rgb_width, rgb_bytes;

        // --- Receive Left RGB Image ---
        char left_header[12];
        if (!recv_all(sock, left_header, 12)) break;
        memcpy(&net_height, left_header, 4);
        memcpy(&net_width,  left_header + 4, 4);
        memcpy(&net_bytes, left_header + 8, 4);
        rgb_height = ntohl(net_height);
        rgb_width  = ntohl(net_width);
        rgb_bytes  = ntohl(net_bytes);

        vector<uchar> left_buffer(rgb_bytes);
        if (!recv_all(sock, (char*)left_buffer.data(), rgb_bytes)) break; // Receive the image data
        cv::Mat left_rgb(rgb_height, rgb_width, CV_8UC3, left_buffer.data()); // Create Mat from buffer
        log_event("Received Left image: height=" + std::to_string(left_rgb.rows) + ", width=" + std::to_string(left_rgb.cols));
        imLeft = left_rgb.clone();  // save for snapshot + debug
        cv::cvtColor(imLeft, imLeftGray, cv::COLOR_RGB2GRAY); // Convert to grayscale

        // --- DEBUG: Log right image properties ---
        {
            std::ostringstream log;
            log << "[DEBUG] Left image received: "
                << "height=" << left_rgb.rows
                << ", width=" << left_rgb.cols
                << ", expected bytes=" << rgb_bytes
                << ", actual buffer size=" << left_buffer.size();
            log_event(log.str());

            // Check image for being empty
            if (imLeft.empty()) {
                log_event("[ERROR] Left image is EMPTY after decoding!");
            }

            // Optional: verify pixel structure
            if (imLeft.type() != CV_8UC3) {
                std::ostringstream tmsg;
                tmsg << "[WARN] Left image has unexpected type: " << imLeft.type();
                log_event(tmsg.str());
            }

            // Optional: pixel checksum
            int checksum = cv::sum(imLeft.reshape(1))[0];
            std::ostringstream csmsg;
            csmsg << "[DEBUG] Left image checksum: " << checksum;
            log_event(csmsg.str());
        }

        // --- Receive Right RGB Image ---
        char right_header[12];
        if (!recv_all(sock, right_header, 12)) break;
        memcpy(&net_height, right_header, 4);
        memcpy(&net_width,  right_header + 4, 4);
        memcpy(&net_bytes, right_header + 8, 4);
        rgb_height = ntohl(net_height);
        rgb_width  = ntohl(net_width);
        rgb_bytes  = ntohl(net_bytes);

        vector<uchar> right_buffer(rgb_bytes); // Allocate buffer for right image
        if (!recv_all(sock, (char*)right_buffer.data(), rgb_bytes)) break; // Receive the image data
        cv::Mat right_rgb(rgb_height, rgb_width, CV_8UC3, right_buffer.data()); // Create Mat from buffer
        log_event("Received Right image: height=" + std::to_string(right_rgb.rows) + ", width=" + std::to_string(right_rgb.cols));
        imRight = right_rgb.clone();  // save for snapshot + debug
        cv::cvtColor(imRight, imRightGray, cv::COLOR_RGB2GRAY); // Convert to grayscale


        // --- DEBUG: Log right image properties ---
        {
            std::ostringstream log;
            log << "[DEBUG] Right image received: "
                << "height=" << rgb_height
                << ", width=" << rgb_width
                << ", expected bytes=" << rgb_bytes
                << ", actual buffer size=" << right_buffer.size();
            log_event(log.str());

            // Check image for being empty
            if (imRight.empty()) {
                log_event("[ERROR] Right image is EMPTY after decoding!");
            }

            // Optional: verify pixel structure
            if (imRight.type() != CV_8UC3) {
                std::ostringstream tmsg;
                tmsg << "[WARN] Right image has unexpected type: " << imRight.type();
                log_event(tmsg.str());
            }

            // Optional: pixel checksum
            int checksum = cv::sum(imRight.reshape(1))[0];
            std::ostringstream csmsg;
            csmsg << "[DEBUG] Right image checksum: " << checksum;
            log_event(csmsg.str());
        }


        if (frame_counter % 10 == 0) {
            std::ostringstream fname_left, fname_right;
            fname_left << "logs/debug_left_" << frame_counter << ".png";
            fname_right << "logs/debug_right_" << frame_counter << ".png";
            cv::imwrite(fname_left.str(), imLeft);
            cv::imwrite(fname_right.str(), imRight);
        }

        cv::imshow("Stereo Left", imLeftGray);
        cv::imshow("Stereo Right", imRightGray);
        cv::waitKey(1);  // Non-blocking; press Esc in the future to close if needed

        // --- Log frame count and timestamp ---
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
            oss << "Calling SLAM.TrackStereo at timestamp=" << std::fixed << std::setprecision(6) << timestamp
                << " | Frame #" << frame_counter;
            log_event(oss.str());
        }
        try {

            // BEFORE calling TrackStereo:
            log_event("Calling SLAM.TrackStereo...");

            // Validate
            if (imLeftGray.empty() || imRightGray.empty()) {
                log_event("[FATAL] Empty grayscale image detected.");
                continue;
            }
            if (imLeftGray.type() != CV_8UC1 || imRightGray.type() != CV_8UC1) {
                log_event("[FATAL] Grayscale image not CV_8UC1 — left=" + std::to_string(imLeftGray.type()) +
                        ", right=" + std::to_string(imRightGray.type()));
                continue;
            }

            // Clone them before sending (defensive copy)
            cv::Mat left_input = imLeftGray.clone();
            cv::Mat right_input = imRightGray.clone();

            // Final check
            if (left_input.empty() || right_input.empty()) {
                log_event("[FATAL] Clone operation failed.");
                continue;
            }

            // Now call SLAM
            log_event("[DEBUG] Calling SLAM.TrackStereo...");

            cv::Mat Tcw;
            try {
                Tcw = SLAM.TrackStereo(left_input, right_input, timestamp);
                log_event("[DEBUG] SLAM.TrackStereo completed.");
            } catch (const std::exception& e) {
                log_event(std::string("[FATAL] SLAM.TrackStereo threw std::exception: ") + e.what());
                continue;
            } catch (...) {
                log_event("[FATAL] SLAM.TrackStereo threw unknown exception.");
                continue;
            }

            
            if (SLAM.GetTracker()->mState == ORB_SLAM2::Tracking::LOST) {
                log_event("[WARN] SLAM tracking lost — resetting system.");
                SLAM.Reset();
            }
            
            // Log the current frame's timestamp
            cv::Mat Tcw_copy = Tcw.clone();

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

            // Defensive checks on Tcw_copy
            bool tcw_valid = true;
            if (Tcw_copy.empty() || Tcw_copy.rows != 4 || Tcw_copy.cols != 4) {
                log_event("[WARN] Tcw_copy is empty or not 4x4.");
                tcw_valid = false;
            }
            std::ostringstream tcw_info;
            tcw_info << "[DEBUG] Tcw_copy rows=" << Tcw_copy.rows
                    << ", cols=" << Tcw_copy.cols
                    << ", type=" << Tcw_copy.type()
                    << ", isEmpty=" << Tcw_copy.empty();
            log_event(tcw_info.str());
            std::ostringstream matrix_stream;
            matrix_stream << Tcw_copy;
            log_event("[DEBUG] Tcw_copy contents:\n" + matrix_stream.str());

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

            // Log the Tcw_copy matrix
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
            oss << "Exception in SLAM.TrackStereo: " << ex.what();
            log_event(oss.str());
            std::cerr << "[ERROR] Exception in SLAM.TrackStereo: " << ex.what() << std::endl;
        } catch (...) {
            log_event("Unknown exception in SLAM.TrackStereo.");
            std::cerr << "[ERROR] Unknown exception in SLAM.TrackStereo." << std::endl;
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

        // Create images directory if it doesn't exist
        #ifdef _WIN32
            _mkdir("H:\\Documents\\AirSimExperiments\\Hybrid_Navigation\\images");
        #else
            mkdir("/mnt/h/Documents/AirSimExperiments/Hybrid_Navigation/images", 0777);
        #endif

        // Optional: save ORB keypoints on RGB image
        if (frame_counter % 25 == 0) {
            std::vector<cv::KeyPoint> orb_kps = tracker->mCurrentFrame.mvKeys;
            cv::Mat rgb_kp;
            cv::drawKeypoints(imLeft, orb_kps, rgb_kp);
            std::ostringstream kp_filename;
            kp_filename << "/mnt/h/Documents/AirSimExperiments/Hybrid_Navigation/images/frame_rgb_kp_" << frame_counter << ".png";
            cv::imwrite(kp_filename.str(), rgb_kp);
        }


        // --- After SLAM.TrackStereo ---
        cv::Mat Tcw = SLAM.GetTracker()->mCurrentFrame.mTcw;
        if (!Tcw.empty() && Tcw.rows == 4 && Tcw.cols == 4 && Tcw.type() == CV_32F) {
            // Log and send pose to the Python receiver
            log_event("[DEBUG] Sending pose matrix to Python receiver...");
            if (pose_sock >= 0) {
                send_pose(pose_sock, Tcw);
            } else {
                log_event("[ERROR] Pose socket is not valid.");
            }
        }


        int track_state = SLAM.GetTracker()->mState;
        
        // Diagnostic: Log the tracking state
        std::ostringstream pose_check;
        pose_check << "[DEBUG] Checking Tcw validity: "
                << "empty=" << Tcw.empty()
                << ", rows=" << Tcw.rows
                << ", cols=" << Tcw.cols
                << ", type=" << Tcw.type()
                << " (expect CV_32F = 5)";
        log_event(pose_check.str());

        // Use Twc (world camera pose) instead of Tcw
        if (!Tcw.empty() && Tcw.rows == 4 && Tcw.cols == 4 && Tcw.type() == CV_32F) {
            if (!(Tcw.type() == CV_32F)) {
                log_event("[WARN] Tcw.type() != CV_32F — skipping pose send.");
            }

            // Write the slam_ready.flag only once, after SLAM becomes valid
            if (!slam_ready_flag_written) {
                std::string flag_path = "/mnt/h/Documents/AirSimExperiments/Hybrid_Navigation/flags/slam_ready.flag";
                std::ofstream flag_file(flag_path);
                if (flag_file.is_open()) {
                    flag_file << "SLAM_READY" << std::endl;
                    flag_file.close();
                    slam_ready_flag_written = true;
                    log_event("[INFO] slam_ready.flag created.");
                } else {
                    log_event("[ERROR] Could not create slam_ready.flag.");
                }
            }

            cv::Mat Twc = Tcw.inv();
            float x = Twc.at<float>(0, 3);
            float y = Twc.at<float>(1, 3);
            float z = Twc.at<float>(2, 3);

            std::ostringstream vec_log;
            vec_log << "[FIXED] Twc (world position) — X: " << x
                    << ", Y: " << y
                    << ", Z: " << z;
            log_event(vec_log.str());

            // HACK: Simulate fake motion by modifying Twc translation
            // This simulates a forward motion of 0.05 meters per frame
            // static int fake_motion_counter = 0;
            // fake_motion_counter++;
            // Twc.at<float>(0, 3) = 0.05f * fake_motion_counter;  // Move along X

            // Send Twc instead of Tcw if tracking is good
            // if (pose_sock >= 0 && track_state >= ORB_SLAM2::Tracking::OK && cv::checkRange(Twc)) {
            log_event("[CHECK] About to test pose_sock condition in main().");

            std::ostringstream live_sock_check;
            live_sock_check << "[CHECK] At pose send time, pose_sock=" << pose_sock;
            log_event(live_sock_check.str());

            if (true) {
                log_event("[DEBUG] Entering pose send block unconditionally for testing.");

                // Diagnostic: Confirm matrix validity
                std::ostringstream status;
                status << "[DEBUG] Checking pose send condition: "
                    << "pose_sock=" << pose_sock
                    << ", Tcw.empty()=" << Twc.empty()
                    << ", Tcw.rows=" << Twc.rows
                    << ", Tcw.cols=" << Twc.cols
                    << ", Tcw.type()=" << Twc.type();
                log_event(status.str());

                log_event("[DEBUG] Calling send_pose(...) with Twc_send matrix");

                cv::Mat Twc_send;
                Twc.convertTo(Twc_send, CV_32F); // ensure float32

                if (send_pose(pose_sock, Twc_send)) {
                    log_event("Pose (Twc) sent to Python receiver.");
                } else {
                    log_event("[WARN] send_pose() returned false.");
                }
            }

            } else {
                log_event("[WARN] Tcw is invalid or not 4x4 matrix — skipping pose send.");
            }

        frame_counter++;
    }

    // --- Cleanup and exit ---
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