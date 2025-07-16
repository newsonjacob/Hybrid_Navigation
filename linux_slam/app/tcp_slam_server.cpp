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
#include <sys/stat.h>

// Define grace period parameters at the top of the main function
int grace_frame_count = 0;  // Counter for frames with no detected motion
const int MAX_GRACE_FRAMES = 30;  // Maximum number of frames to allow without motion

// At the top of the file or near your other constants:
const int MAX_IMAGE_WIDTH  = 1920;
const int MAX_IMAGE_HEIGHT = 1080;
const int MAX_IMAGE_BYTES  = MAX_IMAGE_WIDTH * MAX_IMAGE_HEIGHT * 3; // 3 for RGB

using namespace std;

// Forward declaration for log_event
static void log_event(const std::string& msg);

cv::Mat get_pose_covariance_with_inliers(const cv::Mat& current_pose, const cv::Mat& previous_pose) {
    // Ensure that both poses are valid (4x4 matrices)
    if (current_pose.empty() || previous_pose.empty() || current_pose.rows != 4 || current_pose.cols != 4 || previous_pose.rows != 4 || previous_pose.cols != 4) {
        log_event("[ERROR] Invalid pose matrices.");
        return cv::Mat();  // Return empty matrix on error
    }

    // Calculate pose difference (motion uncertainty)
    cv::Mat pose_diff = current_pose - previous_pose;  // Difference between poses

    // Estimate covariance based on the difference (this is a simple placeholder logic)
    double norm = cv::norm(pose_diff, cv::NORM_L2);  // Compute the L2 norm (Euclidean distance)
    
    // Use the norm as an estimate of uncertainty
    double uncertainty = std::max(norm, 0.1);  // Ensure the uncertainty is not too small

    // For simplicity, scale the covariance based on the uncertainty
    cv::Mat covariance = cv::Mat::eye(4, 4, CV_32F) * uncertainty;  // Identity matrix scaled by uncertainty

    // You can modify the above logic if you want to factor in inliers or other factors
    
    // For now, we're returning the basic pose difference-based covariance matrix
    return covariance;
}

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

const char* POSE_RECEIVER_IP = getenv("POSE_RECEIVER_IP") ? getenv("POSE_RECEIVER_IP") : "192.168.1.100"; // Python receiver IP
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
    cv::Mat Tcw_send; // Temporary matrix to hold the converted pose
    if (Tcw.type() != CV_32F) {
        log_event("[WARN] Pose matrix not CV_32F. Converting...");
        Tcw.convertTo(Tcw_send, CV_32F); // Convert to float if not already
    } else {
        Tcw_send = Tcw; // Use the original matrix if it's already in float format
    }
    // Ensure Tcw_send is 3x4
    float data[12];
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 4; ++c)
            data[r * 4 + c] = Tcw_send.at<float>(r, c);


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

// Function to get the number of inliers (features successfully tracked)
int get_feature_inliers(ORB_SLAM2::System &SLAM) {
    // Get the tracker object from the SLAM system
    auto tracker = SLAM.GetTracker();
    
    // Ensure the tracker is valid
    if (!tracker) {
        log_event("[ERROR] Tracker not available in SLAM system.");
        return -1;  // Return -1 to indicate failure
    }

    // Access the current frame's inliers from the tracker
    int inliers = tracker->mCurrentFrame.N; // N is the number of features in the current frame
    
    log_event("[DEBUG] Inliers tracked: " + std::to_string(inliers));
    return inliers;
}

void cleanup_resources(int sock, int server_fd, int pose_sock) {
    if (sock >= 0) {
        log_event("[DEBUG] Closing image stream socket...");
        close(sock);
    }
    if (server_fd >= 0) {
        log_event("[DEBUG] Closing server socket...");
        close(server_fd);
    }
    if (pose_sock >= 0) {
        log_event("[DEBUG] Closing pose sender socket...");
        close(pose_sock);
    }
}

// ------- Main function to set up the TCP server, receive images, and process them with ORB-SLAM2 -------
int main(int argc, char **argv) {
    // Redirect stdout and stderr to log files
    (void)freopen("/mnt/h/Documents/AirSimExperiments/Hybrid_Navigation/logs/slam_console.txt", "w", stdout);
    (void)freopen("/mnt/h/Documents/AirSimExperiments/Hybrid_Navigation/logs/slam_console_err.txt", "w", stderr);
    // Set the locale to C for consistent number formatting
    
    // Add this static variable at the top of your main function
    static cv::Mat prev_Tcw;  // Previous pose (initialize once)

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

    // Create images directory if it doesn't exist
    #ifdef _WIN32
        _mkdir("H:\\Documents\\AirSimExperiments\\Hybrid_Navigation\\images");
    #else
        mkdir("/mnt/h/Documents/AirSimExperiments/Hybrid_Navigation/images", 0777);
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
    log_event("[DEBUG] Creating server socket for AirSim image stream...");
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd == -1) {
        log_event("[ERROR] Socket creation failed for AirSim image stream.");
        return 1;
    }
    log_event("[DEBUG] Server socket created successfully.");

    int opt = 1;
    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt)) < 0) {
        log_event("[WARN] setsockopt failed for server_fd.");
    } else {
        log_event("[DEBUG] setsockopt SO_REUSEADDR | SO_REUSEPORT succeeded.");
    }

    sockaddr_in address;
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(6000);

    log_event("[DEBUG] Binding server socket to port 6000...");
    if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0) {
        log_event("[ERROR] Bind failed for server_fd.");
        close(server_fd);
        return 1;
    }
    log_event("[DEBUG] Bind succeeded for server_fd.");

    log_event("[DEBUG] Listening for incoming connections on port 6000...");
    if (listen(server_fd, 1) < 0) {
        log_event("[ERROR] Listen failed for server_fd.");
        close(server_fd);
        return 1;
    }
    log_event("[DEBUG] Listen succeeded. Waiting for Python streamer...");

    int addrlen = sizeof(address);
    log_event("[DEBUG] Calling accept() to wait for streamer...");
    int sock = accept(server_fd, (struct sockaddr *)&address, (socklen_t*)&addrlen);
    if (sock < 0) {
        log_event("[ERROR] Failed to accept() client connection.");
        close(server_fd);
        return 1;
    }
    log_event("[INFO] Client connection accepted on image stream socket.");
    log_event("[DEBUG] sock value after accept: " + std::to_string(sock));

    // --- Setup pose sender socket ---
    log_event("[DEBUG] Creating pose sender socket...");
    int pose_sock = socket(AF_INET, SOCK_STREAM, 0);
    if (pose_sock < 0) {
        log_event("[ERROR] Could not create pose sender socket.");
        pose_sock = -1;
    } else {
        log_event("[DEBUG] Pose sender socket created successfully.");
        
        // Set up the address for the pose receiver
        sockaddr_in pose_addr;
        pose_addr.sin_family = AF_INET;
        pose_addr.sin_port = htons(POSE_RECEIVER_PORT);
        inet_pton(AF_INET, POSE_RECEIVER_IP, &pose_addr.sin_addr);
        // Log the IP and port of the pose receiver
        char ip_str[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &pose_addr.sin_addr, ip_str, INET_ADDRSTRLEN);
        std::ostringstream sockinfo;
        sockinfo << "[DEBUG] Attempting connection to pose receiver at "
                 << ip_str << ":" << ntohs(pose_addr.sin_port);
        log_event(sockinfo.str());
        // Attempt to connect to the pose receiver
        log_event("[DEBUG] Connecting to Python pose receiver...");
        bool connected = false;
        for (int attempt = 0; attempt < 10; ++attempt) {
            int pose_conn = connect(pose_sock, (struct sockaddr*)&pose_addr, sizeof(pose_addr));
            if (pose_conn >= 0) {
                connected = true;
                log_event("[INFO] Connected to Python pose receiver.");
                break;
            } else {
                std::ostringstream retry_msg;
                retry_msg << "[WARN] Attempt " << (attempt + 1)
                          << " failed to connect to Python pose receiver — retrying in 1s...";
                log_event(retry_msg.str());
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }
        }
        // If connection was not established after 10 attempts, log an error
        if (!connected) {
            log_event("[ERROR] Failed to connect to Python pose receiver after 10 attempts.");
            close(pose_sock);
            pose_sock = -1;
        }
    }
    // --- Main loop to receive images and process them with SLAM ---
    cv::Mat imLeft, imRight; // Matrices to hold the received images
    cv::Mat imLeftGray, imRightGray; // Grayscale versions of the images for SLAM processing
    log_event("[DEBUG] Starting main image receive loop...");
    // Initialize frame counter and flags
    int frame_counter = 0;              // Frame counter to track the number of frames processed
    bool slam_ready_flag_written = false; // Flag to indicate if SLAM is ready to process frames
    const int MIN_INLIERS_THRESHOLD = 0;  // Minimum inliers to consider SLAM stable

    while (true) {       
        log_event("----- Begin image receive loop -----");

        // Log the loop start time
        double loop_timestamp = (double)cv::getTickCount() / cv::getTickFrequency();

        // Get the inliers after processing the frames with SLAM
        int inliers = get_feature_inliers(SLAM);
        log_event("[SLAM] Inliers after TrackStereo: " + std::to_string(inliers));

        // Continue processing if enough inliers are present
        if (inliers < MIN_INLIERS_THRESHOLD) {
            log_event("[WARN] Too few inliers tracked. SLAM may be unstable.");
            // Optionally, reset SLAM or take other actions
            SLAM.Reset(); // Reset the SLAM system (if this is an acceptable approach)
            std::this_thread::sleep_for(std::chrono::seconds(2)); // Wait before retrying to receive images
        }

        // If we have enough inliers, proceed with receiving images
        if (inliers >= MIN_INLIERS_THRESHOLD) { 
            // Process the received images with SLAM
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

            // ----> SANITY CHECKS:
            if (rgb_height <= 0 || rgb_width <= 0 || rgb_bytes <= 0 ||
                rgb_height > MAX_IMAGE_HEIGHT ||
                rgb_width  > MAX_IMAGE_WIDTH  ||
                rgb_bytes  > MAX_IMAGE_BYTES) {
                log_event("[FATAL] Received image header values out of bounds: " +
                    std::to_string(rgb_height) + "x" + std::to_string(rgb_width) + ", " +
                    std::to_string(rgb_bytes) + " bytes.");
                break;
            }
            // Allocate buffer for left image
            vector<uchar> left_buffer(rgb_bytes);
            if (!recv_all(sock, (char*)left_buffer.data(), rgb_bytes)) break; // Receive the image data
            cv::Mat left_rgb(rgb_height, rgb_width, CV_8UC3, left_buffer.data()); // Create Mat from buffer
            if (left_rgb.empty() || left_rgb.rows != rgb_height || left_rgb.cols != rgb_width || left_rgb.type() != CV_8UC3) {
                log_event("[ERROR] left_rgb invalid after construction! Skipping frame.");
                continue;  // or break, depending on your policy
            }

            log_event("Received Left image: height=" + std::to_string(left_rgb.rows) + ", width=" + std::to_string(left_rgb.cols));
            imLeft = left_rgb.clone();  // save for snapshot + debug

            if (!left_rgb.empty() && left_rgb.type() == CV_8UC3) { // Check if left_rgb is valid
                cv::cvtColor(left_rgb, imLeftGray, cv::COLOR_RGB2GRAY);
            } else {
                log_event("[ERROR] Cannot convert left_rgb to grayscale: empty or wrong type.");
                continue;
            }

            // --- DEBUG: Log right image properties ---
            {
                std::ostringstream log;
                log << "[DEBUG] Left image received: "
                    << "height="<< left_rgb.rows
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

            // ----> SANITY CHECKS:
            if (rgb_height <= 0 || rgb_width <= 0 || rgb_bytes <= 0 ||
                rgb_height > MAX_IMAGE_HEIGHT ||
                rgb_width  > MAX_IMAGE_WIDTH  ||
                rgb_bytes  > MAX_IMAGE_BYTES) {
                log_event("[FATAL] Received image header values out of bounds: " +
                    std::to_string(rgb_height) + "x" + std::to_string(rgb_width) + ", " +
                    std::to_string(rgb_bytes) + " bytes.");
                break;
            }

            vector<uchar> right_buffer(rgb_bytes); // Allocate buffer for right image
            if (!recv_all(sock, (char*)right_buffer.data(), rgb_bytes)) break; // Receive the image data
            cv::Mat right_rgb(rgb_height, rgb_width, CV_8UC3, right_buffer.data()); // Create Mat from buffer
            if (right_rgb.empty() || right_rgb.rows != rgb_height || right_rgb.cols != rgb_width || right_rgb.type() != CV_8UC3) {
                log_event("[ERROR] right_rgb invalid after construction! Skipping frame.");
                continue;  // or break, depending on your policy
            }

            log_event("Received Right image: height=" + std::to_string(right_rgb.rows) + ", width=" + std::to_string(right_rgb.cols));
            imRight = right_rgb.clone();  // save for snapshot + debug
            // Check if right_rgb is valid before converting
            if (!right_rgb.empty() && right_rgb.type() == CV_8UC3) {
                cv::cvtColor(right_rgb, imRightGray, cv::COLOR_RGB2GRAY);
            } else {
                log_event("[ERROR] Cannot convert right_rgb to grayscale: empty or wrong type.");
                continue;
            }

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

            // --- Check if images are valid ---
            // Defensive debug image write: only save if not empty and has expected dimensions [FIX 3C]
            if (frame_counter % 10 == 0) {
                if (!imLeft.empty() && imLeft.rows > 0 && imLeft.cols > 0) { // [FIX 3C]
                    std::ostringstream frame_left;
                    frame_left << "logs/debug_left_" << frame_counter << ".png";
                    cv::imwrite(frame_left.str(), imLeft);
                }
                if (!imRight.empty() && imRight.rows > 0 && imRight.cols > 0) { // [FIX 3C]
                    std::ostringstream frame_right;
                    frame_right << "logs/debug_right_" << frame_counter << ".png";
                    cv::imwrite(frame_right.str(), imRight);
                }
            }

            // Defensive image display (imshow)
            // Only display if not empty and correct type
            if (!imLeftGray.empty() && imLeftGray.type() == CV_8UC1) {
                cv::imshow("Stereo Left", imLeftGray);
            }
            if (!imRightGray.empty() && imRightGray.type() == CV_8UC1) {
                cv::imshow("Stereo Right", imRightGray);
            }
            cv::waitKey(1);

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
            {   std::ostringstream oss;
                oss << "Calling SLAM.TrackStereo at timestamp=" << std::fixed << std::setprecision(6) << timestamp
                    << " | Frame #" << frame_counter;
                log_event(oss.str());
            }
            cv::Mat covariance; // Declare BEFORE the try block
            try {
                // BEFORE calling TrackStereo:
                log_event("Calling SLAM.TrackStereo...");

                // Validate grayscale images
                if (imLeftGray.empty() || imRightGray.empty()) {
                    log_event("[FATAL] Empty grayscale image detected.");
                    continue;
                } // Check if images are grayscale
                if (imLeftGray.type() != CV_8UC1 || imRightGray.type() != CV_8UC1) {
                    log_event("[FATAL] Grayscale image not CV_8UC1 — left=" + std::to_string(imLeftGray.type()) +
                            ", right=" + std::to_string(imRightGray.type()));
                    continue;
                }
                // Defensive clone: only clone if not empty [FIX 3C]
                cv::Mat left_input, right_input;
                if (!imLeftGray.empty())
                    left_input = imLeftGray.clone(); // [FIX 3C]
                else {
                    log_event("[ERROR] imLeftGray is empty before clone. Skipping frame."); // [FIX 3C]
                    continue;
                }
                if (!imRightGray.empty())
                    right_input = imRightGray.clone(); // [FIX 3C]
                else {
                    log_event("[ERROR] imRightGray is empty before clone. Skipping frame."); // [FIX 3C]
                    continue;
                }


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

                // --- Handling SLAM results ---
                static bool first_frame = true;
                if (first_frame) {
                    prev_Tcw = Tcw.clone();
                    first_frame = false;
                }

                // Calculate covariance based on pose difference and inliers
                covariance = get_pose_covariance_with_inliers(Tcw, prev_Tcw);  // Calculate covariance

                // Always check for empty and correct type before accessing elements
                if (covariance.empty() || covariance.rows < 1 || covariance.cols < 1) {
                    log_event("[WARN] Covariance matrix is empty or has invalid size, skipping log.");
                } else {
                    cv::Mat covariance_checked;
                    if (covariance.type() != CV_32F) {
                        log_event("[WARN] Covariance matrix not CV_32F, converting...");
                        covariance.convertTo(covariance_checked, CV_32F);
                    } else {
                        covariance_checked = covariance;
                    }
                    log_event("[DEBUG] Pose Covariance: " + std::to_string(covariance_checked.at<float>(0, 0)));
                }

                // Update the previous pose after each frame
                prev_Tcw = Tcw.clone();

                // Log frame count and timestamp for debugging
                double timestamp = (double)cv::getTickCount() / cv::getTickFrequency();
                log_event("[DEBUG] Timestamp at Frame #" + std::to_string(frame_counter) + ": " + std::to_string(timestamp));

                cv::Mat Tcw_copy = Tcw.clone();  // Defensive copy of the pose matrix

                log_event("Tcw_copy rows: " + std::to_string(Tcw_copy.rows) +
                        ", cols: " + std::to_string(Tcw_copy.cols) +
                        ", type: " + std::to_string(Tcw_copy.type()));
                cv::Mat identity = cv::Mat::eye(4, 4, Tcw_copy.type());

                // Check if Tcw_copy is close to identity matrix
                double frobenius_norm = cv::norm(Tcw_copy - identity, cv::NORM_L2);
                static int identity_frame_count = 0;  // Use static so it persists between frames
                // Check if the identity matrix is detected (no motion)
                if (frobenius_norm < 1e-3) {  // Identity matrix detected (no motion)
                    identity_frame_count++;

                    if (identity_frame_count < MAX_GRACE_FRAMES) {
                        // Still within grace period, just log and continue
                        log_event("[INFO] Tcw appears to be an identity matrix — no motion detected. Count: " + std::to_string(identity_frame_count));
                    } else {
                        // Grace period exceeded, reset SLAM if no motion detected
                        log_event("[ERROR] Too many frames with no motion, resetting SLAM.");
                        SLAM.Reset();
                        identity_frame_count = 0;  // Reset the counter after reset
                    }
                }

                if (identity_frame_count >= MAX_GRACE_FRAMES) {
                    log_event("[INFO] Grace period over. Checking for motion.");
                    if (frobenius_norm < 1e-3) {
                        log_event("[WARN] No motion detected. Resetting SLAM.");
                        SLAM.Reset();
                    } else {
                        log_event("[INFO] Motion detected. Continuing normal operation.");
                    }
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
                        // Defensive: ensure Twc is valid and CV_32F before using .at<float> [FIX 3E]
                        if (!Twc.empty() && Twc.rows >= 3 && Twc.cols >= 4) { // [FIX 3E]
                            cv::Mat Twc_checked;
                            if (Twc.type() != CV_32F) { 
                                log_event("[WARN] Twc not CV_32F after inversion, converting..."); // [FIX 3E]
                                Twc.convertTo(Twc_checked, CV_32F); 
                            } else {
                                Twc_checked = Twc;
                            }
                            float x = Twc_checked.at<float>(0, 3);
                            float y = Twc_checked.at<float>(1, 3); 
                            float z = Twc_checked.at<float>(2, 3); 
                            std::ostringstream twc_log;
                            twc_log << "Camera center (twc): " << x << ", " << y << ", " << z;
                            log_event(twc_log.str());
                        } else {
                            log_event("[ERROR] Twc invalid (empty or wrong shape) after inversion, skipping pose extraction."); // [FIX 3E]
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
            if (!tracker) {
                log_event("[ERROR] Tracker is null. Skipping tracker-dependent logging.");
            } else {
                int state = tracker->mState;
                int n_kfs = tracker->GetNumLocalKeyFrames();
                int n_mappoints = tracker->mCurrentFrame.mvpMapPoints.size();

                std::ostringstream oss;
                oss << "TrackingState=" << state
                    << ", KeyFrames=" << n_kfs
                    << ", MapPoints=" << n_mappoints
                    << ", TrackedFramePoints=" << n_mappoints
                    << ", ORBFeatures=" << tracker->mCurrentFrame.N;
                log_event(oss.str());

                // Optional: save ORB keypoints on RGB image
                if (frame_counter % 25 == 0) {
                    std::vector<cv::KeyPoint> orb_kps = tracker->mCurrentFrame.mvKeys;
                    cv::Mat rgb_kp;
                    cv::drawKeypoints(imLeft, orb_kps, rgb_kp);
                    std::ostringstream kp_filename;
                    kp_filename << "/mnt/h/Documents/AirSimExperiments/Hybrid_Navigation/images/frame_rgb_kp_" << frame_counter << ".png";
                    cv::imwrite(kp_filename.str(), rgb_kp);
                }
            }


            // --- After SLAM.TrackStereo ---
            cv::Mat Tcw = SLAM.GetTracker()->mCurrentFrame.mTcw;
                       
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
                // Always ensure Twc is CV_32F before accessing
                cv::Mat Twc_checked;
                if (Twc.type() != CV_32F) {
                    log_event("[WARN] Twc not CV_32F after inversion, converting...");
                    Twc.convertTo(Twc_checked, CV_32F);
                } else {
                    Twc_checked = Twc;
                }
                float x = Twc_checked.at<float>(0, 3);
                float y = Twc_checked.at<float>(1, 3);
                float z = Twc_checked.at<float>(2, 3);


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
                        // --- Send covariance as a single float ---
                        float covariance_value = 0.1f; // Default value
                        if (!covariance.empty() && covariance.rows > 0 && covariance.cols > 0) {
                            covariance_value = covariance.at<float>(0, 0); // Or use your preferred scalar
                        }
                        int cov_sent = send(pose_sock, reinterpret_cast<char*>(&covariance_value), sizeof(float), 0);
                        if (cov_sent != sizeof(float)) {
                            log_event("[ERROR] Failed to send covariance value.");
                        } else {
                            log_event("[DEBUG] Covariance value sent to Python receiver: " + std::to_string(covariance_value));
                        }

                        // --- Send inlier count as an int ---
                        int inlier_count = get_feature_inliers(SLAM);
                        int inlier_sent = send(pose_sock, reinterpret_cast<char*>(&inlier_count), sizeof(int), 0);
                        if (inlier_sent != sizeof(int)) {
                            log_event("[ERROR] Failed to send inlier count.");
                        } else {
                            log_event("[DEBUG] Inlier count sent to Python receiver: " + std::to_string(inlier_count));
                        }
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
    log_event("[DEBUG] Closing sockets and cleaning up...");
    close(sock);
    close(server_fd);
    if (pose_sock >= 0) close(pose_sock);
    if (pose_log_stream.is_open()) pose_log_stream.close();
    log_event("[DEBUG] Sockets closed. SLAM server shutting down.");
    SLAM.Shutdown();

    SLAM.SaveTrajectoryTUM("/mnt/h/Documents/AirSimExperiments/Hybrid_Navigation/CameraTrajectory.txt");
    SLAM.SaveKeyFrameTrajectoryTUM("/mnt/h/Documents/AirSimExperiments/Hybrid_Navigation/KeyFrameTrajectory.txt");

    return 0;
}
