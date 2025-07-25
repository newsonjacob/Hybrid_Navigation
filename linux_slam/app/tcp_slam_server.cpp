#include <System.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <iostream>
#include <vector>
#include <fstream>
#include <ctime>
#include <iomanip>
#include <thread>
#include <chrono>
#include <sstream>
#include "server/logging.hpp"
#include "server/network.hpp"
#include "server/slam_runner.hpp"
#include <filesystem>
#include <sys/stat.h>
#include <cerrno>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <cstdio>
#ifdef _WIN32
#include <direct.h>
#endif

// Define grace period parameters at the top of the main function
int grace_frame_count = 0;  // Counter for frames with no detected motion
const int MAX_GRACE_FRAMES = 30;  // Maximum number of frames to allow without motion

// At the top of the file or near your other constants:
const int MAX_IMAGE_WIDTH  = 1920; // Maximum width of the image
const int MAX_IMAGE_HEIGHT = 1080; // Maximum height of the image
const int MAX_IMAGE_BYTES  = MAX_IMAGE_WIDTH * MAX_IMAGE_HEIGHT * 1; // Assuming 1 byte per pixel for grayscale images

using namespace std;


using slam_server::g_log_file_path;
using slam_server::log_event;
using slam_server::get_feature_inliers;
using slam_server::get_pose_covariance_with_inliers;
using slam_server::recv_all;
using slam_server::send_pose;

// ------- Main function to set up the TCP server, receive images, and process them with ORB-SLAM2 -------
// Simple cross-platform helpers
#ifdef _WIN32
const char PATH_SEP = '\\';
#else
const char PATH_SEP = '/';
#endif

static bool make_dir(const std::string& p) {
#ifdef _WIN32
    return _mkdir(p.c_str()) == 0 || errno == EEXIST;
#else
    return mkdir(p.c_str(), 0755) == 0 || errno == EEXIST;
#endif
}

static bool create_directories(const std::string& path) {
    std::string accum;
    for (size_t i = 0; i < path.size(); ++i) {
        char c = path[i];
        accum += c;
        if (c == '/' || c == '\\') {
            if (!accum.empty()) make_dir(accum);
        }
    }
    if (!accum.empty()) make_dir(accum);
    return true;
}

static std::string join_path(const std::string& a, const std::string& b) {
    if (a.empty()) return b;
    if (a.back() == PATH_SEP) return a + b;
    return a + PATH_SEP + b;
}

static void perform_full_reinit(ORB_SLAM2::System& SLAM,
                                int& frame_counter,
                                bool& slam_ready_flag_written,
                                bool& first_frame,
                                int& identity_frame_count,
                                const std::string& flag_dir) {
    SLAM.Reset();
    first_frame = true;
    frame_counter = 0;
    identity_frame_count = 0;
    slam_ready_flag_written = false;
    std::string flag_path = join_path(flag_dir, "slam_ready.flag");
    std::remove(flag_path.c_str());
    log_event("[INFO] SLAM fully reinitialized. Awaiting first stereo pair...");
}

int main(int argc, char **argv) {

    std::string log_dir = getenv("SLAM_LOG_DIR") ? getenv("SLAM_LOG_DIR") : "logs";
    std::string flag_dir = getenv("SLAM_FLAG_DIR") ? getenv("SLAM_FLAG_DIR") : "flags";
    std::string image_dir = getenv("SLAM_IMAGE_DIR") ? getenv("SLAM_IMAGE_DIR") : join_path(log_dir, "images");
    std::string video_file = getenv("SLAM_VIDEO_FILE") ? getenv("SLAM_VIDEO_FILE") : "";

    std::vector<std::string> args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        const std::string prefix = "--video-file=";
        if (arg.rfind(prefix, 0) == 0) {
            video_file = arg.substr(prefix.size());
        } else {
            args.push_back(arg);
        }
    }

    if (args.size() < 2) {
        cerr << "Usage: ./tcp_slam_server vocab settings [log_dir] [flag_dir] [image_dir] [--video-file=FILE]" << endl;
        return 1;
    }

    if (args.size() >= 3) log_dir = args[2];
    if (args.size() >= 4) flag_dir = args[3];
    if (args.size() >= 5) image_dir = args[4];

    if (video_file.empty())
        video_file = join_path(log_dir, "slam_feed.avi");

    create_directories(log_dir);
    create_directories(flag_dir);
    create_directories(image_dir);

    std::string video_dir = video_file.substr(0, video_file.find_last_of("/\\"));
    if (!video_dir.empty()) {
        create_directories(video_dir);
    }

    // Create metrics CSV for runtime statistics
    std::string metrics_file = join_path(log_dir, "slam_metrics.csv");
    std::ofstream metrics_stream;
    metrics_stream.open(metrics_file);
    // Timestamp will be relative to when the server starts
    double server_start_time = (double)cv::getTickCount() / cv::getTickFrequency();
    if (metrics_stream.is_open()) {
        metrics_stream
            << "frame,timestamp,tracking_state,inliers,covariance,x,y,z\n";
    }
    
    std::string console_log = join_path(log_dir, "slam_console.txt");
    std::string console_err = join_path(log_dir, "slam_console_err.txt");

    (void)freopen(console_log.c_str(), "w", stdout);
    (void)freopen(console_err.c_str(), "w", stderr);

    // Set the locale to C for consistent number formatting

    // Add this static variable at the top of your main function
    static cv::Mat prev_Tcw;  // Previous pose (initialize once)
    cv::VideoWriter slam_video_writer;
    bool video_writer_initialized = false;

    // Get vocabulary and settings file paths from command line arguments
    std::string vocab = args[0];
    std::string settings = args[1];

    // Set log file path with timestamp if not provided
    const char* log_file_env = getenv("SLAM_LOG_FILE");
    if (log_file_env) {
        g_log_file_path = log_file_env;
    } else {
        std::ostringstream oss;
        std::time_t t = std::time(nullptr);
        char timebuf[32];
        std::strftime(timebuf, sizeof(timebuf), "%Y%m%d_%H%M%S", std::localtime(&t));
        oss << join_path(log_dir, std::string("slam_server_debug_") + timebuf + ".log");

        g_log_file_path = oss.str();
    }

    // Log the chosen output path early for troubleshooting
    log_event(std::string("[DEBUG] Video output file: ") + video_file);

    // --- Setup pose sent log file ---
    std::ostringstream pose_log_oss;
    std::time_t pose_log_t = std::time(nullptr);
    char pose_log_timebuf[32];
    std::strftime(pose_log_timebuf, sizeof(pose_log_timebuf), "%Y%m%d_%H%M%S", std::localtime(&pose_log_t));
    pose_log_oss << join_path(log_dir, std::string("pose_sent_") + pose_log_timebuf + ".log");

    std::string pose_log_file_path = pose_log_oss.str();
    std::ofstream pose_log_stream(pose_log_file_path, std::ios::app);

    log_event("=== SLAM TCP Server launched ===");

    // Initialize ORB-SLAM2
    ORB_SLAM2::System SLAM(vocab, settings, ORB_SLAM2::System::STEREO, true);
    log_event("[INFO] SLAM system initialized.");

    // -- Setup TCP server for receiving AirSim images --
    int server_fd = slam_server::create_server_socket(6000); // Create a TCP server socket on port 6000
    if (server_fd < 0) return 1;
    int sock = slam_server::accept_client(server_fd); // Accept a client connection
    if (sock < 0) return 1;

    // --- Setup pose sender socket ---
    int pose_sock = slam_server::connect_pose_sender(slam_server::POSE_RECEIVER_IP,
                                                     slam_server::POSE_RECEIVER_PORT);

    // --- Main loop to receive images and process them with SLAM ---
    cv::Mat imLeft, imRight; // Matrices to hold the received images
    cv::Mat imLeftGray, imRightGray; // Grayscale versions of the images for SLAM processing
    log_event("[DEBUG] Starting main image receive loop...");

    // Initialize frame counter and flags
    int frame_counter = 0;              // Frame counter to track the number of frames processed
    bool slam_ready_flag_written = false; // Flag to indicate if SLAM is ready to process frames
    const int MIN_INLIERS_THRESHOLD = 0;  // Minimum inliers to consider SLAM stable

    static bool first_frame = true;          // Tracks initialization after resets
    static int identity_frame_count = 0;     // Counter for frames showing no motion

    while (true) {
        log_event("----- Begin image receive loop -----");

        // Log the loop start time
        double loop_timestamp = (double)cv::getTickCount() / cv::getTickFrequency();

        // Inlier count will be obtained after SLAM.TrackStereo processes the frame
        int inliers = 0;

        // Process the received images with SLAM
        std::ostringstream oss;
        oss << "Frame #" << frame_counter << " | Loop timestamp: " << std::fixed << std::setprecision(6) << loop_timestamp;
        log_event(oss.str());

            // Receive the left and right images from the TCP socket
            uint32_t net_height, net_width, net_bytes; // Network byte order variables for image dimensions and bytes
            uint32_t rgb_height, rgb_width, rgb_bytes;

            // --- Receive Left Grayscale Image ---
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
            cv::Mat left_gray(rgb_height, rgb_width, CV_8UC1, left_buffer.data()); // Create Mat from buffer
            if (left_gray.empty() || left_gray.rows != rgb_height || left_gray.cols != rgb_width || left_gray.type() != CV_8UC1) {
                log_event("[ERROR] left_gray invalid after construction! Skipping frame.");
                continue;  // or break, depending on your policy
            }

            // Log the received left image properties
            log_event("Received Left image: height=" + std::to_string(left_gray.rows) + ", width=" + std::to_string(left_gray.cols));
            
            // Convert to BGR for visualization
            imLeftGray = left_gray.clone();
            cv::cvtColor(left_gray, imLeft, cv::COLOR_GRAY2BGR); // for debug snapshots

            // --- DEBUG: Log left image properties ---
            {
                std::ostringstream log;
                log << "[DEBUG] Left image received: "
                    << "height="<< left_gray.rows
                    << ", width=" << left_gray.cols
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

            // --- Receive Right Grayscale Image ---
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
            cv::Mat right_gray(rgb_height, rgb_width, CV_8UC1, right_buffer.data()); // Create Mat from buffer
            if (right_gray.empty() || right_gray.rows != rgb_height || right_gray.cols != rgb_width || right_gray.type() != CV_8UC1) {
                log_event("[ERROR] right_gray invalid after construction! Skipping frame.");
                continue;  // or break, depending on your policy
            }

            log_event("Received Right image: height=" + std::to_string(right_gray.rows) + ", width=" + std::to_string(right_gray.cols));
            imRightGray = right_gray.clone();
            cv::cvtColor(right_gray, imRight, cv::COLOR_GRAY2BGR); // for debug snapshots

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
            // Defensive debug image write: only save if not empty and has expected dimensions 
            if (frame_counter % 10 == 0) {
                if (!imLeft.empty() && imLeft.rows > 0 && imLeft.cols > 0) { 
                    std::ostringstream frame_left;
                    frame_left << "logs/debug_left_" << frame_counter << ".png";
                    cv::imwrite(frame_left.str(), imLeft);
                }
                if (!imRight.empty() && imRight.rows > 0 && imRight.cols > 0) { 
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

                    if (!video_writer_initialized) {
                        bool is_color = !imLeft.empty();
                        cv::Size sz = is_color ? imLeft.size() : imLeftGray.size();
                        int fourcc = cv::VideoWriter::fourcc('M','J','P','G');

                        std::ostringstream omsg;
                        omsg << "[DEBUG] Initialising VideoWriter: path=" << video_file
                             << ", size=" << sz.width << "x" << sz.height
                             << ", color=" << (is_color ? "true" : "false");
                        log_event(omsg.str());

                        slam_video_writer.open(video_file, fourcc, 30.0, sz, is_color);
                        if (!slam_video_writer.isOpened()) {
                            log_event("[ERROR] Failed to open video writer: " + video_file);
                        } else {
                            log_event("[INFO] Video writer opened: " + video_file);
                            video_writer_initialized = true;
                        }
                    }
                    if (video_writer_initialized && slam_video_writer.isOpened()) {

                        log_event("[DEBUG] Writing frame with SLAM visualization to video file");
                        
                        // Create enhanced frame with SLAM features
                        cv::Mat enhanced_frame;
                        
                        // Start with the color left image (or convert grayscale to color)
                        if (!imLeft.empty()) {
                            enhanced_frame = imLeft.clone();
                        } else if (!imLeftGray.empty()) {
                            cv::cvtColor(imLeftGray, enhanced_frame, cv::COLOR_GRAY2BGR);
                        } else {
                            log_event("[WARN] No valid image for video recording");
                            continue;
                        }

                        // Get the tracker from SLAM system
                        auto tracker = SLAM.GetTracker();
                        
                        // Add SLAM feature visualization
                        if (tracker && !tracker->mCurrentFrame.mvKeys.empty()) {
                            // Get current frame features
                            std::vector<cv::KeyPoint> current_keypoints = tracker->mCurrentFrame.mvKeys;
                            std::vector<cv::Point2f> tracked_points;
                            
                            // Extract tracked map points
                            for (size_t i = 0; i < tracker->mCurrentFrame.mvpMapPoints.size(); i++) {
                                if (tracker->mCurrentFrame.mvpMapPoints[i] && 
                                    !tracker->mCurrentFrame.mvbOutlier[i]) {
                                    tracked_points.push_back(current_keypoints[i].pt);
                                }
                            }
                            
                            // Draw all detected features (gray circles)
                            for (const auto& kp : current_keypoints) {
                                cv::circle(enhanced_frame, kp.pt, 3, cv::Scalar(128, 128, 128), 1);
                            }
                            
                            // Draw tracked features (green circles)
                            for (const auto& pt : tracked_points) {
                                cv::circle(enhanced_frame, pt, 4, cv::Scalar(0, 255, 0), 2);
                            }
                            
                            // Add tracking state text overlay
                            std::string state_text;
                            switch (tracker->mState) {
                                case ORB_SLAM2::Tracking::SYSTEM_NOT_READY:
                                    state_text = "NOT READY";
                                    break;
                                case ORB_SLAM2::Tracking::NO_IMAGES_YET:
                                    state_text = "NO IMAGES";
                                    break;
                                case ORB_SLAM2::Tracking::NOT_INITIALIZED:
                                    state_text = "INITIALIZING";
                                    break;
                                case ORB_SLAM2::Tracking::OK:
                                    state_text = "TRACKING OK";
                                    break;
                                case ORB_SLAM2::Tracking::LOST:
                                    state_text = "LOST";
                                    break;
                                default:
                                    state_text = "UNKNOWN";
                            }
                            
                            // Add text overlays
                            cv::putText(enhanced_frame, state_text, cv::Point(10, 30), 
                                       cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 255), 2);
                            
                            std::ostringstream info_text;
                            info_text << "Frame: " << frame_counter 
                                     << " | Features: " << current_keypoints.size()
                                     << " | Tracked: " << tracked_points.size();
                            cv::putText(enhanced_frame, info_text.str(), cv::Point(10, 60), 
                                       cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1);
                            
                            // Add pose information if available
                            if (!Tcw.empty() && Tcw.rows == 4 && Tcw.cols == 4) {
                                cv::Mat Twc = Tcw.inv();
                                if (Twc.type() == CV_32F) {
                                    float x = Twc.at<float>(0, 3);
                                    float y = Twc.at<float>(1, 3);
                                    float z = Twc.at<float>(2, 3);
                                    
                                    std::ostringstream pose_text;
                                    pose_text << "Pos: (" << std::fixed << std::setprecision(2) 
                                             << x << ", " << y << ", " << z << ")";
                                    cv::putText(enhanced_frame, pose_text.str(), cv::Point(10, 90), 
                                               cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 0), 1);
                                }
                            }
                            
                            log_event("[DEBUG] Added feature visualization: " + 
                                     std::to_string(current_keypoints.size()) + " features, " +
                                     std::to_string(tracked_points.size()) + " tracked");
                        }
                        
                        // Write the enhanced frame to video
                        slam_video_writer.write(enhanced_frame);
                    }
                } catch (const std::exception& e) {
                    log_event(std::string("[FATAL] SLAM.TrackStereo threw std::exception: ") + e.what());
                    continue;
                } catch (...) {
                    log_event("[FATAL] SLAM.TrackStereo threw unknown exception.");
                    continue;
                }

                // Retrieve inlier count after successful tracking
                inliers = get_feature_inliers(SLAM);
                log_event("[SLAM] Inliers after TrackStereo: " + std::to_string(inliers));

                if (inliers < MIN_INLIERS_THRESHOLD) {
                    log_event("[WARN] Too few inliers tracked. Performing full reinitialization.");
                    perform_full_reinit(SLAM, frame_counter, slam_ready_flag_written,
                                      first_frame, identity_frame_count, flag_dir);
                    continue;
                }

                // --- Handling SLAM results ---
                if (first_frame) {
                    prev_Tcw = Tcw.clone();
                    first_frame = false;
                }

                // Calculate covariance based on pose difference
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


                if (Tcw_copy.empty() || Tcw_copy.rows != 4 || Tcw_copy.cols != 4) {
                    log_event("[WARN] Tcw_copy invalid; skipping identity check.");
                } else {
                    log_event("Tcw_copy rows: " + std::to_string(Tcw_copy.rows) +
                            ", cols: " + std::to_string(Tcw_copy.cols) +
                            ", type: " + std::to_string(Tcw_copy.type()));
                    cv::Mat identity = cv::Mat::eye(4, 4, Tcw_copy.type());

                    // Check if Tcw_copy is close to identity matrix
                    double frobenius_norm = cv::norm(Tcw_copy - identity, cv::NORM_L2);
                    // Check if the identity matrix is detected (no motion)
                    if (frobenius_norm < 1e-3) {  // Identity matrix detected (no motion)
                        identity_frame_count++;

                        if (identity_frame_count < MAX_GRACE_FRAMES) {
                            // Still within grace period, just log and continue
                            log_event("[INFO] Tcw appears to be an identity matrix — no motion detected. Count: " + std::to_string(identity_frame_count));
                        } else {
                            // Grace period exceeded, perform a full reinitialization
                            log_event("[ERROR] Too many frames with no motion, reinitializing SLAM.");
                            perform_full_reinit(SLAM, frame_counter, slam_ready_flag_written,
                                              first_frame, identity_frame_count, flag_dir);
                            continue;
                        }
                    }

                    if (identity_frame_count >= MAX_GRACE_FRAMES) {
                        log_event("[INFO] Grace period over. Checking for motion.");
                        if (frobenius_norm < 1e-3) {
                            log_event("[WARN] No motion detected. Performing full reinitialization.");
                            perform_full_reinit(SLAM, frame_counter, slam_ready_flag_written,
                                              first_frame, identity_frame_count, flag_dir);
                            continue;
                        } else {
                            log_event("[INFO] Motion detected. Continuing normal operation.");
                        }
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
                    kp_filename << join_path(image_dir, std::string("frame_rgb_kp_") + std::to_string(frame_counter) + ".png");

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
                    std::string flag_path = join_path(flag_dir, "slam_ready.flag");

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
                int track_state = SLAM.GetTrackingState();
                log_event("[CHECK] About to test pose_sock condition in main().");

                std::ostringstream live_sock_check;
                live_sock_check << "[CHECK] At pose send time, pose_sock=" << pose_sock
                                 << ", track_state=" << track_state;
                log_event(live_sock_check.str());

                if (pose_sock >= 0 && track_state >= ORB_SLAM2::Tracking::OK && cv::checkRange(Twc)) {
                    log_event("[DEBUG] Entering pose send block.");

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
                        int inlier_sent = send(pose_sock, reinterpret_cast<char*>(&inliers), sizeof(int), 0);
                        if (inlier_sent != sizeof(int)) {
                            log_event("[ERROR] Failed to send inlier count.");
                        } else {
                            log_event("[DEBUG] Inlier count sent to Python receiver: " + std::to_string(inliers));
                        }
                        log_event("Pose (Twc) sent to Python receiver.");

                        // Record metrics for this frame
                        int tracking_state = -1;
                        auto tracker_metrics = SLAM.GetTracker();
                        if (tracker_metrics) tracking_state = tracker_metrics->mState;
                        if (metrics_stream.is_open()) {
                            double relative_time = timestamp - server_start_time;
                            float x = 0.0f, y = 0.0f, z = 0.0f;
                            if (!Twc_send.empty() && Twc_send.rows >= 3 && Twc_send.cols >= 4) {
                                x = Twc_send.at<float>(0, 3);
                                y = Twc_send.at<float>(1, 3);
                                z = Twc_send.at<float>(2, 3);
                            }
                            metrics_stream << std::fixed << std::setprecision(6)
                                           << frame_counter << ',' << relative_time << ','
                                           << tracking_state << ',' << inliers << ','
                                           << covariance_value << ',' << x << ',' << y << ',' << z << '\n';
                        }
                    } else {
                        log_event("[WARN] send_pose() returned false.");
                    }
                } else {
                    log_event("[DEBUG] Pose send conditions not met.");
                }

                } else {
                    log_event("[WARN] Tcw is invalid or not 4x4 matrix — skipping pose send.");
                }

            frame_counter++;
        }

    // --- Cleanup and exit ---
    log_event("[DEBUG] Closing sockets and cleaning up...");
    slam_server::cleanup_resources(sock, server_fd, pose_sock);
    if (pose_log_stream.is_open()) pose_log_stream.close();
    if (metrics_stream.is_open()) metrics_stream.close();
    log_event("[DEBUG] Sockets closed. SLAM server shutting down.");

    if (slam_video_writer.isOpened()) {
        log_event("[DEBUG] Releasing video writer");
        slam_video_writer.release();
    }

    SLAM.Shutdown();

    SLAM.SaveTrajectoryTUM(join_path(log_dir, "CameraTrajectory.txt"));
    SLAM.SaveKeyFrameTrajectoryTUM(join_path(log_dir, "KeyFrameTrajectory.txt"));
    SLAM.SaveMapPoints(join_path(log_dir, "MapPoints.txt"));

    return 0;
}
