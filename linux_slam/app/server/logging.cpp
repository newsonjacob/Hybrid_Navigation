#include "logging.hpp"
#include <fstream>
#include <iostream>
#include <mutex>
#include <iomanip>
#include <ctime>
#include <memory>

namespace slam_server {
std::string g_log_file_path;
static std::mutex g_log_mutex;

void log_event(const std::string& msg) {
    static bool warned = false;
    static std::unique_ptr<std::ofstream> log_stream;

    if (g_log_file_path.empty()) {
        if (!warned) {
            std::cerr << "[WARN] Logging is disabled because g_log_file_path is empty." << std::endl;
            warned = true;
        }
        return;
    }

    std::lock_guard<std::mutex> lock(g_log_mutex);
    if (!log_stream) {
        log_stream = std::make_unique<std::ofstream>(g_log_file_path, std::ios::app);
        if (!log_stream->is_open() && !warned) {
            std::cerr << "[WARN] Could not open log file: " << g_log_file_path << std::endl;
            warned = true;
        }
    }
    if (log_stream && log_stream->is_open()) {
        std::time_t t = std::time(nullptr);
        (*log_stream) << "[" << std::put_time(std::localtime(&t), "%F %T") << "] " << msg << std::endl;
        log_stream->flush();
    }
}
} // namespace slam_server
