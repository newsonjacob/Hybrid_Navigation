#pragma once
#include <string>
namespace slam_server {
extern std::string g_log_file_path;
void log_event(const std::string& msg);
}
