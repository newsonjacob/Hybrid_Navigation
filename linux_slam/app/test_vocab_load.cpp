#include <iostream>
#include "System.h"
#include <chrono>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: ./test_vocab_load path_to_vocab" << std::endl;
        return 1;
    }
    auto start = std::chrono::steady_clock::now();
    ORB_SLAM2::System SLAM(argv[1], "../app/rgbd_settings.yaml", ORB_SLAM2::System::RGBD, false);
    auto end = std::chrono::steady_clock::now();
    std::cout << "Vocabulary loaded in "
              << std::chrono::duration_cast<std::chrono::seconds>(end - start).count()
              << " seconds." << std::endl;
    return 0;
}
