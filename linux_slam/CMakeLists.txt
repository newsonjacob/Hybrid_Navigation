cmake_minimum_required(VERSION 2.8)
cmake_policy(SET CMP0028 NEW)
project(ORB_SLAM2)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

message("Build type: ${CMAKE_BUILD_TYPE}")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3 -march=native ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -O3 -march=native")

add_compile_options(-Wno-deprecated-declarations)

# Use C++14 standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
message(STATUS "Using C++ standard: ${CMAKE_CXX_STANDARD}")

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

# Prefer OpenCV from /usr/local
set(OpenCV_DIR "/usr/local/share/OpenCV")

find_package(OpenCV REQUIRED)
if(NOT OpenCV_FOUND)
   message(FATAL_ERROR "OpenCV not found.")
endif()

message(STATUS "OpenCV version: ${OpenCV_VERSION}")

# Explicitly set Eigen3 and Pangolin search paths
set(Eigen3_DIR "/usr/lib/cmake/eigen3")
set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};/home/jacob/slam_ws/Pangolin/build")

find_package(Eigen3 3.1.0 REQUIRED NO_MODULE)
find_package(Pangolin REQUIRED)

include_directories(
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/include
  ${Pangolin_INCLUDE_DIRS}
)


# Output libraries and executables inside the build folder
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

add_library(${PROJECT_NAME} SHARED
  src/System.cc
  src/Tracking.cc
  src/LocalMapping.cc
  src/LoopClosing.cc
  src/ORBextractor.cc
  src/ORBmatcher.cc
  src/FrameDrawer.cc
  src/Converter.cc
  src/MapPoint.cc
  src/KeyFrame.cc
  src/Map.cc
  src/MapDrawer.cc
  src/Optimizer.cc
  src/PnPsolver.cc
  src/Frame.cc
  src/KeyFrameDatabase.cc
  src/Sim3Solver.cc
  src/Initializer.cc
  src/Viewer.cc
)

target_include_directories(${PROJECT_NAME} PRIVATE
  ${PROJECT_SOURCE_DIR}/include
)


target_link_libraries(${PROJECT_NAME}
  ${OpenCV_LIBS}
  Eigen3::Eigen
  ${Pangolin_LIBRARIES}
  ${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
  ${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.so
)

# --- Build App ---
add_executable(custom_slam app/offline_slam_evaluation.cpp)

set_target_properties(custom_slam PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/app)
target_link_libraries(custom_slam ${PROJECT_NAME})

# --- Build RGB-D Examples ---
add_executable(rgbd_tum Examples/RGB-D/rgbd_tum.cc)
set_target_properties(rgbd_tum PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/Examples/RGB-D)
target_link_libraries(rgbd_tum ${PROJECT_NAME})

# --- Build Stereo Examples ---
add_executable(stereo_kitti Examples/Stereo/stereo_kitti.cc)
set_target_properties(stereo_kitti PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/Examples/Stereo)
target_link_libraries(stereo_kitti ${PROJECT_NAME})

add_executable(stereo_euroc Examples/Stereo/stereo_euroc.cc)
set_target_properties(stereo_euroc PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/Examples/Stereo)
target_link_libraries(stereo_euroc ${PROJECT_NAME})

# --- Build Monocular Examples ---
add_executable(mono_tum Examples/Monocular/mono_tum.cc)
set_target_properties(mono_tum PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/Examples/Monocular)
target_link_libraries(mono_tum ${PROJECT_NAME})

add_executable(mono_kitti Examples/Monocular/mono_kitti.cc)
set_target_properties(mono_kitti PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/Examples/Monocular)
target_link_libraries(mono_kitti ${PROJECT_NAME})

add_executable(mono_euroc Examples/Monocular/mono_euroc.cc)
set_target_properties(mono_euroc PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/Examples/Monocular)
target_link_libraries(mono_euroc ${PROJECT_NAME})

add_executable(test_vocab_load app/test_vocab_load.cpp)
target_include_directories(test_vocab_load PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(test_vocab_load ${PROJECT_NAME})

set_target_properties(test_vocab_load PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/app)

# --- Build TCP SLAM Server ---
add_executable(tcp_slam_server app/tcp_slam_server.cpp)
target_include_directories(tcp_slam_server PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(tcp_slam_server ${PROJECT_NAME})
set_target_properties(tcp_slam_server PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/app)
