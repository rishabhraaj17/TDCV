cmake_minimum_required(VERSION 3.15)
project(Homework2)

add_definitions(-DPROJ_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

find_package(PkgConfig REQUIRED)

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OPENCV_INCLUDE_DIR})

set(CMAKE_CXX_STANDARD 14)

# can rename executables further
add_executable(main src/main.cpp)
target_link_libraries(main ${OpenCV_LIBRARIES})