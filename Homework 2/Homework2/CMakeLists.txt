cmake_minimum_required(VERSION 3.15)
project(Homework2)

add_definitions(-DPROJ_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

find_package(PkgConfig REQUIRED)

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OPENCV_INCLUDE_DIR})

set(CMAKE_CXX_STANDARD 14)

include_directories(${PROJECT_SOURCE_DIR}/include)

set(HEADER_FILES
        include/HOGDescriptor.h
        include/RandomForest.h
        external/RandomRotation.h)

set(SOURCE_FILES
        src/task1/HOGDescriptor.cpp
        src/task2/RandomForest.cpp
        external/RandomRotation.cpp)


# can rename executables further
add_executable(main src/main.cpp)
target_link_libraries(main ${OpenCV_LIBRARIES})

add_executable(HogDescriptor src/task1/task1.cpp ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(HogDescriptor ${OpenCV_LIBRARIES})

add_executable(ObjectClassification src/task2/task2.cpp ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(ObjectClassification ${OpenCV_LIBRARIES})