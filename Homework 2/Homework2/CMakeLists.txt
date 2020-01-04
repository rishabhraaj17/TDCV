cmake_minimum_required(VERSION 3.15)
project(Homework2)

add_definitions(-DPROJ_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

find_package(PkgConfig REQUIRED)

# OpenCV
set(OpenCV_DIR /usr/local/lib/cmake/opencv4)
find_package(OpenCV REQUIRED)
include_directories(${OPENCV_INCLUDE_DIR})

set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
find_package(Boost 1.65.1 COMPONENTS filesystem regex)
#[[if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
    add_executable(progname file1.cxx file2.cxx)
    target_link_libraries(progname ${Boost_LIBRARIES})
endif()]]

set(CMAKE_CXX_STANDARD 14)

include_directories(${Boost_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)

set(HEADER_FILES
        include/HOGDescriptor.h
        include/RandomForest.h
        include/Util.h
        include/NonMaximalSuppression.h)

set(SOURCE_FILES
        src/task1/HOGDescriptor.cpp
        src/task2/RandomForest.cpp
        src/task2/Util.cpp
        src/task3/NonMaximalSuppression.cpp)


# can rename executables further
add_executable(main src/main.cpp)
target_link_libraries(main ${OpenCV_LIBRARIES})

add_executable(HogDescriptor src/task1/task1.cpp ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(HogDescriptor ${OpenCV_LIBRARIES})
target_link_libraries(HogDescriptor ${Boost_LIBRARIES})

add_executable(ObjectClassification src/task2/task2.cpp ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(ObjectClassification ${OpenCV_LIBRARIES})
target_link_libraries(ObjectClassification ${Boost_LIBRARIES})

add_executable(ObjectDetection src/task3/task3.cpp ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(ObjectDetection ${OpenCV_LIBRARIES})
target_link_libraries(ObjectDetection ${Boost_LIBRARIES})

# Debug
#[[
get_cmake_property(_variableNames VARIABLES)
list (SORT _variableNames)
foreach (_variableName ${_variableNames})
    message(STATUS "${_variableName}=${${_variableName}}")
endforeach()]]
