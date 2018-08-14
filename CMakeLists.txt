cmake_minimum_required(VERSION 3.12)
project(ObjectDetection)

set(CMAKE_CXX_STANDARD 14)

add_executable(ObjectDetection main.cpp)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(ObjectDetection ${OpenCV_LIBS})
