cmake_minimum_required(VERSION 3.11 FATAL_ERROR)

find_package(gazebo REQUIRED)
link_directories(${GAZEBO_LIBRARY_DIRS})
list(APPEND CMAKE_CXX_FLAGS "${GAZEBO_CXX_FLAGS}")
include_directories(${GAZEBO_INCLUDE_DIRS})

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../utils)

add_executable(sample sample_uniformly.cpp)
target_link_libraries(sample ${GAZEBO_LIBRARIES})
