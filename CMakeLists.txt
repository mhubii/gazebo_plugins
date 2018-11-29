cmake_minimum_required(VERSION 3.11 FATAL_ERROR)

set(NAVIGATION_DIR $ENV{HOME}/navigation CACHE STRING "Directory of navigation.")
#set(CMAKE_CXX_STANDARD 11)

project(gazebo_plugins)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)

find_package(gazebo REQUIRED)
link_directories(${GAZEBO_LIBRARY_DIRS})
list(APPEND CMAKE_CXX_FLAGS "${GAZEBO_CXX_FLAGS}")
include_directories(${GAZEBO_INCLUDE_DIRS})

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

option(BUILD_VEHICLE_PLUGINS "Build plugins for vehicle." OFF)
option(BUILD_WITH_TORCH "Whether to build with Torch." OFF)

if (${BUILD_VEHICLE_PLUGINS})

	message(STATUS "Building vehicle plugins.")

	add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/vehicle_plugins)

endif ()


