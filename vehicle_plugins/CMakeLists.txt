option(BUILD_WITH_TORCH "Whether to build with Torch. This enables autonomous navigatoin." OFF)

find_package(navigation REQUIRED)

include_directories(${NAVIGATION_INCLUDE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

if (${BUILD_WITH_TORCH})

  find_package(Torch REQUIRED)
  find_package(OpenCV REQUIRED)

  include_directories(${Torch_INCLUDE_DIRS})
  include_directories(${OpenCV_INCLUDE_DIRS})

  list(APPEND VEHICLE_PLUGIN_SRCS 
  list(APPEND VEHICLE_PLUGIN_LIBRARIES ${Torch_LIBRARIES} ${OpenCV_LIBRARIES})

endif ()





add_library(gazeboVehiclePlugin SHARED ${CMAKE_CURRENT_SOURCE_DIR}/vehicle_plugin.cpp)
target_link_libraries(gazeboVehiclePlugin ${GAZEBO_LIBRARIES} ${TORCH_LIBRARIES} ${NAVIGATION_LIBRARIES})
