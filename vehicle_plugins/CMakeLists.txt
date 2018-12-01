find_package(navigation REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(${NAVIGATION_INCLUDE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/vehicle_plugins)

add_library(gazeboVehicleManualControl SHARED ${CMAKE_CURRENT_SOURCE_DIR}/src/manual_control.cpp)
target_link_libraries(gazeboVehicleManualControl ${GAZEBO_LIBRARIES} ${Eigen_LIBRARIES} ${NAVIGATION_LIBRARIES})

if (${BUILD_WITH_TORCH})

  find_package(Torch REQUIRED)
  find_package(OpenCV REQUIRED)

  include_directories(${Torch_INCLUDE_DIRS})
  include_directories(${OpenCV_INCLUDE_DIRS})

  add_library(gazeboVehicleAutonomousControl SHARED ${CMAKE_CURRENT_SOURCE_DIR}/src/autonomous_control.cpp)
  target_link_libraries(gazeboVehicleAutonomousControl ${GAZEBO_LIBRARIES} ${TORCH_LIBRARIES} ${NAVIGATION_LIBRARIES})

endif ()



