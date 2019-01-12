find_package(navigation REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(${NAVIGATION_INCLUDE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/vehicle_plugins)

if (${BUILD_WITH_TORCH})

  find_package(Torch REQUIRED)

  include_directories(${Torch_INCLUDE_DIRS})

  add_library(gazeboVehicleManualControl SHARED ${CMAKE_CURRENT_SOURCE_DIR}/src/manual_control.cpp)
  target_link_libraries(gazeboVehicleManualControl ${GAZEBO_LIBRARIES} ${TORCH_LIBRARIES} ${Eigen_LIBRARIES} ${NAVIGATION_LIBRARIES})

  add_library(gazeboVehicleAutonomousControl SHARED ${CMAKE_CURRENT_SOURCE_DIR}/src/autonomous_control.cpp)
  target_link_libraries(gazeboVehicleAutonomousControl ${GAZEBO_LIBRARIES} ${TORCH_LIBRARIES} ${NAVIGATION_LIBRARIES})

  add_library(gazeboVehicleReinforcementLearning SHARED ${CMAKE_CURRENT_SOURCE_DIR}/src/reinforcement_learning.cpp)
  target_link_libraries(gazeboVehicleReinforcementLearning ${GAZEBO_LIBRARIES} ${TORCH_LIBRARIES} ${NAVIGATION_LIBRARIES})

else ()

  add_library(gazeboVehicleManualControl SHARED ${CMAKE_CURRENT_SOURCE_DIR}/src/manual_control.cpp)
  target_link_libraries(gazeboVehicleManualControl ${GAZEBO_LIBRARIES} ${Eigen_LIBRARIES} ${NAVIGATION_LIBRARIES})

endif ()

if (${BUILD_WITH_ORB_SLAM2})

  find_package(ORB_SLAM2 REQUIRED)

  include_directories(${ORB_SLAM2_INCLUDE_DIR})
  include_directories(${PROJECT_SOURCE_DIR}/third_party/ORB_SLAM2)

  add_library(gazeboVehicleManualControlSlam SHARED ${CMAKE_CURRENT_SOURCE_DIR}/src/manual_control_slam.cpp)
  target_link_libraries(gazeboVehicleManualControlSlam ${GAZEBO_LIBRARIES} ${Eigen_LIBRARIES} ${ORB_SLAM2_LIBRARY} ${NAVIGATION_LIBRARIES})

endif()



