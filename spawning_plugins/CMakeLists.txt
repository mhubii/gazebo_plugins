add_library(gazeboInfinitySpawn SHARED ${CMAKE_CURRENT_SOURCE_DIR}/src/infinity_spawn.cpp)
target_link_libraries(gazeboInfinitySpawn ${GAZEBO_LIBRARIES})
