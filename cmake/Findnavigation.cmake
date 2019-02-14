# Find the navigation library.
find_path(NAVIGATION_INCLUDE_DIR ddpg_continuous_control.h  
                                 keyboard.h 
                                 models.h 
                                 ornstein_uhlenbeck_process.h
                                 q_learning.h 
                                 replay_memory.h ${NAVIGATION_DIR}/include/navigation)

find_library(NAVIGATION_LIBRARIES
    NAMES navigation
    PATHS ${NAVIGATION_DIR}/lib
) 

if (NAVIGATION_INCLUDE_DIR AND NAVIGATION_LIBRARIES)
   set(NAVIGATION_FOUND TRUE)
endif ()

if (NAVIGATION_FOUND)
  if (NOT NAVIGATION_FIND_QUIETLY)
    message(STATUS "Found navigation: ${NAVIGATION_LIBRARIES}")
  endif ()
else (NAVIGATION_FOUND)
  if (NAVIGATION_FIND_REQUIRED)
    message(FATAL_ERROR "Could not find navigation")
  endif ()
endif ()
