cmake_minimum_required(VERSION 2.8)

#-------------------------------------------------------------------------------
# Build type
#-------------------------------------------------------------------------------
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "serial" CACHE STRING
      "Choose the type of build: serial | debug | parallel | pdebug" FORCE)
endif()
string(TOUPPER ${CMAKE_BUILD_TYPE} BUILD_TYPE)

#-------------------------------------------------------------------------------
# Project
#-------------------------------------------------------------------------------
project(mfem CXX)
message(STATUS "C++ compiler ID = ${CMAKE_CXX_COMPILER_ID}")
if(WIN32)
    set(AUX_CXX_FLAGS "/Wall")
else()
  if(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
    #set(AUX_CXX_FLAGS "-Wall -Wextra -Wshadow -Weffc++ -ansi -pedantic -Werror")
    set(AUX_CXX_FLAGS "-std=c++0x -Wall")
  elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
    #set(AUX_CXX_FLAGS "-Wall -Wextra -Wshadow -Weffc++ -pedantic -Werror")
    set(AUX_CXX_FLAGS "-Wall")
  else()
    #set(AUX_CXX_FLAGS "-Wall -pedantic -Werror")
    set(AUX_CXX_FLAGS "-Wall")
  endif()
endif()
if(MFEM_USE_OPENMP)
  if(WIN32)
    set(AUX_CXX_FLAGS "${AUX_CXX_FLAGS} /openmp")
  else()
    set(AUX_CXX_FLAGS "${AUX_CXX_FLAGS} -openmp")
  endif()
endif()

#-------------------------------------------------------------------------------
# MFEM_DEBUG
#-------------------------------------------------------------------------------
if(BUILD_TYPE MATCHES "DEBUG") # DEBUG and PDEBUG go here
  set(MFEM_DEBUG ON)
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${AUX_CXX_FLAGS}")
else() # SERIAL and PARALLEL go here
  set(MFEM_DEBUG OFF)
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${AUX_CXX_FLAGS}")
endif()

#-------------------------------------------------------------------------------
# MFEM_USE_MPI, HYPRE, METIS
#-------------------------------------------------------------------------------
if(BUILD_TYPE STREQUAL PARALLEL OR BUILD_TYPE STREQUAL PDEBUG)
  set(MFEM_USE_MPI ON)
  # MPI
  find_package(MPI REQUIRED)
  include_directories(${MPI_CXX_INCLUDE_PATH})
  set(CMAKE_CXX_COMPILER ${MPI_CXX_COMPILER})
  # Hypre
  find_path(HYPRE_DIR include/HYPRE.h HINTS ENV HYPRE_DIR PATHS "$ENV{HOME}/software/hypre-2.10.1/src" DOC "Hypre directory")
  if(EXISTS ${HYPRE_DIR}/include/HYPRE.h)
    set(HYPRE_FOUND YES)
    message(STATUS "Hypre found ${HYPRE_DIR}")
    set(HYPRE_INCLUDES ${HYPRE_DIR})
    find_path(HYPRE_INCLUDE_DIR HYPRE.h HINTS "${HYPRE_DIR}" PATH_SUFFIXES include NO_DEFAULT_PATH)
    list(APPEND HYPRE_INCLUDES ${HYPRE_INCLUDE_DIR})
    include_directories(${HYPRE_INCLUDES})
    find_library(HYPRE_LIBRARY NAMES hypre HYPRE HINTS "${HYPRE_DIR}/lib" DOC "Hypre library")
    message(STATUS "Hypre lib ${HYPRE_LIBRARY}")
    get_filename_component(HYPRE_LIB_DIR ${HYPRE_LIBRARY} DIRECTORY)
    message(STATUS "HYPRE_LIB_DIR ${HYPRE_LIB_DIR}")
    link_directories(${HYPRE_LIB_DIR})
  else()
    set(HYPRE_FOUND NO)
    message(FATAL_ERROR "Hypre is NOT found")
  endif()
  # Metis
  find_path(METIS_DIR include/metis.h HINTS ENV METIS_DIR PATHS "$ENV{HOME}/software/metis-5.1.0/build/Linux-x86_64" DOC "Metis directory")
  if(EXISTS ${METIS_DIR}/include/metis.h)
    set(METIS_FOUND YES)
    message(STATUS "Metis found ${METIS_DIR}")
    set(METIS_INCLUDES ${METIS_DIR})
    find_path(METIS_INCLUDE_DIR metis.h HINTS "${METIS_DIR}" PATH_SUFFIXES include NO_DEFAULT_PATH)
    list(APPEND METIS_INCLUDES ${METIS_INCLUDE_DIR})
    include_directories(${METIS_INCLUDES})
    find_library(METIS_LIBRARY NAMES metis METIS HINTS "${METIS_DIR}/lib" DOC "Metis library")
    message(STATUS "Metis lib ${METIS_LIBRARY}")
    string(FIND ${METIS_DIR} "metis-5" IS_METIS_5)
    if(IS_METIS_5)
      set(MFEM_USE_METIS_5 ON)
    else()
      set(MFEM_USE_METIS_5 OFF)
    endif()
    get_filename_component(METIS_LIB_DIR ${METIS_LIBRARY} DIRECTORY)
    message(STATUS "METIS_LIB_DIR ${METIS_LIB_DIR}")
    link_directories(${METIS_LIB_DIR})
  else()
    message(FATAL_ERROR "Metis is NOT found")
    set(METIS_FOUND NO)
  endif()
else()
  set(MFEM_USE_MPI OFF)
endif()

#-------------------------------------------------------------------------------
# LAPACK
#-------------------------------------------------------------------------------
if(MFEM_USE_LAPACK)
  enable_language(C)
  find_package(LAPACK REQUIRED)
endif()

#-------------------------------------------------------------------------------
# MFEM_USE_MEMALLOC
#-------------------------------------------------------------------------------
set(MFEM_USE_MEMALLOC ON)

#-------------------------------------------------------------------------------
# MFEM_TIMER_TYPE
#-------------------------------------------------------------------------------
if(NOT MFEM_TIMER_TYPE)
  if(UNIX)
    set(MFEM_TIMER_TYPE 2)
  elseif(WIN32)
    set(MFEM_TIMER_TYPE 3)
  else()
    set(MFEM_TIMER_TYPE 1)
  endif()
endif()

#-------------------------------------------------------------------------------
# Windows specific
#-------------------------------------------------------------------------------
if(WIN32)
  set(_USE_MATH_DEFINES ON)
else()
  set(_USE_MATH_DEFINES OFF)
endif()

#-------------------------------------------------------------------------------
# Config file
#-------------------------------------------------------------------------------
configure_file(
  "${PROJECT_SOURCE_DIR}/config/config.hpp.in_cmake"
  "${PROJECT_SOURCE_DIR}/config/config.hpp")
include_directories(${PROJECT_SOURCE_DIR}/config)

#-------------------------------------------------------------------------------
# Headers and sources
#-------------------------------------------------------------------------------
set(DIRS general linalg mesh fem config)
foreach(dir IN LISTS DIRS)
  file(GLOB SRC "${PROJECT_SOURCE_DIR}/${dir}/*.cpp")
  file(GLOB HDR "${PROJECT_SOURCE_DIR}/${dir}/*.hpp")
  list(APPEND SRC_LIST ${SRC})
  list(APPEND HDR_LIST ${HDR})
endforeach()

#-------------------------------------------------------------------------------
# Put the information about the configuration on the screen
#-------------------------------------------------------------------------------
  message("\n--------------- configuration information ----------------")
  message("build type     : " ${BUILD_TYPE})
  message("compiler       : " ${CMAKE_CXX_COMPILER})
if(MFEM_USE_MPI)
  message("hypre          : " ${HYPRE_PATH})
  message("metis          : " ${METIS_PATH})
endif()
if(MFEM_DEBUG)
  message("compiler flags : " ${CMAKE_CXX_FLAGS_DEBUG})
else()
  message("compiler flags : " ${CMAKE_CXX_FLAGS_RELEASE})
endif()

#-------------------------------------------------------------------------------
# Declaring the library
#-------------------------------------------------------------------------------
add_library(${PROJECT_NAME} ${SRC_LIST} ${HDR_LIST})
if(MFEM_USE_LAPACK)
  target_link_libraries(${PROJECT_NAME} ${LAPACK_LIBRARIES})
endif()
if(CMAKE_SYSTEM_NAME MATCHES "Linux")
  target_link_libraries(${PROJECT_NAME} rt)
endif()
if(MFEM_USE_MPI)
  target_link_libraries(${PROJECT_NAME} ${MPI_CXX_LIBRARIES} ${HYPRE_LIBRARY} ${METIS_LIBRARY})
endif()

#-------------------------------------------------------------------------------
# Installation
#-------------------------------------------------------------------------------
install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_INSTALL_PREFIX}/${BUILD_TYPE}/lib)
install(FILES ${PROJECT_SOURCE_DIR}/mfem.hpp DESTINATION ${CMAKE_INSTALL_PREFIX}/${BUILD_TYPE}/include)
foreach(dir IN LISTS DIRS)
  file(GLOB HDR "${PROJECT_SOURCE_DIR}/${dir}/*.hpp")
  install(FILES ${HDR} DESTINATION ${CMAKE_INSTALL_PREFIX}/${BUILD_TYPE}/include/${dir})
endforeach()
