cmake_minimum_required(VERSION 2.8.11)

# Load user settings before the defaults - this way the defaults will not
# overwrite the user set options. If the user has not set all options, we still
# have the defaults.
include(${CMAKE_CURRENT_SOURCE_DIR}/config/user.cmake OPTIONAL)
include(${CMAKE_CURRENT_SOURCE_DIR}/config/defaults.cmake)

# Allow overwriting of the compiler by setting CXX/MPICXX on the command line
# or in user.cmake.
if (CXX)
  set(CMAKE_CXX_COMPILER ${CXX})
  # Avoid some issues when CXX is defined
  unset(CXX)
  unset(CXX CACHE)
endif()
if (MFEM_USE_MPI AND MPICXX)
  # In parallel MPICXX takes precedence, if defined.
  set(CMAKE_CXX_COMPILER ${MPICXX})
  # Setting the variables below circumvents autodetection, see FindMPI.cmake.
  set(MPI_CXX_INCLUDE_PATH "")
  set(MPI_CXX_LIBRARIES "")
endif()

project(mfem CXX)

if (CMAKE_VERSION VERSION_LESS 3.2)
   # This seems to be needed by: find_package(BLAS REQUIRED)
   enable_language(C)
endif()

# CMake needs to know where to find things
set(MFEM_CMAKE_PATH ${PROJECT_SOURCE_DIR}/config/cmake)
set(CMAKE_MODULE_PATH ${MFEM_CMAKE_PATH}/modules)

# Define MFEM_BUILD_DIR to point to the build directory.
add_definitions("-DMFEM_BUILD_DIR=${PROJECT_BINARY_DIR}")

# Include the source directory where mfem.hpp and mfem-performance.hpp are.
# TODO: building the library does not require this; just the examples/miniapps.
include_directories(${PROJECT_SOURCE_DIR})

# MFEM_DEBUG
if (CMAKE_BUILD_TYPE MATCHES "Debug|debug|DEBUG")
  set(MFEM_DEBUG ON)
else()
  set(MFEM_DEBUG OFF)
endif()

# MPI -> hypre, METIS
if (MFEM_USE_MPI)
  find_package(MPI REQUIRED)
  include_directories(${MPI_CXX_INCLUDE_PATH})
  # Parallel MFEM depends on Hypre and Metis
  find_package(HYPRE REQUIRED)
  include_directories(${HYPRE_INCLUDE_DIRS})
  find_package(METIS REQUIRED)
  include_directories(${METIS_INCLUDE_DIRS})
endif()

# Backtrace with libunwind
if (MFEM_USE_LIBUNWIND)
  set(MFEMBacktrace_REQUIRED_PACKAGES "Libunwind" "LIBDL" "CXXABIDemangle")
  find_package(MFEMBacktrace REQUIRED)
  include_directories(${LIBUNWIND_INCLUDE_DIRS})
endif()

# BLAS, LAPACK
if (MFEM_USE_LAPACK)
  find_package(BLAS REQUIRED)
  find_package(LAPACK REQUIRED)
endif()

# OpenMP
if (MFEM_USE_OPENMP)
  if (MFEM_THREAD_SAFE)
    find_package(OpenMP REQUIRED)
  else()
    message(FATAL_ERROR " *** MFEM_USE_OPENMP requires MFEM_THREAD_SAFE=ON.")
  endif()
endif()

# Mesquite
if (MFEM_USE_MESQUITE)
  find_package(Mesquite REQUIRED)
  include_directories(${MESQUITE_INCLUDE_DIRS})
endif()

# SuiteSparse
if (MFEM_USE_SUITESPARSE)
  find_package(SuiteSparse REQUIRED)
  include_directories(${SuiteSparse_INCLUDE_DIRS})
endif()

# SuperLU_DIST can only be enabled if parallel
if (MFEM_USE_SUPERLU)
  if (MFEM_USE_MPI)
    find_package(SuperLUDist REQUIRED)
    include_directories(${SuperLUDist_INCLUDE_DIRS})
  else()
    message(FATAL_ERROR " *** SuperLU_DIST requires that MPI be enabled.")
  endif()
endif()

# Gecko
if (MFEM_USE_GECKO)
  find_package(Gecko REQUIRED)
  include_directories(${GECKO_INCLUDE_DIRS})
endif()

# GnuTLS
if (MFEM_USE_GNUTLS)
  find_package(_GnuTLS REQUIRED)
  include_directories(${GNUTLS_INCLUDE_DIRS})
endif()

# NetCDF
if (MFEM_USE_NETCDF)
  find_package(NetCDF REQUIRED)
  include_directories(${NETCDF_INCLUDE_DIRS})
endif()

# MPFR
if (MFEM_USE_MPFR)
  find_package(MPFR REQUIRED)
  include_directories(${MPFR_INCLUDE_DIRS})
endif()

# MFEM_TIMER_TYPE
if (NOT DEFINED MFEM_TIMER_TYPE)
  if (APPLE)
    set(MFEM_TIMER_TYPE 0) # use std::clock from <ctime>
  elseif (WIN32)
    set(MFEM_TIMER_TYPE 3) # QueryPerformanceCounter from <windows.h>
  else()
    find_package(POSIXClocks)
    if (POSIXCLOCKS_FOUND)
      set(MFEM_TIMER_TYPE 2) # use high-resolution POSIX clocks
    else()
      set(MFEM_TIMER_TYPE 0) # use std::clock from <ctime>
    endif()
  endif()
endif()


# List all posible libraries in order of dependencies.
set(MFEM_TPLS HYPRE OPENMP MEQUITE SuiteSparse SuperLUDist ParMETIS METIS LAPACK
    BLAS GECKO GNUTLS NETCDF MPFR POSIXCLOCKS MFEMBacktrace)
# Add all *_FOUND libraries in the variable TPL_LIBRARIES.
set(TPL_LIBRARIES "")
foreach(TPL IN LISTS MFEM_TPLS)
  if (${TPL}_FOUND)
    message(STATUS "MFEM: using package ${TPL}")
    list(APPEND TPL_LIBRARIES ${${TPL}_LIBRARIES})
  endif()
endforeach(TPL)

if (OPENMP_FOUND)
  message(STATUS "MFEM: using package OpenMP")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

message(STATUS "MFEM build type: ${CMAKE_BUILD_TYPE}")


# Windows specific
set(_USE_MATH_DEFINES ${WIN32})

# Headers and sources
include(MfemCmakeUtilities)
set(SOURCES "")
set(HEADERS "")
add_subdirectory(general)
add_subdirectory(linalg)
add_subdirectory(mesh)
add_subdirectory(fem)
add_subdirectory(config)

# Declaring the library
add_library(mfem ${SOURCES} ${HEADERS})
list(REMOVE_DUPLICATES TPL_LIBRARIES)
# message(STATUS " TPL_LIBRARIES = ${TPL_LIBRARIES}")
if (CMAKE_VERSION VERSION_GREATER 2.8.11)
  target_link_libraries(mfem PUBLIC ${TPL_LIBRARIES})
else()
  target_link_libraries(mfem ${TPL_LIBRARIES})
endif()

# The examples and miniapps are separate
if (MFEM_ENABLE_TESTING)
  enable_testing()
endif()
if (MFEM_ENABLE_EXAMPLES)
  add_subdirectory(examples)
endif()
if (MFEM_ENABLE_MINIAPPS)
  add_subdirectory(miniapps)
endif()

# Generate configuration file in the build directory: config/_config.hpp.
configure_file(
  "${PROJECT_SOURCE_DIR}/config/config.hpp.in_cmake"
  "${PROJECT_BINARY_DIR}/config/_config.hpp")

# Create substitute mfem.hpp and mfem-performance.hpp in the build directory,
# if it is different from the source directory.
if (NOT ("${PROJECT_SOURCE_DIR}" STREQUAL "${PROJECT_BINARY_DIR}"))
  foreach(Header mfem.hpp mfem-performance.hpp)
    message(STATUS
      "Writing substitute header --> \"${Header}\"")
    file(WRITE "${PROJECT_BINARY_DIR}/${Header}"
"// Auto-generated file.
#define MFEM_BUILD_DIR ${PROJECT_BINARY_DIR}
#include \"${PROJECT_SOURCE_DIR}/${Header}\"
")
  endforeach()
endif()

# Create the "test" and "check" targets. "make test"/"ctest" should
# follow the existing makefile structure. "make check" should just
# build ex1(p) and run it in serial/parallel.

#-------------------------------------------------------------------------------
# Installation
#-------------------------------------------------------------------------------
#string(TOLOWER ${BUILD_TYPE} INSTALL_DIR)
#set(INSTALL_DIR ${CMAKE_INSTALL_PREFIX}/${INSTALL_DIR})
#install(TARGETS ${PROJECT_NAME} DESTINATION ${INSTALL_DIR}/lib)
#install(FILES ${PROJECT_SOURCE_DIR}/mfem.hpp DESTINATION ${INSTALL_DIR}/include)
#foreach(dir IN LISTS DIRS)
#  file(GLOB HDR "${PROJECT_SOURCE_DIR}/${dir}/*.hpp")
#  install(FILES ${HDR} DESTINATION ${INSTALL_DIR}/include/${dir})
#endforeach()
