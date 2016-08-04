cmake_minimum_required(VERSION 2.8.11)

project(mfem)

# CMake needs to know where to find things
set(MFEM_CMAKE_PATH ${PROJECT_SOURCE_DIR}/config/cmake)
set(CMAKE_MODULE_PATH ${MFEM_CMAKE_PATH}/modules)

# MFEM options
option(MFEM_USE_MPI "Enable MPI parallel build" OFF)
option(MFEM_USE_LAPACK "Enable LAPACK usage" OFF)
option(MFEM_USE_OPENMP "Enable OpenMP usage" OFF)
option(MFEM_USE_MESQUITE "Enable MESQUITE usage" OFF)
option(MFEM_USE_SUITESPARSE "Enable SuiteSparse usage" OFF)
option(MFEM_USE_GECKO "Enable GECKO usage" OFF)
option(MFEM_USE_SUPERLU "Enable SuperLU-dist usage" OFF)
option(MFEM_USE_GNUTLS "Enable GNUTLS usage" OFF)
option(MFEM_USE_NETCDF "Enable NETCDF usage" OFF)
option(MFEM_USE_MEMALLOC "Enable MEMALLOC usage" ON)
option(MFEM_THREAD_SAFE "Enable thread safety" OFF)

# MFEM_DEBUG
if (CMAKE_BUILD_TYPE MATCHES "DEBUG")
  set(MFEM_DEBUG ON)
else()
  set(MFEM_DEBUG OFF)
endif()

# MPI
if (MFEM_USE_MPI)
  find_package(MPI REQUIRED)
  include_directories(${MPI_CXX_INCLUDE_PATH})
  
  # Parallel MFEM depends on Hypre and Metis
  find_package(HYPRE REQUIRED)
  set(MFEM_USE_HYPRE ON)
  include_directories(${HYPRE_INCLUDE_DIRS})
  
  find_package(METIS REQUIRED)
  include_directories(${METIS_INCLUDE_DIRS})
  set(MFEM_USE_METIS ON)
endif()

# SuperLU_dist can only be enabled if parallel
if (MFEM_USE_SUPERLU)
  if (MFEM_USE_MPI)
    find_package(SuperLUDist REQUIRED)
    include_directories(${SuperLUDist_INCLUDE_DIRS})
  else()
    message(FATAL_ERROR "SuperLU_DIST requires that MPI be enabled")
  endif()
endif()

# LAPACK
if(MFEM_USE_LAPACK)
  find_package(BLAS REQUIRED)
  find_package(LAPACK REQUIRED)
endif()

# OpenMP
if (MFEM_USE_OPENMP)
  find_package(OPENMP REQUIRED)
  if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  else()
    message(STATUS "OpenMP NOT FOUND; Content disabled.")
  endif()
endif()

# MESQUITE
if (MFEM_USE_MESQUITE)
  find_package(Mesquite REQUIRED)
  include_directories(${Mesquite_INCLUDE_DIRS})
endif()

# SuiteSparse
if (MFEM_USE_SUITESPARSE)
  find_package(SuiteSparse REQUIRED)
  include_directories(${SuiteSparse_INCLUDE_DIRS})
endif()

# GECKO
if (MFEM_USE_GECKO)
  find_package(Gecko REQUIRED)
  include_directories(${Gecko_INCLUDE_DIRS})
endif()

# GnuTLS
if (MFEM_USE_GNUTLS)
  find_package(GnuTLS REQUIRED)
  include_directories(${GNUTLS_INCLUDE_DIR})
endif()

# NetCDF
if (MFEM_USE_NETCDF)
  find_package(NetCDF REQUIRED)
  include_directories(${NetCDF_INCLUDE_DIRS})
endif()

set(TPL_LIBRARIES "")
if (SuiteSparse_FOUND)
  list(APPEND TPL_LIBRARIES ${SuiteSparse_LIBRARIES})
endif()
if (HYPRE_FOUND)
  list(APPEND TPL_LIBRARIES ${HYPRE_LIBRARIES})
endif()
if (SuperLUDist_FOUND)
  list(APPEND TPL_LIBRARIES ${SuperLUDist_LIBRARIES})
endif()
if (ParMETIS_FOUND)
  list(APPEND TPL_LIBRARIES ${ParMETIS_LIBRARIES})
endif()
if (METIS_FOUND)
  list(APPEND TPL_LIBRARIES ${METIS_LIBRARIES})
endif()
if (LAPACK_FOUND)
  list(APPEND TPL_LIBRARIES ${LAPACK_LIBRARIES})
endif()
if (BLAS_FOUND)
  list(APPEND TPL_LIBRARIES ${BLAS_LIBRARIES})
endif()

# MFEM_TIMER_TYPE
if(NOT MFEM_TIMER_TYPE)
  if (APPLE)
    set(MFEM_TIMER_TYPE 0)
  elseif (UNIX)
    set(MFEM_TIMER_TYPE 2)
  elseif (WIN32)
    # TRB: I don't know what this means, but Mikhail had it there, so
    # it stayed.
    set(MFEM_TIMER_TYPE 3)
  else()
    set(MFEM_TIMER_TYPE 1)
  endif()
endif()

# Windows specific
if (WIN32)
  set(_USE_MATH_DEFINES ON)
else()
  set(_USE_MATH_DEFINES OFF)
endif()

include_directories(${PROJECT_SOURCE_DIR})

# Configuration file
configure_file(
  "${PROJECT_SOURCE_DIR}/config/config.hpp.in_cmake"
  "${PROJECT_SOURCE_DIR}/config/config.hpp")
include_directories(${PROJECT_SOURCE_DIR}/config)

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
target_link_libraries(mfem PUBLIC ${TPL_LIBRARIES})

# The examples and miniapps are separate
add_subdirectory(examples)
#add_subdirectory(miniapps)

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
