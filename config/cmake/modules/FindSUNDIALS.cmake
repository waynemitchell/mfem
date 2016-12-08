# Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at the
# Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights reserved.
# See file COPYRIGHT for details.
#
# This file is part of the MFEM library. For more information and source code
# availability see http://mfem.org.
#
# MFEM is free software; you can redistribute it and/or modify it under the
# terms of the GNU Lesser General Public License (as published by the Free
# Software Foundation) version 2.1 dated February 1999.

# Defines the following variables:
#   - SUNDIALS_FOUND
#   - SUNDIALS_LIBRARIES
#   - SUNDIALS_INCLUDE_DIRS

include(MfemCmakeUtilities)

# Using module provided by Slaven

# Find SUNDIALS installation path
find_path (SUNDIALS_BASE_DIR include/sundials/sundials_config.h
  HINTS /usr/local/sundials ${SUNDIALS_DIR})
message (STATUS "Found SUNDIALS in ${SUNDIALS_DIR}")

# Required SUNDIALS headers
set(SUNDIALS_HEADERS
  nvector/nvector_serial.h
  nvector/nvector_parallel.h
  nvector/nvector_parhyp.h
  cvode/cvode.h
  arkode/arkode.h
  kinsol/kinsol.h
  )

# SUNDIALS modules needed for the build
# The order matters in case of static build!
set(SUNDIALS_MODULES 
  sundials_kinsol 
  sundials_arkode 
  sundials_cvode 
  sundials_nvecserial 
  sundials_nvecparallel 
  sundials_nvecparhyp 
)

# Search for each header and make sure all are found
set(SUNDIALS_INCLUDE_DIRS)
foreach (hdr ${SUNDIALS_HEADERS})
  get_filename_component(_TMP_NAME ${hdr} NAME_WE)
  
  find_path(_TMP_${_TMP_NAME}_INCLUDE_DIR ${hdr}
    HINTS ${SUNDIALS_DIR} ${SUNDIALS_BASE_DIR}
    PATH_SUFFIXES include
    NO_DEFAULT_PATH)
  find_path(_TMP_${_TMP_NAME}_INCLUDE_DIR ${hdr})

  if (_TMP_${_TMP_NAME}_INCLUDE_DIR)
    list(APPEND SUNDIALS_INCLUDE_DIRS ${_TMP_${_TMP_NAME}_INCLUDE_DIR})
  else()
    message(FATAL_ERROR "Failed to find required SUNDIALS header: ${hdr}")
  endif (_TMP_${_TMP_NAME}_INCLUDE_DIR)
endforeach (hdr ${SUNDIALS_HEADERS})

list(REMOVE_DUPLICATES SUNDIALS_INCLUDE_DIRS)

message (STATUS "Found SUNDIALS headers in ${SUNDIALS_INCLUDE_DIRS}")

# Find each SUNDIALS module and add it to the list of libraries to link
set(SUNDIALS_LIBRARIES)
foreach(mod ${SUNDIALS_MODULES})
  message(STATUS "SUNDIALS: Looking for required component: ${mod}")
  
  find_library(SUNDIALS_${mod}_LIBRARY ${mod}
    HINTS ${SUNDIALS_DIR} ${SUNDIALS_LIBRARY_DIR} ${SUNDIALS_BASE_DIR}
    PATH_SUFFIXES lib
    NO_DEFAULT_PATH
    )
  find_library(SUNDIALS_${mod}_LIBRARY ${mod})
  
  if(SUNDIALS_${mod}_LIBRARY)
    list(APPEND SUNDIALS_LIBRARIES ${SUNDIALS_${mod}_LIBRARY})
  else()
    message(FATAL_ERROR "Failed to find require library: ${mod}.\nPlease set SUNDIALS_LIBRARY_DIR and rerun CMake.")
  endif()
endforeach()
message (STATUS "Following SUNDIALS libraries found: ${SUNDIALS_LIBRARIES}")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(SUNDIALS
  " *** SUNDIALS not found. Please set SUNDIALS_DIR."
  SUNDIALS_INCLUDE_DIRS SUNDIALS_LIBRARIES)
