# Sets the following variables
#   - SuiteSparse_FOUND
#   - SuiteSparse_INCLUDE_DIRS
#   - SuiteSparse_LIBRARY_DIRS
#   - SuiteSparse_LIBRARIES
#
# We need the following libraries:
#   umfpack, cholmod, amd, camd, colamd, ccolamd, suitesparseconfig, klu, btf
#
# TODO: This should split into separate FindXXX.cmake files so that if
# a crazy person has <libA> at <libA_prefix> and libB at
# <libB_prefix>, we still succeed.

find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
find_package(METIS REQUIRED)
# librt?

if (METIS_VERSION_5)
  message(FATAL_ERROR " *** SuiteSparse requires METIS v4!")
endif()

# Quick return
if (SuiteSparse_LIBRARIES AND SuiteSparse_INCLUDE_DIRS AND
    SuiteSparse_LIBRARY_DIRS)
  set(SuiteSparse_FOUND TRUE)
  set(SUITESPARSE_FOUND TRUE)
  return()
endif()

# UMFPack is the most obvious dependency, so start there.
find_path(SS_UMFPACK_INCLUDE_DIRS umfpack.h
  HINTS ${SuiteSparse_DIR} $ENV{SuiteSparse_DIR}
  PATH_SUFFIXES suitesparse include
  NO_DEFAULT_PATH
  DOC "Directory where SuiteSparse headers live.")
find_path(SS_UMFPACK_INCLUDE_DIRS umfpack.h
  PATH_SUFFIXES suitesparse)

find_library(SS_UMFPACK_LIBRARY umfpack
  HINTS ${SuiteSparse_DIR} $ENV{SuiteSparse_DIR}
  PATH_SUFFIXES lib
  NO_DEFAULT_PATH
  DOC "The SuiteSparse/UMFPack library.")
find_library(SS_UMFPACK_LIBRARY umfpack)

if(CMAKE_VERSION VERSION_GREATER 2.8.11)
  get_filename_component(SS_UMFPACK_LIB_DIR ${SS_UMFPACK_LIBRARY} DIRECTORY)
else()
  get_filename_component(SS_UMFPACK_LIB_DIR ${SS_UMFPACK_LIBRARY} PATH)
endif()

# Now using umfpack as a guide, find the rest:
# AMD
find_path(SS_AMD_INCLUDE_DIRS amd.h
  HINTS ${SS_UMFPACK_INCLUDE_DIRS} ${SuiteSparse_DIR} $ENV{SuiteSparse_DIR}
  PATH_SUFFIXES suitesparse include
  NO_DEFAULT_PATH
  DOC "Directory where SuiteSparse headers live.")
find_path(SS_AMD_INCLUDE_DIRS amd.h
  PATH_SUFFIXES suitesparse)

find_library(SS_AMD_LIBRARY amd
  HINTS ${SS_UMFPACK_LIB_DIR} ${SuiteSparse_DIR} $ENV{SuiteSparse_DIR}
  PATH_SUFFIXES lib
  NO_DEFAULT_PATH
  DOC "The SuiteSparse/AMD Library.")
find_library(SS_AMD_LIBRARY amd)

if(CMAKE_VERSION VERSION_GREATER 2.8.11)
  get_filename_component(SS_AMD_LIB_DIR ${SS_AMD_LIBRARY} DIRECTORY)
else()
  get_filename_component(SS_AMD_LIB_DIR ${SS_AMD_LIBRARY} PATH)
endif()

# CHOLMOD
find_path(SS_CHOLMOD_INCLUDE_DIRS cholmod.h
  HINTS ${SS_UMFPACK_INCLUDE_DIRS} ${SS_AMD_INCLUDE_DIRS} ${SuiteSparse_DIR}
  $ENV{SuiteSparse_DIR}
  PATH_SUFFIXES suitesparse include
  NO_DEFAULT_PATH
  DOC "Directory where SuiteSparse headers live.")
find_path(SS_CHOLMOD_INCLUDE_DIRS cholmod.h
  PATH_SUFFIXES suitesparse)

find_library(SS_CHOLMOD_LIBRARY cholmod
  HINTS ${SS_UMFPACK_LIB_DIR} ${SS_AMD_LIB_DIR} ${SuiteSparse_DIR}
  $ENV{SuiteSparse_DIR}
  PATH_SUFFIXES lib
  NO_DEFAULT_PATH
  DOC "The SuiteSparse/Cholmod Library.")
find_library(SS_CHOLMOD_LIBRARY cholmod)

if(CMAKE_VERSION VERSION_GREATER 2.8.11)
  get_filename_component(SS_CHOLMOD_LIB_DIR ${SS_CHOLMOD_LIBRARY} DIRECTORY)
else()
  get_filename_component(SS_CHOLMOD_LIB_DIR ${SS_CHOLMOD_LIBRARY} PATH)
endif()

# COLAMD
find_path(SS_COLAMD_INCLUDE_DIRS colamd.h
  HINTS
  ${SS_UMFPACK_INCLUDE_DIRS} ${SS_AMD_INCLUDE_DIRS} ${SS_CHOLMOD_INCLUDE_DIRS}
  ${SuiteSparse_DIR} $ENV{SuiteSparse_DIR}
  PATH_SUFFIXES suitesparse include
  NO_DEFAULT_PATH
  DOC "Directory where SuiteSparse headers live.")
find_path(SS_COLAMD_INCLUDE_DIRS colamd.h
  PATH_SUFFIXES suitesparse)

find_library(SS_COLAMD_LIBRARY colamd
  HINTS
    ${SS_UMFPACK_LIB_DIR} ${SS_AMD_LIB_DIR} ${SS_CHOLMOD_LIB_DIR}
    ${SuiteSparse_DIR} $ENV{SuiteSparse_DIR}
  PATH_SUFFIXES lib
  NO_DEFAULT_PATH
  DOC "The SuiteSparse/Colamd Library.")
find_library(SS_COLAMD_LIBRARY colamd)

if(CMAKE_VERSION VERSION_GREATER 2.8.11)
  get_filename_component(SS_COLAMD_LIB_DIR ${SS_COLAMD_LIBRARY} DIRECTORY)
else()
  get_filename_component(SS_COLAMD_LIB_DIR ${SS_COLAMD_LIBRARY} PATH)
endif()

# SuiteSparseConfig
find_path(SS_CONFIG_INCLUDE_DIRS SuiteSparse_config.h
  HINTS
  ${SS_UMFPACK_INCLUDE_DIRS} ${SS_AMD_INCLUDE_DIRS}
  ${SS_COLAMD_INCLUDE_DIRS} ${SS_CHOLMOD_INCLUDE_DIRS}
  ${SuiteSparse_DIR} $ENV{SuiteSparse_DIR}
  PATH_SUFFIXES suitesparse include
  NO_DEFAULT_PATH
  DOC "Directory where SuiteSparse headers live.")
find_path(SS_CONFIG_INCLUDE_DIRS SuiteSparse_config.h
  PATH_SUFFIXES suitesparse)

find_library(SS_CONFIG_LIBRARY suitesparseconfig
  HINTS
  ${SS_UMFPACK_LIB_DIR} ${SS_AMD_LIB_DIR}
  ${SS_COLAMD_LIB_DIR} ${SS_CHOLMOD_LIB_DIR}
  ${SuiteSparse_DIR} $ENV{SuiteSparse_DIR}
  PATH_SUFFIXES lib
  NO_DEFAULT_PATH  
  DOC "The SuiteSparseConfig Library.")
find_library(SS_CONFIG_LIBRARY suitesparseconfig)

if(CMAKE_VERSION VERSION_GREATER 2.8.11)
  get_filename_component(SS_CONFIG_LIB_DIR ${SS_CONFIG_LIBRARY} DIRECTORY)
else()
  get_filename_component(SS_CONFIG_LIB_DIR ${SS_CONFIG_LIBRARY} PATH)
endif()

# KLU
find_library(SS_KLU_LIBRARY klu
  HINTS
  ${SS_UMFPACK_LIB_DIR} ${SS_AMD_LIB_DIR}
  ${SS_COLAMD_LIB_DIR} ${SS_CHOLMOD_LIB_DIR}
  ${SuiteSparse_DIR} $ENV{SuiteSparse_DIR}
  PATH_SUFFIXES lib
  NO_DEFAULT_PATH
  DOC "The SuiteSparse/KLU Library.")
find_library(SS_KLU_LIBRARY klu)

if(CMAKE_VERSION VERSION_GREATER 2.8.11)
  get_filename_component(SS_KLU_LIB_DIR ${SS_KLU_LIBRARY} DIRECTORY)
else()
  get_filename_component(SS_KLU_LIB_DIR ${SS_KLU_LIBRARY} PATH)
endif()

# BTF
find_library(SS_BTF_LIBRARY btf
  HINTS
  ${SS_UMFPACK_LIB_DIR} ${SS_AMD_LIB_DIR}
  ${SS_COLAMD_LIB_DIR} ${SS_CHOLMOD_LIB_DIR}
  ${SuiteSparse_DIR} $ENV{SuiteSparse_DIR}
  PATH_SUFFIXES lib
  NO_DEFAULT_PATH
  DOC "The SuiteSparse/BTF Library.")
find_library(SS_BTF_LIBRARY btf)

if(CMAKE_VERSION VERSION_GREATER 2.8.11)
  get_filename_component(SS_BTF_LIB_DIR ${SS_BTF_LIBRARY} DIRECTORY)
else()
  get_filename_component(SS_BTF_LIB_DIR ${SS_BTF_LIBRARY} PATH)
endif()

# CAMD
find_library(SS_CAMD_LIBRARY camd
  HINTS
  ${SS_UMFPACK_LIB_DIR} ${SuiteSparse_DIR} $ENV{SuiteSparse_DIR}
  PATH_SUFFIXES lib
  NO_DEFAULT_PATH
  DOC "The SuiteSparse/CAMD library.")
find_library(SS_CAMD_LIBRARY camd)

if(CMAKE_VERSION VERSION_GREATER 2.8.11)
  get_filename_component(SS_CAMD_LIB_DIR ${SS_CAMD_LIBRARY} DIRECTORY)
else()
  get_filename_component(SS_CAMD_LIB_DIR ${SS_CAMD_LIBRARY} PATH)
endif()

# CCOLAMD
find_library(SS_CCOLAMD_LIBRARY ccolamd
  HINTS
  ${SS_UMFPACK_LIB_DIR} ${SuiteSparse_DIR} $ENV{SuiteSparse_DIR}
  PATH_SUFFIXES lib
  NO_DEFAULT_PATH
  DOC "The SuiteSparse/CCOLAMD library.")
find_library(SS_CCOLAMD_LIBRARY ccolamd)

if(CMAKE_VERSION VERSION_GREATER 2.8.11)
  get_filename_component(SS_CCOLAMD_LIB_DIR ${SS_CCOLAMD_LIBRARY} DIRECTORY)
else()
  get_filename_component(SS_CCOLAMD_LIB_DIR ${SS_CCOLAMD_LIBRARY} PATH)
endif()

if (SS_UMFPACK_LIBRARY AND SS_KLU_LIBRARY AND SS_BTF_LIBRARY AND SS_AMD_LIBRARY
    AND SS_CAMD_LIBRARY AND SS_CHOLMOD_LIBRARY AND SS_COLAMD_LIBRARY AND
    SS_CCOLAMD_LIBRARY AND SS_CONFIG_LIBRARY)

  # Sub-libraries, in order of dependence:
  set(SuiteSparse_LIBRARIES
    ${SS_KLU_LIBRARY} ${SS_BTF_LIBRARY} ${SS_UMFPACK_LIBRARY}
    ${SS_CHOLMOD_LIBRARY} ${SS_COLAMD_LIBRARY} ${SS_AMD_LIBRARY}
    ${SS_CAMD_LIBRARY} ${SS_CCOLAMD_LIBRARY} ${SS_CONFIG_LIBRARY})
  set(SuiteSparse_LIBRARY_DIRS
    ${SS_UMFPACK_LIB_DIR} ${SS_AMD_LIB_DIR} ${SS_CAMD_LIB_DIR}
    ${SS_CHOLMOD_LIB_DIR} ${SS_COLAMD_LIB_DIR} ${SS_CCOLAMD_LIB_DIR}
    ${SS_CONFIG_LIB_DIR} ${SS_KLU_LIB_DIR} ${SS_BTF_LIB_DIR})
  set(SuiteSparse_INCLUDE_DIRS
    ${SS_UMFPACK_INCLUDE_DIRS} ${SS_AMD_INCLUDE_DIRS} ${SS_CHOLMOD_INCLUDE_DIRS}
    ${SS_COLAMD_INCLUDE_DIRS} ${SS_CONFIG_INCLUDE_DIRS})

  LIST(REMOVE_DUPLICATES SuiteSparse_LIBRARIES)
  LIST(REMOVE_DUPLICATES SuiteSparse_INCLUDE_DIRS)
  LIST(REMOVE_DUPLICATES SuiteSparse_LIBRARY_DIRS)

  # string(CONCAT SuiteSparse_LIBRARIES
  #        "-L${SS_UMFPACK_LIB_DIR} -lklu -lbtf -lumfpack -lcholmod -lcolamd "
  #        "-lamd -lcamd -lccolamd -lsuitesparseconfig")

  set(SuiteSparse_LIBRARIES ${SuiteSparse_LIBRARIES} CACHE STRING
      "List of all SuiteSparse libraries.")
  set(SuiteSparse_LIBRARY_DIRS ${SuiteSparse_LIBRARY_DIRS} CACHE PATH
      "Path to the SuiteSparse libraries.")
  set(SuiteSparse_INCLUDE_DIRS ${SuiteSparse_INCLUDE_DIRS} CACHE PATH
      "Path to the SuiteSparse headers.")

endif()

# This handles "REQUIRED" etc keywords
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(SuiteSparse
  "SuiteSparse could not be found. Be sure to set SuiteSparse_DIR."
  SuiteSparse_LIBRARIES SuiteSparse_INCLUDE_DIRS SuiteSparse_LIBRARY_DIRS)
# For older cmake versions
set(SuiteSparse_FOUND ${SUITESPARSE_FOUND})
