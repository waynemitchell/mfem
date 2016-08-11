# Sets the following variables
#   - SuiteSparse_FOUND
#   - SuiteSparse_INCLUDE_DIRS
#   - SuiteSparse_LIBRARY_DIRS
#   - SuiteSparse_LIBRARIES
#
# We need the following libraries:
#   spqr, umfpack, cholmod, amd, colamd, suitesparseconfig, klu, btf
#
# TODO: This should split into separate FindXXX.cmake files so that if
# a crazy person has <libA> at <libA_prefix> and libB at
# <libB_prefix>, we still succeed.

# UMFPack is the most obvious dependency, so start there.
find_path(UMFPACK_INCLUDE_DIRS umfpack.h
  HINTS ${SuiteSparse_DIR} $ENV{SuiteSparse_DIR}
  PATH_SUFFIXES suitesparse include
  NO_DEFAULT_PATH
  DOC "Directory where SuiteSparse headers live.")
find_path(UMFPACK_INCLUDE_DIRS umfpack.h
  PATH_SUFFIXES suitesparse)

find_library(UMFPACK_LIBRARY umfpack
  HINTS ${SuiteSparse_DIR} $ENV{SuiteSparse_DIR}
  PATH_SUFFIXES lib
  NO_DEFAULT_PATH
  DOC "The UMFPack Library.")
find_library(UMFPACK_LIBRARY umfpack)

if(CMAKE_VERSION VERSION_GREATER 2.8.11)
  get_filename_component(UMFPACK_LIB_DIR ${UMFPACK_LIBRARY} DIRECTORY)
else()
  get_filename_component(UMFPACK_LIB_DIR ${UMFPACK_LIBRARY} PATH)
endif()

# Now using umfpack as a guide, find the rest:
find_path(AMD_INCLUDE_DIRS amd.h
  HINTS ${UMFPACK_INCLUDE_DIRS} ${SuiteSparse_DIR} $ENV{SuiteSparse_DIR}
  PATH_SUFFIXES suitesparse include
  NO_DEFAULT_PATH
  DOC "Directory where SuiteSparse headers live.")
find_path(AMD_INCLUDE_DIRS amd.h
  PATH_SUFFIXES suitesparse)

find_library(AMD_LIBRARY amd
  HINTS ${UMFPACK_LIB_DIR} ${SuiteSparse_DIR} $ENV{SuiteSparse_DIR}
  PATH_SUFFIXES lib
  NO_DEFAULT_PATH
  DOC "The Amd Library.")
find_library(AMD_LIBRARY amd)

if(CMAKE_VERSION VERSION_GREATER 2.8.11)
  get_filename_component(AMD_LIB_DIR ${AMD_LIBRARY} DIRECTORY)
else()
  get_filename_component(AMD_LIB_DIR ${AMD_LIBRARY} PATH)
endif()

# CHOLMOD
find_path(CHOLMOD_INCLUDE_DIRS cholmod.h
  HINTS ${UMFPACK_INCLUDE_DIRS} ${AMD_INCLUDE_DIRS} ${SuiteSparse_DIR} $ENV{SuiteSparse_DIR}
  PATH_SUFFIXES suitesparse include
  NO_DEFAULT_PATH
  DOC "Directory where SuiteSparse headers live.")
find_path(CHOLMOD_INCLUDE_DIRS cholmod.h
  PATH_SUFFIXES suitesparse)

find_library(CHOLMOD_LIBRARY cholmod
  HINTS ${UMFPACK_LIB_DIR} ${AMD_LIB_DIR} ${SuiteSparse_DIR} $ENV{SuiteSparse_DIR}
  PATH_SUFFIXES lib
  NO_DEFAULT_PATH
  DOC "The Cholmod Library.")
find_library(CHOLMOD_LIBRARY cholmod)

if(CMAKE_VERSION VERSION_GREATER 2.8.11)
  get_filename_component(CHOLMOD_LIB_DIR ${CHOLMOD_LIBRARY} DIRECTORY)
else()
  get_filename_component(CHOLMOD_LIB_DIR ${CHOLMOD_LIBRARY} PATH)
endif()

# COLAMD
find_path(COLAMD_INCLUDE_DIRS colamd.h
  HINTS
  ${UMFPACK_INCLUDE_DIRS} ${AMD_INCLUDE_DIRS} ${CHOLMOD_INCLUDE_DIRS}
  ${SuiteSparse_DIR} $ENV{SuiteSparse_DIR}
  PATH_SUFFIXES suitesparse include
  NO_DEFAULT_PATH
  DOC "Directory where SuiteSparse headers live.")
find_path(COLAMD_INCLUDE_DIRS colamd.h
  PATH_SUFFIXES suitesparse)

find_library(COLAMD_LIBRARY colamd
  HINTS
    ${UMFPACK_LIB_DIR} ${AMD_LIB_DIR} ${CHOLMOD_LIB_DIR}
    ${SuiteSparse_DIR} $ENV{SuiteSparse_DIR}
  PATH_SUFFIXES lib
  NO_DEFAULT_PATH
  DOC "The Colamd Library.")
find_library(COLAMD_LIBRARY colamd)

if(CMAKE_VERSION VERSION_GREATER 2.8.11)
  get_filename_component(COLAMD_LIB_DIR ${COLAMD_LIBRARY} DIRECTORY)
else()
  get_filename_component(COLAMD_LIB_DIR ${COLAMD_LIBRARY} PATH)
endif()

# SuiteSparseConfig
find_path(CONFIG_INCLUDE_DIRS SuiteSparse_config.h
  HINTS
  ${UMFPACK_INCLUDE_DIRS} ${AMD_INCLUDE_DIRS}
  ${COLAMD_INCLUDE_DIRS} ${CHOLMOD_INCLUDE_DIRS}
  ${SuiteSparse_DIR} $ENV{SuiteSparse_DIR}
  PATH_SUFFIXES suitesparse include
  NO_DEFAULT_PATH
  DOC "Directory where SuiteSparse headers live.")
find_path(CONFIG_INCLUDE_DIRS SuiteSparse_config.h
  PATH_SUFFIXES suitesparse)

find_library(CONFIG_LIBRARY suitesparseconfig
  HINTS
  ${UMFPACK_LIB_DIR} ${AMD_LIB_DIR}
  ${COLAMD_LIB_DIR} ${CHOLMOD_LIB_DIR}
  ${SuiteSparse_DIR} $ENV{SuiteSparse_DIR}
  PATH_SUFFIXES lib
  NO_DEFAULT_PATH  
  DOC "The SuiteSparseConfig Library.")
find_library(CONFIG_LIBRARY suitesparseconfig)

if(CMAKE_VERSION VERSION_GREATER 2.8.11)
  get_filename_component(CONFIG_LIB_DIR ${CONFIG_LIBRARY} DIRECTORY)
else()
  get_filename_component(CONFIG_LIB_DIR ${CONFIG_LIBRARY} PATH)
endif()

find_library(KLU_LIBRARY klu
  HINTS
  ${UMFPACK_LIB_DIR} ${AMD_LIB_DIR}
  ${COLAMD_LIB_DIR} ${CHOLMOD_LIB_DIR}
  ${SuiteSparse_DIR} $ENV{SuiteSparse_DIR}
  PATH_SUFFIXES lib
  NO_DEFAULT_PATH
  DOC "The KLU Library.")
find_library(KLU_LIBRARY klu)

if(CMAKE_VERSION VERSION_GREATER 2.8.11)
  get_filename_component(KLU_LIB_DIR ${KLU_LIBRARY} DIRECTORY)
else()
  get_filename_component(KLU_LIB_DIR ${KLU_LIBRARY} PATH)
endif()

find_library(BTF_LIBRARY btf
  HINTS
  ${UMFPACK_LIB_DIR} ${AMD_LIB_DIR}
  ${COLAMD_LIB_DIR} ${CHOLMOD_LIB_DIR}
  ${SuiteSparse_DIR} $ENV{SuiteSparse_DIR}
  PATH_SUFFIXES lib
  NO_DEFAULT_PATH
  DOC "The BTF Library.")
find_library(BTF_LIBRARY btf)

if(CMAKE_VERSION VERSION_GREATER 2.8.11)
  get_filename_component(BTF_LIB_DIR ${BTF_LIBRARY} DIRECTORY)
else()
  get_filename_component(BTF_LIB_DIR ${BTF_LIBRARY} PATH)
endif()

set(SuiteSparse_LIBRARIES
  ${UMFPACK_LIBRARY} ${KLU_LIBRARY} ${BTF_LIBRARY} ${AMD_LIBRARY}
  ${CHOLMOD_LIBRARY} ${COLAMD_LIBRARY} ${CONFIG_LIBRARY})

set(SuiteSparse_LIBRARY_DIRS
  ${UMFPACK_LIB_DIR} ${AMD_LIB_DIR} ${CHOLMOD_LIB_DIR} ${COLAMD_LIB_DIR}
  ${CONFIG_LIB_DIR} ${KLU_LIB_DIR} ${BTF_LIB_DIR})

set(SuiteSparse_INCLUDE_DIRS
  ${UMFPACK_INCLUDE_DIRS} ${AMD_INCLUDE_DIRS} ${CHOLMOD_INCLUDE_DIRS}
  ${COLAMD_INCLUDE_DIRS} ${CONFIG_INCLUDE_DIRS})

LIST(REMOVE_DUPLICATES SuiteSparse_LIBRARIES)
LIST(REMOVE_DUPLICATES SuiteSparse_INCLUDE_DIRS)
LIST(REMOVE_DUPLICATES SuiteSparse_LIBRARY_DIRS)

# This handles "REQUIRED" etc keywords
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(SuiteSparse
  "SuiteSparse could not be found. Be sure to set SuiteSparse_DIR."
  SuiteSparse_LIBRARIES SuiteSparse_INCLUDE_DIRS SuiteSparse_LIBRARY_DIRS)
