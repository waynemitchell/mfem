# Sets the following variables
#   - ParMETIS_FOUND
#   - ParMETIS_INCLUDE_DIRS
#   - ParMETIS_LIBRARY_DIRS
#   - ParMETIS_LIBRARIES
#
# We need the following libraries:
#   parmetis

find_package(METIS QUIET)

if (NOT METIS_VERSION_5)
  message(FATAL_ERROR " *** ParMETIS requires METIS v5!")
endif()

# Find the header
find_path(ParMETIS_INCLUDE_DIRS parmetis.h
  HINTS ${ParMETIS_DIR} $ENV{ParMETIS_DIR} ${METIS_DIR} $ENV{METIS_DIR}
  PATH_SUFFIXES include
  NO_DEFAULT_PATH
  DOC "Directory where ParMETIS headers live.")
find_path(ParMETIS_INCLUDE_DIRS parmetis.h
  HINTS ${METIS_INCLUDE_DIRS}
  NO_DEFAULT_PATH)
find_path(ParMETIS_INCLUDE_DIRS parmetis.h)

find_library(ParMETIS_LIBRARIES parmetis
  HINTS ${ParMETIS_DIR} $ENV{ParMETIS_DIR} ${METIS_DIR} $ENV{METIS_DIR}
  PATH_SUFFIXES lib
  NO_DEFAULT_PATH
  DOC "The ParMETIS Library.")
find_library(ParMETIS_LIBRARIES parmetis
  HINTS ${METIS_LIBRARY_DIRS}
  NO_DEFAULT_PATH)
find_library(ParMETIS_LIBRARIES parmetis)

if(CMAKE_VERSION VERSION_GREATER 2.8.11)
  get_filename_component(ParMETIS_LIBRARY_DIRS ${ParMETIS_LIBRARIES} DIRECTORY)
else()
  get_filename_component(ParMETIS_LIBRARY_DIRS ${ParMETIS_LIBRARIES} PATH)
endif()

# This handles "REQUIRED" etc keywords
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ParMETIS
  " *** ParMETIS library not found. Please set ParMETIS_DIR."
  ParMETIS_LIBRARIES ParMETIS_INCLUDE_DIRS ParMETIS_LIBRARY_DIRS)
# For older cmake versions
set(ParMETIS_FOUND ${PARMETIS_FOUND})
