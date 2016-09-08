# Defines the following variables:
#   - GECKO_FOUND
#   - GECKO_LIBRARIES
#   - GECKO_INCLUDE_DIRS
#   - GECKO_LIBRARY_DIRS

find_path(GECKO_INCLUDE_DIRS graph.h
  HINTS ${GECKO_DIR} $ENV{GECKO_DIR}
  PATH_SUFFIXES include inc
  NO_DEFAULT_PATH
  DOC "Directory with Gecko graph.h header.")
find_path(GECKO_INCLUDE_DIRS graph.h)

find_library(GECKO_LIBRARIES gecko
  HINTS ${GECKO_DIR}/lib $ENV{GECKO_DIR}
  NO_DEFAULT_PATH
  DOC "The Gecko library.")
find_library(GECKO_LIBRARIES gecko)

if(CMAKE_VERSION VERSION_GREATER 2.8.11)
  get_filename_component(GECKO_LIBRARY_DIRS ${GECKO_LIBRARIES} DIRECTORY)
else()
  get_filename_component(GECKO_LIBRARY_DIRS ${GECKO_LIBRARIES} PATH)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Gecko
  " *** Gecko library not found. Please set GECKO_DIR."
  GECKO_LIBRARIES GECKO_INCLUDE_DIRS GECKO_LIBRARY_DIRS)

