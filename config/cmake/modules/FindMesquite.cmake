# Defines the following variables:
#   - MESQUITE_FOUND
#   - MESQUITE_LIBRARIES
#   - MESQUITE_INCLUDE_DIRS
#   - MESQUITE_LIBRARY_DIRS

find_path(MESQUITE_INCLUDE_DIRS Mesquite_all_headers.hpp
  HINTS ${MESQUITE_DIR} $ENV{MESQUITE_DIR}
  PATH_SUFFIXES include
  NO_DEFAULT_PATH
  DOC "Directory with Mesquite master header.")
find_path(MESQUITE_INCLUDE_DIRS Mesquite_all_headers.hpp)

find_library(MESQUITE_LIBRARIES mesquite
  HINTS ${MESQUITE_DIR}/lib $ENV{MESQUITE_DIR}
  NO_DEFAULT_PATH
  DOC "The Mesquite library.")
find_library(MESQUITE_LIBRARIES mesquite)

if(CMAKE_VERSION VERSION_GREATER 2.8.11)
  get_filename_component(MESQUITE_LIBRARY_DIRS ${MESQUITE_LIBRARIES} DIRECTORY)
else()
  get_filename_component(MESQUITE_LIBRARY_DIRS ${MESQUITE_LIBRARIES} PATH)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MESQUITE
  DEFAULT_MSG
  MESQUITE_LIBRARIES MESQUITE_INCLUDE_DIRS MESQUITE_LIBRARY_DIRS)

