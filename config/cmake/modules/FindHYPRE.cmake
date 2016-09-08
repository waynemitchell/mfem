# Defines the following variables:
#   - HYPRE_FOUND
#   - HYPRE_LIBRARIES
#   - HYPRE_INCLUDE_DIRS
#   - HYPRE_LIBRARY_DIRS

find_path(HYPRE_INCLUDE_DIRS HYPRE.h
  HINTS ${HYPRE_DIR} $ENV{HYPRE_DIR}
  PATH_SUFFIXES include
  NO_DEFAULT_PATH
  DOC "Directory with HYPRE header.")
find_path(HYPRE_INCLUDE_DIRS HYPRE.h)

find_library(HYPRE_LIBRARIES HYPRE
  HINTS ${HYPRE_DIR} $ENV{HYPRE_DIR} ${HYPRE_LIBRARY_DIRS}
  PATH_SUFFIXES lib
  NO_DEFAULT_PATH
  DOC "The HYPRE library.")
find_library(HYPRE_LIBRARIES HYPRE)

if(CMAKE_VERSION VERSION_GREATER 2.8.11)
  get_filename_component(HYPRE_LIBRARY_DIRS ${HYPRE_LIBRARIES} DIRECTORY)
else()
  get_filename_component(HYPRE_LIBRARY_DIRS ${HYPRE_LIBRARIES} PATH)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(HYPRE
  " *** HYPRE library not found. Please set HYPRE_DIR, e.g. ~/hypre-2.10.0b/src/hypre"
  HYPRE_LIBRARIES HYPRE_INCLUDE_DIRS HYPRE_LIBRARY_DIRS)

