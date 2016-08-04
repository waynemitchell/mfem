# Sets the following variables
#   - SuperLUDist_FOUND
#   - SuperLUDist_INCLUDE_DIRS
#   - SuperLUDist_LIBRARY_DIRS
#   - SuperLUDist_LIBRARIES
#   - HAVE_SuperLUDist_VERSION_5

# Really, I *want* to depend on ParMETIS, but I guess I can default to
# natural ordering or metis-based ordering, since metis is a required
# dependency.

find_package(ParMETIS REQUIRED)

# Look for superlu_defs.h
find_path(SuperLUDist_INCLUDE_DIRS superlu_defs.h
  HINTS ${SuperLUDist_DIR} $ENV{SuperLUDist_DIR}
  PATH_SUFFIXES include SRC SuperLU
  DOC "Directory where SuperLUDist defs headers live."
  NO_DEFAULT_PATH)
find_path(SuperLUDist_INCLUDE_DIRS slu_ddefs.h)

# Now try to find the library
find_library(SuperLUDist_LIBRARIES
  NAMES superludist superlu_dist superlu_dist_4.3 NAMES_PER_DIR
  HINTS ${SuperLUDist_DIR} $ENV{SuperLUDist_DIR} ${SuperLUDist_LIBRARY_DIRS}
  PATH_SUFFIXES lib SRC
  DOC "Distributed SuperLU library."
  NO_DEFAULT_PATH)
find_library(SuperLUDist_LIBRARIES
  NAMES superludist superlu_dist superlu_dist_4.3 NAMES_PER_DIR)


if(CMAKE_VERSION VERSION_GREATER 2.8.11)
  get_filename_component(SuperLUDist_LIBRARY_DIRS ${SuperLUDist_LIBRARIES} DIRECTORY)
else()
  get_filename_component(SuperLUDist_LIBRARY_DIRS ${SuperLUDist_LIBRARIES} PATH)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(SuperLUDist
  DEFAULT_MSG
  SuperLUDist_LIBRARIES SuperLUDist_INCLUDE_DIRS SuperLUDist_LIBRARY_DIRS)
