# Sets the following variables
#   - SuperLUDist_FOUND
#   - SuperLUDist_INCLUDE_DIRS
#   - SuperLUDist_LIBRARY_DIRS
#   - SuperLUDist_LIBRARIES

find_package(OpenMP)
find_package(BLAS REQUIRED)
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

# Verify a sufficient version of SuperLU_DIST (i.e. 5.*.*)
include(CheckCXXSourceCompiles)
function(check_superlu_dist_version VAR)
  set(TEST_SOURCE
    "
#include <superlu_defs.h>

int main()
{
  superlu_dist_options_t opts;
  return 0;
}
")
  set(CMAKE_REQUIRED_INCLUDES ${MPI_CXX_INCLUDE_PATH} ${SuperLUDist_INCLUDE_DIRS})
  set(CMAKE_REQUIRED_LIBRARIES ${MPI_CXX_LIBRARIES})
  check_cxx_source_compiles("${TEST_SOURCE}" ${VAR})
endfunction()

check_superlu_dist_version(SuperLUDist_VERSION_OK)

if(CMAKE_VERSION VERSION_GREATER 2.8.11)
  get_filename_component(SuperLUDist_LIBRARY_DIRS ${SuperLUDist_LIBRARIES} DIRECTORY)
else()
  get_filename_component(SuperLUDist_LIBRARY_DIRS ${SuperLUDist_LIBRARIES} PATH)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(SuperLUDist
  " *** SuperLU_DIST library not found. Please set SuperLUDist_DIR."
  SuperLUDist_LIBRARIES SuperLUDist_INCLUDE_DIRS SuperLUDist_LIBRARY_DIRS SuperLUDist_VERSION_OK)
# For older cmake versions
set(SuperLUDist_FOUND ${SUPERLUDIST_FOUND})
