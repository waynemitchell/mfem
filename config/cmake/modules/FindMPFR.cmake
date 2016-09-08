# Find MPFR.
# Defines the following variables:
#   - MPFR_FOUND
#   - MPFR_LIBRARIES
#   - MPFR_INCLUDE_DIRS (may be empty)

if (MPFR_LIBRARIES)
  set(MPFR_FOUND TRUE)
  return()
endif()

include(CheckCXXSourceRuns)
function(check_mpfr VAR)
  set(TEST_SOURCE
    "
#include <mpfr.h>
int main()
{
  mpfr_t one;
  mpfr_init2(one, 128);
  mpfr_set_si(one, 1, GMP_RNDN);
  mpfr_clear(one);
  return 0;
}
")
  set(CMAKE_REQUIRED_INCLUDES ${MPFR_TEST_INCLUDE_DIRS})
  set(CMAKE_REQUIRED_LIBRARIES ${MPFR_TEST_LIBRARIES})
  set(CMAKE_REQUIRED_QUIET TRUE)
  check_cxx_source_runs("${TEST_SOURCE}" ${VAR})
endfunction()

set(MPFR_TEST_INCLUDE_DIRS "")
set(MPFR_TEST_LIBRARIES "mpfr")
check_mpfr(MPFR_STANDARD_LOCATION)

if (MPFR_STANDARD_LOCATION)
  set(MPFR_INCLUDE_DIRS ${MPFR_TEST_INCLUDE_DIRS} CACHE STRING
      "Directory with MPFR header.")
  set(MPFR_LIBRARIES ${MPFR_TEST_LIBRARIES} CACHE STRING "The MPFR librarry.")
else (MPFR_STANDARD_LOCATION)
  find_path(MPFR_INCLUDE_DIRS mpfr.h
    HINTS ${MPFR_DIR} $ENV{MPFR_DIR}
    PATH_SUFFIXES include
    DOC "Directory with MPFR header.")
  find_library(MPFR_LIBRARIES mpfr
    HINTS ${MPFR_DIR} $ENV{MPFR_DIR}
    PATH_SUFFIXES lib
    DOC "The MPFR librarry.")
endif (MPFR_STANDARD_LOCATION)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MPFR
  " *** MPFR library not found. Please set MPFR_DIR."
  MPFR_LIBRARIES)
