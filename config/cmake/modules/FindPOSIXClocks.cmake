# Find POSIX clocks.
# Defines:
#    POSIXCLOCKS_FOUND
#    POSIXCLOCKS_LIBRARIES

include(CheckSymbolExists)
if (NOT DEFINED POSIXCLOCKS_FOUND)
  set(POSIXCLOCKS_LIBRARIES "")
  # First check without -lrt
  check_symbol_exists(clock_gettime "time.h" POSIXCLOCKS_FOUND)
  if (NOT POSIXCLOCKS_FOUND)
    # Then check with -lrt
    set(CMAKE_REQUIRED_LIBRARIES "rt")
    check_symbol_exists(clock_gettime "time.h" POSIXCLOCKS_FOUND)
    if (POSIXCLOCKS_FOUND)
      set(POSIXCLOCKS_LIBRARIES "rt")
    endif (POSIXCLOCKS_FOUND)
  endif()
  set(POSIXCLOCKS_LIBRARIES ${POSIXCLOCKS_LIBRARIES} CACHE STRING
      "Library needed by POSIX clocks.")
endif()

# Handle REQUIRED etc
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(POSIXClocks
  " *** POSIX clocks not found." POSIXCLOCKS_FOUND)
