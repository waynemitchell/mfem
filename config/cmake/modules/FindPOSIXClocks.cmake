# Find POSIX clocks.
# Defines:
#    POSIXCLOCKS_FOUND
#    POSIXCLOCKS_LIBRARIES

include(MfemCmakeUtilities)
mfem_find_library(POSIXClocks POSIXCLOCKS "rt"
  "Library required by POSIX clocks." POSIXCLOCKS_BUILD
  "
#include <time.h>
int main()
{
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return 0;
}
")
