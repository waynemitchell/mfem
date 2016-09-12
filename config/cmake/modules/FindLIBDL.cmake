# Find libdl.
# Defines:
#    LIBDL_FOUND
#    LIBDL_LIBRARIES (if needed)

include(MfemCmakeUtilities)
mfem_find_library(LIBDL LIBDL "dl" "The dynamic library." LIBDL_BUILD
  "
#define _GNU_SOURCE
#include <dlfcn.h>
int main()
{
  Dl_info info;
  dladdr((void*)0, &info);
  return 0;
}
")
