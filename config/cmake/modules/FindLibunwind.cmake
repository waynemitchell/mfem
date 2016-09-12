# Find libunwind.
# Defines the following variables:
#   - LIBUNWIND_FOUND
#   - LIBUNWIND_LIBRARIES    (if needed)
#   - LIBUNWIND_INCLUDE_DIRS (if needed)

include(MfemCmakeUtilities)
set(Libunwind_SKIP_FPHSA TRUE)
mfem_find_library(Libunwind LIBUNWIND "unwind" "The libunwind library."
  LIBUNWIND_BUILD
  "
#define UNW_LOCAL_ONLY
#include <libunwind.h>
int main()
{
  unw_context_t uc;
  unw_getcontext(&uc);
  return 0;
}
")
unset(Libunwind_SKIP_FPHSA)

set(Libunwind_SKIP_LOOKING_MSG TRUE)
mfem_find_package(Libunwind LIBUNWIND LIBUNWIND_DIR "include" libunwind.h
  "lib" unwind
  "Paths to headers required by libunwind." "Libraries required by libunwind.")
unset(Libunwind_SKIP_LOOKING_MSG)
