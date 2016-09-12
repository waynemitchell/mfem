# Find MPFR.
# Defines the following variables:
#   - MPFR_FOUND
#   - MPFR_LIBRARIES    (if needed)
#   - MPFR_INCLUDE_DIRS (if needed)

include(MfemCmakeUtilities)
set(MPFR_SKIP_STANDARD TRUE)
set(MPFR_SKIP_FPHSA TRUE)
mfem_find_library(MPFR MPFR "mpfr" "The MPFR library." MPFR_BUILD
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
unset(MPFR_SKIP_FPHSA)
unset(MPFR_SKIP_STANDARD)

set(MPFR_SKIP_LOOKING_MSG TRUE)
mfem_find_package(MPFR MPFR MPFR_DIR "include" mpfr.h "lib" mpfr
  "Paths to headers required by MPFR." "Libraries required by MPFR.")
unset(MPFR_SKIP_LOOKING_MSG)
