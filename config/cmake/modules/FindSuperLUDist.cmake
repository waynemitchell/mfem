# Sets the following variables:
#   - SuperLUDist_FOUND
#   - SuperLUDist_INCLUDE_DIRS
#   - SuperLUDist_LIBRARIES

include(MfemCmakeUtilities)
mfem_find_package(SuperLUDist SuperLUDist SuperLUDist_DIR
  "include;SRC;SuperLU" "superlu_defs.h;slu_ddefs.h"
  "lib;SRC" "superludist;superlu_dist;superlu_dist_4.3" # add NAMES_PER_DIR?
  "Paths to headers required by SuperLU_DIST."
  "Libraries required by SuperLU_DIST."
  CHECK_BUILD SuperLUDist_VERSION_OK TRUE
  "
#include <superlu_defs.h>
int main()
{
  superlu_dist_options_t opts;
  return 0;
}
")
