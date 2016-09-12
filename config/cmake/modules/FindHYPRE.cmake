# Defines the following variables:
#   - HYPRE_FOUND
#   - HYPRE_LIBRARIES
#   - HYPRE_INCLUDE_DIRS

include(MfemCmakeUtilities)
mfem_find_package(HYPRE HYPRE HYPRE_DIR "include" "HYPRE.h" "lib" "HYPRE"
  "Paths to headers required by HYPRE." "Libraries required by HYPRE.")
