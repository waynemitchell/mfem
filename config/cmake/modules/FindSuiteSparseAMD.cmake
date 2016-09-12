# Sets the following variables:
#   - SuiteSparseAMD_FOUND
#   - SuiteSparseAMD_INCLUDE_DIRS
#   - SuiteSparseAMD_LIBRARIES

include(MfemCmakeUtilities)
mfem_find_package(SuiteSparseAMD SuiteSparseAMD SuiteSparse_DIR
  "include;suitesparse" amd.h "lib" amd
  "Paths to headers required by SuiteSparse/AMD."
  "Libraries required by SuiteSparse/AMD.")
