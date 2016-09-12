# Sets the following variables:
#   - SuiteSparseCAMD_FOUND
#   - SuiteSparseCAMD_INCLUDE_DIRS
#   - SuiteSparseCAMD_LIBRARIES

include(MfemCmakeUtilities)
mfem_find_package(SuiteSparseCAMD SuiteSparseCAMD SuiteSparse_DIR
  "include;suitesparse" camd.h "lib" camd
  "Paths to headers required by SuiteSparse/CAMD."
  "Libraries required by SuiteSparse/CAMD.")
