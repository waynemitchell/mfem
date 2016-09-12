# Sets the following variables:
#   - SuiteSparseKLU_FOUND
#   - SuiteSparseKLU_INCLUDE_DIRS
#   - SuiteSparseKLU_LIBRARIES

include(MfemCmakeUtilities)
mfem_find_package(SuiteSparseKLU SuiteSparseKLU SuiteSparse_DIR
  "include;suitesparse" klu.h "lib" klu
  "Paths to headers required by SuiteSparse/KLU."
  "Required SuiteSparse/KLU libraries.")
