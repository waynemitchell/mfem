# Sets the following variables:
#   - SuiteSparseUMFPACK_FOUND
#   - SuiteSparseUMFPACK_INCLUDE_DIRS
#   - SuiteSparseUMFPACK_LIBRARIES

include(MfemCmakeUtilities)
mfem_find_package(SuiteSparseUMFPACK SuiteSparseUMFPACK SuiteSparse_DIR
  "include;suitesparse" umfpack.h "lib" umfpack
  "Paths to headers required by SuiteSparse/UMFPACK."
  "Required SuiteSparse/UMFPACK libraries.")
