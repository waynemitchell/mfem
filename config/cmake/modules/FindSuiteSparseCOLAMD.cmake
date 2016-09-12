# Sets the following variables:
#   - SuiteSparseCOLAMD_FOUND
#   - SuiteSparseCOLAMD_INCLUDE_DIRS
#   - SuiteSparseCOLAMD_LIBRARIES

include(MfemCmakeUtilities)
mfem_find_package(SuiteSparseCOLAMD SuiteSparseCOLAMD SuiteSparse_DIR
  "include;suitesparse" colamd.h "lib" colamd
  "Paths to headers required by SuiteSparse/COLAMD."
  "Libraries required by SuiteSparse/COLAMD.")
