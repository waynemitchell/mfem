# Sets the following variables:
#   - SuiteSparseCCOLAMD_FOUND
#   - SuiteSparseCCOLAMD_INCLUDE_DIRS
#   - SuiteSparseCCOLAMD_LIBRARIES

include(MfemCmakeUtilities)
mfem_find_package(SuiteSparseCCOLAMD SuiteSparseCCOLAMD SuiteSparse_DIR
  "include;suitesparse" ccolamd.h "lib" ccolamd
  "Paths to headers required by SuiteSparse/CCOLAMD."
  "Libraries required by SuiteSparse/CCOLAMD.")
