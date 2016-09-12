# Sets the following variables:
#   - SuiteSparseCHOLMOD_FOUND
#   - SuiteSparseCHOLMOD_INCLUDE_DIRS
#   - SuiteSparseCHOLMOD_LIBRARIES

include(MfemCmakeUtilities)
mfem_find_package(SuiteSparseCHOLMOD SuiteSparseCHOLMOD SuiteSparse_DIR
  "include;suitesparse" cholmod.h "lib" cholmod
  "Paths to headers required by SuiteSparse/CHOLMOD."
  "Libraries required by SuiteSparse/CHOLMOD.")
