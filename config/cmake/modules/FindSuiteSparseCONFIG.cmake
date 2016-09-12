# Sets the following variables:
#   - SuiteSparseCONFIG_FOUND
#   - SuiteSparseCONFIG_INCLUDE_DIRS
#   - SuiteSparseCONFIG_LIBRARIES

include(MfemCmakeUtilities)
mfem_find_package(SuiteSparseCONFIG SuiteSparseCONFIG SuiteSparse_DIR
  "include;suitesparse" SuiteSparse_config.h "lib" suitesparseconfig
  "Paths to headers required by SuiteSparse/CONFIG."
  "Required SuiteSparse/CONFIG libraries.")
