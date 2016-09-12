# Sets the following variables:
#   - SuiteSparseBTF_FOUND
#   - SuiteSparseBTF_INCLUDE_DIRS
#   - SuiteSparseBTF_LIBRARIES

include(MfemCmakeUtilities)
mfem_find_package(SuiteSparseBTF SuiteSparseBTF SuiteSparse_DIR
  "include;suitesparse" btf.h "lib" btf
  "Paths to headers required by SuiteSparse/BTF."
  "Libraries required by SuiteSparse/BTF.")
