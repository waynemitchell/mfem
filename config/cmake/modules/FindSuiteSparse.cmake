# Sets the following variables
#   - SuiteSparse_FOUND
#   - SuiteSparse_INCLUDE_DIRS
#   - SuiteSparse_LIBRARIES
#
# We need the following libraries:
#   umfpack, cholmod, amd, camd, colamd, ccolamd, suitesparseconfig, klu, btf
#

include(MfemCmakeUtilities)
mfem_find_package(SuiteSparse SuiteSparse SuiteSparse_DIR "" "" "" ""
  "Paths to headers required by SuiteSparse."
  "Libraries required by SuiteSparse.")

if (SuiteSparse_FOUND AND METIS_VERSION_5)
  message(STATUS " *** Warning: using SuiteSparse with METIS v5!")
endif()
