# Sets the following variables
#   - ParMETIS_FOUND
#   - ParMETIS_INCLUDE_DIRS
#   - ParMETIS_LIBRARIES
#
# We need the following libraries:
#   parmetis

include(MfemCmakeUtilities)
mfem_find_package(ParMETIS ParMETIS ParMETIS_DIR "include" parmetis.h
  "lib" parmetis
  "Paths to headers required by ParMETIS." "Libraries required by ParMETIS.")
