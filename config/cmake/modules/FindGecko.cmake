# Defines the following variables:
#   - GECKO_FOUND
#   - GECKO_LIBRARIES
#   - GECKO_INCLUDE_DIRS

include(MfemCmakeUtilities)
mfem_find_package(Gecko GECKO GECKO_DIR "include;inc" graph.h "lib" gecko
  "Paths to headers required by Gecko." "Libraries required by Gecko.")
