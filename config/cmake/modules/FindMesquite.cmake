# Defines the following variables:
#   - MESQUITE_FOUND
#   - MESQUITE_LIBRARIES
#   - MESQUITE_INCLUDE_DIRS

include(MfemCmakeUtilities)
mfem_find_package(Mesquite MESQUITE MESQUITE_DIR
  "include" "Mesquite_all_headers.hpp" "lib" "mesquite"
  "Paths to headers required by Mesquite." "Libraries required by Mesquite.")
