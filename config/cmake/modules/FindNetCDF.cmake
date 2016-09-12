# Defines the following variables:
#   - NETCDF_FOUND
#   - NETCDF_LIBRARIES
#   - NETCDF_INCLUDE_DIRS

include(MfemCmakeUtilities)
mfem_find_package(NetCDF NETCDF NETCDF_DIR "include" netcdf.h "lib" netcdf
  "Paths to headers required by NetCDF." "Libraries required by NetCDF.")
