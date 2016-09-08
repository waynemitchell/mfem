# Defines the following variables:
#   - NETCDF_FOUND
#   - NETCDF_LIBRARIES
#   - NETCDF_INCLUDE_DIRS
#   - NETCDF_LIBRARY_DIRS

find_path(NETCDF_INCLUDE_DIRS netcdf.h
  HINTS ${NETCDF_DIR} $ENV{NETCDF_DIR}
  PATH_SUFFIXES include
  NO_DEFAULT_PATH
  DOC "Directory with NETCDF header.")
find_path(NETCDF_INCLUDE_DIRS netcdf.h
  DOC "Directory with NETCDF header.")

find_library(NETCDF_LIBRARIES netcdf
  HINTS ${NETCDF_DIR} $ENV{NETCDF_DIR} ${NETCDF_LIBRARY_DIRS}
  PATH_SUFFIXES lib
  NO_DEFAULT_PATH
  DOC "The NETCDF library.")
find_library(NETCDF_LIBRARIES netcdf
  DOC "The NETCDF library.")

if(CMAKE_VERSION VERSION_GREATER 2.8.11)
  get_filename_component(NETCDF_LIBRARY_DIRS ${NETCDF_LIBRARIES} DIRECTORY)
else()
  get_filename_component(NETCDF_LIBRARY_DIRS ${NETCDF_LIBRARIES} PATH)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NetCDF
  " *** NetCDF libarary not found. Please set NETCDF_DIR."
  NETCDF_LIBRARIES NETCDF_INCLUDE_DIRS NETCDF_LIBRARY_DIRS)

