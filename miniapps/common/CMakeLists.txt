set(MFEM_MINIAPPS_COMMON_SOURCES
  fem_extras.cpp)
set(MFEM_MINIAPPS_COMMON_HEADERS
  fem_extras.hpp)

if (MFEM_USE_MPI)
  list(APPEND MFEM_MINIAPPS_COMMON_SOURCES
    pfem_extras.cpp)
  list(APPEND MFEM_MINIAPPS_COMMON_HEADERS
    pfem_extras.hpp)
endif()

add_library(mfem_miniapps_common ${MFEM_MINIAPPS_COMMON_SOURCES} ${MFEM_MINIAPPS_COMMON_HEADERS})

