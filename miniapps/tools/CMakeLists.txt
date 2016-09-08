add_mfem_executable(display-basis
  MAIN display-basis.cpp
  EXTRA_HEADERS ${MFEM_MINIAPPS_COMMON_HEADERS}
  LIBRARIES mfem mfem_miniapps_common)

# Add tests?
