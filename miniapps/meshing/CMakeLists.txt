add_mfem_executable(klein-bottle
  MAIN klein-bottle.cpp
  LIBRARIES mfem mfem_miniapps_common)

add_mfem_executable(mesh-explorer
  MAIN mesh-explorer.cpp
  LIBRARIES mfem mfem_miniapps_common)

add_mfem_executable(mobius-strip
  MAIN mobius-strip.cpp
  LIBRARIES mfem mfem_miniapps_common)
