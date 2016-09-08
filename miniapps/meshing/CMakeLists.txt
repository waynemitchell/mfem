add_mfem_executable(klein-bottle
  MAIN klein-bottle.cpp
  LIBRARIES mfem)

add_mfem_executable(mesh-explorer
  MAIN mesh-explorer.cpp
  LIBRARIES mfem)

add_mfem_executable(mobius-strip
  MAIN mobius-strip.cpp
  LIBRARIES mfem)

# Add tests?
