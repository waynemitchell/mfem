
# Include the source directory where mfem.hpp and mfem-performance.hpp are.
include_directories(${PROJECT_BINARY_DIR})

set(MINIAPP_COMMON_HEADERS)
add_subdirectory(common)
add_subdirectory(electromagnetics)
add_subdirectory(meshing)
add_subdirectory(performance)
add_subdirectory(tools)
