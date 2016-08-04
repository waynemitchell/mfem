set(SRCS
  element.cpp
  hexahedron.cpp
  mesh.cpp
  mesh_operators.cpp
  mesh_readers.cpp
  ncmesh.cpp
  nurbs.cpp
  point.cpp
  quadrilateral.cpp
  segment.cpp
  tetrahedron.cpp
  triangle.cpp
  vertex.cpp
  )

set(HDRS
  element.hpp
  hexahedron.hpp
  mesh.hpp
  mesh_headers.hpp
  mesh_operators.hpp
  ncmesh.hpp
  nurbs.hpp
  point.hpp
  quadrilateral.hpp
  segment.hpp
  tetrahedron.hpp
  tmesh.hpp
  triangle.hpp
  vertex.hpp
  )

if (MFEM_USE_MPI)
  list(APPEND SRCS
    pmesh.cpp
    pncmesh.cpp)
  list(APPEND HDRS
    pmesh.hpp
    pncmesh.hpp)
endif()

if (MFEM_USE_MESQUITE)
  list(APPEND SRCS mesquite.cpp)
  list(APPEND HDRS mesquite.hpp)
endif()

convert_filenames_to_full_paths(SRCS)
convert_filenames_to_full_paths(HDRS)

set(SOURCES ${SOURCES} ${SRCS} PARENT_SCOPE)
set(HEADERS ${HEADERS} ${HDRS} PARENT_SCOPE)
