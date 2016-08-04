list(APPEND SRCS
  blockmatrix.cpp
  blockoperator.cpp
  blockvector.cpp
  densemat.cpp
  matrix.cpp
  ode.cpp
  operator.cpp
  solvers.cpp
  sparsemat.cpp
  sparsesmoothers.cpp
  vector.cpp
  )

list(APPEND HDRS
  blockmatrix.hpp
  blockoperator.hpp
  blockvector.hpp
  densemat.hpp
  linalg.hpp
  matrix.hpp
  ode.hpp
  operator.hpp
  solvers.hpp
  sparsemat.hpp
  sparsesmoothers.hpp
  tlayout.hpp
  tmatrix.hpp
  ttensor.hpp
  vector.hpp
  )

if (MFEM_USE_MPI)
  list(APPEND SRCS
    hypre.cpp
    hypre_parcsr.cpp)
  list(APPEND HDRS
    hypre.hpp
    hypre_parcsr.hpp)
endif()
    
if (MFEM_USE_SUPERLU)
  list(APPEND SRCS superlu.cpp)
  list(APPEND HDRS superlu.hpp)
endif()

convert_filenames_to_full_paths(SRCS)
convert_filenames_to_full_paths(HDRS)

set(SOURCES ${SOURCES} ${SRCS} PARENT_SCOPE)
set(HEADERS ${HEADERS} ${HDRS} PARENT_SCOPE)

