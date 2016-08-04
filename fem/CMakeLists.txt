set(SRCS
  bilinearform.cpp
  bilininteg.cpp
  coefficient.cpp
  datacollection.cpp
  eltrans.cpp
  estimators.cpp
  fe.cpp
  fe_coll.cpp
  fespace.cpp
  geom.cpp
  gridfunc.cpp
  hybridization.cpp
  intrules.cpp
  linearform.cpp
  lininteg.cpp
  nonlinearform.cpp
  nonlininteg.cpp
  staticcond.cpp
  )

set(HDRS
  bilinearform.hpp
  bilininteg.hpp
  coefficient.hpp
  datacollection.hpp
  eltrans.hpp
  estimators.hpp
  fe.hpp
  fe_coll.hpp
  fem.hpp
  fespace.hpp
  geom.hpp
  gridfunc.hpp
  hybridization.hpp
  intrules.hpp
  linearform.hpp
  lininteg.hpp
  nonlinearform.hpp
  nonlininteg.hpp
  staticcond.hpp
  tbilinearform.hpp
  tbilininteg.hpp
  tcoefficient.hpp
  teltrans.hpp
  tevaluator.hpp
  tfe.hpp
  tfespace.hpp
  tintrules.hpp
  )

if (MFEM_USE_MPI)
  list(APPEND SRCS
    pbilinearform.cpp
    pfespace.cpp
    pgridfunc.cpp
    plinearform.cpp
    pnonlinearform.cpp)
  list(APPEND HDRS
    pbilinearform.hpp
    pfespace.hpp
    pgridfunc.hpp
    plinearform.hpp
    pnonlinearform.hpp)
endif()
    
convert_filenames_to_full_paths(SRCS)
convert_filenames_to_full_paths(HDRS)

set(SOURCES ${SOURCES} ${SRCS} PARENT_SCOPE)
set(HEADERS ${HEADERS} ${HDRS} PARENT_SCOPE)
