list(APPEND SRCS
  array.cpp
  error.cpp
  isockstream.cpp
  optparser.cpp
  osockstream.cpp
  sets.cpp
  socketstream.cpp
  stable3d.cpp
  table.cpp
  tic_toc.cpp
  )

list(APPEND HDRS
  array.hpp
  error.hpp
  hash.hpp
  isockstream.hpp
  mem_alloc.hpp
  optparser.hpp
  osockstream.hpp
  sets.hpp
  socketstream.hpp
  sort_pairs.hpp
  stable3d.hpp
  table.hpp
  tassign.hpp
  tic_toc.hpp
  )

if (MFEM_USE_MPI)
  list(APPEND SRCS communication.cpp)
  list(APPEND HDRS communication.hpp)
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIRECTORY})

convert_filenames_to_full_paths(SRCS)
convert_filenames_to_full_paths(HDRS)

set(SOURCES ${SOURCES} ${SRCS} PARENT_SCOPE)
set(HEADERS ${HEADERS} ${HDRS} PARENT_SCOPE)
