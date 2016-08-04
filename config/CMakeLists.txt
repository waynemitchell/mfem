set(HDRS
 config.hpp
 tconfig.hpp
 )

include_directories(${CMAKE_CURRENT_SOURCE_DIRECTORY})

convert_filenames_to_full_paths(HDRS)

set(HEADERS ${HEADERS} ${HDRS} PARENT_SCOPE)
