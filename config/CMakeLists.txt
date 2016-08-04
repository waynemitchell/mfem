set(HDRS
 config.hpp
 tconfig.hpp
 )

convert_filenames_to_full_paths(HDRS)

set(HEADERS ${HEADERS} ${HDRS} PARENT_SCOPE)
