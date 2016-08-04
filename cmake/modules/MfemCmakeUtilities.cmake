# A handy function to add the current source directory to a local
# filename. To be used for creating a list of sources.
function(convert_filenames_to_full_paths NAMES)
  unset(tmp_names)
  foreach(name ${${NAMES}})
    list(APPEND tmp_names ${CMAKE_CURRENT_SOURCE_DIR}/${name})
  endforeach()
  set(${NAMES} ${tmp_names} PARENT_SCOPE)
endfunction()

function(add_mfem_examples EXE_SRCS)
  foreach(SRC_FILE IN LISTS ${EXE_SRCS})
    get_filename_component(SRC_FILENAME ${SRC_FILE} NAME)
    
    string(REPLACE ".cpp" ".exe" EXE_NAME ${SRC_FILENAME})
    add_executable(${EXE_NAME} ${SRC_FILE})
    
    target_link_libraries(${EXE_NAME} mfem)
    if (MFEM_USE_MPI)
      target_link_libraries(${EXE_NAME} ${MPI_CXX_LIBRARIES})
    endif()
    
    # Language-specific include directories:
    target_include_directories(${EXE_NAME} PRIVATE ${MPI_CXX_INCLUDE_PATH})
    if (MPI_CXX_COMPILE_FLAGS)
      target_compile_options(${EXE_NAME} PRIVATE ${MPI_CXX_COMPILE_FLAGS})
    endif()
    
    if (MPI_CXX_LINK_FLAGS)
      set_target_properties(${EXE_NAME} PROPERTIES
        LINK_FLAGS "${MPI_CXX_LINK_FLAGS}")
    endif()
  endforeach(SRC_FILE)
endfunction()
