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
    
    string(REPLACE ".cpp" "" EXE_NAME ${SRC_FILENAME})
    add_executable(${EXE_NAME} ${SRC_FILE})
    
    target_link_libraries(${EXE_NAME} mfem)
    if (MFEM_USE_MPI)
      target_link_libraries(${EXE_NAME} ${MPI_CXX_LIBRARIES})
    
      # Language-specific include directories:
      target_include_directories(${EXE_NAME} PRIVATE ${MPI_CXX_INCLUDE_PATH})
      if (MPI_CXX_COMPILE_FLAGS)
        target_compile_options(${EXE_NAME} PRIVATE ${MPI_CXX_COMPILE_FLAGS})
      endif()
      
      if (MPI_CXX_LINK_FLAGS)
        set_target_properties(${EXE_NAME} PROPERTIES
          LINK_FLAGS "${MPI_CXX_LINK_FLAGS}")
      endif()
    endif()
  endforeach(SRC_FILE)
endfunction()

# A slightly more versitile function for adding executables to MFEM
function(add_mfem_executable MFEM_EXE_NAME)
  # Parse the input arguments looking for the things we need
  set(POSSIBLE_ARGS "MAIN" "EXTRA_SOURCES" "EXTRA_HEADERS" "EXTRA_OPTIONS" "EXTRA_DEFINES" "LIBRARIES")
  set(CURRENT_ARG)
  foreach(arg ${ARGN})
    list(FIND POSSIBLE_ARGS ${arg} is_arg_name)
    if (${is_arg_name} GREATER -1)
      set(CURRENT_ARG ${arg})
      set(${CURRENT_ARG}_LIST)
    else()
      list(APPEND ${CURRENT_ARG}_LIST ${arg})
    endif()
  endforeach()

  # Actually add the test
  add_executable(${MFEM_EXE_NAME} ${MAIN_LIST}
    ${EXTRA_SOURCES_LIST} ${EXTRA_HEADERS_LIST})

  # Append the additional libraries and options
  target_link_libraries(${MFEM_EXE_NAME} PRIVATE ${LIBRARIES_LIST})
  target_compile_options(${MFEM_EXE_NAME} PRIVATE ${EXTRA_OPTIONS_LIST})
  target_compile_definitions(${MFEM_EXE_NAME} PRIVATE ${EXTRA_DEFINES_LIST})
  
  # Handle the MPI separately
  if (MFEM_USE_MPI)
    target_link_libraries(${MFEM_EXE_NAME} PRIVATE ${MPI_CXX_LIBRARIES})
    
    target_include_directories(${MFEM_EXE_NAME} PRIVATE ${MPI_CXX_INCLUDE_PATH})
    if (MPI_CXX_COMPILE_FLAGS)
      target_compile_options(${MFEM_EXE_NAME} PRIVATE ${MPI_CXX_COMPILE_FLAGS})
    endif()
    
    if (MPI_CXX_LINK_FLAGS)
      set_target_properties(${MFEM_EXE_NAME} PROPERTIES
        LINK_FLAGS "${MPI_CXX_LINK_FLAGS}")
    endif()
  endif()
endfunction()
