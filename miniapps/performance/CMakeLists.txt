set(PERFORMANCE_CXX_OPTIONS)
if (${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")
  list(APPEND PERFORMANCE_CXX_OPTIONS
    "-march=native"
    "-std=c++11"
    "-Wpedantic"
    "-Wall"
    "-fcolor-diagnostics"
    "-fvectorize"
    "-fslp-vectorize"
    "-fslp-vectorize-aggressive"
    "-ffp-contract=fast")
elseif (${CMAKE_CXX_COMPILER_ID} MATCHES "GNU")
  list(APPEND PERFORMANCE_CXX_OPTIONS
    "-march=native"
    "-std=c++11"
    "-pedantic"
    "-Wall"
    "--param" "max-completely-peel-times=3")
endif()

add_mfem_executable(performance_ex1
  MAIN ex1.cpp
  LIBRARIES mfem mfem_miniapps_common
  EXTRA_OPTIONS ${PERFORMANCE_CXX_OPTIONS})

add_test(NAME performance_ex1_ser
  COMMAND performance_ex1 -no-vis
  WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/miniapps)

if (MFEM_USE_MPI)
  add_mfem_executable(performance_ex1p
    MAIN ex1p.cpp 
    LIBRARIES mfem mfem_miniapps_common
    EXTRA_OPTIONS ${PERFORMANCE_CXX_OPTIONS})

  add_test(NAME performance_ex1p_np=4
    COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4
    ${MPIEXEC_PREFLAGS} $<TARGET_FILE:performance_ex1p> -no-vis
    ${MPIEXEC_POSTFLAGS}
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/miniapps)
endif()
