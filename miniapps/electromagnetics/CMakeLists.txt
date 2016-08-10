if (MFEM_USE_MPI)
  add_mfem_executable(tesla
    MAIN tesla.cpp
    EXTRA_SOURCES tesla_solver.cpp
    EXTRA_HEADERS tesla_solver.hpp ${MFEM_MINIAPPS_COMMON_HEADERS}
    LIBRARIES mfem mfem_miniapps_common)
  
  add_mfem_executable(volta
    MAIN volta.cpp
    EXTRA_SOURCES volta_solver.cpp
    EXTRA_HEADERS volta_solver.hpp ${MFEM_MINIAPPS_COMMON_HEADERS}
    LIBRARIES mfem mfem_miniapps_common)

  # Add the corresponding tests to the "test" target
  add_test(NAME tesla_np=4
    COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4
    ${MPIEXEC_PREFLAGS}
    $<TARGET_FILE:tesla> -maxit 2 -cr "0 0 -0.2 0 0 0.2 0.2 0.4 1"
    ${MPIEXEC_POSTFLAGS}
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/miniapps)

  add_test(NAME volta_np=4
    COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4
    ${MPIEXEC_PREFLAGS}
    $<TARGET_FILE:volta> -maxit 2 -dbcs 1 -dbcg -ds "0.0 0.0 0.0 0.2 8.0"
    ${MPIEXEC_POSTFLAGS}
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/miniapps)

endif()
