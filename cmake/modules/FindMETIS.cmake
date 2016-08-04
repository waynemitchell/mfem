# Defines the following variables:
#   - METIS_FOUND
#   - METIS_LIBRARIES
#   - METIS_INCLUDE_DIRS
#   - METIS_LIBRARY_DIRS

find_path(METIS_INCLUDE_DIRS metis.h
  HINTS ${METIS_DIR} $ENV{METIS_DIR}
  PATH_SUFFIXES include
  NO_DEFAULT_PATH
  DOC "Directory with METIS header.")
find_path(METIS_INCLUDE_DIRS metis.h
  DOC "Directory with METIS header.")

find_library(METIS_LIBRARIES metis
  HINTS ${METIS_DIR} $ENV{METIS_DIR}
  PATH_SUFFIXES lib
  NO_DEFAULT_PATH
  DOC "The METIS library.")
find_library(METIS_LIBRARIES metis
  DOC "The METIS library.")

include(CheckCXXSourceCompiles)
function(check_for_metis_5 VARNAME)
  set(TEST_SOURCE
    "
#include <metis.h>
#include <iostream>//So NULL is defined
typedef int idx_t;
typedef float real_t;
extern \"C\" {
    int METIS_PartGraphRecursive(
        idx_t *nvtxs, idx_t *ncon, idx_t *xadj,
        idx_t *adjncy, idx_t *vwgt, idx_t *vsize, idx_t *adjwgt,
        idx_t *nparts, real_t *tpwgts, real_t *ubvec, idx_t *options,
        idx_t *edgecut, idx_t *part);
    int METIS_PartGraphKway(
        idx_t *nvtxs, idx_t *ncon, idx_t *xadj,
        idx_t *adjncy, idx_t *vwgt, idx_t *vsize, idx_t *adjwgt,
        idx_t *nparts, real_t *tpwgts, real_t *ubvec, idx_t *options,
        idx_t *edgecut, idx_t *part);
    int METIS_SetDefaultOptions(idx_t *options);
}

int main()
{

    int n = 10;
    int nparts = 5;
    int edgecut;
    int* partitioning = new int[10];
    int* I = partitioning,
       * J = partitioning;

    int ncon = 1;
    int err;
    int options[40];

    METIS_SetDefaultOptions(options);
    options[10] = 1; // set METIS_OPTION_CONTIG

    err = METIS_PartGraphKway(&n,
                              &ncon,
                              I,
                              J,
                              (idx_t *) NULL,
                              (idx_t *) NULL,
                              (idx_t *) NULL,
                              &nparts,
                              (real_t *) NULL,
                              (real_t *) NULL,
                              options,
                              &edgecut,
                              partitioning);
    return err;
}
")
  set(CMAKE_REQUIRED_INCLUDES ${METIS_INCLUDE_DIRS})
  set(CMAKE_REQUIRED_LIBRARIES ${METIS_LIBRARIES})
  check_cxx_source_compiles("${TEST_SOURCE}" ${VARNAME})
endfunction()

# Decide if we're using METIS 5
check_for_metis_5(MFEM_USE_METIS_5)

if (CMAKE_VERSION VERSION_GREATER 2.8.11)
  get_filename_component(METIS_LIBRARY_DIRS ${METIS_LIBRARIES} DIRECTORY)
else()
  get_filename_component(METIS_LIBRARY_DIRS ${METIS_LIBRARIES} PATH)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(METIS
  DEFAULT_MSG
  METIS_LIBRARIES METIS_INCLUDE_DIRS METIS_LIBRARY_DIRS)
