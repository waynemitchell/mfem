# Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at the
# Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights reserved.
# See file COPYRIGHT for details.
#
# This file is part of the MFEM library. For more information and source code
# availability see http://mfem.org.
#
# MFEM is free software; you can redistribute it and/or modify it under the
# terms of the GNU Lesser General Public License (as published by the Free
# Software Foundation) version 2.1 dated February 1999.

# Sets the following variables:
#   - SuiteSparseUMFPACK_FOUND
#   - SuiteSparseUMFPACK_INCLUDE_DIRS
#   - SuiteSparseUMFPACK_LIBRARIES

include(MfemCmakeUtilities)
mfem_find_package(SuiteSparseUMFPACK SuiteSparseUMFPACK SuiteSparse_DIR
  "include;suitesparse" umfpack.h "lib" umfpack
  "Paths to headers required by SuiteSparse/UMFPACK."
  "Required SuiteSparse/UMFPACK libraries.")
