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
#   - SuiteSparseKLU_FOUND
#   - SuiteSparseKLU_INCLUDE_DIRS
#   - SuiteSparseKLU_LIBRARIES

include(MfemCmakeUtilities)
mfem_find_package(SuiteSparseKLU SuiteSparseKLU SuiteSparse_DIR
  "include;suitesparse" klu.h "lib" klu
  "Paths to headers required by SuiteSparse/KLU."
  "Required SuiteSparse/KLU libraries.")
