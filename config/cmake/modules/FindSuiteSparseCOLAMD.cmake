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
#   - SuiteSparseCOLAMD_FOUND
#   - SuiteSparseCOLAMD_INCLUDE_DIRS
#   - SuiteSparseCOLAMD_LIBRARIES

include(MfemCmakeUtilities)
mfem_find_package(SuiteSparseCOLAMD SuiteSparseCOLAMD SuiteSparse_DIR
  "include;suitesparse" colamd.h "lib" colamd
  "Paths to headers required by SuiteSparse/COLAMD."
  "Libraries required by SuiteSparse/COLAMD.")
