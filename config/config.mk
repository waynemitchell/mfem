# Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at the
# Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights reserved.
# See file COPYRIGHT for details.
#
# This file is part of the MFEM library. For more information and source code
# availability see http://mfem.googlecode.com.
#
# MFEM is free software; you can redistribute it and/or modify it under the
# terms of the GNU Lesser General Public License (as published by the Free
# Software Foundation) version 2.1 dated February 1999.

# Variables corresponding to defines in config.hpp (values 'YES' or 'NO')
MFEM_USE_MPI          = NO
MFEM_USE_METIS_5      = NO
MFEM_DEBUG            = NO
MFEM_USE_POSIX_CLOCKS = YES
MFEM_USE_LAPACK       = NO
MFEM_THREAD_SAFE      = NO
MFEM_USE_OPENMP       = NO
MFEM_USE_MESQUITE     = NO
MFEM_USE_SUITESPARSE  = NO
MFEM_USE_MEMALLOC     = YES

# Compiler, compile options, and link options
MFEM_CXX      = g++
MFEM_CPPFLAGS =
MFEM_CXXFLAGS = -O3
MFEM_LIBFLAGS = -I$(MFEM_DIR)
MFEM_FLAGS    = $(MFEM_CPPFLAGS) $(MFEM_CXXFLAGS) $(MFEM_LIBFLAGS)
MFEM_LIBS     = -L$(MFEM_DIR) -lmfem -lrt
MFEM_LIB_FILE = $(MFEM_DIR)/libmfem.a
