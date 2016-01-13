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

# Variables corresponding to defines in config.hpp (YES, NO, or value)
MFEM_USE_MPI         = NO
MFEM_USE_METIS_5     = NO
MFEM_DEBUG           = NO
MFEM_USE_LAPACK      = NO
MFEM_THREAD_SAFE     = NO
MFEM_USE_OPENMP      = NO
MFEM_USE_MESQUITE    = NO
MFEM_USE_SUITESPARSE = NO
MFEM_USE_MEMALLOC    = YES
MFEM_USE_GECKO       = NO
MFEM_TIMER_TYPE      = 2

# Compiler, compile options, and link options
MFEM_CXX       = g++
MFEM_CPPFLAGS  =
MFEM_CXXFLAGS  = -O3
MFEM_INCFLAGS  = -I$(MFEM_INC_DIR) -I$(MFEM_DIR)/../sidre/include -I$(MFEM_DIR)/../conduit/include
MFEM_FLAGS     = $(MFEM_CPPFLAGS) $(MFEM_CXXFLAGS) $(MFEM_INCFLAGS)
MFEM_LIBS      = -L$(MFEM_LIB_DIR) -lmfem -L$(MFEM_DIR)/../sidre/lib -lsidre -lslic -lcommon -L$(MFEM_DIR)/../conduit/lib -lconduit -Wl,-rpath $(MFEM_DIR)/../conduit/lib -lrt
MFEM_LIB_FILE  = $(MFEM_LIB_DIR)/libmfem.a
MFEM_BUILD_TAG = Linux cab670 x86_64
MFEM_PREFIX    = ./mfem
MFEM_INC_DIR   = $(MFEM_DIR)
MFEM_LIB_DIR   = $(MFEM_DIR)
