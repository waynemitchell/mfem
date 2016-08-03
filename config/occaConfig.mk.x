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
MFEM_USE_SUPERLU     = NO
MFEM_USE_MEMALLOC    = YES
MFEM_USE_GECKO       = NO
MFEM_TIMER_TYPE      = 2

# Compiler, compile options, and link options
MFEM_CXX       = g++
MFEM_CPPFLAGS  = -D LINUX_OS=1 -D OSX_OS=2 -D WINDOWS_OS=4 -D WINUX_OS=5  -D OCCA_OS=LINUX_OS -fopenmp  -O3 -D __extern_always_inline=inline -DOCCA_DEBUG_ENABLED=0 -DNDEBUG=1 -DOCCA_SHOW_WARNINGS=0 -DOCCA_CHECK_ENABLED=1 -DOCCA_OPENMP_ENABLED=1 -DOCCA_OPENCL_ENABLED=1 -DOCCA_CUDA_ENABLED=1 -DOCCA_HSA_ENABLED=0 -DOCCA_COI_ENABLED=0
MFEM_CPPFLAGS  +=  -I/g/g17/vargas45/occa/lib -I/g/g17/vargas45/occa/include -L/g/g17/vargas45/occa/lib   -I/opt/cudatoolkit-7.5/include -I/opt/cudatoolkit-7.5/include  -locca -lm -lrt -ldl -L/opt/cudatoolkit-7.5/lib64 -lOpenCL -L/usr/lib64/nvidia -lcuda
MFEM_CXXFLAGS  = -O3
MFEM_INCFLAGS  = -I$(MFEM_INC_DIR)
MFEM_FLAGS     = $(MFEM_CPPFLAGS) $(MFEM_CXXFLAGS) $(MFEM_INCFLAGS)
MFEM_LIBS      = -L$(MFEM_LIB_DIR) -lmfem -lrt
MFEM_LIB_FILE  = $(MFEM_LIB_DIR)/libmfem.a
MFEM_BUILD_TAG = Linux rzhasgpu18 x86_64
MFEM_PREFIX    = ./mfem
MFEM_INC_DIR   = $(MFEM_DIR)
MFEM_LIB_DIR   = $(MFEM_DIR)
