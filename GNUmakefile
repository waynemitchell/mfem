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

# Serial compiler
CC         = g++
CCOPTS     =
DEBUG_OPTS = -g -DMFEM_DEBUG
OPTIM_OPTS = -O3
DEPCC      = $(CC)

# Parallel compiler
MPICC      = mpicxx
MPIOPTS    = $(CCOPTS) -I$(HYPRE_DIR)/include

# The HYPRE library (needed to build the parallel version)
HYPRE_DIR  = ../../hypre-2.8.0b/src/hypre

# Enable experimental OpenMP support
USE_OPENMP = NO

# Which version of the METIS library should be used, 4 (default) or 5?
USE_METIS_5 = NO

# The LAPACK library
USE_LAPACK = NO

# The MESQUITE library
USE_MESQUITE = NO
MESQUITE_DIR = ../../mesquite-2.99

# The SuiteSparse library
USE_SUITESPARSE = NO
SUITESPARSE_DIR = ../../SuiteSparse

# Internal mfem options
USE_MEMALLOC = YES
# Use high-resolution POSIX clocks, link with -lrt
USE_POSIX_CLOCKS = NO

USE_LAPACK_NO  =
USE_LAPACK_YES = -DMFEM_USE_LAPACK
USE_LAPACK_DEF = $(USE_LAPACK_$(USE_LAPACK))

USE_MEMALLOC_NO  =
USE_MEMALLOC_YES = -DMFEM_USE_MEMALLOC
USE_MEMALLOC_DEF = $(USE_MEMALLOC_$(USE_MEMALLOC))

USE_OPENMP_NO  =
USE_OPENMP_YES = -fopenmp -DMFEM_USE_OPENMP -DMFEM_THREAD_SAFE
USE_OPENMP_DEF = $(USE_OPENMP_$(USE_OPENMP))

USE_METIS_5_NO  =
USE_METIS_5_YES = -DMFEM_USE_METIS_5
USE_METIS_5_DEF = $(USE_METIS_5_$(USE_METIS_5))

USE_MESQUITE_DEF_NO   =
USE_MESQUITE_DEF_YES  = -DMFEM_USE_MESQUITE
USE_MESQUITE_DEF      = $(USE_MESQUITE_DEF_$(USE_MESQUITE))
USE_MESQUITE_OPTS_NO  =
USE_MESQUITE_OPTS_YES = -I$(MESQUITE_DIR)/include
USE_MESQUITE_OPTS     = $(USE_MESQUITE_OPTS_$(USE_MESQUITE))

SUITESPARSE_OPT_NO  =
SUITESPARSE_OPT_YES = -DMFEM_USE_SUITESPARSE -I$(SUITESPARSE_DIR)/include
SUITESPARSE_OPT     = $(SUITESPARSE_OPT_$(USE_SUITESPARSE))

POSIX_CLOCKS_DEF_NO  =
POSIX_CLOCKS_DEF_YES = -DMFEM_USE_POSIX_CLOCKS
POSIX_CLOCKS_DEF = $(POSIX_CLOCKS_DEF_$(USE_POSIX_CLOCKS))

DEFS = $(USE_LAPACK_DEF) $(USE_MEMALLOC_DEF) $(USE_OPENMP_DEF) \
       $(USE_METIS_5_DEF) $(USE_MESQUITE_DEF) $(POSIX_CLOCKS_DEF)

CCC = $(CC) $(MODE_OPTS) $(DEFS) $(CCOPTS) $(USE_MESQUITE_OPTS) \
      $(SUITESPARSE_OPT)

# Compiler and options used for generating deps.mk
DEPCCC = $(DEPCC) $(CCOPTS) $(MODE_OPTS) $(DEFS)

DEFINES = $(subst -D,,$(filter -DMFEM_%,$(CCC)))

# Source dirs in logical order
DIRS = general linalg mesh fem library/picojson
SOURCE_FILES = $(foreach dir,$(DIRS),$(wildcard $(dir)/*.cpp))
OBJECT_FILES = $(SOURCE_FILES:.cpp=.o)

.SUFFIXES: .cpp .o
.cpp.o:
	cd $(<D); $(CCC) -c $(<F)

# Serial build
serial: opt

# Parallel build
parallel pdebug: CCC=$(MPICC) $(MODE_OPTS) $(DEFS) -DMFEM_USE_MPI $(MPIOPTS) $(USE_MESQUITE_OPTS) $(SUITESPARSE_OPT)
parallel: opt
pdebug: debug

lib: libmfem.a mfem_defs.hpp

debug deps_debug: MODE_OPTS = $(DEBUG_OPTS)
debug: lib

opt deps_opt: MODE_OPTS = $(OPTIM_OPTS)
opt: lib

-include deps.mk

libmfem.a: $(OBJECT_FILES)
	ar cruv libmfem.a $(OBJECT_FILES)
	ranlib libmfem.a

mfem_defs.hpp:
	@echo "Generating 'mfem_defs.hpp' ..."
	@echo "// Auto-generated file." > mfem_defs.hpp
	for i in $(DEFINES); do \
		echo "#define "$${i} >> mfem_defs.hpp; done

deps deps_debug deps_opt:
	rm -f deps.mk
	for i in $(SOURCE_FILES:.cpp=); do \
		$(DEPCCC) -MM -MT $${i}.o $${i}.cpp >> deps.mk; done

clean:
	rm -f */*.o */*~ *~ libmfem.a mfem_defs.hpp deps.mk
	cd examples; $(MAKE) clean
