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
CC = g++
OPTS = -O3 -I$(MFEM_DIR)
DEBUG_OPTS = -g -I$(MFEM_DIR)
LIBS = $(MFEM_LIB) $(LAPACK_LIBS)
PROF_LIBS = $(MFEM_LIB) $(LAPACK_LIBS) $(PROFILER_LIB)

# Parallel compiler
MPICC = mpiCC
MPIOPTS = $(OPTS) -I$(HYPRE_DIR)/include
MPIDEBUG_OPTS = $(DEBUG_OPTS) -I$(HYPRE_DIR)/include
MPILIBS = $(LIBS) -L$(METIS_DIR) -lmetis -L$(HYPRE_DIR)/lib -lHYPRE

# The MFEM library
MFEM_DIR = ..
MFEM_LIB = -L$(MFEM_DIR) -lmfem

# The METIS and HYPRE libraries (needed for the parallel examples)
METIS_DIR = ../../metis-4.0
HYPRE_DIR = ../../hypre-2.7.0b/src/hypre

# The LAPACK and BLAS libraries (needed if MFEM was compiled with LAPACK support)
USE_LAPACK = YES
LAPACK_DIR = $(HOME)/lapack
LAPACK_LIB = -L$(LAPACK_DIR) -llapack
BLAS_DIR   = $(HOME)/lapack
BLAS_LIB   = -L$(LAPACK_DIR) -lblas -lgfortran
# on a Mac:
# BLAS_LIB   = -L$(LAPACK_DIR) -lblas
LAPACK_LIBS_NO  =
LAPACK_LIBS_YES = $(LAPACK_LIB) $(BLAS_LIB)
LAPACK_LIBS     = $(LAPACK_LIBS_$(USE_LAPACK))

# google profiler library
PROFILER_DIR = $(HOME)/local/lib
PROFILER_LIB = -L$(PROFILER_DIR) -lprofiler


serial: ex1x ex4x test_simplex_intrules print_quadrature_rule view_basis ex5 ex6

parallel: ex5p

ex1x: ex1x.cpp
	$(CC) $(OPTS) ex1x.cpp -o ex1x $(LIBS)

ex4x: ex4x.cpp
	$(CC) $(OPTS) ex4x.cpp -o ex4x $(LIBS)

test_simplex_intrules: test_simplex_intrules.cpp
	$(CC) $(OPTS) test_simplex_intrules.cpp -o test_simplex_intrules $(LIBS)

print_quadrature_rule: print_quadrature_rule.cpp
	$(CC) $(OPTS) print_quadrature_rule.cpp -o print_quadrature_rule $(LIBS)

view_basis: view_basis.cpp
	$(CC) $(OPTS) view_basis.cpp -o view_basis $(LIBS)

ex5: ex5.cpp
	$(CC) $(OPTS) ex5.cpp -o ex5 $(LIBS)

ex5p: ex5p.cpp
	$(MPICC) $(MPIOPTS) ex5p.cpp -o ex5p $(MPILIBS)

ex6: ex6.cpp
	$(CC) $(OPTS) ex6.cpp -o ex6 $(LIBS)

debug:
	make "OPTS=$(DEBUG_OPTS)" serial

pdebug:
	make "MPIOPTS=$(MPIDEBUG_OPTS)" parallel

prof:
	make "LIBS=$(PROF_LIBS)" serial

prof-debug:
	make "OPTS=$(DEBUG_OPTS)" "LIBS=$(PROF_LIBS)" serial

clean:
	rm -f *.o *~ ex1x ex4x test_simplex_intrules print_quadrature_rule
	rm -f view_basis ex5 ex5p ex6
	rm -f refined.mesh sol.gf
