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
OPTS = -O3
DEBUG_OPTS = -g
LIB_OPTS = -I$(MFEM_DIR) $(SUITESPARSE_OPT)
COPTS = $(OPTS) $(LIB_OPTS)
LIBS = $(MFEM_LIB) $(LAPACK_LIBS) $(OPENMP_LIBS) $(SUITESPARSE_LIB)\
 $(POSIX_CLOCKS_LIB)

# Parallel compiler
MPICC = mpicxx
MPIOPTS = $(OPTS) $(LIB_OPTS) -I$(HYPRE_DIR)/include
MPIDEBUG_OPTS = $(DEBUG_OPTS) $(LIB_OPTS) -I$(HYPRE_DIR)/include
MPILIBS = $(LIBS) -L$(METIS_DIR) -lmetis -L$(HYPRE_DIR)/lib -lHYPRE

# The MFEM library
MFEM_DIR = ..
MFEM_LIB = -L$(MFEM_DIR) -lmfem

# The METIS and HYPRE libraries (needed for the parallel examples)
METIS_DIR = ../../metis-4.0
HYPRE_DIR = ../../hypre-2.8.0b/src/hypre

# Enable experimental OpenMP support
USE_OPENMP = NO
OPENMP_LIBS_NO  =
OPENMP_LIBS_YES = -fopenmp
OPENMP_LIBS     = $(OPENMP_LIBS_$(USE_OPENMP))

# The LAPACK and BLAS libraries (needed if MFEM was compiled with LAPACK support)
USE_LAPACK = NO
LAPACK_DIR = $(HOME)/lapack
LAPACK_LIB = -L$(LAPACK_DIR) -llapack
BLAS_DIR   = $(HOME)/lapack
BLAS_LIB   = -L$(LAPACK_DIR) -lblas -lgfortran
# on a Mac:
# BLAS_LIB   = -L$(LAPACK_DIR) -lblas
LAPACK_LIBS_NO  =
LAPACK_LIBS_YES = $(LAPACK_LIB) $(BLAS_LIB)
LAPACK_LIBS     = $(LAPACK_LIBS_$(USE_LAPACK))

# The SuiteSparse library
USE_SUITESPARSE = NO
SUITESPARSE_DIR = ../../SuiteSparse
SUITESPARSE_OPT_NO  =
SUITESPARSE_OPT_YES = -I$(SUITESPARSE_DIR)/include
SUITESPARSE_OPT     = $(SUITESPARSE_OPT_$(USE_SUITESPARSE))
SUITESPARSE_LIB_NO  =
SUITESPARSE_LIB_YES = -L$(SUITESPARSE_DIR)/lib -lumfpack -lcholmod -lcolamd\
 -lamd -lcamd -lccolamd -lsuitesparseconfig -lrt -L$(METIS_DIR) -lmetis\
 $(LAPACK_LIB) $(BLAS_LIB)
SUITESPARSE_LIB     = $(SUITESPARSE_LIB_$(USE_SUITESPARSE))

# The high-resolution POSIX clocks may need -lrt
USE_POSIX_CLOCKS = NO
POSIX_CLOCKS_LIB_NO  =
POSIX_CLOCKS_LIB_YES = -lrt
POSIX_CLOCKS_LIB = $(POSIX_CLOCKS_LIB_$(USE_POSIX_CLOCKS))

serial: ex1 ex2 ex3 ex4 ex5 ex8

parallel: ex1p ex2p ex3p ex4p ex5p

ex1: ex1.cpp
	$(CC) $(COPTS) ex1.cpp -o ex1 $(LIBS)

ex1p: ex1p.cpp
	$(MPICC) $(MPIOPTS) ex1p.cpp -o ex1p $(MPILIBS)

ex2: ex2.cpp
	$(CC) $(COPTS) ex2.cpp -o ex2 $(LIBS)

ex2p: ex2p.cpp
	$(MPICC) $(MPIOPTS) ex2p.cpp -o ex2p $(MPILIBS)

ex3: ex3.cpp
	$(CC) $(COPTS) ex3.cpp -o ex3 $(LIBS)

ex3p: ex3p.cpp
	$(MPICC) $(MPIOPTS) ex3p.cpp -o ex3p $(MPILIBS)

ex4: ex4.cpp
	$(CC) $(COPTS) ex4.cpp -o ex4 $(LIBS)

ex4p: ex4p.cpp
	$(MPICC) $(MPIOPTS) ex4p.cpp -o ex4p $(MPILIBS)

ex5: ex5.cpp
	$(CC) $(COPTS) ex5.cpp -o ex5 $(LIBS)

ex5p: ex5p.cpp
	$(MPICC) $(MPIOPTS) ex5p.cpp -o ex5p $(MPILIBS)

ex8: ex8.cpp
	$(CC) $(COPTS) ex8.cpp -o ex8 $(LIBS)

debug:
	$(MAKE) "OPTS=$(DEBUG_OPTS)" serial

pdebug:
	$(MAKE) "MPIOPTS=$(MPIDEBUG_OPTS)" parallel

clean:
	rm -f *.o *~ ex1 ex1p ex2 ex2p ex3 ex3p ex4 ex4p ex5 ex5p ex8
	rm -f refined.mesh displaced.mesh sol.* mesh.* ex5.mesh sol_u.* sol_p.*
