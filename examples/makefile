CC = g++
OPTS = -O3
DEBUG_OPTS = -g -DDEBUG

MFEM_DIR = ..
MFEM_LIB = -L$(MFEM_DIR) -lmfem

LAPACK_DIR = $(HOME)/lapack
LAPACK_LIB = -L$(LAPACK_DIR) -llapack

BLAS_DIR   = $(HOME)/lapack
BLAS_LIB   = -L$(LAPACK_DIR) -lblas -lgfortran
# on a Mac:
# BLAS_LIB   = -L$(LAPACK_DIR) -lblas

LIBS = $(MFEM_LIB) $(LAPACK_LIB) $(BLAS_LIB)

all: ex1

ex1: ex1.cpp
	$(CC) $(OPTS) -I$(MFEM_DIR) ex1.cpp -o ex1 $(LIBS)

debug:
	make "OPTS=$(DEBUG_OPTS)"

clean:
	rm -f *.o ex1 sol.gf *~
