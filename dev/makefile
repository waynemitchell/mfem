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

# Use the MFEM build directory
MFEM_DIR = ..
CONFIG_MK = $(MFEM_DIR)/config/config.mk

ifneq (clean,$(MAKECMDGOALS))
   -include $(CONFIG_MK)
endif

# MFEM_CXXFLAGS = -O2
# MFEM_CXXFLAGS = -O3
# --- icpc (intel compiler)
# MFEM_CXXFLAGS += -xHost
# MFEM_CXXFLAGS += -xSSE3
# MFEM_CXXFLAGS += -xAVX
# MFEM_CXXFLAGS += -xCORE-AVX2
# --- xlcxx (ibm compiler)
# MFEM_CXXFLAGS = -O5 -qmaxmem=-1 -qnoeh
# MFEM_CXXFLAGS = -O5 -qnounwind -qstrict
# MFEM_CXXFLAGS += -qsuppress=1540-1088:1540-1090:1540-1101

ifeq ($(MFEM_CXX),g++)
# ifeq ($(MFEM_CXX),mpicxx)
   MFEM_CXXFLAGS += -march=native
   # MFEM_CXXFLAGS += -march=sandybridge -mtune=native
   # MFEM_CXXFLAGS += -fno-exceptions
   # MFEM_CXXFLAGS += -std=c++03
   MFEM_CXXFLAGS += -std=c++11
   MFEM_CXXFLAGS += -pedantic -Wall

   # MFEM_CXXFLAGS += -Winline
   # MFEM_CXXFLAGS += -fno-tree-vectorize -fno-tree-slp-vectorize
   # MFEM_CXXFLAGS += -fno-peel-loops -fno-unroll-loops
   # MFEM_CXXFLAGS += -fno-strict-aliasing
   # MFEM_CXXFLAGS += -Wstrict-aliasing=1
   # Note: -ffast-math implies -funsafe-math-optimizations;
   # -funsafe-math-optimizations implies -fassociative-math, -fno-signed-zeros,
   # and -fno-trapping-math; -fassociative-math requires -fno-signed-zeros and
   # -fno-trapping-math
   # MFEM_CXXFLAGS += -fno-signed-zeros
   # MFEM_CXXFLAGS += -fassociative-math -fno-signed-zeros -fno-trapping-math
   # MFEM_CXXFLAGS += -funsafe-math-optimizations
   # MFEM_CXXFLAGS += -ffast-math

   # Default values in gcc 4.9.2/5.1.0 (from params.def)
   #  max-unrolled-insns                     . 200
   #  max-average-unrolled-insns             . 80
   #  max-unroll-times                       . 8
   #  max-peeled-insns                       . 100
   #  max-peel-times                         . 16
   #  max-peel-branches                      . 32
   #  max-completely-peeled-insns            . 100/200
   #  max-completely-peel-times              . 16
   #  max-once-peeled-insns (new?)           . 400
   #  max-completely-peel-loop-nest-depth    . 8

   # MFEM_CXXFLAGS += -fpeel-loops\
   #  --param max-unrolled-insns=400\
   #  --param max-average-unrolled-insns=160\
   #  --param max-unroll-times=8\
   #  --param max-peeled-insns=200\
   #  --param max-peel-times=16\
   #  --param max-peel-branches=32\
   #  --param max-completely-peeled-insns=200\
   #  --param max-completely-peel-times=16\
   #  --param max-completely-peel-loop-nest-depth=8
   # MFEM_CXXFLAGS += --param large-function-growth=200
   # MFEM_CXXFLAGS += --param max-unroll-times=2
   # MFEM_CXXFLAGS += --param max-peel-times=3
   MFEM_CXXFLAGS += --param max-completely-peel-times=3

   # MFEM_CXXFLAGS += -fopt-info-all=all.info
   MFEM_CXXFLAGS += -fdump-tree-optimized-blocks
   # MFEM_CXXFLAGS += -fdump-tree-vect-details
   # MFEM_CXXFLAGS += -fdump-passes
endif

ifeq ($(MFEM_CXX),clang++)
   MFEM_CXXFLAGS += -march=native
   # MFEM_CXXFLAGS += -march=sandybridge -mtune=native
   MFEM_CXXFLAGS += -std=c++11
   MFEM_CXXFLAGS += -pedantic -Wall

   MFEM_CXXFLAGS += -fcolor-diagnostics
   MFEM_CXXFLAGS += -fvectorize
   MFEM_CXXFLAGS += -fslp-vectorize
   MFEM_CXXFLAGS += -fslp-vectorize-aggressive
   MFEM_CXXFLAGS += -ffp-contract=fast

   # MFEM_CXXFLAGS += -emit-llvm
   # MFEM_CXXFLAGS += -emit-ast
   # MFEM_CXXFLAGS += -Rpass=.*
   # MFEM_CXXFLAGS += -Rpass=inline
   # MFEM_CXXFLAGS += -Rpass=loop-unroll
endif

ifeq ($(COUNT_FLOPS),YES)
   MFEM_CXXFLAGS += -DMFEM_COUNT_FLOPS
endif

ifdef MFEM_CXX
   $(info Compiler info: $(value MFEM_CXX) -v)
   $(info $(shell $(MFEM_CXX) -v))
endif

TESTS_CPP = $(wildcard *test*.cpp) ex1t.cpp
TESTS = $(TESTS_CPP:.cpp=)


.PHONY: all clean

all: $(TESTS)

# Remove built-in rule
%: %.cpp

# Replace the default implicit rule for *.cpp files
%: %.cpp $(CONFIG_MK) $(MFEM_LIB_FILE)
	$(MFEM_CXX) $(MFEM_FLAGS) $(EIGEN_FLAGS) $(@).cpp -o $@ $(MFEM_LIBS)

# Remove built-in rule
%.s: %.cpp

# Replace the default implicit rule for *.cpp files
%.s: %.cpp $(CONFIG_MK) $(MFEM_LIB_FILE)
	$(MFEM_CXX) $(MFEM_FLAGS) $(EIGEN_FLAGS) -S $< -o $@

# Remove built-in rule
%.o: %.cpp

# Replace the default implicit rule for *.cpp files
%.o: %.cpp $(CONFIG_MK) $(MFEM_LIB_FILE)
	$(MFEM_CXX) $(MFEM_FLAGS) $(EIGEN_FLAGS) -c $< -o $@

# Generate an error message if the MFEM library is not built and exit
$(CONFIG_MK) $(MFEM_LIB_FILE):
	$(error The MFEM library is not built)

clean:
	rm -rf $(TESTS) *.s *.o *~ *.dSYM
	rm -rf *.info *.optimized *.vect
