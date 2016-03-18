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
MFEM_DIR = ../..
CONFIG_MK = $(MFEM_DIR)/config/config.mk
# Use the MFEM install directory
# MFEM_DIR = ../../mfem
# CONFIG_MK = $(MFEM_DIR)/config.mk

ifneq (clean,$(MAKECMDGOALS))
   -include $(CONFIG_MK)
endif

# Compiler specific optimizations.
# For best performance, GCC 5 (or newer) is recommended.
ifneq (,$(findstring $(MFEM_CXX),g++ mpicxx))
   MFEM_CXXFLAGS += -march=native
   # MFEM_CXXFLAGS += -std=c++03
   MFEM_CXXFLAGS += -std=c++11
   MFEM_CXXFLAGS += -pedantic -Wall

   MFEM_CXXFLAGS += --param max-completely-peel-times=3
   # MFEM_CXXFLAGS += -fdump-tree-optimized-blocks

   $(info ----------)
   $(info using gnu g++ options: $(MFEM_CXXFLAGS))
   $(info ----------)
endif

ifeq ($(MFEM_CXX),clang++)
   MFEM_CXXFLAGS += -march=native
   # MFEM_CXXFLAGS += -std=c++03
   MFEM_CXXFLAGS += -std=c++11
   MFEM_CXXFLAGS += -pedantic -Wall

   MFEM_CXXFLAGS += -fcolor-diagnostics
   MFEM_CXXFLAGS += -fvectorize
   MFEM_CXXFLAGS += -fslp-vectorize
   MFEM_CXXFLAGS += -fslp-vectorize-aggressive
   MFEM_CXXFLAGS += -ffp-contract=fast

   $(info ----------)
   $(info using clang++ options: $(MFEM_CXXFLAGS))
   $(info ----------)
endif

SEQ_MINIAPPS = ex1
ifeq ($(MFEM_USE_MPI),NO)
   MINIAPPS = $(SEQ_MINIAPPS)
else
   MINIAPPS = $(SEQ_MINIAPPS) ex1p
endif

.PHONY: all clean style

# Remove built-in rule
%: %.cpp

# Replace the default implicit rule for *.cpp files
%: %.cpp $(CONFIG_MK) $(MFEM_LIB_FILE)
	$(MFEM_CXX) $(MFEM_FLAGS) $(@).cpp -o $@ $(MFEM_LIBS)

all: $(MINIAPPS)

# Generate an error message if the MFEM library is not built and exit
$(CONFIG_MK) $(MFEM_LIB_FILE):
	$(error The MFEM library is not built)

clean:
	rm -f *.o *~ ex1
	rm -f refined.mesh sol.*
	rm -rf *.dSYM *.TVD.*breakpoints
