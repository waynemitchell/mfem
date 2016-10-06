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
TEST_MK = $(MFEM_DIR)/config/test.mk
# Use the MFEM install directory
# MFEM_DIR = ../../mfem
# CONFIG_MK = $(MFEM_DIR)/config.mk
# TEST_MK = $(MFEM_DIR)/test.mk

MFEM_LIB_FILE = mfem_is_not_built
-include $(CONFIG_MK)

SEQ_MINIAPPS = display-basis
PAR_MINIAPPS =
ifeq ($(MFEM_USE_MPI),NO)
   MINIAPPS = $(SEQ_MINIAPPS)
else
   MINIAPPS = $(PAR_MINIAPPS) $(SEQ_MINIAPPS)
endif

.PHONY: all clean clean-build clean-exec
.PRECIOUS: %.o

COMMON_HPP=../common/fem_extras.hpp ../common/mesh_extras.hpp
COMMON_O=$(COMMON_HPP:.hpp=.o)

# Remove built-in rule
%: %.cpp

# Replace the default implicit rule for *.cpp files
%: %.cpp $(COMMON_O) $(MFEM_LIB_FILE) $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) $< -o $@ $(COMMON_O) $(MFEM_LIBS)

# Replace the default implicit rule for *.cpp files
%.o: %.cpp %.hpp $(COMMON_HPP) $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) -c $(<) -o $(@)

all: $(MINIAPPS)

MFEM_TESTS = MINIAPPS
include $(TEST_MK)

# Testing: Parallel vs. serial runs
%-test-par: %
	@$(call mfem-test,$<, mpirun -np 4, Tools miniapp)
%-test-seq: %
	@$(call mfem-test,$<,, Tools miniapp)

# Testing: Specific execution options
# Do not test display-basis:
display-basis-test-seq:
	@true

# Testing: "test" target and mfem-test* variables are defined in config/test.mk

# Generate an error message if the MFEM library is not built and exit
$(MFEM_LIB_FILE):
	$(error The MFEM library is not built)

clean: clean-build clean-exec

clean-build:
	rm -f *.o *~ $(SEQ_MINIAPPS) $(PAR_MINIAPPS)
	rm -rf *.dSYM *.TVD.*breakpoints

clean-exec:
	@true
