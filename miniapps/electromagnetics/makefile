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

SEQ_MINIAPPS =
PAR_MINIAPPS = volta tesla
ifeq ($(MFEM_USE_MPI),NO)
   MINIAPPS = $(SEQ_MINIAPPS)
else
   MINIAPPS = $(PAR_MINIAPPS) $(SEQ_MINIAPPS)
endif

.PHONY: all clean

COMMON_HPP=../common/pfem_extras.hpp
COMMON_O=$(COMMON_HPP:.hpp=.o)

# Remove built-in rule
%: %.cpp

# Replace the default implicit rule for *.cpp files
%: %.cpp %_solver.o $(COMMON_O) $(CONFIG_MK) $(MFEM_LIB_FILE)
	$(MFEM_CXX) $(MFEM_FLAGS) $< -o $@ $@_solver.o $(COMMON_O) $(MFEM_LIBS)

# Remove built-in rule
%.o: %.cpp

# Replace the default implicit rule for *.cpp files
%.o: %.cpp %.hpp $(COMMON_HPP) $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) -c $(<) -o $(@)

all: $(MINIAPPS)

# Test default runs of the miniapps
mfem-test = \
   $(if $(shell if ($(2) $(1) -no-vis $(4) > /dev/null); then echo "ok"; fi), \
   echo "   $(3)$(2)$(1): OK", echo "   $(3)$(2)$(1): FAILED")

# Testing: Parallel vs. serial runs
%-test-par: %
	@$(call mfem-test, $<, mpirun -np 4, Electromagnetic miniapp)
%-test-seq: %
	@$(call mfem-test, $<, , Electromagnetic miniapp)

# Testing: Specific execution options
volta-test-par: volta
	@$(call mfem-test, $<, mpirun -np 4, Electromagnetic miniapp, -maxit 2 -dbcs 1 -dbcg -ds '0.0 0.0 0.0 0.2 8.0')
tesla-test-par: tesla
	@$(call mfem-test, $<, mpirun -np 4, Electromagnetic miniapp, -maxit 2 -cr '0 0 -0.2 0 0 0.2 0.2 0.4 1')

# Testing: What sets of tests to run in serial and parallel
test-par-YES: $(PAR_MINIAPPS:=-test-par) $(SEQ_MINIAPPS:=-test-seq)
test-par-NO:  $(SEQ_MINIAPPS:=-test-seq)
test:         test-par-$(MFEM_USE_MPI) clean-exec

# Generate an error message if the MFEM library is not built and exit
$(CONFIG_MK) $(MFEM_LIB_FILE):
	$(error The MFEM library is not built)

clean: clean-build clean-exec

clean-build:
	rm -f *.o *~ volta tesla
	rm -rf *.dSYM *.TVD.*breakpoints

clean-exec:
	@rm -rf Volta-AMR* Tesla-AMR*
