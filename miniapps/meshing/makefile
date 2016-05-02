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

SEQ_MINIAPPS = mobius-strip klein-bottle mesh-explorer
PAR_MINIAPPS =
ifeq ($(MFEM_USE_MPI),NO)
   MINIAPPS = $(SEQ_MINIAPPS)
else
   MINIAPPS = $(PAR_MINIAPPS) $(SEQ_MINIAPPS)
endif

.PHONY: all clean style

# Remove built-in rule
%: %.cpp

# Replace the default implicit rule for *.cpp files
%: %.cpp $(CONFIG_MK) $(MFEM_LIB_FILE)
	$(MFEM_CXX) $(MFEM_FLAGS) $(@).cpp -o $@ $(MFEM_LIBS)

all: $(MINIAPPS)

# Test default runs of the miniapps
mfem-meshing-test = \
   $(if $(shell if ($(2)$(1) -no-vis > /dev/null; test -a $(1).mesh); then echo "ok"; fi), \
   echo "   $(3)$(2)$(1): OK", echo "   $(3)$(2)$(1): FAILED")

# Testing: Parallel vs. serial runs
%-test-par: %
	@$(call mfem-meshing-test, $<, mpirun -np 4, Meshing miniapp)
%-test-seq: %
	@$(call mfem-meshing-test, $<, , Meshing miniapp)

# Testing: Specific execution options
mesh-explorer-test-seq:
	

# Testing: What sets of tests to run in serial and parallel
test-par-YES: $(PAR_MINIAPPS:=-test-par) $(SEQ_MINIAPPS:=-test-seq)
test-par-NO:  $(SEQ_MINIAPPS:=-test-seq)
test:         test-par-$(MFEM_USE_MPI) clean-exec

# Generate an error message if the MFEM library is not built and exit
$(CONFIG_MK) $(MFEM_LIB_FILE):
	$(error The MFEM library is not built)

clean: clean-build clean-exec

clean-build:
	rm -f *.o *~ mobius-strip klein-bottle mesh-explorer
	rm -rf *.dSYM *.TVD.*breakpoints

clean-exec:
	@rm -f mobius-strip.mesh klein-bottle.mesh mesh-explorer.mesh
	@rm -f partitioning.txt
