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

MFEM_DIR = ..
CONFIG_MK = $(MFEM_DIR)/config/config.mk
ifneq (clean,$(MAKECMDGOALS))
   -include $(CONFIG_MK)
endif

ifeq ($(MFEM_USE_MPI),NO)
   EXAMPLES = ex1 ex2 ex3 ex4 ex5 ex6 ex7 ex8
else
   EXAMPLES = ex1p ex2p ex3p ex4p ex5p ex7p ex8p
endif

.PHONY: all clean

# Remove built-in rule
%: %.cpp

# Replace the default implicit rule for *.cpp files
%: %.cpp $(CONFIG_MK) $(MFEM_LIB_FILE)
	$(MFEM_CXX) $(MFEM_FLAGS) $(@).cpp -o $@ $(MFEM_LIBS)

all: $(EXAMPLES)

# Generate an error message if the MFEM library is not built and exit
$(CONFIG_MK) $(MFEM_LIB_FILE):
	$(error The MFEM library is not built)

clean:
	rm -f *.o *~ ex1 ex1p ex2 ex2p ex3 ex3p ex4 ex4p ex5 ex5p ex6 ex7 ex7p
	rm -f ex8 ex8p
	rm -f refined.mesh displaced.mesh mesh.* ex5.mesh sphere_refined.*
	rm -f sol.* sol_u.* sol_p.*
