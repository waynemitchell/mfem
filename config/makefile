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

# Helper print-info function
mfem-info = $(if $(filter YES,$(VERBOSE)),$(info *** [info]$(1)),)

# The list of all defines that may be enabled in config.hpp and config.mk
# is kept in the MFEM_DEFINES variable, exported here from ../makefile

# Extract the enabled (!=NO) defines
DEFINES = $(shell for def in $${MFEM_DEFINES}; do\
   if [ "NO" != "$${!def}" ]; then echo $${def}; fi; done)

# The list of variables to set in config.mk using their current values
# is kept in the MFEM_CONFIG_VARS variable, exported here from ../makefile

# Debug output
# $(foreach def,$(MFEM_DEFINES),$(info $(def) = $(value $(def))))
# $(foreach var,$(MFEM_CONFIG_VARS),$(info $(var) = $(value $(var))))

# Output path + file name for the config header
CONFIG_HPP = config.hpp

# Output path + file name for the config makefile
CONFIG_MK = config.mk

ifeq ($(shell uname -s),Darwin)
   SED_I = sed -i ''
else
   SED_I = sed -i
endif

.PHONY: all header makefile

all: header makefile

header:
	$(call mfem-info, Writing $(CONFIG_HPP) ...)
	@cp -f config.hpp.in $(CONFIG_HPP) && \
	for def in $(DEFINES); do \
	   $(SED_I) -e "s|// \(#define $${def}\)|\1|" \
	            -e "s#@$${def}@#$${!def}#g" \
	            $(CONFIG_HPP); done

# The target below uses the values of the shell variables listed in
# the shell variables MFEM_DEFINES and MFEM_CONFIG_VARS
makefile:
	$(call mfem-info, Writing $(CONFIG_MK) ...)
	@cp -f config.mk.in $(CONFIG_MK) && \
	for ref in $${MFEM_DEFINES} $${MFEM_CONFIG_VARS}; do \
	   $(SED_I) "s#@$${ref}@#$${!ref}#g" $(CONFIG_MK); done && \
	$(SED_I) -e 's/@\([^@]*\)@/$$(\1)/g' -e 's/ *$$//g' $(CONFIG_MK)

clean:
	rm -f $(CONFIG_HPP) $(CONFIG_MK)
