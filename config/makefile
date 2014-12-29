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

# List of all defines that may be enabled in config.hpp and config.mk.
# The actual value is set in ../makefile and exported to here.
# MFEM_DEFINES = ...

# Extract the enabled (=YES) defines
DEFINES := $(foreach var,$(MFEM_DEFINES),$(var).$($(var)))
DEFINES := $(DEFINES:%.YES=%)
DEFINES := $(DEFINES:%.NO=)

# List of variables to set in config.mk using their current values:
# The actual value is set in ../makefile and exported to here.
# MFEM_CONFIG_VARS = ...

# Debug output
# $(foreach def,$(MFEM_DEFINES),$(info $(def) = $(value $(def))))
# $(foreach var,$(MFEM_CONFIG_VARS),$(info $(var) = $(value $(var))))

# Output path + file name for the config header
CONFIG_HPP = ../config.hpp

# Output path + file name for the config makefile
CONFIG_MK = config.mk

.PHONY: all header makefile

all: header makefile

header:
	$(call mfem-info, Writing $(CONFIG_HPP) ...)
	@cp -f config.hpp.in $(CONFIG_HPP) && \
	for def in $(DEFINES); do \
	   sed "s|// \(#define $${def}\)|\1|" -i $(CONFIG_HPP); done

# Using the values of the shell variables listed in the shell variables
# MFEM_DEFINES and MFEM_CONFIG_VARS
makefile:
	$(call mfem-info, Writing $(CONFIG_MK) ...)
	@cp -f config.mk.in $(CONFIG_MK) && \
	for ref in $${MFEM_DEFINES} $${MFEM_CONFIG_VARS}; do \
	   # echo "@$${ref}@   --->   $${!ref}" && \
	   sed "s#@$${ref}@#$${!ref}#g" -i $(CONFIG_MK); done && \
	sed -e 's/@\([^@]*\)@/$$(\1)/g' -e 's/ *$$//g' -i $(CONFIG_MK)
