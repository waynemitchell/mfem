# -----------------------------------------------------------------
# $Revision: 4456 $
# $Date: 2015-03-28 20:22:03 -0700 (Sat, 28 Mar 2015) $
# -----------------------------------------------------------------
# Programmer: Radu Serban @ LLNL
# -----------------------------------------------------------------
# Copyright (c) 2002, The Regents of the University of California.
# Produced at the Lawrence Livermore National Laboratory.
# All rights reserved.
# For details, see the LICENSE file.
# -----------------------------------------------------------------
# Makefile for  serial examples.
#
# This file is generated from a template using various variables
# set at configuration time. It can be used as a template for
# other user Makefiles.
#
# Note: if the solver was successfully configured with Blas/Lapack
# support, the Blas/Lapack libraries are specified through the 
# variable LIBRARIES_BL. Otherwise, this variable should contain
# an empty string. We include LIBRARIES_BL in the link line for
# all examples, whether they use the Lapack module or not, to
# address the case in which the SUNDIALS libraries are shared 
# objects. In that case, the solver library references Lapack 
# symbols which must be always resolved by linking against the
# Blas/Lapack libraries. If only static SUNDIALS libraries have 
# been built, it is not required to link the Blas/Lapack libraries
# for examples that do not use that module...
# -----------------------------------------------------------------

SHELL = sh

prefix       = /home/sexton10/Summer15/sundials_svn_install
exec_prefix  = $(prefix)
includedir   = $(prefix)/include
libdir       = $(prefix)/lib

# ------------------------------------------------------------------
MFEM_DIR     = /home/sexton10/Summer15/mfem-dev
CONFIG_MK    = $(MFEM_DIR)/config/config.mk
ifneq (clean,$(MAKECMDGOALS))
   -include $(CONFIG_MK)
endif
# ------------------------------------------------------------------

CPP      = /usr/bin/cc
CPPFLAGS = -O3 -DNDEBUG -g -Wno-write-strings
CC       = /usr/bin/cc
CFLAGS   = -O3 -DNDEBUG -g -Wno-write-strings
LDFLAGS  = 
LIBS     =  -lm /usr/lib64/librt.so 

LINKFLAGS = -Wl,-rpath,$(libdir)

INCLUDES = -I${includedir}
LIBRARIES = -lsundials_arkode -lsundials_cvode -lsundials_nvecserial -lsundials_nvecserial ${LIBS}

EXAMPLES =
EXAMPLE_DEPENDENCIES = 
#EXAMPLES =  test_nvector_serial 
#EXAMPLES_DEPENDENCIES =  test_nvector sundials_nvector 

OBJECTS = ${EXAMPLES:=.o}
OBJECTS_DEPENDENCIES = ${EXAMPLES_DEPENDENCIES:=.o}
# -----------------------------------------------------------------------------------------

.SUFFIXES : .o .c

.c.o :
	${CC} ${CPPFLAGS} ${CFLAGS} ${INCLUDES} -c $<

# -----------------------------------------------------------------------------------------

all: ${OBJECTS} ex9_sundials ex9_arkode_rk ex9_mfem_rk ex9_stripped
	@for i in ${EXAMPLES} ; do \
	  echo "${CC} -o $${i} $${i}.o ${OBJECTS_DEPENDENCIES} ${CFLAGS} ${LDFLAGS} ${INCLUDES} -L${libdir} ${LIBRARIES} ${LINKFLAGS}" ; \
	  ${CC} -o $${i} $${i}.o ${OBJECTS_DEPENDENCIES} ${CFLAGS} ${LDFLAGS} ${INCLUDES} -L${libdir} ${LIBRARIES} ${LINKFLAGS} ; \
	done

${OBJECTS}: ${OBJECTS_DEPENDENCIES}

# Add example compile
% : %.cpp $(CONFIG_MK) $(MFEM_LIB_FILE) sundials.cpp sundials.hpp Makefile
	$(MFEM_CXX) $(MFEM_FLAGS) -g -Wno-write-strings sundials.cpp $(@).cpp -o $@ $(MFEM_LIBS) ${LDFLAGS} ${INCLUDES} -L${libdir} ${LIBRARIES} ${LINKFLAGS}

clean:
	rm -f ${OBJECTS_DEPENDENCIES}
	rm -f ${OBJECTS}
	rm -f ${EXAMPLES}
	rm -f ex9.mesh ex9-mesh.* ex9-init.* ex9-final.*
	rm -f ex9 ex9_sundials ex9_cvode_euler ex9_mfem_euler ex9_arkode_rk ex9_mfem_rk ex9_stripped
# -----------------------------------------------------------------------------------------


