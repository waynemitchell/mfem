# Copyright (c) 2010,  Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# This file is part of the MFEM library.  See file COPYRIGHT for details.
#
# MFEM is free software; you can redistribute it and/or modify it under the
# terms of the GNU Lesser General Public License (as published by the Free
# Software Foundation) version 2.1 dated February 1999.

Help("""
       Type: 'scons' to build the production library,
             'scons -c' to clean the build,
             'scons debug=1' to build the debug version.
       """)

env = Environment()

CC_OPTS    = '-O3'
DEBUG_OPTS = '-g -DMFEM_DEBUG -Wall'

# MFEM-specific options
env.Append(CPPDEFINES = ['MFEM_USE_MEMALLOC'])

# Debug options
debug = ARGUMENTS.get('debug', 0)
if int(debug):
   env.Append(CCFLAGS = DEBUG_OPTS)
else:
   env.Append(CCFLAGS = CC_OPTS)

conf = Configure(env)

# Check for LAPACK
if conf.CheckLib('lapack', 'dsyevr_'):
   env.Append(CPPDEFINES = ['MFEM_USE_LAPACK'])
   print 'Using LAPACK'
else:
   print 'Did not find LAPACK, continuing without it'

env = conf.Finish()

env.Append(CPPPATH = ['.', 'general', 'linalg', 'mesh', 'fem'])

# general, linalg, mesh and fem sources
general_src = Glob('general/*.cpp')
linalg_src = Glob('linalg/*.cpp')
mesh_src = Glob('mesh/*.cpp')
fem_src = Glob('fem/*.cpp')

# libmfem.a library
env.Library('mfem',[general_src,linalg_src,mesh_src,fem_src])
