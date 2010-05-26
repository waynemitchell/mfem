# Copyright (c) 2010,  Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# This file is part of the MFEM library.  See file COPYRIGHT for details.
#
# MFEM is free software; you can redistribute it and/or modify it under the
# terms of the GNU Lesser General Public License (as published by the Free
# Software Foundation) version 2.1 dated February 1999.

Help("""
       Type: 'scons' to build the production library,
             'scons debug=1' to build the debug version.
       """)


CCOPTS     = '-Wall'
DEBUG_OPTS = '-g -DMFEM_DEBUG'
OPTIM_OPTS = '-O3'
USE_IOS_FMTFLAGS_DEF 	= '-DMFEM_IOS_FMTFLAGS'
USE_MEMALLOC_DEF 	= '-DMFEM_USE_MEMALLOC'
USE_LAPACK_DEF 		= '-DMFEM_USE_LAPACK'

env = Environment()
env.Append(CCFLAGS = [CCOPTS, OPTIM_OPTS, USE_IOS_FMTFLAGS_DEF, USE_MEMALLOC_DEF, USE_LAPACK_DEF])

debug = ARGUMENTS.get('debug', 0)
if int(debug):
    env.Append(CCFLAGS = DEBUG_OPTS)

env.Append(CPPPATH = ['.', 'fem', 'general', 'linalg', 'mesh'])

general_src = Glob('general/*.cpp')
fem_src = Glob('fem/*.cpp')
linalg_src = Glob('linalg/*.cpp')
mesh_src = Glob('mesh/*.cpp')

env.Library('mfem',[fem_src,general_src,linalg_src,mesh_src])
