// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.googlecode.com.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#include <cstdlib>
#include <iostream>
#include "error.hpp"

#ifdef MFEM_USE_MPI
#include <mpi.h>
#endif

MFEM_NAMESPACE_BEGIN();

void mfem_error (const char *msg)
{
   if (msg)
   {
      // NOTE: This endl also flushes the I/O stream, which can be a very bad
      // thing if all your processors try to do it at the same time.
      std::cerr << msg << std::endl;
   }
#ifdef MFEM_USE_MPI
   MPI_Abort(MPI_COMM_WORLD, 1);
#else
   std::abort(); // force crash by calling abort
#endif

}

MFEM_NAMESPACE_END();
