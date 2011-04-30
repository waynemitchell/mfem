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

#ifdef MFEM_USE_MPI

#include "fem.hpp"

HypreParMatrix *ParBilinearForm::ParallelAssemble(SparseMatrix *m)
{
   if (m == NULL)
      return NULL;

   // construct a parallel block-diagonal wrapper matrix A based on m
   HypreParMatrix *A = new HypreParMatrix(pfes->GlobalVSize(),
                                          pfes->GetDofOffsets(), m);

   HypreParMatrix *rap = RAP(A, pfes->Dof_TrueDof_Matrix());

   delete A;

   return rap;
}

#endif
