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

   HypreParMatrix *A;
   if (fbfi.Size() == 0)
   {
      // construct a parallel block-diagonal wrapper matrix A based on m
      A = new HypreParMatrix(pfes->GlobalVSize(), pfes->GetDofOffsets(), m);
   }
   else
   {
      // handle the case when 'm' contains offdiagonal
      int  lvsize = pfes->GetVSize();
      int *face_nbr_glob_ldof = pfes->GetFaceNbrGlobalDofMap();
      int ldof_offset = pfes->GetMyDofOffset();

      Array<int> glob_J(m->NumNonZeroElems());
      int *J = m->GetJ();
      for (int i = 0; i < glob_J.Size(); i++)
         if (J[i] < lvsize)
            glob_J[i] = J[i] + ldof_offset;
         else
            glob_J[i] = face_nbr_glob_ldof[J[i] - lvsize];

      A = new HypreParMatrix(pfes->GetComm(), lvsize, pfes->GlobalVSize(),
                             pfes->GlobalVSize(), m->GetI(), glob_J, m->GetData(),
                             pfes->GetDofOffsets(), pfes->GetDofOffsets());
   }

   HypreParMatrix *rap = RAP(A, pfes->Dof_TrueDof_Matrix());

   delete A;

   return rap;
}

void ParBilinearForm::AssembleSharedFaces(int skip_zeros)
{
   ParMesh *pmesh = pfes->GetParMesh();
   FaceElementTransformations *T;
   Array<int> vdofs1, vdofs2, vdofs_all;
   DenseMatrix elemmat;

   int nfaces = pmesh->GetNSharedFaces();
   for (int i = 0; i < nfaces; i++)
   {
      T = pmesh->GetSharedFaceTransformations(i);
      pfes->GetElementVDofs(T->Elem1No, vdofs1);
      pfes->GetFaceNbrElementVDofs(T->Elem2No, vdofs2);
      vdofs1.Copy(vdofs_all);
      for (int j = 0; j < vdofs2.Size(); j++)
         vdofs2[j] += size;
      vdofs_all.Append(vdofs2);
      for (int k = 0; k < fbfi.Size(); k++)
      {
         fbfi[k]->AssembleFaceMatrix(*pfes->GetFE(T->Elem1No),
                                     *pfes->GetFaceNbrFE(T->Elem2No),
                                     *T, elemmat);
         if (keep_nbr_block)
            mat->AddSubMatrix(vdofs_all, vdofs_all, elemmat, skip_zeros);
         else
            mat->AddSubMatrix(vdofs1, vdofs_all, elemmat, skip_zeros);
      }
   }
}

void ParBilinearForm::Assemble(int skip_zeros)
{
   if (mat == NULL && fbfi.Size() > 0)
   {
      pfes->ExchangeFaceNbrData();
      int nbr_size = pfes->GetFaceNbrVSize();
      if (keep_nbr_block)
         mat = new SparseMatrix(size + nbr_size, size + nbr_size);
      else
         mat = new SparseMatrix(size, size + nbr_size);
   }

   BilinearForm::Assemble(skip_zeros);

   if (fbfi.Size() > 0)
      AssembleSharedFaces(skip_zeros);
}

HypreParMatrix *ParDiscreteLinearOperator::ParallelAssemble(SparseMatrix *m)
{
   if (m == NULL)
      return NULL;

   int *I = m->GetI();
   int *J = m->GetJ();
   double *data = m->GetData();

   // remap to tdof local row and tdof global column indices
   SparseMatrix local(range_fes->TrueVSize(), domain_fes->GlobalTrueVSize());
   for (int i = 0; i < m->Size(); i++)
   {
      int lti = range_fes->GetLocalTDofNumber(i);
      if (lti >= 0)
         for (int j = I[i]; j < I[i+1]; j++)
            local.Set(lti, domain_fes->GetGlobalTDofNumber(J[j]), data[j]);
   }
   local.Finalize();

   // construct and return a global ParCSR matrix by splitting the local matrix
   // into diag and offd parts
   return new HypreParMatrix(range_fes->GetComm(),
                             range_fes->TrueVSize(),
                             range_fes->GlobalTrueVSize(),
                             domain_fes->GlobalTrueVSize(),
                             local.GetI(), local.GetJ(), local.GetData(),
                             range_fes->GetTrueDofOffsets(),
                             domain_fes->GetTrueDofOffsets());
}

void ParDiscreteLinearOperator::GetParBlocks(Array2D<HypreParMatrix *> &blocks) const
{
   int rdim = range_fes->GetVDim();
   int ddim = domain_fes->GetVDim();

   blocks.SetSize(rdim, ddim);

   int i, j, n;

   // construct the scalar versions of the row/coll offset arrays
   int *row_starts, *col_starts;
   if (HYPRE_AssumedPartitionCheck())
      n = 2;
   else
      n = range_fes->GetNRanks()+1;
   row_starts = new int[n];
   col_starts = new int[n];
   for (i = 0; i < n; i++)
   {
      row_starts[i] = (range_fes->GetTrueDofOffsets())[i] / rdim;
      col_starts[i] = (domain_fes->GetTrueDofOffsets())[i] / ddim;
   }

   Array2D<SparseMatrix *> lblocks;
   GetBlocks(lblocks);

   for (int bi = 0; bi < rdim; bi++)
      for (int bj = 0; bj < ddim; bj++)
      {
         int *I = lblocks(bi,bj)->GetI();
         int *J = lblocks(bi,bj)->GetJ();
         double *data = lblocks(bi,bj)->GetData();

         // remap to tdof local row and tdof global column indices
         SparseMatrix local(range_fes->TrueVSize()/rdim,
                            domain_fes->GlobalTrueVSize()/ddim);
         for (i = 0; i < lblocks(bi,bj)->Size(); i++)
         {
            int lti = range_fes->GetLocalTDofNumber(i);
            if (lti >= 0)
               for (j = I[i]; j < I[i+1]; j++)
                  local.Set(lti,
                            domain_fes->GetGlobalScalarTDofNumber(J[j]),
                            data[j]);
         }
         local.Finalize();

         delete lblocks(bi,bj);

         // construct and return a global ParCSR matrix by splitting the local
         // matrix into diag and offd parts
         blocks(bi,bj) = new HypreParMatrix(range_fes->GetComm(),
                                            range_fes->TrueVSize()/rdim,
                                            range_fes->GlobalTrueVSize()/rdim,
                                            domain_fes->GlobalTrueVSize()/ddim,
                                            local.GetI(), local.GetJ(), local.GetData(),
                                            row_starts, col_starts);
      }

   delete row_starts;
   delete col_starts;
}

#endif
