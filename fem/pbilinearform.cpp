// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#include "../config/config.hpp"

#ifdef MFEM_USE_MPI

#include "fem.hpp"
#include "../general/sort_pairs.hpp"

namespace mfem
{

void ParBilinearForm::pAllocMat()
{
   int nbr_size = pfes->GetFaceNbrVSize();

   if (precompute_sparsity == 0 || fes->GetVDim() > 1)
   {
      if (keep_nbr_block)
      {
         mat = new SparseMatrix(height + nbr_size, width + nbr_size);
      }
      else
      {
         mat = new SparseMatrix(height, width + nbr_size);
      }
      return;
   }

   // the sparsity pattern is defined from the map: face->element->dof
   fes->BuildElementToDofTable();
   const Table &lelem_ldof = fes->GetElementToDofTable(); // <-- dofs
   const Table &nelem_ndof = pfes->face_nbr_element_dof; // <-- vdofs
   Table elem_dof; // element + nbr-element <---> dof
   if (nbr_size > 0)
   {
      // merge lelem_ldof and nelem_ndof into elem_dof
      int s1 = lelem_ldof.Size(), s2 = nelem_ndof.Size();
      const int *I1 = lelem_ldof.GetI(), *J1 = lelem_ldof.GetJ();
      const int *I2 = nelem_ndof.GetI(), *J2 = nelem_ndof.GetJ();
      const int nnz1 = I1[s1], nnz2 = I2[s2];

      elem_dof.SetDims(s1 + s2, nnz1 + nnz2);

      int *I = elem_dof.GetI(), *J = elem_dof.GetJ();
      for (int i = 0; i <= s1; i++)
      {
         I[i] = I1[i];
      }
      for (int j = 0; j < nnz1; j++)
      {
         J[j] = J1[j];
      }
      for (int i = 0; i <= s2; i++)
      {
         I[s1+i] = I2[i] + nnz1;
      }
      for (int j = 0; j < nnz2; j++)
      {
         J[nnz1+j] = J2[j] + height;
      }
   }
   //   dof_elem x  elem_face x face_elem x elem_dof  (keep_nbr_block = true)
   // ldof_lelem x lelem_face x face_elem x elem_dof  (keep_nbr_block = false)
   Table dof_dof;
   {
      Table face_dof; // face_elem x elem_dof
      {
         Table *face_elem = pfes->GetParMesh()->GetFaceToAllElementTable();
         if (nbr_size > 0)
         {
            mfem::Mult(*face_elem, elem_dof, face_dof);
         }
         else
         {
            mfem::Mult(*face_elem, lelem_ldof, face_dof);
         }
         delete face_elem;
         if (nbr_size > 0)
         {
            elem_dof.Clear();
         }
      }

      if (keep_nbr_block)
      {
         Table dof_face;
         Transpose(face_dof, dof_face, height + nbr_size);
         mfem::Mult(dof_face, face_dof, dof_dof);
      }
      else
      {
         Table ldof_face;
         {
            Table face_ldof;
            Table *face_lelem = fes->GetMesh()->GetFaceToElementTable();
            mfem::Mult(*face_lelem, lelem_ldof, face_ldof);
            delete face_lelem;
            Transpose(face_ldof, ldof_face, height);
         }
         mfem::Mult(ldof_face, face_dof, dof_dof);
      }
   }

   int *I = dof_dof.GetI();
   int *J = dof_dof.GetJ();
   int nrows = dof_dof.Size();
   double *data = new double[I[nrows]];

   mat = new SparseMatrix(I, J, data, nrows, height + nbr_size);
   *mat = 0.0;

   dof_dof.LoseData();
}

HypreParMatrix *ParBilinearForm::ParallelAssemble(SparseMatrix *m)
{
   if (m == NULL)
   {
      return NULL;
   }

   HypreParMatrix *A;
   if (fbfi.Size() == 0)
   {
      // construct a parallel block-diagonal wrapper matrix A based on m
      A = new HypreParMatrix(pfes->GetComm(),
                             pfes->GlobalVSize(), pfes->GetDofOffsets(), m);
   }
   else
   {
      // handle the case when 'm' contains offdiagonal
      int lvsize = pfes->GetVSize();
      const HYPRE_Int *face_nbr_glob_ldof = pfes->GetFaceNbrGlobalDofMap();
      HYPRE_Int ldof_offset = pfes->GetMyDofOffset();

      Array<HYPRE_Int> glob_J(m->NumNonZeroElems());
      int *J = m->GetJ();
      for (int i = 0; i < glob_J.Size(); i++)
         if (J[i] < lvsize)
         {
            glob_J[i] = J[i] + ldof_offset;
         }
         else
         {
            glob_J[i] = face_nbr_glob_ldof[J[i] - lvsize];
         }

      A = new HypreParMatrix(pfes->GetComm(), lvsize, pfes->GlobalVSize(),
                             pfes->GlobalVSize(), m->GetI(), glob_J,
                             m->GetData(), pfes->GetDofOffsets(),
                             pfes->GetDofOffsets());
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
      {
         vdofs2[j] += height;
      }
      vdofs_all.Append(vdofs2);
      for (int k = 0; k < fbfi.Size(); k++)
      {
         fbfi[k]->AssembleFaceMatrix(*pfes->GetFE(T->Elem1No),
                                     *pfes->GetFaceNbrFE(T->Elem2No),
                                     *T, elemmat);
         if (keep_nbr_block)
         {
            mat->AddSubMatrix(vdofs_all, vdofs_all, elemmat, skip_zeros);
         }
         else
         {
            mat->AddSubMatrix(vdofs1, vdofs_all, elemmat, skip_zeros);
         }
      }
   }
}

void ParBilinearForm::Assemble(int skip_zeros)
{
   if (mat == NULL && fbfi.Size() > 0)
   {
      pfes->ExchangeFaceNbrData();
      pAllocMat();
   }

   BilinearForm::Assemble(skip_zeros);

   if (fbfi.Size() > 0)
   {
      AssembleSharedFaces(skip_zeros);
   }
}

void ParBilinearForm::TrueAddMult(const Vector &x, Vector &y, const double a)
const
{
   MFEM_VERIFY(fbfi.Size() == 0, "the case of interior face integrators is not"
               " implemented");

   if (X.ParFESpace() != pfes)
   {
      X.Update(pfes);
      Y.Update(pfes);
   }

   X.Distribute(&x);
   mat->Mult(X, Y);
   pfes->Dof_TrueDof_Matrix()->MultTranspose(a, Y, 1.0, y);
}


HypreParMatrix *ParDiscreteLinearOperator::ParallelAssemble(
   SparseMatrix *m, HYPRE_Int *true_row_starts, HYPRE_Int *true_col_starts,
   bool scalar) const
{
   // For a vector space (vdim > 1) and scalar == true, the ordering is assumed
   // to be Ordering::byNODES, e.g. when using GetLocalTDofNumber().

   if (m == NULL) { return NULL; }

   int *I = m->GetI();
   int *J = m->GetJ();
   double *data = m->GetData();

   int rdim = scalar ? range_fes->GetVDim() : 1;
   int ddim = scalar ? domain_fes->GetVDim() : 1;

   int  range_ldofs =  range_fes->GetVSize()/rdim; // == m->Height()
   int domain_ldofs = domain_fes->GetVSize()/ddim; // == m->Width()

   int num_rows = range_fes->TrueVSize()/rdim;

   HYPRE_Int *diag_i, *diag_j, *offd_i, *offd_j;
   double *diag_data, *offd_data;

   int offd_num_cols = 0;
   Array<int> col_ldof_marker(domain_ldofs);
   col_ldof_marker = -1;

   diag_i = new HYPRE_Int[num_rows+1];
   offd_i = new HYPRE_Int[num_rows+1];
   // count the number of entries in each row of diag and offd;
   // at the same time, mark and count the columns used by offd
   for (int i = 0; i <= num_rows; i++)
   {
      diag_i[i] = 0;
      offd_i[i] = 0;
   }
   for (int i = 0; i < range_ldofs; i++)
   {
      int lti = range_fes->GetLocalTDofNumber(i);
      if (lti >= 0)
      {
         for (int j = I[i]; j < I[i+1]; j++)
         {
            int k = J[j];
            int ltk = domain_fes->GetLocalTDofNumber(k);
            if (ltk >= 0)
            {
               diag_i[lti]++;
            }
            else
            {
               offd_i[lti]++;
               if (col_ldof_marker[k] < 0)
               {
                  col_ldof_marker[k] = 1;
                  offd_num_cols++;
               }
            }
         }
      }
   }
   // define offd_col_map -- the local-to-global column mapping for offd
   // define col_ldof_marker -- the map from domain ldof to offd column index
   HYPRE_Int *offd_col_map = new HYPRE_Int[offd_num_cols];
   {
      Array<Pair<HYPRE_Int, int> > cmap_j_offd(offd_num_cols);
      int edof_counter = 0;
      for (int i = 0; i < domain_ldofs; i++)
      {
         if (col_ldof_marker[i] > 0)
         {
            cmap_j_offd[edof_counter].one =
               scalar ?
               domain_fes->GetGlobalScalarTDofNumber(i) :
               domain_fes->GetGlobalTDofNumber(i);
            cmap_j_offd[edof_counter].two = i;
            edof_counter++;
         }
      }
      SortPairs<HYPRE_Int, int>(cmap_j_offd, offd_num_cols);
      for (int i = 0; i < offd_num_cols; i++)
      {
         offd_col_map[i] = cmap_j_offd[i].one;
         // col_ldof_marker is the inverse of the map i -> cmap_j_offd[i].two
         col_ldof_marker[cmap_j_offd[i].two] = i;
      }
   }
   // in diag_i and offd_i, convert row sizes into row offsets
   HYPRE_Int diag_offset = 0, offd_offset = 0;
   for (int i = 0; i < num_rows; i++)
   {
      HYPRE_Int diag_row_size = diag_i[i];
      HYPRE_Int offd_row_size = offd_i[i];
      diag_i[i] = diag_offset;
      offd_i[i] = offd_offset;
      diag_offset += diag_row_size;
      offd_offset += offd_row_size;
   }
   diag_i[num_rows] = diag_offset;
   offd_i[num_rows] = offd_offset;
   // allocate the j and data arrays of diag and offd
   diag_j = new HYPRE_Int[diag_offset];
   diag_data = new double[diag_offset];
   offd_j = new HYPRE_Int[offd_offset];
   offd_data = new double[offd_offset];
   // set the entries of the j and data arrays of diag and offd
   for (int i = 0; i < range_ldofs; i++)
   {
      int lti = range_fes->GetLocalTDofNumber(i);
      if (lti >= 0)
      {
         for (int j = I[i]; j < I[i+1]; j++)
         {
            int k = J[j];
            int ltk = domain_fes->GetLocalTDofNumber(k);
            if (ltk >= 0)
            {
               diag_j[diag_i[lti]] = ltk;
               diag_data[diag_i[lti]] = data[j];
               diag_i[lti]++;
            }
            else
            {
               offd_j[offd_i[lti]] = col_ldof_marker[k];
               offd_data[offd_i[lti]] = data[j];
               offd_i[lti]++;
            }
         }
      }
   }
   // shift back the i arrays of diag and offd
   diag_offset = offd_offset = 0;
   for (int i = 0; i < num_rows; i++)
   {
      Swap(diag_i[i], diag_offset);
      Swap(offd_i[i], offd_offset);
   }

   HypreParMatrix *glob_m =
      new HypreParMatrix(range_fes->GetComm(),
                         range_fes->GlobalTrueVSize()/rdim,
                         domain_fes->GlobalTrueVSize()/ddim,
                         true_row_starts, true_col_starts,
                         diag_i, diag_j, diag_data,
                         offd_i, offd_j, offd_data,
                         offd_num_cols, offd_col_map);

   return glob_m;
}

void ParDiscreteLinearOperator::GetParBlocks(Array2D<HypreParMatrix *> &blocks)
const
{
   int rdim = range_fes->GetVDim();
   int ddim = domain_fes->GetVDim();

   blocks.SetSize(rdim, ddim);

   // construct the scalar versions of the row/col offset arrays
   int n = HYPRE_AssumedPartitionCheck() ? 2 : range_fes->GetNRanks()+1;
   HYPRE_Int *row_starts = new HYPRE_Int[n];
   HYPRE_Int *col_starts = new HYPRE_Int[n];
   for (int i = 0; i < n; i++)
   {
      row_starts[i] = (range_fes->GetTrueDofOffsets())[i] / rdim;
      col_starts[i] = (domain_fes->GetTrueDofOffsets())[i] / ddim;
   }

   Array2D<SparseMatrix *> lblocks;
   GetBlocks(lblocks);

   for (int bi = 0; bi < rdim; bi++)
      for (int bj = 0; bj < ddim; bj++)
      {
         blocks(bi,bj) = ParallelAssemble(lblocks(bi,bj),
                                          row_starts, col_starts, true);

         if (bi == 0 && bj == 0)
         {
            // transfer ownership of row_starts and col_starts to blocks(0,0)
            // (since ownership is given to hypre, the arrays need to be
            // re-allocated)
            blocks(0,0)->CopyRowStarts();
            blocks(0,0)->CopyColStarts();
            delete [] row_starts;
            delete [] col_starts;
            row_starts = blocks(0,0)->GetRowStarts();
            col_starts = blocks(0,0)->GetColStarts();
         }

         delete lblocks(bi, bj);
      }
}

HypreParMatrix *ParMixedBilinearForm::ParallelAssemble()
{
   // construct the block-diagonal matrix A
   HypreParMatrix *A =
      new HypreParMatrix(trial_pfes->GetComm(),
                         test_pfes->GlobalVSize(),
                         trial_pfes->GlobalVSize(),
                         test_pfes->GetDofOffsets(),
                         trial_pfes->GetDofOffsets(),
                         mat);

   HypreParMatrix *rap = RAP(test_pfes->Dof_TrueDof_Matrix(), A,
                             trial_pfes->Dof_TrueDof_Matrix());

   delete A;

   return rap;
}

/// Compute y += a (P^t A P) x, where x and y are vectors on the true dofs
void ParMixedBilinearForm::TrueAddMult(const Vector &x, Vector &y,
                                       const double a) const
{
   if (X.ParFESpace() != trial_pfes)
   {
      X.Update(trial_pfes);
      Y.Update(test_pfes);
   }

   X.Distribute(&x);
   mat->Mult(X, Y);
   test_pfes->Dof_TrueDof_Matrix()->MultTranspose(a, Y, 1.0, y);
}

}

#endif
