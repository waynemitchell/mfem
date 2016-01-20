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

#include "hybridization.hpp"
#include "gridfunc.hpp"

#include <map>
// #include <fstream> // uncomment for debuging: write C and P to file

namespace mfem
{

Hybridization::Hybridization(FiniteElementSpace *fespace,
                             FiniteElementSpace *mu_fespace)
   : fes(fespace), mu_fes(mu_fespace), c_bfi(NULL), Ct(NULL),
     Af_data(NULL), Af_ipiv(NULL)
{
}

Hybridization::~Hybridization()
{
   delete [] Af_ipiv;
   delete [] Af_data;
   delete H;
   delete Ct;
   delete c_bfi;
}

void Hybridization::Init(const Array<int> &ess_cdofs_marker)
{
   if (Ct) { return; }

   // Assemble the constraint matrix C

   // count the number of dofs in the discontinuous version of fes:
   const int NE = fes->GetNE();
   Array<int> vdofs, mu_vdofs;
   int num_hat_dofs = 0;
   hat_offsets.SetSize(NE+1);
   hat_offsets[0] = 0;
   for (int i = 0; i < NE; i++)
   {
      fes->GetElementVDofs(i, vdofs);
      num_hat_dofs += vdofs.Size();
      hat_offsets[i+1] = num_hat_dofs;
   }

   Ct = new SparseMatrix(num_hat_dofs, mu_fes->GetVSize());

   if (c_bfi)
   {
      const int skip_zeros = 1;
      DenseMatrix elmat;
      FaceElementTransformations *FTr;
      Mesh *mesh = fes->GetMesh();
      int num_faces = mesh->GetNumFaces();
      for (int i = 0; i < num_faces; i++)
      {
         FTr = mesh->GetInteriorFaceTransformations(i);
         if (!FTr) { continue; }

         int o1 = hat_offsets[FTr->Elem1No];
         int s1 = hat_offsets[FTr->Elem1No+1] - o1;
         int o2 = hat_offsets[FTr->Elem2No];
         int s2 = hat_offsets[FTr->Elem2No+1] - o2;
         vdofs.SetSize(s1 + s2);
         for (int j = 0; j < s1; j++)
         {
            vdofs[j] = o1 + j;
         }
         for (int j = 0; j < s2; j++)
         {
            vdofs[s1+j] = o2 + j;
         }
         mu_fes->GetFaceVDofs(i, mu_vdofs);
         c_bfi->AssembleFaceMatrix(*mu_fes->GetFaceElement(i),
                                   *fes->GetFE(FTr->Elem1No),
                                   *fes->GetFE(FTr->Elem2No),
                                   *FTr, elmat);
         // zero-out small elements in elmat
         {
            double eps = 1e-12 * elmat.MaxMaxNorm();
            for (int col = 0; col < elmat.Width(); col++)
            {
               for (int row = 0; row < elmat.Height(); row++)
               {
                  if (std::abs(elmat(row,col)) <= eps)
                  {
                     elmat(row,col) = 0.0;
                  }
               }
            }
         }
         Ct->AddSubMatrix(vdofs, mu_vdofs, elmat, skip_zeros);
      }
      Ct->Finalize(skip_zeros);
#if 0
      const SparseMatrix *mu_cP = mu_fes->GetConformingProlongation();
      if (mu_cP)
      {
         SparseMatrix *conf_Ct = mfem::Mult(*Ct, *mu_cP);
         delete Ct;
         Ct = conf_Ct;
      }
#endif
   }
   else
   {
      MFEM_ABORT("TODO: algebraic definition of C");
   }

#if 0
   // Debug: write C and P to file
   {
      std::ofstream C_file("C_matrix.txt");
      SparseMatrix *C = Transpose(*Ct);
      C->PrintMatlab(C_file);
      delete C;

      const SparseMatrix *P = fes->GetConformingProlongation();
      if (P)
      {
         std::ofstream P_file("P_matrix.txt");
         P->PrintMatlab(P_file);
      }
   }
#endif

   // Define the "free" (0) and "essential" (1) hat_dofs.
   // The "essential" hat_dofs are those that depend only on essential cdofs;
   // all other hat_dofs are "free".
   const SparseMatrix *cP = fes->GetConformingProlongation();
   hat_dofs_marker.SetSize(num_hat_dofs);
   for (int i = 0; i < NE; i++)
   {
      fes->GetElementVDofs(i, vdofs);
      FiniteElementSpace::AdjustVDofs(vdofs);
      for (int j = 0; j < vdofs.Size(); j++)
      {
         bool all_cols_marked;
         if (!cP)
         {
            all_cols_marked = ess_cdofs_marker[vdofs[j]];
         }
         else
         {
            const int *cols = cP->GetRowColumns(vdofs[j]);
            const int ncols = cP->RowSize(vdofs[j]);
            all_cols_marked = true;
            for (int k = 0; k < ncols; k++)
            {
               if (!ess_cdofs_marker[cols[k]])
               {
                  all_cols_marked = false;
                  break;
               }
            }
         }
         hat_dofs_marker[hat_offsets[i]+j] = all_cols_marked;
      }
   }
   // Split the "free" (0) hat_dofs into "internal" (0) or "boundary" (-1).
   // The "internal" hat_dofs are those "free" hat_dofs for which the
   // corresponding column in C is zero; otherwise the free hat_dof is
   // "boundary".
   for (int i = 0; i < num_hat_dofs; i++)
   {
      // skip "essential" hat_dofs and empty rows in Ct
      if (hat_dofs_marker[i] != 1 && Ct->RowSize(i) > 0)
      {
         hat_dofs_marker[i] = -1; // mark this hat_dof as "boundary"
      }
   }

   H = new SparseMatrix(Ct->Width());

   // Define Af_offsets and Af_f_offsets
   Af_offsets.SetSize(NE+1);
   Af_offsets[0] = 0;
   Af_f_offsets.SetSize(NE+1);
   Af_f_offsets[0] = 0;
   for (int i = 0; i < NE; i++)
   {
      int f_size = 0; // count the "free" hat_dofs in element i
      for (int j = hat_offsets[i]; j < hat_offsets[i+1]; j++)
      {
         if (hat_dofs_marker[j] != 1) { f_size++; }
      }
      Af_offsets[i+1] = Af_offsets[i] + f_size*f_size;
      Af_f_offsets[i+1] = Af_f_offsets[i] + f_size;
   }

   Af_data = new double[Af_offsets[NE]];
   Af_ipiv = new int[Af_f_offsets[NE]];

#ifdef MFEM_DEBUG
   // check that Ref = 0
   const SparseMatrix *cR = fes->GetConformingRestriction();
   if (!cR) { return; }
   Array<int> vdof_marker(fes->GetVSize()); // 0 - f, 1 - e
   vdof_marker = 0;
   for (int i = 0; i < NE; i++)
   {
      fes->GetElementVDofs(i, vdofs);
      FiniteElementSpace::AdjustVDofs(vdofs);
      for (int j = 0; j < vdofs.Size(); j++)
      {
         if (hat_dofs_marker[hat_offsets[i]+j] == 1) // "essential" hat dof
         {
            vdof_marker[vdofs[j]] = 1;
         }
      }
   }
   for (int cdof = 0; cdof < cR->Height(); cdof++)
   {
      if (!ess_cdofs_marker[cdof]) { continue; }

      const int ncols = cR->RowSize(cdof);
      const int *cols = cR->GetRowColumns(cdof);
      const double *vals = cR->GetRowEntries(cdof);
      for (int j = 0; j < ncols; j++)
      {
         if (std::abs(vals[j]) != 0.0 && vdof_marker[cols[j]] == 0)
         {
            MFEM_ABORT("Ref != 0");
         }
      }
   }
#endif
}

void Hybridization::GetIBDofs(
   int el, Array<int> &i_dofs, Array<int> &b_dofs) const
{
   int h_start, h_end;

   h_start = hat_offsets[el];
   h_end = hat_offsets[el+1];
   i_dofs.Reserve(h_end-h_start);
   i_dofs.SetSize(0);
   b_dofs.Reserve(h_end-h_start);
   b_dofs.SetSize(0);
   for (int i = h_start; i < h_end; i++)
   {
      int mark = hat_dofs_marker[i];
      if (mark == 0) { i_dofs.Append(i-h_start); }
      else if (mark == -1) { b_dofs.Append(i-h_start); }
   }
}

void Hybridization::AssembleMatrix(int el, const Array<int> &vdofs,
                                   const DenseMatrix &A)
{
   Array<int> i_dofs, b_dofs;

   GetIBDofs(el, i_dofs, b_dofs);

   DenseMatrix A_ii(Af_data + Af_offsets[el], i_dofs.Size(), i_dofs.Size());
   DenseMatrix A_ib(A_ii.Data() + i_dofs.Size()*i_dofs.Size(),
                    i_dofs.Size(), b_dofs.Size());
   DenseMatrix A_bi(A_ib.Data() + i_dofs.Size()*b_dofs.Size(),
                    b_dofs.Size(), i_dofs.Size());
   DenseMatrix A_bb(A_bi.Data() + b_dofs.Size()*i_dofs.Size(),
                    b_dofs.Size(), b_dofs.Size());

   for (int j = 0; j < i_dofs.Size(); j++)
   {
      int j_dof = i_dofs[j];
      for (int i = 0; i < i_dofs.Size(); i++)
      {
         A_ii(i,j) = A(i_dofs[i],j_dof);
      }
      for (int i = 0; i < b_dofs.Size(); i++)
      {
         A_bi(i,j) = A(b_dofs[i],j_dof);
      }
   }
   for (int j = 0; j < b_dofs.Size(); j++)
   {
      int j_dof = b_dofs[j];
      for (int i = 0; i < i_dofs.Size(); i++)
      {
         A_ib(i,j) = A(i_dofs[i],j_dof);
      }
      for (int i = 0; i < b_dofs.Size(); i++)
      {
         A_bb(i,j) = A(b_dofs[i],j_dof);
      }
   }

   LUFactors LU_ii(A_ii.Data(), Af_ipiv + Af_f_offsets[el]);
   LUFactors LU_bb(A_bb.Data(), LU_ii.ipiv + i_dofs.Size());

   LU_ii.Factor(i_dofs.Size());
   LU_ii.BlockFactor(i_dofs.Size(), b_dofs.Size(),
                     A_ib.Data(), A_bi.Data(), A_bb.Data());
   LU_bb.Factor(b_dofs.Size());

   // Extract Cb_t from Ct
   std::map<int,int> Cb_g2l;
   for (int i = 0; i < b_dofs.Size(); i++)
   {
      const int row = hat_offsets[el] + b_dofs[i];
      const int ncols = Ct->RowSize(row);
      const int *cols = Ct->GetRowColumns(row);
      for (int j = 0; j < ncols; j++)
      {
         const std::pair<const int,int> p(cols[j], (int)Cb_g2l.size());
         Cb_g2l.insert(p);
      }
   }
   Array<int> mu_dofs((int)Cb_g2l.size());
   for (std::map<int, int>::iterator it = Cb_g2l.begin();
        it != Cb_g2l.end(); ++it)
   {
      mu_dofs[it->second] = it->first;
   }
   DenseMatrix Cb_t(b_dofs.Size(), mu_dofs.Size()); // Cb_t is init with 0
   for (int i = 0; i < b_dofs.Size(); i++)
   {
      const int row = hat_offsets[el] + b_dofs[i];
      const int ncols = Ct->RowSize(row);
      const int *cols = Ct->GetRowColumns(row);
      const double *vals = Ct->GetRowEntries(row);
      for (int j = 0; j < ncols; j++)
      {
         Cb_t(i,Cb_g2l[cols[j]]) = vals[j];
      }
   }

   // Compute Hb = Cb Sb^{-1} Cb^t
   DenseMatrix Sb_inv_Cb_t(Cb_t);
   LU_bb.Solve(Cb_t.Height(), Cb_t.Width(), Sb_inv_Cb_t.Data());
   DenseMatrix Hb(Cb_t.Width());
   MultAtB(Cb_t, Sb_inv_Cb_t, Hb);

   // Assemble Hb into H
   const int skip_zeros = 1;
   H->AddSubMatrix(mu_dofs, mu_dofs, Hb, skip_zeros);
}

void Hybridization::MultAfInv(const Vector &b, const Vector &lambda, Vector &bf,
                              int mode) const
{
   // b1 = Rf^t b (assuming that Ref = 0)
   Vector b1;
   const SparseMatrix *cR = fes->GetConformingRestriction();
   if (!cR)
   {
      b1.SetDataAndSize(b.GetData(), b.Size());
   }
   else
   {
      b1.SetSize(fes->GetVSize());
      cR->MultTranspose(b, b1);
   }

   const int NE = fes->GetMesh()->GetNE();
   Array<int> vdofs, i_dofs, b_dofs;
   Vector el_vals, bf_i, i_vals, b_vals;
   bf.SetSize(hat_offsets[NE]);
   if (mode == 1)
   {
      Ct->Mult(lambda, bf);
   }
   // Apply Af^{-1}
   Array<bool> vdof_marker(b1.Size());
   vdof_marker = false;
   for (int i = 0; i < NE; i++)
   {
      fes->GetElementVDofs(i, vdofs);
      b1.GetSubVector(vdofs, el_vals);
      for (int j = 0; j < vdofs.Size(); j++)
      {
         int vdof = vdofs[j];
         if (vdof < 0) { vdof = -1 - vdof; }
         if (vdof_marker[vdof]) { el_vals(j) = 0.0; }
         else { vdof_marker[vdof] = true; }
      }
      bf_i.SetDataAndSize(&bf[hat_offsets[i]], vdofs.Size());
      if (mode == 1)
      {
         el_vals -= bf_i;
      }
      GetIBDofs(i, i_dofs, b_dofs);
      el_vals.GetSubVector(i_dofs, i_vals);
      el_vals.GetSubVector(b_dofs, b_vals);

      LUFactors LU_ii(Af_data + Af_offsets[i], Af_ipiv + Af_f_offsets[i]);
      double *U_ib = LU_ii.data + i_dofs.Size()*i_dofs.Size();
      double *L_bi = U_ib + i_dofs.Size()*b_dofs.Size();
      LUFactors LU_bb(L_bi + b_dofs.Size()*i_dofs.Size(),
                      LU_ii.ipiv + i_dofs.Size());
      LU_ii.BlockForwSolve(i_dofs.Size(), b_dofs.Size(), 1, L_bi,
                           i_vals.GetData(), b_vals.GetData());
      LU_bb.Solve(b_dofs.Size(), 1, b_vals.GetData());
      bf_i = 0.0;
      if (mode == 1)
      {
         LU_ii.BlockBackSolve(i_dofs.Size(), b_dofs.Size(), 1, U_ib,
                              b_vals.GetData(), i_vals.GetData());
         bf_i.SetSubVector(i_dofs, i_vals);
      }
      bf_i.SetSubVector(b_dofs, b_vals);
   }
}

void Hybridization::ReduceRHS(const Vector &b, Vector &b_r) const
{
   // bf = Af^{-1} Rf^t b
   Vector bf;
   MultAfInv(b, b, bf, 0);

   // b_r = Cf bf
   b_r.SetSize(Ct->Width());
   Ct->MultTranspose(bf, b_r);
}

void Hybridization::ComputeSolution(const Vector &b, const Vector &sol_r,
                                    Vector &sol) const
{
   // bf = Af^{-1} ( Rf^t - Cf^t sol_r )
   Vector bf;
   MultAfInv(b, sol_r, bf, 1);

   // sol = Rf bf
   GridFunction s;
   const SparseMatrix *cR = fes->GetConformingRestriction();
   if (!cR)
   {
      MFEM_ASSERT(sol.Size() == fes->GetVSize(), "");
      s.Update(fes, sol, 0);
   }
   else
   {
      s.Update(fes);
#if 0
      const SparseMatrix *cP = fes->GetConformingProlongation();
      cP->Mult(sol, s); // interpolate b.c. from sol
#else
      cR->MultTranspose(sol, s);
#endif
   }
   const int NE = fes->GetMesh()->GetNE();
   Array<int> vdofs;
   for (int i = 0; i < NE; i++)
   {
      fes->GetElementVDofs(i, vdofs);
      for (int j = hat_offsets[i]; j < hat_offsets[i+1]; j++)
      {
         if (hat_dofs_marker[j] == 1) { continue; } // skip essential b.c.
         int vdof = vdofs[j-hat_offsets[i]];
         if (vdof >= 0) { s(vdof) = bf(j); }
         else { s(-1-vdof) = -bf(j); }
      }
   }
   if (cR)
   {
      cR->Mult(s, sol); // assuming that Ref = 0
   }
}

}
