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

#ifndef MFEM_HYBRIDIZATION
#define MFEM_HYBRIDIZATION

#include "../config/config.hpp"
#include "fespace.hpp"
#include "bilininteg.hpp"

#ifdef MFEM_USE_MPI
#include "../linalg/hypre.hpp"
#endif

namespace mfem
{

class Hybridization
{
protected:
   FiniteElementSpace *fes, *mu_fes;
   BilinearFormIntegrator *c_bfi;

   SparseMatrix *Ct, *H;

   Array<int> hat_offsets, hat_dofs_marker;
   Array<int> Af_offsets, Af_f_offsets;
   double *Af_data;
   int *Af_ipiv;

#ifdef MFEM_USE_MPI
   HypreParMatrix *pH;
#endif

   void GetIBDofs(int el, Array<int> &i_dofs, Array<int> &b_dofs) const;

   // Compute depending on mode:
   // - mode 0: bf = Af^{-1} Rf^t b, where
   //           the non-"boundary" part of bf is set to 0;
   // - mode 1: bf = Af^{-1} ( Rf^t b - Cf^t lambda ), where
   //           the "essential" part of bf is set to 0.
   // Input: size(b)      =    fes->GetConformingVSize()
   //        size(lambda) = mu_fes->GetConformingVSize()
   void MultAfInv(const Vector &b, const Vector &lambda, Vector &bf,
                  int mode) const;

public:
   /// Constructor
   Hybridization(FiniteElementSpace *fespace, FiniteElementSpace *mu_fespace);
   /// Destructor
   ~Hybridization();

   ///
   void SetConstraintIntegrator(BilinearFormIntegrator *c_integ)
   { delete c_bfi; c_bfi = c_integ; }

   ///
   void Init(const Array<int> &ess_cdofs_marker);

   ///
   void AssembleMatrix(int el, const Array<int> &vdofs, const DenseMatrix &A);

   ///
   void Finalize();

   ///
   SparseMatrix &GetMatrix() { return *H; }

#ifdef MFEM_USE_MPI
   HypreParMatrix &GetParallelMatrix() { return *pH; }
#endif

   ///
   void ReduceRHS(const Vector &b, Vector &b_r) const;

   /// assuming 'sol' has the right essential b.c.
   void ComputeSolution(const Vector &b, const Vector &sol_r,
                        Vector &sol) const;
};

}

#endif
