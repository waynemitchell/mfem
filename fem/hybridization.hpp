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

/** Auxiliary class Hybridization, used to implement BilinearForm hybridization.

    Hybridization can be viewed as a technique for solving linear systems
    obtained through finite element assembly. The assembled matrix A can be
    written as:
       A = P^T \hat{A} P
    where P is the matrix mapping the conforming finite element space to the
    purely local finite element space without any inter-element constraints
    imposed, and \hat{A} is the block-diagonal matrix of all element matrices.

    We assume that:
    * \hat{A} is invertible
    * P has a left inverse R, such that R P = I
    * a constraint matrix C can be constructed, such that Ker(C) = Im(P).

    Under these conditions, the linear system A x = b can be solved using the
    following procedure:
    * solve for lambda in the linear system:
       (C \hat{A}^{-1} C^T) \lambda = C \hat{A}^{-1} R^T b
    * compute x = R \hat{A}^{-1} (R^T b - C^T \lambda)
    The advantage of hybridization is that the matrix H = (C \hat{A}^{-1} C^T)
    of the hybridized system may either be smaller than the original system, or
    be simpler to invert with a known method.

    In some cases, e.g. high-order elements, the matrix C can be written as
       C = [ 0  C_b ]
    and then the hybridized matrix H can be assembled using the identity
       H = C_b S_b^{-1} C_b^T
    where S_b is the Schur complement of \hat{A} with respect to the same
    decomposition as the columns of C:
       S_b = \hat{A}_b - \hat{A}_{bf} \hat{A}_{f}^{-1} \hat{A}_{fb}.

    Hybridization can also be viewed as a discretization method for imposing
    (weak) continuity constraints between neighboring elements. */
class Hybridization
{
protected:
   FiniteElementSpace *fes, *c_fes;
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
   // Input: size(b)      =   fes->GetConformingVSize()
   //        size(lambda) = c_fes->GetConformingVSize()
   void MultAfInv(const Vector &b, const Vector &lambda, Vector &bf,
                  int mode) const;

public:
   /// Constructor
   Hybridization(FiniteElementSpace *fespace, FiniteElementSpace *c_fespace);
   /// Destructor
   ~Hybridization();

   /** Set the integrator that will be used to construct the constraint matrix
       C. The Hybridization object assumes ownership of the integrator, i.e. it
       will delete the integrator when destroyed. */
   void SetConstraintIntegrator(BilinearFormIntegrator *c_integ)
   { delete c_bfi; c_bfi = c_integ; }

   /// Prepare the Hybridization object for assembly.
   void Init(const Array<int> &ess_tdof_list);

   /// Assemble the element matrix A into the hybridized system matrix.
   void AssembleMatrix(int el, const Array<int> &vdofs, const DenseMatrix &A);

   /// Finalize the construction of the hybridized matrix.
   void Finalize();

   /// Return the serial hybridized matrix.
   SparseMatrix &GetMatrix() { return *H; }

#ifdef MFEM_USE_MPI
   /// Return the parallel hybridized matrix.
   HypreParMatrix &GetParallelMatrix() { return *pH; }
#endif

   /** Perform the reduction of the given r.h.s. vector, b, to a r.h.s vector,
       b_r, for the hybridized system. */
   void ReduceRHS(const Vector &b, Vector &b_r) const;

   /** Reconstruct the solution of the original system, sol, from solution of
       the hybridized system, sol_r, and the original r.h.s. vector, b.
       It is assumed that the vector sol has the right essential b.c. */
   void ComputeSolution(const Vector &b, const Vector &sol_r,
                        Vector &sol) const;
};

}

#endif
