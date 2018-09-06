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

#ifndef MFEM_BACKENDS_HYPRE_SOLVERS_HPP
#define MFEM_BACKENDS_HYPRE_SOLVERS_HPP

#include "../../config/config.hpp"
#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_MPI)

#include "../../linalg/operator.hpp"
#include "parmatrix.hpp"
#include "HYPRE_parcsr_ls.h"
// #include "_hypre_parcsr_ls.h"
#include "_hypre_parcsr_mv.h"

#if defined(HYPRE_BIGINT)
#error Recompile without HYPRE_BIGINT and read hypre.cpp
#endif

namespace mfem
{

namespace hypre
{

class AMGSolver : public Solver
{
   typedef mfem::hypre::ParMatrix MatrixType;
   ParMatrix *A;

   HYPRE_Solver solver;
   hypre_ParVector *x_vec, *y_vec;

public:
   AMGSolver(ParMatrix *A_)
      : Solver(A_->InLayout()->As<Layout>(), A_->OutLayout()->As<Layout>()),
        A(NULL), x_vec(NULL), y_vec(NULL)
   {
      HYPRE_BoomerAMGCreate(&solver);
      Setup(A_);
   }

   void Setup(ParMatrix *A_)
   {
      A = A_;
      if (x_vec != NULL) hypre_ParVectorDestroy(x_vec);
      if (y_vec != NULL) hypre_ParVectorDestroy(y_vec);
      x_vec = InitializeVector(A->InLayout()->As<Layout>());
      y_vec = InitializeVector(A->OutLayout()->As<Layout>());
      HYPRE_BoomerAMGSetup(solver, A->HypreMatrix(), x_vec, y_vec);
   }

   virtual void SetOperator(const Operator &op)
   {
      mfem_error("Not supported");
   }

   virtual ~AMGSolver()
   {
      hypre_ParVectorDestroy(x_vec);
      hypre_ParVectorDestroy(y_vec);
      HYPRE_BoomerAMGDestroy(solver);
   }

   virtual void Mult(const mfem::Vector &x, mfem::Vector &y) const {
      hypre_VectorData(hypre_ParVectorLocalVector(x_vec)) = (HYPRE_Complex *) x.Get_PVector()->GetData();
      hypre_VectorData(hypre_ParVectorLocalVector(y_vec)) = (HYPRE_Complex *) y.Get_PVector()->GetData();
      HYPRE_BoomerAMGSolve(solver, A->HypreMatrix(), x_vec, y_vec);
   }

};

} // namespace mfem::hypre

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_MPI)

#endif // MFEM_BACKENDS_HYPRE_SOLVERS_HPP
