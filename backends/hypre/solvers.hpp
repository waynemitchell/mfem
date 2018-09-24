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
#include "../../general/tic_toc.hpp"

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

   mutable StopWatch sw;
   mutable double setup_time, solve_time;

public:
   AMGSolver(ParMatrix *A_);

   void Setup(ParMatrix *A_);
   virtual void SetOperator(const Operator &op)
   {
      mfem_error("Not supported");
   }

   virtual ~AMGSolver();
   virtual void Mult(const mfem::Vector &x, mfem::Vector &y) const;

   double SetupTime() const { return setup_time; }
   double SolveTime() const { return solve_time; }
};

} // namespace mfem::hypre

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_MPI)

#endif // MFEM_BACKENDS_HYPRE_SOLVERS_HPP
