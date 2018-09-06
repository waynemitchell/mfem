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

#ifndef MFEM_BACKENDS_HYPRE_PARMATRIX_HPP
#define MFEM_BACKENDS_HYPRE_PARMATRIX_HPP

#include "../../config/config.hpp"
#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_MPI)

#include "../../linalg/operator.hpp"
#include "../../linalg/sparsemat.hpp"
#include "layout.hpp"
#include "_hypre_parcsr_mv.h"

#if defined(HYPRE_BIGINT)
#error Recompile without HYPRE_BIGINT and read hypre.cpp
#endif

namespace mfem
{

namespace hypre
{

/* HYPRE_Int HYPRE_ParCSRMatrixCreate(
MPI_Comm comm ,
HYPRE_Int global_num_rows ,
HYPRE_Int global_num_cols ,
HYPRE_Int *row_starts ,
HYPRE_Int *col_starts ,
HYPRE_Int num_cols_offd ,
HYPRE_Int num_nonzeros_diag ,
HYPRE_Int num_nonzeros_offd ,
HYPRE_ParCSRMatrix *matrix );
*/

class ParMatrix : public mfem::Operator
{
protected:
   hypre_ParCSRMatrix *mat;
   hypre_ParVector *x_vec, *y_vec;

public:
   // Make a block-diagonal matrix
   ParMatrix(Layout &layout, mfem::SparseMatrix &spmat);

   ParMatrix(Layout &in_layout, Layout &out_layout,
             HYPRE_Int *i_diag, HYPRE_Int *j_diag,
             HYPRE_Int *i_offd, HYPRE_Int *j_offd,
             HYPRE_Int *cmap, HYPRE_Int cmap_size);

   ParMatrix(mfem::hypre::Layout &in_layout, mfem::hypre::Layout &out_layout, hypre_ParCSRMatrix *mat_);

   explicit ParMatrix(ParMatrix& other);

   virtual ~ParMatrix() {
      hypre_ParCSRMatrixDestroy(mat);
      hypre_ParVectorDestroy(x_vec);
      hypre_ParVectorDestroy(y_vec);
   }

   virtual void Mult(const mfem::Vector &x, mfem::Vector &y) const {
      // Assumes the PVector inside x and y use compatible layouts with x_vec and y_vec
      // y = alpha*A*x + beta*y
      hypre_VectorData(hypre_ParVectorLocalVector(x_vec)) = (HYPRE_Complex *) x.Get_PVector()->GetData();
      hypre_VectorData(hypre_ParVectorLocalVector(y_vec)) = (HYPRE_Complex *) y.Get_PVector()->GetData();
      hypre_ParCSRMatrixMatvec(1.0, mat, x_vec, 0.0, y_vec);
   }

   hypre_ParCSRMatrix *HypreMatrix() const { return const_cast<hypre_ParCSRMatrix*>(mat); }

   // *this = alpha * A + beta * B
   // ASSUMPTION: the sparsity pattern of *this, A, and B are the same.
   void HypreAxpy(const double alpha, const ParMatrix& A, const double beta, const ParMatrix& B);

   void Print(const char *filename) {
      hypre_ParCSRMatrixPrintIJ(mat, 0, 0, filename);
   }
};

hypre_ParVector *InitializeVector(Layout &layout);

ParMatrix *MakePtAP(const ParMatrix &A, const ParMatrix &P);

// C = alpha * A + beta * B
ParMatrix *Add(const double alpha, const ParMatrix &A, const double beta, const ParMatrix &B);

} // namespace mfem::hypre

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_MPI)

#endif // MFEM_BACKENDS_HYPRE_PARMATRIX_HPP
