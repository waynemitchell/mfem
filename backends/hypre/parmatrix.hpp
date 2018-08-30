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
#include "vector.hpp"
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

   hypre_ParVector *InitVector(Layout &layout);

public:
   // Make a block-diagonal matrix
   ParMatrix(Layout &layout, mfem::SparseMatrix &spmat);

   ParMatrix(Layout &in_layout, Layout &out_layout,
             HYPRE_Int *i_diag, HYPRE_Int *j_diag,
             HYPRE_Int *i_offd, HYPRE_Int *j_offd,
             HYPRE_Int *cmap, HYPRE_Int cmap_size);

   ParMatrix(mfem::hypre::Layout &layout, hypre_ParCSRMatrix *mat_);

   virtual ~ParMatrix() {
      hypre_ParCSRMatrixDestroy(mat);
      hypre_ParVectorDestroy(x_vec);
      hypre_ParVectorDestroy(y_vec);
   }

   virtual void Mult(const mfem::Vector &x, mfem::Vector &y) const {
      // Assumes the PVector inside x and y use compatible layouts with x_vec and y_vec
      // y = alpha*A*x + beta*y
      // HYPRE_Int hypre_ParCSRMatrixMatvec ( HYPRE_Complex alpha , hypre_ParCSRMatrix *A , hypre_ParVector *x , HYPRE_Complex beta , hypre_ParVector *y );
      hypre_VectorData(hypre_ParVectorLocalVector(x_vec)) = (HYPRE_Complex *) x.Get_PVector()->GetData();
      hypre_VectorData(hypre_ParVectorLocalVector(y_vec)) = (HYPRE_Complex *) y.Get_PVector()->GetData();
      hypre_ParCSRMatrixMatvec(1.0, mat, x_vec, 0.0, y_vec);
   }

   friend ParMatrix *MakePtAP(const ParMatrix &A, const ParMatrix &P);

   void Print(const char *filename) {
// HYPRE_Int hypre_ParCSRMatrixPrint ( hypre_ParCSRMatrix *matrix , const char *file_name );
      hypre_ParCSRMatrixPrintIJ(mat, 0, 0, filename);
   }
};

ParMatrix *MakePtAP(const ParMatrix &A, const ParMatrix &P);

} // namespace mfem::hypre

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_MPI)

#endif // MFEM_BACKENDS_HYPRE_PARMATRIX_HPP
