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

#include "../config/config.hpp"

#ifdef MFEM_USE_MPI

#include "eliminate.hpp"
#include <limits>

namespace mfem
{
namespace internal
{

/*
  Function:  hypre_CSRMatrixEliminateBC

  Eliminate the rows and columns of A corresponding to the
  given sorted (!) list of rows. Put I on the eliminated diagonal,
  subtract columns times X from B.
*/
void hypre_CSRMatrixEliminateBC(hypre_CSRMatrix *A,
                                HYPRE_Int nrows_to_eliminate,
                                HYPRE_Int *rows_to_eliminate,
                                hypre_Vector *X,
                                hypre_Vector *B)
{
   HYPRE_Int  i, j;
   HYPRE_Int  irow, jcol, ibeg, iend, pos;
   HYPRE_Real a;

   HYPRE_Int  *Ai    = hypre_CSRMatrixI(A);
   HYPRE_Int  *Aj    = hypre_CSRMatrixJ(A);
   HYPRE_Real *Adata = hypre_CSRMatrixData(A);
   HYPRE_Int   nrows = hypre_CSRMatrixNumRows(A);

   HYPRE_Real *Xdata = hypre_VectorData(X);
   HYPRE_Real *Bdata = hypre_VectorData(B);

   /* eliminate the columns */
   for (i = 0; i < nrows; i++)
   {
      ibeg = Ai[i];
      iend = Ai[i+1];
      for (j = ibeg; j < iend; j++)
      {
         jcol = Aj[j];
         pos = hypre_BinarySearch(rows_to_eliminate, jcol, nrows_to_eliminate);
         if (pos != -1)
         {
            a = Adata[j];
            Adata[j] = 0.0;
            Bdata[i] -= a * Xdata[jcol];
         }
      }
   }

   /* remove the rows and set the diagonal equal to 1 */
   for (i = 0; i < nrows_to_eliminate; i++)
   {
      irow = rows_to_eliminate[i];
      ibeg = Ai[irow];
      iend = Ai[irow+1];
      for (j = ibeg; j < iend; j++)
      {
         if (Aj[j] == irow)
         {
            Adata[j] = 1.0;
         }
         else
         {
            Adata[j] = 0.0;
         }
      }
   }
}

/*
  Function:  hypre_CSRMatrixEliminateCols

  Eliminate the given sorted (!) list of columns of A, subtract them from B.
*/
void hypre_CSRMatrixEliminateOffdCols(hypre_CSRMatrix *A,
                                      HYPRE_Int ncols_to_eliminate,
                                      HYPRE_Int *eliminate_cols,
                                      HYPRE_Real *eliminate_coefs,
                                      hypre_Vector *B)
{
   HYPRE_Int i, j;
   HYPRE_Int ibeg, iend, pos;
   HYPRE_Real a;

   HYPRE_Int *Ai = hypre_CSRMatrixI(A);
   HYPRE_Int *Aj = hypre_CSRMatrixJ(A);
   HYPRE_Real *Adata = hypre_CSRMatrixData(A);
   HYPRE_Int nrows = hypre_CSRMatrixNumRows(A);

   HYPRE_Real *Bdata = hypre_VectorData(B);

   for (i = 0; i < nrows; i++)
   {
      ibeg = Ai[i];
      iend = Ai[i+1];
      for (j = ibeg; j < iend; j++)
      {
         pos = hypre_BinarySearch(eliminate_cols, Aj[j], ncols_to_eliminate);
         if (pos != -1)
         {
            a = Adata[j];
            Adata[j] = 0.0;
            Bdata[i] -= a * eliminate_coefs[pos];
         }
      }
   }
}

/*
  Function:  hypre_CSRMatrixEliminateRows

  Eliminate (zero) the given list of rows of A.
*/
void hypre_CSRMatrixEliminateOffdRows(hypre_CSRMatrix *A,
                                      HYPRE_Int  nrows_to_eliminate,
                                      HYPRE_Int *rows_to_eliminate)
{
   HYPRE_Int  *Ai    = hypre_CSRMatrixI(A);
   HYPRE_Real *Adata = hypre_CSRMatrixData(A);

   HYPRE_Int i, j;
   HYPRE_Int irow, ibeg, iend;

   for (i = 0; i < nrows_to_eliminate; i++)
   {
      irow = rows_to_eliminate[i];
      ibeg = Ai[irow];
      iend = Ai[irow+1];
      for (j = ibeg; j < iend; j++)
      {
         Adata[j] = 0.0;
      }
   }
}


/*
  Function:  hypre_ParCSRMatrixEliminateBC

  This function eliminates the global rows and columns of a matrix
  A corresponding to given lists of sorted (!) local row numbers,
  so that the solution to the system A*X = B is X_b for the given rows.

  The elimination is done as follows:

                    (input)                  (output)

                / A_ii | A_ib \          / A_ii |  0   \
            A = | -----+----- |   --->   | -----+----- |
                \ A_bi | A_bb /          \   0  |  I   /

                        / X_i \          / X_i \
                    X = | --- |   --->   | --- |  (no change)
                        \ X_b /          \ X_b /

                        / B_i \          / B_i - A_ib * X_b \
                    B = | --- |   --->   | ---------------- |
                        \ B_b /          \        X_b       /

*/
void hypre_ParCSRMatrixEliminateBC(hypre_ParCSRMatrix *A,
                                   HYPRE_Int nrows_to_eliminate,
                                   HYPRE_Int *rows_to_eliminate,
                                   hypre_ParVector *X,
                                   hypre_ParVector *B)
{
   hypre_CSRMatrix *diag = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrix *offd = hypre_ParCSRMatrixOffd(A);
   HYPRE_Int diag_nrows  = hypre_CSRMatrixNumRows(diag);
   HYPRE_Int offd_ncols  = hypre_CSRMatrixNumCols(offd);

   hypre_Vector *Xlocal = hypre_ParVectorLocalVector(X);
   hypre_Vector *Blocal = hypre_ParVectorLocalVector(B);

   HYPRE_Real   *Bdata  = hypre_VectorData(Blocal);
   HYPRE_Real   *Xdata  = hypre_VectorData(Xlocal);

   HYPRE_Int  ncols_to_eliminate;
   HYPRE_Int  *cols_to_eliminate;
   HYPRE_Real *eliminate_coefs;

   /* figure out which offd cols should be eliminated and with what coef */
   hypre_ParCSRCommHandle *comm_handle;
   hypre_ParCSRCommPkg *comm_pkg;
   HYPRE_Int num_sends;
   HYPRE_Int index, start;
   HYPRE_Int i, j, k, irow;

   HYPRE_Real *eliminate_row = hypre_CTAlloc(HYPRE_Real, diag_nrows);
   HYPRE_Real *eliminate_col = hypre_CTAlloc(HYPRE_Real, offd_ncols);
   HYPRE_Real *buf_data, coef;

   /* make sure A has a communication package */
   comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   if (!comm_pkg)
   {
      hypre_MatvecCommPkgCreate(A);
      comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   }

   /* HACK: rows that shouldn't be eliminated are marked with quiet NaN;
      those that should are set to the boundary value from X; this is to
      avoid sending complex type (int+double) or communicating twice. */
   for (i = 0; i < diag_nrows; i++)
   {
      eliminate_row[i] = std::numeric_limits<HYPRE_Real>::quiet_NaN();
   }
   for (i = 0; i < nrows_to_eliminate; i++)
   {
      irow = rows_to_eliminate[i];
      eliminate_row[irow] = Xdata[irow];
   }

   /* use a Matvec communication pattern to find (in eliminate_col)
      which of the local offd columns are to be eliminated */
   num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
   buf_data = hypre_CTAlloc(HYPRE_Real,
                            hypre_ParCSRCommPkgSendMapStart(comm_pkg,
                                                            num_sends));
   index = 0;
   for (i = 0; i < num_sends; i++)
   {
      start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
      for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
      {
         k = hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j);
         buf_data[index++] = eliminate_row[k];
      }
   }
   comm_handle = hypre_ParCSRCommHandleCreate(1, comm_pkg,
                                              buf_data, eliminate_col);

   /* do sequential part of the elimination while stuff is getting sent */
   hypre_CSRMatrixEliminateBC(diag, nrows_to_eliminate, rows_to_eliminate,
                              Xlocal, Blocal);

   /* finish the communication */
   hypre_ParCSRCommHandleDestroy(comm_handle);

   /* received eliminate_col[], count offd columns to eliminate */
   ncols_to_eliminate = 0;
   for (i = 0; i < offd_ncols; i++)
   {
      coef = eliminate_col[i];
      if (coef == coef) // test for NaN
      {
         ncols_to_eliminate++;
      }
   }

   cols_to_eliminate = hypre_CTAlloc(HYPRE_Int, ncols_to_eliminate);
   eliminate_coefs = hypre_CTAlloc(HYPRE_Real, ncols_to_eliminate);

   /* get a list of offd column indices and coefs */
   ncols_to_eliminate = 0;
   for (i = 0; i < offd_ncols; i++)
   {
      coef = eliminate_col[i];
      if (coef == coef) // test for NaN
      {
         cols_to_eliminate[ncols_to_eliminate] = i;
         eliminate_coefs[ncols_to_eliminate] = coef;
         ncols_to_eliminate++;
      }
   }

   hypre_TFree(buf_data);
   hypre_TFree(eliminate_row);
   hypre_TFree(eliminate_col);

   /* eliminate the off-diagonal part */
   hypre_CSRMatrixEliminateOffdCols(offd, ncols_to_eliminate, cols_to_eliminate,
                                    eliminate_coefs, Blocal);

   hypre_CSRMatrixEliminateOffdRows(offd, nrows_to_eliminate, rows_to_eliminate);

   /* set boundary values in the rhs */
   for (int i = 0; i < nrows_to_eliminate; i++)
   {
      irow = rows_to_eliminate[i];
      Bdata[irow] = Xdata[irow];
   }

   hypre_TFree(cols_to_eliminate);
   hypre_TFree(eliminate_coefs);
}

}
} // namespace mfem::internal


#endif // MFEM_USE_MPI
