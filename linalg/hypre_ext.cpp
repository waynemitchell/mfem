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

#include "hypre_ext.hpp"
#include <limits>

namespace mfem
{
namespace internal
{

/*--------------------------------------------------------------------------
 *                        A*X = B style elimination
 *--------------------------------------------------------------------------*/

/*
  Function:  hypre_CSRMatrixEliminateAXB

  Eliminate the rows and columns of A corresponding to the
  given sorted (!) list of rows. Put I on the eliminated diagonal,
  subtract columns times X from B.
*/
void hypre_CSRMatrixEliminateAXB(hypre_CSRMatrix *A,
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
  Function:  hypre_CSRMatrixEliminateOffdColsAXB

  Eliminate the given sorted (!) list of columns of A, subtract them from B.
*/
void hypre_CSRMatrixEliminateOffdColsAXB(hypre_CSRMatrix *A,
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
  Function:  hypre_CSRMatrixEliminateOffdRowsAXB

  Eliminate (zero) the given list of rows of A.
*/
void hypre_CSRMatrixEliminateOffdRowsAXB(hypre_CSRMatrix *A,
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
  Function:  hypre_ParCSRMatrixEliminateAXB

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
void hypre_ParCSRMatrixEliminateAXB(hypre_ParCSRMatrix *A,
                                    HYPRE_Int num_rowscols_to_elim,
                                    HYPRE_Int *rowscols_to_elim,
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

   HYPRE_Int  num_offd_cols_to_elim;
   HYPRE_Int  *offd_cols_to_elim;
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
   for (i = 0; i < num_rowscols_to_elim; i++)
   {
      irow = rowscols_to_elim[i];
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
   hypre_CSRMatrixEliminateAXB(diag, num_rowscols_to_elim, rowscols_to_elim,
                               Xlocal, Blocal);

   /* finish the communication */
   hypre_ParCSRCommHandleDestroy(comm_handle);

   /* received eliminate_col[], count offd columns to eliminate */
   num_offd_cols_to_elim = 0;
   for (i = 0; i < offd_ncols; i++)
   {
      coef = eliminate_col[i];
      if (coef == coef) // test for NaN
      {
         num_offd_cols_to_elim++;
      }
   }

   offd_cols_to_elim = hypre_CTAlloc(HYPRE_Int, num_offd_cols_to_elim);
   eliminate_coefs = hypre_CTAlloc(HYPRE_Real, num_offd_cols_to_elim);

   /* get a list of offd column indices and coefs */
   num_offd_cols_to_elim = 0;
   for (i = 0; i < offd_ncols; i++)
   {
      coef = eliminate_col[i];
      if (coef == coef) // test for NaN
      {
         offd_cols_to_elim[num_offd_cols_to_elim] = i;
         eliminate_coefs[num_offd_cols_to_elim] = coef;
         num_offd_cols_to_elim++;
      }
   }

   hypre_TFree(buf_data);
   hypre_TFree(eliminate_row);
   hypre_TFree(eliminate_col);

   /* eliminate the off-diagonal part */
   hypre_CSRMatrixEliminateOffdColsAXB(offd, num_offd_cols_to_elim,
                                       offd_cols_to_elim,
                                       eliminate_coefs, Blocal);

   hypre_CSRMatrixEliminateOffdRowsAXB(offd, num_rowscols_to_elim,
                                       rowscols_to_elim);

   /* set boundary values in the rhs */
   for (int i = 0; i < num_rowscols_to_elim; i++)
   {
      irow = rowscols_to_elim[i];
      Bdata[irow] = Xdata[irow];
   }

   hypre_TFree(offd_cols_to_elim);
   hypre_TFree(eliminate_coefs);
}


/*--------------------------------------------------------------------------
 *                        (A + Ae) style elimination
 *--------------------------------------------------------------------------*/

/*
  Function:  hypre_CSRMatrixCreateDiagAe


*/
void hypre_CSRMatrixCreateDiagAe(hypre_CSRMatrix *A,
                                 hypre_CSRMatrix *Ae,
                                 HYPRE_Int num_rowscols_to_elim,
                                 HYPRE_Int *rowscols_to_elim)
{
   HYPRE_Int  i, j;
   HYPRE_Int  A_beg, A_end;

   HYPRE_Int  *A_i     = hypre_CSRMatrixI(A);
   HYPRE_Int  *A_j     = hypre_CSRMatrixJ(A);
   HYPRE_Int   nrows   = hypre_CSRMatrixNumRows(A);

   hypre_CSRMatrixI(Ae) = new HYPRE_Int[nrows+1];

   HYPRE_Int  *Ae_i    = hypre_CSRMatrixI(Ae);
   HYPRE_Int   nnz     = 0;

   for (i = 0; i < nrows; i++)
   {
      Ae_i[i] = nnz;

      A_beg = A_i[i];
      A_end = A_i[i+1];

      if (hypre_BinarySearch(rowscols_to_elim, i, num_rowscols_to_elim) >= 0)
      {
         /* full row */
         nnz += A_end - A_beg;
      }
      else
      {
         /* count columns */
         for (j = A_beg; j < A_end; j++)
         {
            if (hypre_BinarySearch(rowscols_to_elim, A_j[j],
                                   num_rowscols_to_elim) >= 0) { nnz++; }
         }
      }
   }
   Ae_i[nrows] = nnz;

   hypre_CSRMatrixJ(Ae) = new HYPRE_Int[nnz];
   hypre_CSRMatrixData(Ae) = new double[nnz];
   hypre_CSRMatrixNumNonzeros(Ae) = nnz;
}

/*
  Function:  hypre_CSRMatrixEliminateDiagAe


*/
void hypre_CSRMatrixEliminateDiagAe(hypre_CSRMatrix *A,
                                    hypre_CSRMatrix *Ae,
                                    HYPRE_Int num_rowscols_to_elim,
                                    HYPRE_Int *rowscols_to_elim)
{
   HYPRE_Int  i, j, k, col;
   HYPRE_Int  A_beg, Ae_beg, A_end;
   HYPRE_Int  nrc = num_rowscols_to_elim;
   HYPRE_Real a;

   HYPRE_Int  *A_i     = hypre_CSRMatrixI(A);
   HYPRE_Int  *A_j     = hypre_CSRMatrixJ(A);
   HYPRE_Real *A_data  = hypre_CSRMatrixData(A);
   HYPRE_Int   nrows   = hypre_CSRMatrixNumRows(A);

   HYPRE_Int  *Ae_i    = hypre_CSRMatrixI(Ae);
   HYPRE_Int  *Ae_j    = hypre_CSRMatrixJ(Ae);
   HYPRE_Real *Ae_data = hypre_CSRMatrixData(Ae);

   for (i = 0; i < nrows; i++)
   {
      A_beg = A_i[i];
      A_end = A_i[i+1];
      Ae_beg = Ae_i[i];

      if (hypre_BinarySearch(rowscols_to_elim, i, nrc) >= 0)
      {
         /* eliminate row */
         for (j = A_beg, k = Ae_beg; j < A_end; j++)
         {
            col = A_j[j];
            Ae_j[k] = col;
            a = (col == i) ? 1.0 : 0.0;
            Ae_data[k++] = A_data[j] - a;
            A_data[j] = a;
         }
      }
      else
      {
         /* eliminate columns */
         for (j = A_beg, k = Ae_beg; j < A_end; j++)
         {
            col = A_j[j];
            if (hypre_BinarySearch(rowscols_to_elim, col, nrc) >= 0)
            {
               Ae_j[k] = col;
               Ae_data[k++] = A_data[j];
               A_data[j] = 0.0;
            }
         }
      }
   }
}


/*
  Function:  hypre_ParCSRMatrixEliminateAAe


*/
void hypre_ParCSRMatrixEliminateAAe(hypre_ParCSRMatrix *A,
                                    hypre_ParCSRMatrix **Ae,
                                    HYPRE_Int num_rowscols_to_elim,
                                    HYPRE_Int *rowscols_to_elim)
{
   hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrix *A_offd = hypre_ParCSRMatrixOffd(A);
   HYPRE_Int diag_nrows  = hypre_CSRMatrixNumRows(A_diag);
   HYPRE_Int offd_ncols  = hypre_CSRMatrixNumCols(A_offd);

   *Ae = hypre_ParCSRMatrixCreate(hypre_ParCSRMatrixComm(A),
                                  hypre_ParCSRMatrixGlobalNumRows(A),
                                  hypre_ParCSRMatrixGlobalNumCols(A),
                                  hypre_ParCSRMatrixRowStarts(A),
                                  hypre_ParCSRMatrixColStarts(A),
                                  0, 0, 0);

   hypre_CSRMatrix *Ae_diag = hypre_ParCSRMatrixDiag(*Ae);
   hypre_CSRMatrix *Ae_offd = hypre_ParCSRMatrixOffd(*Ae);

   HYPRE_Int  num_offd_cols_to_elim;
   HYPRE_Int  *offd_cols_to_elim;

   HYPRE_Int  nrc = num_rowscols_to_elim;

   /* figure out which offd cols should be eliminated */
   {
      hypre_ParCSRCommHandle *comm_handle;
      hypre_ParCSRCommPkg *comm_pkg;
      HYPRE_Int num_sends, *int_buf_data;
      HYPRE_Int index, start;
      HYPRE_Int i, j, k, irow;

      HYPRE_Real *eliminate_row = hypre_CTAlloc(HYPRE_Real, diag_nrows);
      HYPRE_Real *eliminate_col = hypre_CTAlloc(HYPRE_Real, offd_ncols);

      /* make sure A has a communication package */
      comm_pkg = hypre_ParCSRMatrixCommPkg(A);
      if (!comm_pkg)
      {
         hypre_MatvecCommPkgCreate(A);
         comm_pkg = hypre_ParCSRMatrixCommPkg(A);
      }

      /* which of the local rows are to be eliminated */
      for (i = 0; i < diag_nrows; i++)
      {
         eliminate_row[i] = 0;
      }
      for (i = 0; i < num_rowscols_to_elim; i++)
      {
         eliminate_row[rowscols_to_elim[i]] = 1;
      }

      /* use a Matvec communication pattern to find (in eliminate_col)
         which of the local offd columns are to be eliminated */
      num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
      int_buf_data = hypre_CTAlloc(HYPRE_Int,
                                   hypre_ParCSRCommPkgSendMapStart(comm_pkg,
                                                                   num_sends));
      index = 0;
      for (i = 0; i < num_sends; i++)
      {
         start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
         for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
         {
            k = hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j);
            int_buf_data[index++] = eliminate_row[k];
         }
      }
      comm_handle = hypre_ParCSRCommHandleCreate(11, comm_pkg,
                                                 int_buf_data, eliminate_col);

      /* eliminate diagonal part while messages are getting sent */
      hypre_CSRMatrixCreateDiagAe(A_diag, Ae_diag, nrc, rowscols_to_elim);
      hypre_CSRMatrixEliminateDiagAe(A_diag, Ae_diag, nrc, rowscols_to_elim);

      /* finish the communication */
      hypre_ParCSRCommHandleDestroy(comm_handle);

      /* received eliminate_col[], count offd columns to eliminate */
      num_offd_cols_to_elim = 0;
      for (i = 0; i < offd_ncols; i++)
      {
         if (eliminate_col[i]) { num_offd_cols_to_elim++; }
      }

      offd_cols_to_elim = hypre_CTAlloc(HYPRE_Int, num_offd_cols_to_elim);

      /* get a list of offd column indices and coefs */
      num_offd_cols_to_elim = 0;
      for (i = 0; i < offd_ncols; i++)
      {
         if (eliminate_col[i])
         {
            offd_cols_to_elim[num_offd_cols_to_elim++] = i;
         }
      }

      hypre_TFree(int_buf_data);
      hypre_TFree(eliminate_row);
      hypre_TFree(eliminate_col);
   }


   /* eliminate the off-diagonal part */
   /*hypre_CSRMatrixEliminateOffdColsAXB(A_offd, num_offd_cols_to_elim, offd_cols_to_elim,
                                    eliminate_coefs, Blocal);

   hypre_CSRMatrixEliminateOffdRowsAXB(A_offd, num_rowscols_to_elim,
                                    rowscols_to_elim);
*/

   hypre_TFree(offd_cols_to_elim);
}

}

} // namespace mfem::internal

#endif // MFEM_USE_MPI
