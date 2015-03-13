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

#include "eliminate.hpp"

namespace mfem
{
namespace internal
{

/*
  Function:  hypre_CSRMatrixEliminateRowsColsDiag

  Eliminate the rows and columns of Adiag corresponding to the
  given sorted (!) list of rows. Put I on the eliminated diagonal.
*/
HYPRE_Int hypre_CSRMatrixEliminateRowsColsDiag (hypre_ParCSRMatrix *A,
                                          HYPRE_Int nrows_to_eliminate,
                                          HYPRE_Int *rows_to_eliminate)
{
   HYPRE_Int ierr = 0;

   MPI_Comm          comm      = hypre_ParCSRMatrixComm(A);
   hypre_CSRMatrix  *Adiag     = hypre_ParCSRMatrixDiag(A);

   HYPRE_Int         i, j;
   HYPRE_Int         irow, ibeg, iend;

   HYPRE_Int         nnz       = hypre_CSRMatrixNumNonzeros(Adiag);
   HYPRE_Int        *Ai        = hypre_CSRMatrixI(Adiag);
   HYPRE_Int        *Aj        = hypre_CSRMatrixJ(Adiag);
   HYPRE_Real       *Adata     = hypre_CSRMatrixData(Adiag);

   HYPRE_Int        *local_rows;

   HYPRE_Int         myproc;

   hypre_MPI_Comm_rank(comm, &myproc);
   ibeg= 0;

   /* grab local rows to eliminate */
   local_rows= hypre_TAlloc(HYPRE_Int, nrows_to_eliminate);
   for (i= 0; i< nrows_to_eliminate; i++)
   {
      local_rows[i]= rows_to_eliminate[i]-ibeg;
   }

   /* remove the columns */
   for (i = 0; i < nnz; i++)
   {
      irow = hypre_BinarySearch(local_rows, Aj[i],
                                nrows_to_eliminate);
      if (irow != -1)
         Adata[i] = 0.0;
   }

   /* remove the rows and set the diagonal equal to 1 */
   for (i = 0; i < nrows_to_eliminate; i++)
   {
      irow = local_rows[i];
      ibeg = Ai[irow];
      iend = Ai[irow+1];
      for (j = ibeg; j < iend; j++)
         if (Aj[j] == irow)
            Adata[j] = 1.0;
         else
            Adata[j] = 0.0;
   }

   hypre_TFree(local_rows);

   return ierr;
}

/*
  Function:  hypre_CSRMatrixEliminateRowsOffd

  Eliminate the given list of rows of Aoffd.
*/
HYPRE_Int hypre_CSRMatrixEliminateRowsOffd (hypre_ParCSRMatrix *A,
                                      HYPRE_Int  nrows_to_eliminate,
                                      HYPRE_Int *rows_to_eliminate)
{
   HYPRE_Int ierr = 0;

   MPI_Comm         comm      = hypre_ParCSRMatrixComm(A);

   hypre_CSRMatrix *Aoffd     = hypre_ParCSRMatrixOffd(A);
   HYPRE_Int       *Ai        = hypre_CSRMatrixI(Aoffd);

   HYPRE_Real      *Adata     = hypre_CSRMatrixData(Aoffd);

   HYPRE_Int i, j;
   HYPRE_Int ibeg, iend;

   HYPRE_Int *local_rows;
   HYPRE_Int myproc;

   hypre_MPI_Comm_rank(comm, &myproc);
   ibeg= 0;

   /* grab local rows to eliminate */
   local_rows= hypre_TAlloc(HYPRE_Int, nrows_to_eliminate);
   for (i= 0; i< nrows_to_eliminate; i++)
   {
      local_rows[i]= rows_to_eliminate[i]-ibeg;
   }

   for (i = 0; i < nrows_to_eliminate; i++)
   {
      ibeg = Ai[local_rows[i]];
      iend = Ai[local_rows[i]+1];
      for (j = ibeg; j < iend; j++)
         Adata[j] = 0.0;
   }

   hypre_TFree(local_rows);

   return ierr;
}

/*
  Function:  hypre_CSRMatrixEliminateColsOffd

  Eliminate the given sorted (!) list of columns of Aoffd.
*/
HYPRE_Int hypre_CSRMatrixEliminateColsOffd (hypre_CSRMatrix *Aoffd,
                                      HYPRE_Int ncols_to_eliminate,
                                      HYPRE_Int *cols_to_eliminate)
{
   HYPRE_Int ierr = 0;

   HYPRE_Int i;
   HYPRE_Int icol;

   HYPRE_Int nnz = hypre_CSRMatrixNumNonzeros(Aoffd);
   HYPRE_Int *Aj = hypre_CSRMatrixJ(Aoffd);
   HYPRE_Real *Adata = hypre_CSRMatrixData(Aoffd);

   for (i = 0; i < nnz; i++)
   {
      icol = hypre_BinarySearch(cols_to_eliminate, Aj[i], ncols_to_eliminate);
      if (icol != -1)
         Adata[i] = 0.0;
   }

   return ierr;
}

/*
  Function:  hypre_ParCSRMatrixEliminateBC

  This function eliminates the global rows and columns of a matrix
  A corresponding to given lists of sorted (!) local row numbers.

  The elimination is done as follows:

                / A_ii | A_ib \          / A_ii |  0   \
    (input) A = | -----+----- |   --->   | -----+----- | (output)
                \ A_bi | A_bb /          \   0  |  I   /
*/
HYPRE_Int hypre_ParCSRMatrixEliminateBC(hypre_ParCSRMatrix *A,
                                        HYPRE_Int nrows_to_eliminate,
                                        HYPRE_Int *rows_to_eliminate)
{
   HYPRE_Int ierr = 0;

   MPI_Comm         comm      = hypre_ParCSRMatrixComm(A);

   hypre_CSRMatrix *diag      = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrix *offd      = hypre_ParCSRMatrixOffd(A);
   HYPRE_Int diag_nrows             = hypre_CSRMatrixNumRows(diag);
   HYPRE_Int offd_ncols             = hypre_CSRMatrixNumCols(offd);

   HYPRE_Int ncols_to_eliminate;
   HYPRE_Int *cols_to_eliminate;

   HYPRE_Int       myproc;
   HYPRE_Int       ibeg;

   hypre_MPI_Comm_rank(comm, &myproc);
   ibeg= 0;


   /* take care of the diagonal part (sequential elimination) */
   hypre_CSRMatrixEliminateRowsColsDiag (A, nrows_to_eliminate,
                                         rows_to_eliminate);

   /* eliminate the off-diagonal rows */
   hypre_CSRMatrixEliminateRowsOffd (A, nrows_to_eliminate,
                                     rows_to_eliminate);

   /* figure out which offd cols should be eliminated */
   {
      hypre_ParCSRCommHandle *comm_handle;
      hypre_ParCSRCommPkg *comm_pkg;
      HYPRE_Int num_sends, *int_buf_data;
      HYPRE_Int index, start;
      HYPRE_Int i, j, k;

      HYPRE_Int *eliminate_row = hypre_CTAlloc(HYPRE_Int, diag_nrows);
      HYPRE_Int *eliminate_col = hypre_CTAlloc(HYPRE_Int, offd_ncols);

      /* make sure A has a communication package */
      comm_pkg = hypre_ParCSRMatrixCommPkg(A);
      if (!comm_pkg)
      {
         hypre_MatvecCommPkgCreate(A);
         comm_pkg = hypre_ParCSRMatrixCommPkg(A);
      }

      /* which of the local rows are to be eliminated */
      for (i = 0; i < diag_nrows; i++)
         eliminate_row[i] = 0;
      for (i = 0; i < nrows_to_eliminate; i++)
         eliminate_row[rows_to_eliminate[i]-ibeg] = 1;

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
            k = hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j);
            int_buf_data[index++] = eliminate_row[k];
         }
      }
      comm_handle = hypre_ParCSRCommHandleCreate(11, comm_pkg,
                                                 int_buf_data, eliminate_col);
      hypre_ParCSRCommHandleDestroy(comm_handle);

      /* set the array cols_to_eliminate */
      ncols_to_eliminate = 0;
      for (i = 0; i < offd_ncols; i++)
         if (eliminate_col[i])
            ncols_to_eliminate++;

      cols_to_eliminate = hypre_CTAlloc(HYPRE_Int, ncols_to_eliminate);

      ncols_to_eliminate = 0;
      for (i = 0; i < offd_ncols; i++)
         if (eliminate_col[i])
            cols_to_eliminate[ncols_to_eliminate++] = i;

      hypre_TFree(int_buf_data);
      hypre_TFree(eliminate_row);
      hypre_TFree(eliminate_col);
   }

   /* eliminate the off-diagonal columns */
   hypre_CSRMatrixEliminateColsOffd (offd, ncols_to_eliminate,
                                     cols_to_eliminate);

   hypre_TFree(cols_to_eliminate);

   return ierr;
}

} } // namespace mfem::internal
