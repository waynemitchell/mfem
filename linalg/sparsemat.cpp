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

// Implementation of sparse matrix

#include <iostream>
#include <iomanip>
#include <math.h>

#include "linalg.hpp"
#include "../general/table.hpp"
#include "../general/sort_pairs.hpp"

SparseMatrix::SparseMatrix(int nrows, int ncols)
   : Matrix(nrows)
{
   I = NULL;
   J = NULL;
   A = NULL;

   Rows = new RowNode *[nrows];
   width = (ncols) ? (ncols) : (nrows);
   for (int i = 0; i < nrows; i++)
      Rows[i] = NULL;
   ColPtr.Node = NULL;

#ifdef MFEM_USE_MEMALLOC
   NodesMem = new RowNodeAlloc;
#endif
}

SparseMatrix::SparseMatrix(int *i, int *j, double *data, int m, int n)
   : Matrix (m), I(i), J(j), width(n), A(data)
{
   Rows = NULL;
   ColPtr.J = NULL;
#ifdef MFEM_USE_MEMALLOC
   NodesMem = new RowNodeAlloc;
#endif
}

int SparseMatrix::RowSize(const int i) const
{
   if (I)
      return I[i+1]-I[i];

   int s = 0;
   RowNode *row = Rows[i];
   for ( ; row != NULL; row = row->Prev)
      if (row->Value != 0.0)
         s++;
   return s;
}

int SparseMatrix::MaxRowSize() const
{
   int out=0;
   int rowSize=0;
   if (I)
   {
      for (int i=0; i < size; ++i)
      {
         rowSize = I[i+1]-I[i];
         out = (out > rowSize) ? out : rowSize;
      }
   }
   else
   {
      for (int i=0; i < size; ++i)
      {
         rowSize = RowSize(i);
         out = (out > rowSize) ? out : rowSize;
      }
   }

   return out;
}

int *SparseMatrix::GetRowColumns(const int row)
{
   if (Rows)
      mfem_error("SparseMatrix::GetRowColumns : matrix is not Finalized!");

   return J + I[row];
}

double *SparseMatrix::GetRowEntries(const int row)
{
   if (Rows)
      mfem_error("SparseMatrix::GetRowEntries : matrix is not Finalized!");

   return A + I[row];
}

void SparseMatrix::SetWidth(int newWidth)
{
   if(newWidth == width)
   {
      // Nothing to be done here
      return;
   }
   else if( newWidth == -1)
   {
      // Compute the actual width
      width = actualWidth();
      // No need to reset the ColPtr, since the new ColPtr will be shorter.
   }
   else if(newWidth > width)
   {
      // We need to reset ColPtr, since now we may have additional columns.
      if (Rows != NULL)
      {
         delete [] ColPtr.Node;
         ColPtr.Node = static_cast<RowNode **>(NULL);
      }
      else
      {
         delete [] ColPtr.J;
         ColPtr.J = static_cast<int *>(NULL);
      }
      width = newWidth;
   }
   else
   {
      // Check that the new width is bigger or equal to the actual width.
      MFEM_ASSERT( newWidth >= actualWidth(), "The new width needs to be bigger or equal to the actual width");
      width = newWidth;
   }
}


void SparseMatrix::SortColumnIndices()
{
   if (Rows)
      mfem_error("SparseMatrix::SortColumnIndices : matrix is not Finalized!");

   Array<Pair<int,double> > row;
   for (int j = 0, i = 0; i < size; i++)
   {
      int end = I[i+1];
      row.SetSize(end - j);
      for (int k = 0; k < row.Size(); k++)
      {
         row[k].one = J[j+k];
         row[k].two = A[j+k];
      }
      SortPairs<int,double>(row, row.Size());
      for (int k = 0; k < row.Size(); k++, j++)
      {
         J[j] = row[k].one;
         A[j] = row[k].two;
      }
   }
}

double &SparseMatrix::Elem(int i, int j)
{
   return operator()(i,j);
}

const double &SparseMatrix::Elem(int i, int j) const
{
   return operator()(i,j);
}

double &SparseMatrix::operator()(int i, int j)
{
   int k, end;

#ifdef MFEM_DEBUG
   if (i >= size || i < 0 || j >= width || j < 0)
      mfem_error("SparseMatrix::operator() #1");
#endif

   if (A == NULL)
      mfem_error("SparseMatrix::operator() #2");

   end = I[i+1];
   for (k = I[i]; k < end; k++)
      if (J[k] == j)
         return A[k];

   mfem_error("SparseMatrix::operator() #3");
   return A[0];
}

const double &SparseMatrix::operator()(int i, int j) const
{
   int k, end;
   static const double zero = 0.0;

#ifdef MFEM_DEBUG
   if (i >= size || i < 0 || j >= width || j < 0)
      mfem_error("SparseMatrix::operator() const #1");
#endif

   if (A == NULL)
      mfem_error("SparseMatrix::operator() const #2");
   end = I[i+1];
   for (k = I[i]; k < end; k++)
      if (J[k] == j)
         return A[k];

   return zero;
}

void SparseMatrix::GetDiag(Vector & d) const
{

   if (size != width)
      mfem_error("SparseMatrix::GetDiag(Vector & d) this must be a square"
                 " matrix");

   if (A == NULL)
      mfem_error("SparseMatrix::GetDiag(Vector & d) not implemented for non"
                 " assembled matrices");

   d.SetSize(size);

   int j, end;
   for (int i = 0; i < size; i++)
   {

      end = I[i+1];
      for (j = I[i]; j < end; j++)
      {
         if (J[j] == i)
         {
            d[i] = A[j];
            break;
         }
      }
      if (j == end)
         d[i] = 0.;
   }
}

void SparseMatrix::Mult(const Vector &x, Vector &y) const
{
   y = 0.0;
   AddMult(x, y);
}

void SparseMatrix::AddMult(const Vector &x, Vector &y, const double a) const
{
#ifdef MFEM_DEBUG
   if (( width != x.Size() ) || ( size != y.Size() ))
      mfem_error("SparseMatrix::AddMult() #1");
#endif

   int i, j, end;
   double *Ap = A, *yp = y.GetData();
   const double *xp = x.GetData();

   if (Ap == NULL)
   {
      //  The matrix is not finalized, but multiplication is still possible
      for (i = 0; i < size; i++)
      {
         RowNode *row = Rows[i];
         double b = 0.0;
         for ( ; row != NULL; row = row->Prev)
            b += row->Value * xp[row->Column];
         *yp += a * b;
         yp++;
      }
      return;
   }

   int *Jp = J, *Ip = I;

   if (a == 1.0)
   {
#ifndef MFEM_USE_OPENMP
      for (i = j = 0; i < size; i++)
      {
         double d = 0.0;
         for (end = Ip[i+1]; j < end; j++)
         {
            d += Ap[j] * xp[Jp[j]];
         }
         yp[i] += d;
      }
#else
#pragma omp parallel for private(j,end)
      for (i = 0; i < size; i++)
      {
         double d = 0.0;
         for (j = Ip[i], end = Ip[i+1]; j < end; j++)
         {
            d += Ap[j] * xp[Jp[j]];
         }
         yp[i] += d;
      }
#endif
   }
   else
      for (i = j = 0; i < size; i++)
      {
         double d = 0.0;
         for (end = Ip[i+1]; j < end; j++)
         {
            d += Ap[j] * xp[Jp[j]];
         }
         yp[i] += a * d;
      }
}

void SparseMatrix::MultTranspose(const Vector &x, Vector &y) const
{
   y = 0.0;
   AddMultTranspose(x, y);
}

void SparseMatrix::AddMultTranspose(const Vector &x, Vector &y,
                                    const double a) const
{
#ifdef MFEM_DEBUG
   if (( size != x.Size() ) || ( width != y.Size() ))
      mfem_error("SparseMatrix::AddMultTranspose() #1");
#endif

   int i, j, end;
   double *yp = y.GetData();

   if (A == NULL)
   {
      // The matrix is not finalized, but multiplication is still possible
      for (i = 0; i < size; i++)
      {
         RowNode *row = Rows[i];
         double b = a * x(i);
         for ( ; row != NULL; row = row->Prev)
            yp[row->Column] += row->Value * b;
      }
      return;
   }

   for (i = 0; i < size; i++)
   {
      double xi = a * x(i);
      end = I[i+1];
      for(j = I[i]; j < end; j++)
      {
         yp[J[j]] += A[j]*xi;
      }
   }
}

void SparseMatrix::PartMult(
   const Array<int> &rows, const Vector &x, Vector &y) const
{
   if (A)
   {
      for (int i = 0; i < rows.Size(); i++)
      {
         int r = rows[i];
         int end = I[r+1];
         double a = 0.0;
         for (int j = I[r]; j < end; j++)
            a += A[j] * x(J[j]);
         y(r) = a;
      }
   }
   else
   {
      mfem_error("SparseMatrix::PartMult");
   }
}

void SparseMatrix::PartAddMult(
   const Array<int> &rows, const Vector &x, Vector &y, const double a) const
{
   if (A)
   {
      for (int i = 0; i < rows.Size(); i++)
      {
         int r = rows[i];
         int end = I[r+1];
         double val = 0.0;
         for (int j = I[r]; j < end; j++)
            val += A[j] * x(J[j]);
         y(r) += a*val;
      }
   }
   else
   {
      mfem_error("SparseMatrix::PartAddMult");
   }
}

double SparseMatrix::InnerProduct(const Vector &x, const Vector &y) const
{
   double prod = 0.0;
   for (int i = 0; i < size; i++)
   {
      double a = 0.0;
      if (A)
         for (int j = I[i], end = I[i+1]; j < end; j++)
            a += A[j] * x(J[j]);
      else
         for (RowNode *np = Rows[i]; np != NULL; np = np->Prev)
            a += np->Value * x(np->Column);
      prod += a * y(i);
   }

   return prod;
}

void SparseMatrix::GetRowSums(Vector &x) const
{
   for (int i = 0; i < size; i++)
   {
      double a = 0.0;
      if (A)
         for (int j = I[i], end = I[i+1]; j < end; j++)
            a += A[j];
      else
         for (RowNode *np = Rows[i]; np != NULL; np = np->Prev)
            a += np->Value;
      x(i) = a;
   }
}

void SparseMatrix::Finalize(int skip_zeros)
{
   int i, j, nr, nz;
   RowNode *aux;

   if (A != NULL) return;

   delete [] ColPtr.Node;
   ColPtr.J = NULL;

   I = new int[size+1];
   I[0] = 0;
   for (i = 1; i <= size; i++)
   {
      nr = 0;
      for (aux = Rows[i-1]; aux != NULL; aux = aux->Prev)
         if (!skip_zeros || aux->Value != 0.0)
            nr++;
      I[i] = I[i-1] + nr;
   }
   nz = I[size];
   J = new int[nz];
   A = new double[nz];
   for (j = i = 0; i < size; i++)
      for (aux = Rows[i]; aux != NULL; aux = aux->Prev)
         if (!skip_zeros || aux->Value != 0.0)
         {
            J[j] = aux->Column;
            A[j] = aux->Value;
            j++;
         }
#ifdef MFEM_USE_MEMALLOC
   NodesMem->Clear();
#else
   for (i = 0; i < size; i++)
   {
      RowNode *node_p = Rows[i];
      while (node_p != NULL)
      {
         aux = node_p;
         node_p = node_p->Prev;
         delete aux;
      }
   }
#endif
   delete [] Rows;
   Rows = NULL;
}

void SparseMatrix::GetBlocks(Array2D<SparseMatrix *> &blocks) const
{
   if (A)
      mfem_error("SparseMatrix::GetBlocks : matrix is finalized!");

   int br = blocks.NumRows(), bc = blocks.NumCols();
   int nr = (size + br - 1)/br, nc = (width + bc - 1)/bc;

   for (int j = 0; j < bc; j++)
      for (int i = 0; i < br; i++)
      {
         int *bI = new int[nr + 1];
         for (int k = 0; k <= nr; k++)
            bI[k] = 0;
         blocks(i,j) = new SparseMatrix(bI, NULL, NULL, nr, nc);
      }

   for (int gr = 0; gr < size; gr++)
   {
      int bi = gr/nr, i = gr%nr + 1;
      for (RowNode *n_p = Rows[gr]; n_p != NULL; n_p = n_p->Prev)
         if (n_p->Value != 0.0)
            blocks(bi,n_p->Column/nc)->I[i]++;
   }

   for (int j = 0; j < bc; j++)
      for (int i = 0; i < br; i++)
      {
         SparseMatrix &b = *blocks(i,j);
         int nnz = 0, rs;
         for (int k = 1; k <= nr; k++)
            rs = b.I[k], b.I[k] = nnz, nnz += rs;
         b.J = new int[nnz];
         b.A = new double[nnz];
      }

   for (int gr = 0; gr < size; gr++)
   {
      int bi = gr/nr, i = gr%nr + 1;
      for (RowNode *n_p = Rows[gr]; n_p != NULL; n_p = n_p->Prev)
         if (n_p->Value != 0.0)
         {
            SparseMatrix &b = *blocks(bi,n_p->Column/nc);
            b.J[b.I[i]] = n_p->Column % nc;
            b.A[b.I[i]] = n_p->Value;
            b.I[i]++;
         }
   }
}

double SparseMatrix::IsSymmetric() const
{
   if (A == NULL)
      mfem_error("SparseMatrix::IsSymmetric()");

   int i, j;
   double a, max;

   max = 0.0;
   for (i = 1; i < size; i++)
      for (j = I[i]; j < I[i+1]; j++)
         if (J[j] < i)
         {
            a = fabs ( A[j] - (*this)(J[j],i) );
            if (max < a)
               max = a;
         }

   return max;
}

void SparseMatrix::Symmetrize()
{
   if (A == NULL)
      mfem_error("SparseMatrix::Symmetrize()");

   int i, j;
   for (i = 1; i < size; i++)
      for (j = I[i]; j < I[i+1]; j++)
         if (J[j] < i)
         {
            A[j] += (*this)(J[j],i);
            A[j] *= 0.5;
            (*this)(J[j],i) = A[j];
         }
}

int SparseMatrix::NumNonZeroElems() const
{
   if (A != NULL)  //  matrix is finalized
   {
      return I[size];
   }
   else
   {
      int nnz = 0;

      for (int i = 0; i < size; i++)
         for (RowNode *node_p = Rows[i]; node_p != NULL; node_p = node_p->Prev)
            nnz++;

      return nnz;
   }
}

double SparseMatrix::MaxNorm() const
{
   double m = 0.0;

   if (A)
   {
      int nnz = I[size];
      for (int j = 0; j < nnz; j++)
         m = fmax(m, fabs(A[j]));
   }
   else
   {
      for (int i = 0; i < size; i++)
         for (RowNode *n_p = Rows[i]; n_p != NULL; n_p = n_p->Prev)
            m = fmax(m, fabs(n_p->Value));
   }
   return m;
}

int SparseMatrix::CountSmallElems(double tol) const
{
   int i, counter = 0;

   if (A)
   {
      int nz = I[size];
      double *Ap = A;

      for (i = 0; i < nz; i++)
         if (fabs(Ap[i]) < tol)
            counter++;
   }
   else
   {
      RowNode *aux;

      for (i = 0; i < size; i++)
         for (aux = Rows[i]; aux != NULL; aux = aux->Prev)
            if (fabs(aux -> Value) < tol)
               counter++;
   }

   return counter;
}

MatrixInverse *SparseMatrix::Inverse() const
{
   return NULL;
}

void SparseMatrix::EliminateRow(int row, const double sol, Vector &rhs)
{
   RowNode *aux;

#ifdef MFEM_DEBUG
   if ( row >= size || row < 0 )
      mfem_error("SparseMatrix::EliminateRow () #1");
#endif

   if (Rows == NULL)
      mfem_error("SparseMatrix::EliminateRow () #2");

   for (aux = Rows[row]; aux != NULL; aux = aux->Prev)
   {
      rhs(aux->Column) -= sol * aux->Value;
      aux->Value = 0.0;
   }
}

void SparseMatrix::EliminateRow(int row, int setOneDiagonal)
{
   RowNode *aux;

#ifdef MFEM_DEBUG
   if ( row >= size || row < 0 )
      mfem_error("SparseMatrix::EliminateRow () #1");

   if( setOneDiagonal && size != width )
      mfem_error("SparseMatrix::EliminateRow () #2");
#endif

   if (Rows == NULL)
      for (int i=I[row]; i < I[row+1]; ++i)
         A[i]=0.0;
   else
      for (aux = Rows[row]; aux != NULL; aux = aux->Prev)
         aux->Value = 0.0;

   if(setOneDiagonal)
      SearchRow(row, row) = 1.;
}

void SparseMatrix::EliminateCol(int col)
{
   RowNode *aux;

   if (Rows == NULL)
      mfem_error("SparseMatrix::EliminateCol () #1");

   for (int i = 0; i < size; i++)
      for (aux = Rows[i]; aux != NULL; aux = aux->Prev)
         if (aux -> Column == col)
            aux->Value = 0.0;
}

void SparseMatrix::EliminateCols(Array<int> &cols, Vector *x, Vector *b)
{
   if (Rows == NULL)
   {
      for (int i = 0; i < size; i++)
         for (int jpos = I[i]; jpos != I[i+1]; ++jpos)
            if (cols[ J[jpos]] )
            {
               if (x && b)
                  (*b)(i) -= A[jpos] * (*x)( J[jpos] );
               A[jpos] = 0.0;
            }
   }
   else
   {
      RowNode *aux;
      for (int i = 0; i < size; i++)
         for (aux = Rows[i]; aux != NULL; aux = aux->Prev)
            if (cols[aux -> Column])
            {
               if (x && b)
                  (*b)(i) -= aux -> Value * (*x)(aux -> Column);
               aux->Value = 0.0;
            }
   }
}

void SparseMatrix::EliminateRowCol(int rc, const double sol, Vector &rhs,
                                   int d)
{
   int col;

#ifdef MFEM_DEBUG
   if ( rc >= size || rc < 0 )
      mfem_error("SparseMatrix::EliminateRowCol () #1");
#endif

   if (Rows == NULL)
      for (int j = I[rc]; j < I[rc+1]; j++)
         if ((col = J[j]) == rc)
            if (d)
            {
               rhs(rc) = A[j] * sol;
            }
            else
            {
               A[j] = 1.0;
               rhs(rc) = sol;
            }
         else
         {
            A[j] = 0.0;
            for (int k = I[col]; 1; k++)
               if (k == I[col+1])
               {
                  mfem_error("SparseMatrix::EliminateRowCol () #2");
               }
               else if (J[k] == rc)
               {
                  rhs(col) -= sol * A[k];
                  A[k] = 0.0;
                  break;
               }
         }
   else
      for (RowNode *aux = Rows[rc]; aux != NULL; aux = aux->Prev)
         if ((col = aux->Column) == rc)
            if (d)
            {
               rhs(rc) = aux->Value * sol;
            }
            else
            {
               aux->Value = 1.0;
               rhs(rc) = sol;
            }
         else
         {
            aux->Value = 0.0;
            for (RowNode *node = Rows[col]; 1; node = node->Prev)
               if (node == NULL)
               {
                  mfem_error("SparseMatrix::EliminateRowCol () #3");
               }
               else if (node->Column == rc)
               {
                  rhs(col) -= sol * node->Value;
                  node->Value = 0.0;
                  break;
               }
         }
}

void SparseMatrix::EliminateRowColMultipleRHS(int rc, const Vector &sol,
                                              DenseMatrix &rhs, int d)
{
   int col;
   int num_rhs = rhs.Width();

#ifdef MFEM_DEBUG
   if (rc >= size || rc < 0)
      mfem_error("SparseMatrix::EliminateRowColMultipleRHS() #1");
   if (sol.Size() != num_rhs)
      mfem_error("SparseMatrix::EliminateRowColMultipleRHS() #2");
#endif

   if (Rows == NULL)
      for (int j = I[rc]; j < I[rc+1]; j++)
         if ((col = J[j]) == rc)
            if (d)
            {
               for (int r = 0; r < num_rhs; r++)
                  rhs(rc,r) = A[j] * sol(r);
            }
            else
            {
               A[j] = 1.0;
               for (int r = 0; r < num_rhs; r++)
                  rhs(rc,r) = sol(r);
            }
         else
         {
            A[j] = 0.0;
            for (int k = I[col]; 1; k++)
               if (k == I[col+1])
               {
                  mfem_error("SparseMatrix::EliminateRowColMultipleRHS() #3");
               }
               else if (J[k] == rc)
               {
                  for (int r = 0; r < num_rhs; r++)
                     rhs(col,r) -= sol(r) * A[k];
                  A[k] = 0.0;
                  break;
               }
         }
   else
      for (RowNode *aux = Rows[rc]; aux != NULL; aux = aux->Prev)
         if ((col = aux->Column) == rc)
            if (d)
            {
               for (int r = 0; r < num_rhs; r++)
                  rhs(rc,r) = aux->Value * sol(r);
            }
            else
            {
               aux->Value = 1.0;
               for (int r = 0; r < num_rhs; r++)
                  rhs(rc,r) = sol(r);
            }
         else
         {
            aux->Value = 0.0;
            for (RowNode *node = Rows[col]; 1; node = node->Prev)
               if (node == NULL)
               {
                  mfem_error("SparseMatrix::EliminateRowColMultipleRHS() #4");
               }
               else if (node->Column == rc)
               {
                  for (int r = 0; r < num_rhs; r++)
                     rhs(col,r) -= sol(r) * node->Value;
                  node->Value = 0.0;
                  break;
               }
         }
}

void SparseMatrix::EliminateRowCol(int rc, int d)
{
   int col;

#ifdef MFEM_DEBUG
   if ( rc >= size || rc < 0 )
      mfem_error("SparseMatrix::EliminateRowCol() #1");
#endif

   if (Rows == NULL)
   {
      for (int j = I[rc]; j < I[rc+1]; j++)
         if ((col = J[j]) == rc)
         {
            if (d == 0)
               A[j] = 1.0;
         }
         else
         {
            A[j] = 0.0;
            for (int k = I[col]; 1; k++)
               if (k == I[col+1])
               {
                  mfem_error("SparseMatrix::EliminateRowCol() #2");
               }
               else if (J[k] == rc)
               {
                  A[k] = 0.0;
                  break;
               }
         }
   }
   else
   {
      RowNode *aux, *node;

      for (aux = Rows[rc]; aux != NULL; aux = aux->Prev)
      {
         if ((col = aux->Column) == rc)
         {
            if (d == 0)
               aux->Value = 1.0;
         }
         else
         {
            aux->Value = 0.0;
            for (node = Rows[col]; 1; node = node->Prev)
               if (node == NULL)
               {
                  mfem_error("SparseMatrix::EliminateRowCol() #3");
               }
               else if (node->Column == rc)
               {
                  node->Value = 0.0;
                  break;
               }
         }
      }
   }
}

void SparseMatrix::EliminateRowCol(int rc, SparseMatrix &Ae, int d)
{
   int col;

   if (Rows)
   {
      RowNode *nd, *nd2;
      for (nd = Rows[rc]; nd != NULL; nd = nd->Prev)
      {
         if ((col = nd->Column) == rc)
         {
            if (d == 0)
            {
               Ae.Add(rc, rc, nd->Value - 1.0);
               nd->Value = 1.0;
            }
         }
         else
         {
            Ae.Add(rc, col, nd->Value);
            nd->Value = 0.0;
            for (nd2 = Rows[col]; 1; nd2 = nd2->Prev)
            {
               if (nd2 == NULL)
               {
                  mfem_error("SparseMatrix::EliminateRowCol");
               }
               else if (nd2->Column == rc)
               {
                  Ae.Add(col, rc, nd2->Value);
                  nd2->Value = 0.0;
                  break;
               }
            }
         }
      }
   }
   else
   {
      for (int j = I[rc]; j < I[rc+1]; j++)
         if ((col = J[j]) == rc)
         {
            if (d == 0)
            {
               Ae.Add(rc, rc, A[j] - 1.0);
               A[j] = 1.0;
            }
         }
         else
         {
            Ae.Add(rc, col, A[j]);
            A[j] = 0.0;
            for (int k = I[col]; true; k++)
               if (k == I[col+1])
               {
                  mfem_error("SparseMatrix::EliminateRowCol");
               }
               else if (J[k] == rc)
               {
                  Ae.Add(col, rc, A[k]);
                  A[k] = 0.0;
                  break;
               }
         }
   }
}

void SparseMatrix::SetDiagIdentity()
{
   for (int i = 0; i < size; i++)
      if (I[i+1] == I[i]+1 && fabs(A[I[i]]) < 1e-16)
         A[I[i]] = 1.0;
}

void SparseMatrix::EliminateZeroRows()
{
   int i, j;
   double zero;

   for (i = 0; i < size; i++) {
      zero = 0.0;
      for (j = I[i]; j < I[i+1]; j++)
         zero += fabs(A[j]);
      if (zero < 1e-12) {
         for (j = I[i]; j < I[i+1]; j++)
            if (J[j] == i)
               A[j] = 1.0;
            else
               A[j] = 0.0;
      }
   }
}

void SparseMatrix::Gauss_Seidel_forw(const Vector &x, Vector &y) const
{
   int c, i, s = size;
   double sum, *yp = y.GetData();
   const double *xp = x.GetData();

   if (A == NULL)
   {
      RowNode *diag_p, *n_p, **R = Rows;

      for (i = 0; i < s; i++)
      {
         sum = 0.0;
         diag_p = NULL;
         for (n_p = R[i]; n_p != NULL; n_p = n_p->Prev)
            if ((c = n_p->Column) == i)
               diag_p = n_p;
            else
               sum += n_p->Value * yp[c];

         if (diag_p != NULL && diag_p->Value != 0.0)
            yp[i] = (xp[i] - sum) / diag_p->Value;
         else
            if (xp[i] == sum)
               yp[i] = sum;
            else
               mfem_error("SparseMatrix::Gauss_Seidel_forw()");
      }
   }
   else
   {
      int j, end, d, *Ip = I, *Jp = J;
      double *Ap = A;

      j = Ip[0];
      for (i = 0; i < s; i++)
      {
         end = Ip[i+1];
         sum = 0.0;
         d = -1;
         for ( ; j < end; j++)
            if ((c = Jp[j]) == i)
               d = j;
            else
               sum += Ap[j] * yp[c];

         if (d >= 0 && Ap[d] != 0.0)
            yp[i] = (xp[i] - sum) / Ap[d];
         else
            if (xp[i] == sum)
               yp[i] = sum;
            else
               mfem_error("SparseMatrix::Gauss_Seidel_forw(...) #2");
      }
   }
}

void SparseMatrix::Gauss_Seidel_back(const Vector &x, Vector &y) const
{
   int i, c;
   double sum, *yp = y.GetData();
   const double *xp = x.GetData();

   if (A == NULL)
   {
      RowNode *diag_p, *n_p, **R = Rows;

      for (i = size-1; i >= 0; i--)
      {
         sum = 0.;
         diag_p = NULL;
         for (n_p = R[i]; n_p != NULL; n_p = n_p->Prev)
            if ((c = n_p->Column) == i)
               diag_p = n_p;
            else
               sum += n_p->Value * yp[c];

         if (diag_p != NULL && diag_p->Value != 0.0)
            yp[i] = (xp[i] - sum) / diag_p->Value;
         else
            if (xp[i] == sum)
               yp[i] = sum;
            else
               mfem_error("SparseMatrix::Gauss_Seidel_back()");
      }
   }
   else
   {
      int j, beg, d, *Ip = I, *Jp = J;
      double *Ap = A;

      j = Ip[size]-1;
      for (i = size-1; i >= 0; i--)
      {
         beg = Ip[i];
         sum = 0.;
         d = -1;
         for( ; j >= beg; j--)
            if ((c = Jp[j]) == i)
               d = j;
            else
               sum += Ap[j] * yp[c];

         if (d >= 0 && Ap[d] != 0.0)
            yp[i] = (xp[i] - sum) / Ap[d];
         else
            if (xp[i] == sum)
               yp[i] = sum;
            else
               mfem_error("SparseMatrix::Gauss_Seidel_back(...) #2");
      }
   }
}

double SparseMatrix::GetJacobiScaling() const
{
   if (A == NULL)
      mfem_error("SparseMatrix::GetJacobiScaling()");

   double sc = 1.0;
   for (int i = 0; i < size; i++)
   {
      int d = -1;
      double norm = 0.0;
      for (int j = I[i]; j < I[i+1]; j++)
      {
         if (J[j] == i)
            d = j;
         norm += fabs(A[j]);
      }
      if (d >= 0 && A[d] != 0.0)
      {
         double a = 1.8 * fabs(A[d]) / norm;
         if (a < sc)
            sc = a;
      }
      else
         mfem_error("SparseMatrix::GetJacobiScaling() #2");
   }
   return sc;
}

void SparseMatrix::Jacobi(const Vector &b, const Vector &x0, Vector &x1,
                          double sc) const
{
   if (A == NULL)
      mfem_error("SparseMatrix::Jacobi(...)");

   for (int i = 0; i < size; i++)
   {
      int d = -1;
      double sum = b(i);
      for (int j = I[i]; j < I[i+1]; j++)
      {
         if (J[j] == i)
            d = j;
         else
            sum -= A[j] * x0(J[j]);
      }
      if (d >= 0 && A[d] != 0.0)
         x1(i) = sc * (sum / A[d]) + (1.0 - sc) * x0(i);
      else
         mfem_error("SparseMatrix::Jacobi(...) #2");
   }
}

void SparseMatrix::DiagScale(const Vector &b, Vector &x, double sc) const
{
   if (A == NULL)
      mfem_error("SparseMatrix::DiagScale(...)");

   bool scale = (sc != 1.0);
   for (int i = 0, j = 0; i < size; i++)
   {
      int end = I[i+1];
      for ( ; true; j++)
      {
         if (j == end)
            goto diagscale_error;
         if (J[j] == i)
         {
            if (A[j] == 0.0)
               goto diagscale_error;
            if (scale)
               x(i) = sc * b(i) / A[j];
            else
               x(i) = b(i) / A[j];
            break;
         }
      }
      j = end;
   }
   return;

diagscale_error:
   mfem_error("SparseMatrix::DiagScale(...) #2");
}

void SparseMatrix::Jacobi2(const Vector &b, const Vector &x0, Vector &x1,
                           double sc) const
{
   if (A == NULL)
      mfem_error("SparseMatrix::Jacobi2(...)");

   for (int i = 0; i < size; i++)
   {
      double resi = b(i), norm = 0.0;
      for (int j = I[i]; j < I[i+1]; j++)
      {
         resi -= A[j] * x0(J[j]);
         norm += fabs(A[j]);
      }
      if (norm > 0.0)
         x1(i) = x0(i) + sc * resi / norm;
      else
         mfem_error("SparseMatrix::Jacobi2(...) #2");
   }
}

void SparseMatrix::AddSubMatrix(const Array<int> &rows, const Array<int> &cols,
                                const DenseMatrix &subm, int skip_zeros)
{
   int i, j, gi, gj, s, t;
   double a;

   for (i = 0; i < rows.Size(); i++)
   {
      if ((gi=rows[i]) < 0) gi = -1-gi, s = -1; else s = 1;
#ifdef MFEM_DEBUG
      if (gi >= size)
         mfem_error("SparseMatrix::AddSubMatrix(...) #1");
#endif
      SetColPtr(gi);
      for (j = 0; j < cols.Size(); j++)
      {
         if ((gj=cols[j]) < 0) gj = -1-gj, t = -s; else t = s;
#ifdef MFEM_DEBUG
         if (gj >= width)
            mfem_error("SparseMatrix::AddSubMatrix(...) #2");
#endif
         a = subm(i, j);
         if (skip_zeros && a == 0.0)
         {
            // if the element is zero do not assemble it unless this breaks
            // the symmetric structure
            if (&rows != &cols || subm(j, i) == 0.0)
               continue;
         }
         if (t < 0)  a = -a;
         _Add_(gj, a);
      }
      ClearColPtr();
   }
}

void SparseMatrix::Set(const int i, const int j, const double A)
{
   double a = A;
   int gi, gj, s, t;

   if ((gi=i) < 0) gi = -1-gi, s = -1; else s = 1;
#ifdef MFEM_DEBUG
   if (gi >= size)
      mfem_error("SparseMatrix::Set (...) #1");
#endif
   if ((gj=j) < 0) gj = -1-gj, t = -s; else t = s;
#ifdef MFEM_DEBUG
   if (gj >= width)
      mfem_error("SparseMatrix::Set (...) #2");
#endif
   if (t < 0)  a = -a;
   _Set_(gi, gj, a);
}

void SparseMatrix::Add(const int i, const int j, const double A)
{
   int gi, gj, s, t;
   double a = A;

   if ((gi=i) < 0) gi = -1-gi, s = -1; else s = 1;
#ifdef MFEM_DEBUG
   if (gi >= size)
      mfem_error("SparseMatrix::Add (...) #1");
#endif
   if ((gj=j) < 0) gj = -1-gj, t = -s; else t = s;
#ifdef MFEM_DEBUG
   if (gj >= width)
      mfem_error("SparseMatrix::Add (...) #2");
#endif
   if (t < 0)  a = -a;
   _Add_(gi, gj, a);
}

void SparseMatrix::SetSubMatrix(const Array<int> &rows, const Array<int> &cols,
                                const DenseMatrix &subm, int skip_zeros)
{
   int i, j, gi, gj, s, t;
   double a;

   for (i = 0; i < rows.Size(); i++)
   {
      if ((gi=rows[i]) < 0) gi = -1-gi, s = -1; else s = 1;
#ifdef MFEM_DEBUG
      if (gi >= size)
         mfem_error("SparseMatrix::SetSubMatrix(...) #1");
#endif
      SetColPtr(gi);
      for (j = 0; j < cols.Size(); j++)
      {
         a = subm(i, j);
         if (skip_zeros && a == 0.0)
            continue;
         if ((gj=cols[j]) < 0) gj = -1-gj, t = -s; else t = s;
#ifdef MFEM_DEBUG
         if (gj >= width)
            mfem_error("SparseMatrix::SetSubMatrix(...) #2");
#endif
         if (t < 0)  a = -a;
         _Set_(gj, a);
      }
      ClearColPtr();
   }
}

void SparseMatrix::SetSubMatrixTranspose(const Array<int> &rows,
                                         const Array<int> &cols,
                                         const DenseMatrix &subm,
                                         int skip_zeros)
{
   int i, j, gi, gj, s, t;
   double a;

   for (i = 0; i < rows.Size(); i++)
   {
      if ((gi=rows[i]) < 0) gi = -1-gi, s = -1; else s = 1;
#ifdef MFEM_DEBUG
      if (gi >= size)
         mfem_error("SparseMatrix::SetSubMatrixTranspose (...) #1");
#endif
      SetColPtr(gi);
      for (j = 0; j < cols.Size(); j++)
      {
         a = subm(j, i);
         if (skip_zeros && a == 0.0)
            continue;
         if ((gj=cols[j]) < 0) gj = -1-gj, t = -s; else t = s;
#ifdef MFEM_DEBUG
         if (gj >= width)
            mfem_error("SparseMatrix::SetSubMatrixTranspose (...) #2");
#endif
         if (t < 0)  a = -a;
         _Set_(gj, a);
      }
      ClearColPtr();
   }
}

void SparseMatrix::GetSubMatrix(const Array<int> &rows, const Array<int> &cols,
                                DenseMatrix &subm)
{
   int i, j, gi, gj, s, t;
   double a;

   for (i = 0; i < rows.Size(); i++)
   {
      if ((gi=rows[i]) < 0) gi = -1-gi, s = -1; else s = 1;
#ifdef MFEM_DEBUG
      if (gi >= size)
         mfem_error("SparseMatrix::GetSubMatrix(...) #1");
#endif
      SetColPtr(gi);
      for (j = 0; j < cols.Size(); j++)
      {
         if ((gj=cols[j]) < 0) gj = -1-gj, t = -s; else t = s;
#ifdef MFEM_DEBUG
         if (gj >= width)
            mfem_error("SparseMatrix::GetSubMatrix(...) #2");
#endif
         a = _Get_(gj);
         subm(i, j) = (t < 0) ? (-a) : (a);
      }
      ClearColPtr();
   }
}

bool SparseMatrix::RowIsEmpty(const int row) const
{
   int gi;

   if ((gi=row) < 0)
      gi = -1-gi;
#ifdef MFEM_DEBUG
   if (gi >= size)
      mfem_error("SparseMatrix::RowIsEmpty(...) #1");
#endif
   if (Rows)
      return (Rows[gi] == NULL);
   else
      return (I[gi] == I[gi+1]);
}

void SparseMatrix::GetRow(const int row, Array<int> &cols, Vector &srow) const
{
   RowNode *n;
   int j, gi;

   if ((gi=row) < 0) gi = -1-gi;
#ifdef MFEM_DEBUG
   if (gi >= size)
      mfem_error("SparseMatrix::GetRow(...) #1");
#endif
   if (Rows)
   {
      for (n = Rows[gi], j = 0; n; n = n->Prev)
         j++;
      cols.SetSize(j);
      srow.SetSize(j);
      for (n = Rows[gi], j = 0; n; n = n->Prev, j++)
      {
         cols[j] = n->Column;
         srow(j) = n->Value;
      }
      if (row < 0)
         srow.Neg();
   }
   else
   {
      j = I[gi];
      cols.MakeRef(J + j, I[gi+1]-j);
      srow.NewDataAndSize(A + j, cols.Size());
#ifdef DEBUG
      if (row < 0)
         mfem_error("SparseMatrix::GetRow(...) #2");
#endif
   }
}

void SparseMatrix::SetRow(const int row, const Array<int> &cols,
                          const Vector &srow)
{
   int j, gi, gj, s, t;
   double a;

   if (Rows == NULL)
      mfem_error("SparseMatrix::SetRow(...) #0");

   if ((gi=row) < 0) gi = -1-gi, s = -1; else s = 1;
#ifdef MFEM_DEBUG
   if (gi >= size)
      mfem_error("SparseMatrix::SetRow(...) #1");
#endif
   SetColPtr(gi);
   for (j = 0; j < cols.Size(); j++)
   {
      if ((gj=cols[j]) < 0) gj = -1-gj, t = -s; else t = s;
#ifdef MFEM_DEBUG
      if (gj >= width)
         mfem_error("SparseMatrix::SetRow(...) #2");
#endif
      a = srow(j);
      if (t < 0)  a = -a;
      _Set_(gj, a);
   }
   ClearColPtr();
}

void SparseMatrix::AddRow(const int row, const Array<int> &cols,
                          const Vector &srow)
{
   int j, gi, gj, s, t;
   double a;

   if (Rows == NULL)
      mfem_error("SparseMatrix::AddRow(...) #0");

   if ((gi=row) < 0) gi = -1-gi, s = -1; else s = 1;
#ifdef MFEM_DEBUG
   if (gi >= size)
      mfem_error("SparseMatrix::AddRow(...) #1");
#endif
   SetColPtr(gi);
   for (j = 0; j < cols.Size(); j++)
   {
      if ((gj=cols[j]) < 0) gj = -1-gj, t = -s; else t = s;
#ifdef MFEM_DEBUG
      if (gj >= width)
         mfem_error("SparseMatrix::AddRow(...) #2");
#endif
      a = srow(j);
      if (a == 0.0)
         continue;
      if (t < 0)  a = -a;
      _Add_(gj, a);
   }
   ClearColPtr();
}

void SparseMatrix::ScaleRow(const int row, const double scale)
{
   int i;

   if ((i=row) < 0)
      i = -1-i;
   if (Rows != NULL)
   {
      RowNode *aux;

      for (aux = Rows[i]; aux != NULL; aux = aux -> Prev)
         aux -> Value *= scale;
   }
   else
   {
      int j, end = I[i+1];

      for (j = I[i]; j < end; j++)
         A[j] *= scale;
   }
}

void SparseMatrix::ScaleRows(const Vector & sl)
{
   double scale;
   if (Rows != NULL)
   {
      RowNode *aux;
      for (int i=0; i < size; ++i)
      {
         scale = sl(i);
         for (aux = Rows[i]; aux != NULL; aux = aux -> Prev)
            aux -> Value *= scale;
      }
   }
   else
   {
      int j, end;

      for (int i=0; i < size; ++i)
      {
         end = I[i+1];
         scale = sl(i);
         for (j = I[i]; j < end; j++)
            A[j] *= scale;
      }
   }
}

void SparseMatrix::ScaleColumns(const Vector & sr)
{
   if (Rows != NULL)
   {
      RowNode *aux;
      for (int i=0; i < size; ++i)
      {
         for (aux = Rows[i]; aux != NULL; aux = aux -> Prev)
            aux -> Value *= sr(aux->Column);
      }
   }
   else
   {
      int j, end;

      for (int i=0; i < size; ++i)
      {
         end = I[i+1];
         for (j = I[i]; j < end; j++)
            A[j] *= sr(J[j]);
      }
   }
}

SparseMatrix &SparseMatrix::operator+=(SparseMatrix &B)
{
#ifdef MFEM_DEBUG
   if (size != B.size || width != B.width)
      mfem_error("SparseMatrix::operator+=(...) #1");
#endif

   for (int i = 0; i < size; i++)
   {
      SetColPtr(i);
      if (B.Rows)
      {
         for (RowNode *aux = B.Rows[i]; aux != NULL; aux = aux->Prev)
         {
            _Add_(aux->Column, aux->Value);
         }
      }
      else
      {
         for (int j = B.I[i]; j < B.I[i+1]; j++)
         {
            _Add_(B.J[j], B.A[j]);
         }
      }
      ClearColPtr();
   }

   return (*this);
}

void SparseMatrix::Add(const double a, SparseMatrix &B)
{
   for (int i = 0; i < size; i++)
   {
      B.SetColPtr(i);
      if (Rows)
      {
         for (RowNode *np = Rows[i]; np != NULL; np = np->Prev)
            np->Value += a * B._Get_(np->Column);
      }
      else
      {
         for (int j = I[i]; j < I[i+1]; j++)
            A[j] += a * B._Get_(J[j]);
      }
      B.ClearColPtr();
   }
}

SparseMatrix &SparseMatrix::operator=(double a)
{
   if (Rows == NULL)
      for (int i = 0, nnz = I[size]; i < nnz; i++)
         A[i] = a;
   else
      for (int i = 0; i < size; i++)
         for (RowNode *node_p = Rows[i]; node_p != NULL;
              node_p = node_p -> Prev)
            node_p -> Value = a;

   return (*this);
}

SparseMatrix &SparseMatrix::operator*=(double a)
{
   if (Rows == NULL)
      for (int i = 0, nnz = I[size]; i < nnz; i++)
         A[i] *= a;
   else
      for (int i = 0; i < size; i++)
         for (RowNode *node_p = Rows[i]; node_p != NULL;
              node_p = node_p -> Prev)
            node_p -> Value *= a;

   return (*this);
}

void SparseMatrix::Print(ostream & out, int _width) const
{
   int i, j;

   if (A == NULL)
   {
      RowNode *nd;
      for (i = 0; i < size; i++)
      {
         out << "[row " << i << "]\n";
         for (nd = Rows[i], j = 0; nd != NULL; nd = nd->Prev, j++)
         {
            out << " (" << nd->Column << ","<< nd->Value << ")";
            if ( !((j+1) % _width) )
               out << '\n';
         }
         if (j % _width)
            out << '\n';
      }
      return;
   }

   for (i = 0; i < size; i++)
   {
      out << "[row " << i << "]\n";
      for (j = I[i]; j < I[i+1]; j++)
      {
         out << " (" << J[j] << ","<< A[j] << ")";
         if ( !((j+1-I[i]) % _width) )
            out << '\n';
      }
      if ((j-I[i]) % _width)
         out << '\n';
   }
}

void SparseMatrix::PrintMatlab(ostream & out) const
{
   out << "% size " << size << " " << width << "\n";
   out << "% Non Zeros " << NumNonZeroElems() << "\n";
   int i, j;
   ios::fmtflags old_fmt = out.flags();
   out.setf(ios::scientific);
   int old_prec = out.precision(14);

   for(i = 0; i < size; i++)
      for (j = I[i]; j < I[i+1]; j++)
         out << i+1 << " " << J[j]+1 << " " << A[j] << endl;
   out.precision(old_prec);
   out.flags(old_fmt);
}

void SparseMatrix::PrintMM(ostream & out) const
{
   int i, j;
   ios::fmtflags old_fmt = out.flags();
   out.setf(ios::scientific);
   int old_prec = out.precision(14);

   out << "%%MatrixMarket matrix coordinate real general" << endl
       << "% Generated by MFEM" << endl;

   out << size << " " << width << " " << NumNonZeroElems() << endl;
   for(i = 0; i < size; i++)
      for (j = I[i]; j < I[i+1]; j++)
         out << i+1 << " " << J[j]+1 << " " << A[j] << endl;
   out.precision(old_prec);
   out.flags(old_fmt);
}

void SparseMatrix::PrintCSR(ostream & out) const
{
   if (A == NULL)
      mfem_error("SparseMatrix::PrintCSR()");

   int i;

   out << size << '\n';  // number of rows

   for (i = 0; i <= size; i++)
      out << I[i]+1 << '\n';

   for (i = 0; i < I[size]; i++)
      out << J[i]+1 << '\n';

   for (i = 0; i < I[size]; i++)
      out << A[i] << '\n';
}

void SparseMatrix::PrintCSR2(ostream & out) const
{
   if (A == NULL)
      mfem_error("SparseMatrix::PrintCSR2()");

   int i;

   out << size << '\n';  // number of rows
   out << width << '\n';  // number of columns

   for (i = 0; i <= size; i++)
      out << I[i] << '\n';

   for (i = 0; i < I[size]; i++)
      out << J[i] << '\n';

   for (i = 0; i < I[size]; i++)
      out << A[i] << '\n';
}

SparseMatrix::~SparseMatrix ()
{
#ifdef MFEM_USE_MEMALLOC
	delete NodesMem;
#endif
   if (Rows != NULL)
   {
      delete [] ColPtr.Node;
#ifdef MFEM_USE_MEMALLOC
      // Do nothing
#else
      for (int i = 0; i < size; i++)
      {
         RowNode *aux, *node_p = Rows[i];
         while (node_p != NULL)
         {
            aux = node_p;
            node_p = node_p->Prev;
            delete aux;
         }
      }
#endif
      delete [] Rows;
   }
   if (A != NULL)
   {
      delete [] ColPtr.J;
      delete [] I;
      delete [] J;
      delete [] A;
   }
}

int SparseMatrix::actualWidth()
{
   int awidth = 0;
   if(A)
   {
      int * start_j(J);
      int * end_j(J + I[size]);
      for(int * jptr(start_j); jptr != end_j; ++jptr)
         awidth = (*jptr > awidth) ? *jptr : awidth;
   }
   else
   {
      RowNode *aux;
      for (int i = 0; i < size; i++)
         for (aux = Rows[i]; aux != NULL; aux = aux->Prev)
            awidth =(aux->Column > awidth) ? aux->Column : awidth;
   }
   ++awidth;

   return awidth;

}

void SparseMatrixFunction (SparseMatrix & S, double (*f)(double))
{
   int n = S.NumNonZeroElems();
   double * s = S.GetData();

   for (int i = 0; i < n; i++)
      s[i] = f(s[i]);
}

SparseMatrix *Transpose (const SparseMatrix &A)
{
   if (!A.Finalized())
      mfem_error("Finalize must be called before Transpose. Use"
                 " TransposeRowMatrix instead");

   int i, j, end;
   int m, n, nnz, *A_i, *A_j, *At_i, *At_j;
   double *A_data, *At_data;

   m      = A.Size();   // number of rows of A
   n      = A.Width();  // number of columns of A
   nnz    = A.NumNonZeroElems();
   A_i    = A.GetI();
   A_j    = A.GetJ();
   A_data = A.GetData();

   At_i = new int[n+1];
   At_j = new int[nnz];
   At_data = new double[nnz];

   for (i = 0; i <= n; i++)
      At_i[i] = 0;
   for (i = 0; i < nnz; i++)
      At_i[A_j[i]+1]++;
   for (i = 1; i < n; i++)
      At_i[i+1] += At_i[i];

   for (i = j = 0; i < m; i++)
   {
      end = A_i[i+1];
      for ( ; j < end; j++)
      {
         At_j[At_i[A_j[j]]] = i;
         At_data[At_i[A_j[j]]] = A_data[j];
         At_i[A_j[j]]++;
      }
   }

   for (i = n; i > 0; i--)
      At_i[i] = At_i[i-1];
   At_i[0] = 0;

   return  new SparseMatrix (At_i, At_j, At_data, n, m);
}

SparseMatrix *TransposeRowMatrix (const SparseMatrix &A, int useActualWidth)
{
   int i, j;
   int m, n, nnz, *At_i, *At_j;
   double *At_data;
   Array<int> Acols;
   Vector Avals;

   m = A.Size(); // number of rows of A
   if (useActualWidth)
   {
      n = 0;
      int tmp;
      for (i = 0; i < m; i++)
      {
         A.GetRow(i, Acols, Avals);
         if (Acols.Size())
         {
            tmp = Acols.Max();
            if (tmp > n)
               n = tmp;
         }
      }
      ++n;
   }
   else
   {
      n = A.Width(); // number of columns of A
   }
   nnz = A.NumNonZeroElems();

   At_i = new int[n+1];
   At_j = new int[nnz];
   At_data = new double[nnz];

   for (i = 0; i <= n; i++)
      At_i[i] = 0;

   for (i = 0; i < m; i++)
   {
      A.GetRow(i, Acols, Avals);
      for (j = 0; j<Acols.Size(); ++j)
         At_i[Acols[j]+1]++;
   }
   for (i = 1; i < n; i++)
      At_i[i+1] += At_i[i];

   for (i = 0; i < m; i++)
   {
      A.GetRow(i, Acols, Avals);
      for (j = 0; j<Acols.Size(); ++j)
      {
         At_j[At_i[Acols[j]]] = i;
         At_data[At_i[Acols[j]]] = Avals[j];
         At_i[Acols[j]]++;
      }
   }

   for (i = n; i > 0; i--)
      At_i[i] = At_i[i-1];
   At_i[0] = 0;

   return new SparseMatrix(At_i, At_j, At_data, n, m);
}


SparseMatrix *Mult (const SparseMatrix &A, const SparseMatrix &B,
                    SparseMatrix *OAB)
{
   int nrowsA, ncolsA, nrowsB, ncolsB;
   int *A_i, *A_j, *B_i, *B_j, *C_i, *C_j, *B_marker;
   double *A_data, *B_data, *C_data;
   int ia, ib, ic, ja, jb, num_nonzeros;
   int row_start, counter;
   double a_entry, b_entry;
   SparseMatrix *C;

   nrowsA = A.Size();
   ncolsA = A.Width();
   nrowsB = B.Size();
   ncolsB = B.Width();

   if (ncolsA != nrowsB)
      mfem_error("Sparse matrix multiplication, Mult (...) #1");

   A_i    = A.GetI();
   A_j    = A.GetJ();
   A_data = A.GetData();
   B_i    = B.GetI();
   B_j    = B.GetJ();
   B_data = B.GetData();

   B_marker = new int[ncolsB];

   for (ib = 0; ib < ncolsB; ib++)
      B_marker[ib] = -1;

   if (OAB == NULL)
   {
      C_i = new int[nrowsA+1];

      C_i[0] = num_nonzeros = 0;
      for (ic = 0; ic < nrowsA; ic++)
      {
         for (ia = A_i[ic]; ia < A_i[ic+1]; ia++)
         {
            ja = A_j[ia];
            for (ib = B_i[ja]; ib < B_i[ja+1]; ib++)
            {
               jb = B_j[ib];
               if (B_marker[jb] != ic)
               {
                  B_marker[jb] = ic;
                  num_nonzeros++;
               }
            }
         }
         C_i[ic+1] = num_nonzeros;
      }

      C_j    = new int[num_nonzeros];
      C_data = new double[num_nonzeros];

      C = new SparseMatrix (C_i, C_j, C_data, nrowsA, ncolsB);

      for (ib = 0; ib < ncolsB; ib++)
         B_marker[ib] = -1;
   }
   else
   {
      C = OAB;

      if (nrowsA != C -> Size() || ncolsB != C -> Width())
         mfem_error("Sparse matrix multiplication, Mult (...) #2");

      C_i    = C -> GetI();
      C_j    = C -> GetJ();
      C_data = C -> GetData();
   }

   counter = 0;
   for (ic = 0; ic < nrowsA; ic++)
   {
      // row_start = C_i[ic];
      row_start = counter;
      for (ia = A_i[ic]; ia < A_i[ic+1]; ia++)
      {
         ja = A_j[ia];
         a_entry = A_data[ia];
         for (ib = B_i[ja]; ib < B_i[ja+1]; ib++)
         {
            jb = B_j[ib];
            b_entry = B_data[ib];
            if (B_marker[jb] < row_start)
            {
               B_marker[jb] = counter;
               if (OAB == NULL)
                  C_j[counter] = jb;
               C_data[counter] = a_entry*b_entry;
               counter++;
            }
            else
               C_data[B_marker[jb]] += a_entry*b_entry;
         }
      }
   }

   if (OAB != NULL && counter != OAB -> NumNonZeroElems())
      mfem_error("Sparse matrix multiplication, Mult (...) #3");

   delete [] B_marker;

   return C;
}

SparseMatrix *MultRowMatrix (const SparseMatrix &A, const SparseMatrix &B)
{
   int nrowsA, ncolsA, nrowsB, ncolsB;
   int *C_i, *C_j, *B_marker;
   double *C_data;
   int ia, ib, ic, ja, jb, num_nonzeros;
   int row_start, counter;
   double a_entry, b_entry;
   SparseMatrix *C;

   nrowsA = A.Size();
   ncolsA = A.Width();
   nrowsB = B.Size();
   ncolsB = B.Width();

   if (ncolsA != nrowsB)
      mfem_error("Sparse matrix multiplication, Mult (...) #1");


   B_marker = new int[ncolsB];

   for (ib = 0; ib < ncolsB; ib++)
      B_marker[ib] = -1;

   C_i = new int[nrowsA+1];

   C_i[0] = num_nonzeros = 0;

   Array<int> colsA, colsB;
   Vector dataA, dataB;
   for (ic = 0; ic < nrowsA; ic++)
   {
      A.GetRow(ic, colsA, dataA);
      for (ia = 0; ia < colsA.Size(); ia++)
      {
         ja = colsA[ia];
         B.GetRow(ja, colsB, dataB);
         for (ib = 0; ib < colsB.Size(); ib++)
         {
            jb = colsB[ib];
            if (B_marker[jb] != ic)
            {
               B_marker[jb] = ic;
               num_nonzeros++;
            }
         }
      }
      C_i[ic+1] = num_nonzeros;
   }

   C_j    = new int[num_nonzeros];
   C_data = new double[num_nonzeros];

   C = new SparseMatrix(C_i, C_j, C_data, nrowsA, ncolsB);

   for (ib = 0; ib < ncolsB; ib++)
      B_marker[ib] = -1;

   counter = 0;
   for (ic = 0; ic < nrowsA; ic++)
   {
      row_start = counter;
      A.GetRow(ic, colsA, dataA);
      for (ia = 0; ia < colsA.Size(); ia++)
      {
         ja = colsA[ia];
         a_entry = dataA[ia];
         B.GetRow(ja, colsB, dataB);
         for (ib = 0; ib < colsB.Size(); ib++)
         {
            jb = colsB[ib];
            b_entry = dataB[ib];
            if (B_marker[jb] < row_start)
            {
               B_marker[jb] = counter;
               C_j[counter] = jb;
               C_data[counter] = a_entry*b_entry;
               counter++;
            }
            else
               C_data[B_marker[jb]] += a_entry*b_entry;
         }
      }
   }

   delete [] B_marker;

   return C;
}

SparseMatrix *RAP (const SparseMatrix &A, const SparseMatrix &R,
                   SparseMatrix *ORAP)
{
   SparseMatrix *P  = Transpose (R);
   SparseMatrix *AP = Mult (A, *P);
   delete P;
   SparseMatrix *_RAP = Mult (R, *AP, ORAP);
   delete AP;
   return _RAP;
}

SparseMatrix *Mult_AtDA (const SparseMatrix &A, const Vector &D,
                         SparseMatrix *OAtDA)
{
   int i, At_nnz, *At_j;
   double *At_data;

   SparseMatrix *At = Transpose (A);
   At_nnz  = At -> NumNonZeroElems();
   At_j    = At -> GetJ();
   At_data = At -> GetData();
   for (i = 0; i < At_nnz; i++)
      At_data[i] *= D(At_j[i]);
   SparseMatrix *AtDA = Mult (*At, A, OAtDA);
   delete At;
   return AtDA;
}

void Swap(SparseMatrix & A, SparseMatrix & B)
{
   int width(A.width); A.width = B.width; B.width = width;
   int size(A.size);   A.size = B.size; B.size = size;
   int * I(A.I);       A.I = B.I;  B.I = I;
   int * J(A.J);       A.J = B.J;  B.J = J;
   double * val(A.A);  A.A = B.A;  B.A = val;
   RowNode **rows(A.Rows); A.Rows = B.Rows; B.Rows = rows;
   int current_row(A.current_row); A.current_row = B.current_row; B.current_row = current_row;
   J = A.ColPtr.J; A.ColPtr.J = B.ColPtr.J; B.ColPtr.J = J;
#ifdef MFEM_USE_MEMALLOC
   SparseMatrix::RowNodeAlloc * NodesMem(A.NodesMem); A.NodesMem = B.NodesMem; B.NodesMem = NodesMem;
#endif
}
