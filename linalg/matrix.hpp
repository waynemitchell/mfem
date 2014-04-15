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

#ifndef MFEM_MATRIX
#define MFEM_MATRIX

// Abstract data types matrix, inverse matrix

#include "../general/array.hpp"
#include "operator.hpp"

class  MatrixInverse;

/// Abstract data type matrix
class Matrix : public Operator
{
   friend class MatrixInverse;
public:
   /// Creates matrix of width s.
   explicit Matrix (int s) { size=s; }

   /// Returns reference to a_{ij}.  Index i, j = 0 .. size-1
   virtual double& Elem (int i, int j) = 0;

   /// Returns constant reference to a_{ij}.  Index i, j = 0 .. size-1
   virtual const double& Elem (int i, int j) const = 0;

   /// Returns a pointer to (approximation) of the matrix inverse.
   virtual MatrixInverse * Inverse() const = 0;

   /// Finalizes the matrix initialization.
   virtual void Finalize(int) { }

   /// Prints matrix to stream out.
   virtual void Print (ostream & out = cout, int width = 4) const;

   /// Destroys matrix.
   virtual ~Matrix() { }
};


/// Abstract data type for matrix inverse
class MatrixInverse : public Solver
{
public:
   MatrixInverse() { }

   /// Creates approximation of the inverse of square matrix
   MatrixInverse(const Matrix &mat)
      : Solver(mat.size) { }
};

/// Abstract data type for sparse matrices
class SparseRowMatrix : public Matrix
{
public:
	   /// Creates matrix of width s.
	   explicit SparseRowMatrix (int s):Matrix(s){};
	   /// Returns the Width of the matrix
	   virtual int Width() const = 0;
	   /// Returns the number of non-zeros in a matrix
	   virtual int NumNonZeroElems() const = 0;

	   /// Gets the columns indeces and values for row *row*. Returns 0 if cols and srow are copies, 1 if they are references.
	   virtual int GetRow(const int row, Array<int> &cols, Vector &srow) const = 0;
	   /// If the matrix is square, it will place 1 on the diagonal (i,i) if row i has "almost" zero l1-norm.
	   /*
	    * If entry (i,i) does not belong to the sparsity pattern of A, then a error will occur.
	    */
	   virtual void EliminateZeroRows() = 0;

	   /// Matrix-Vector Multiplication y = A*x
	   virtual void Mult(const Vector & x, Vector & y) const = 0;
	   /// Matrix-Vector Multiplication y += A*x
	   virtual void AddMult(const Vector & x, Vector & y, const double val = 1.) const = 0;
	   /// MatrixTranspose-Vector Multiplication y = A'*x
	   virtual void MultTranspose(const Vector & x, Vector & y) const = 0;
	   /// MatrixTranspose-Vector Multiplication y += A'*x
	   virtual void AddMultTranspose(const Vector & x, Vector & y, const double val = 1.) const = 0;


	   /// Destroys SparseRowMatrix.
	   virtual ~SparseRowMatrix() { };
};

#endif
