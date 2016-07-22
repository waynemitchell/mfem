// Copyright (c) 2016, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#ifndef MFEM_PETSC
#define MFEM_PETSC

#include "../config/config.hpp"

#ifdef MFEM_USE_PETSC
#ifdef MFEM_USE_MPI
#include <petsc.h>

namespace mfem
{

class ParFiniteElementSpace;
class PetscParMatrix;

/// Wrapper for PETSc's vector class
class PetscParVector : public Vector
{
protected:
   /// The actual object
   Vec x;

   friend class PetscParMatrix;
   friend class PetscSolver;

   // Set Vector::data and Vector::size from x
   void _SetDataAndSize_();

public:
   /** Creates vector with given global size and partitioning of the columns.
       Processor P owns columns [col[P],col[P+1]) */
   PetscParVector(MPI_Comm comm, PetscInt glob_size, PetscInt *col);

   /// Calls PETSc's destroy function
   ~PetscParVector();

   /** Creates vector with given global size, partitioning of the columns,
       and data. The data must be allocated and destroyed outside.
       If _data is NULL, a dummy vector without a valid data array will
       be created. */
   PetscParVector(MPI_Comm comm, PetscInt glob_size, PetscScalar *_data,
                  PetscInt *col);

   /// Creates vector compatible with y
   PetscParVector(const PetscParVector &y);

   /// Creates vector compatible with (i.e. in the domain of) A or A^T
   explicit PetscParVector(const PetscParMatrix &A, int transpose = 0);

   /// Creates vector wrapping y
   explicit PetscParVector(Vec y);

   /// Create a true dof parallel vector on a given ParFiniteElementSpace
   explicit PetscParVector(ParFiniteElementSpace *pfes);

   /// MPI communicator
   MPI_Comm GetComm() const { return PetscObjectComm((PetscObject)x); }

   /// Returns the global number of rows
   PetscInt GlobalSize();

   /// Typecasting to PETSc's Vec
   operator Vec() const { return x; }

   /// Create HypreParVector referencing the data in Vec
   HypreParVector* GetHypreParVector(Vec y);

   /// Create HypreParVector compatible with B or B^T (data not allocated)
   HypreParVector* GetHypreParVector(Mat B, bool transpose);

   /// Returns the global vector in each processor
   Vector* GlobalVector() const;

   /// Set constant values
   PetscParVector& operator= (PetscScalar d);

   /// Define '=' for PETSc vectors.
   PetscParVector& operator= (const PetscParVector &y);

   /// Replace PetscParVector array
   void SetData(PetscScalar *_data);

   /// Restore PetscParVector array to its orignal state
   void ResetData();

   /// Set random values
   void Randomize(PetscInt seed);

   /// Prints the locally owned rows in parallel
   void Print(const char *fname) const;
};

/// Wrapper for PETSc's matrix class
class PetscParMatrix : public Operator
{
protected:
   /// The actual object
   Mat A;

   /// Auxiliary vectors for typecasting
   mutable PetscParVector *X, *Y;

   /// Initialize with defaults. Does not initialize inherited members.
   void Init();

   /// Delete all owned data. Does not perform re-initialization with defaults.
   void Destroy();

   /// Creates a wrapper around HypreParMatrix using PETSc's MATSHELL object
   void MakeWrapper(const HypreParMatrix* hmat);

   friend class PetscSolver;

public:
   /// An empty matrix to be used as a reference to an existing matrix
   PetscParMatrix();

   /// Calls PETSc's destroy function
   virtual ~PetscParMatrix() { Destroy(); }

   /// Creates PetscParMatrix out of Mat
   PetscParMatrix(Mat a);

   /// Converts HypreParMatrix to PetscParMatrix
   PetscParMatrix(const HypreParMatrix a);

   /// MatMult
   virtual void Mult(const Vector &x, Vector &y) const;

   /// MPI communicator
   MPI_Comm GetComm() const { return PetscObjectComm((PetscObject)A); }

   /// Typecasting to PETSc's Mat
   operator Mat() { return A; }

   /// Returns the global number of rows
   PetscInt M();

   /// Returns the global number of columns
   PetscInt N();

};

/// Abstract class for PETSc's solvers and preconditioners
class PetscSolver : public Solver
{
protected:
   /// The Krylov object
   KSP ksp;

   /// Right-hand side and solution vector
   mutable PetscParVector *B, *X;

private:
   bool wrap; // internal flag to handle HypreParMatrix conversion or not

   void Init();

public:
   /// Initialize protected objects to NULL
   PetscSolver();

   /// Constructs a solver using a PetscParMatrix
   PetscSolver(PetscParMatrix &_A);

   /// Customization
   void SetTol(double tol);
   void SetMaxIter(int max_iter);
   void SetPrintLevel(int plev);

   /// Constructs a solver using a HypreParMatrix. If wrap is true, then the MatMult ops of HypreParMatrix are wrapped. No preconditioner can be automatically constructed from PETSc. If wrap is false, the HypreParMatrix is converted into PETSc format.
   PetscSolver(HypreParMatrix &_A,bool wrap=true);

   /// Set the solver to be used as a preconditioner
   void SetPreconditioner(Solver &precond);

   /// Typecast to KSP -- returns the inner solver
   //virtual operator KSP() const = 0;

   virtual void SetOperator(const Operator &op);

   /// Solve the linear system Ax=b
   virtual void Mult(const PetscParVector &b, PetscParVector &x) const;
   virtual void Mult(const Vector &b, Vector &x) const;

   virtual ~PetscSolver();
};

}

#endif // MFEM_USE_MPI
#endif // MFEM_USE_PETSC

#endif
