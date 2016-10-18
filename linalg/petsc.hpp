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

// Author: Stefano Zampini <stefano.zampini@gmail.com>

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
   friend class PetscBDDCSolver;

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

   /// Creates PetscParVector out of PETSc's Mat
   /// If ref is true, we increase the reference count of the PETSc's Vec object
   explicit PetscParVector(Vec y, bool ref=false);

   /// Create a true dof parallel vector on a given ParFiniteElementSpace
   explicit PetscParVector(ParFiniteElementSpace *pfes);

   /// MPI communicator
   MPI_Comm GetComm() const { return PetscObjectComm((PetscObject)x); }

   /// Returns the global number of rows
   PetscInt GlobalSize();

   /// Typecasting to PETSc's Vec
   operator Vec() const { return x; }

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

   /// Prints the vector (to stdout if fname is NULL)
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

   /// Creates a wrapper around a mfem::Operator op using PETSc's MATSHELL object
   /// and returns the Mat in B. This does not take any reference to op, that
   /// should not be destroyed until B is needed.
   void MakeWrapper(MPI_Comm comm, const Operator* op, Mat *B);

   /// Convert a mfem::Operator into a Mat B
   /// Op can be destroyed
   void ConvertOperator(MPI_Comm comm, const Operator& op, Mat *B);

   friend class PetscLinearSolver;
   friend class PetscPreconditioner;

private:
   /// Constructs a block-diagonal Mat object
   void BlockDiagonalConstructor(MPI_Comm comm, PetscInt global_num_rows,
                                 PetscInt global_num_cols, PetscInt *row_starts,
                                 PetscInt *col_starts, SparseMatrix *diag,
                                 bool assembled, Mat *A);

public:
   /// An empty matrix to be used as a reference to an existing matrix.
   PetscParMatrix();

   /// Calls PETSc's destroy function.
   virtual ~PetscParMatrix() { Destroy(); }

   /// Creates PetscParMatrix out of PETSc's Mat.
   /// If ref is true, we increase the reference count of PETSc's Mat object.
   PetscParMatrix(Mat a, bool ref=false);

   /// Converts HypreParMatrix to PetscParMatrix
   /// If wrap is false, a PETSc's MATAIJ (resp. MATIS) object
   /// is created if assembled is true (resp. false).
   /// If wrap is true, the matvec operations of the HypreParMatrix are wrapped
   /// through PETSc's MATSHELL object. In this case, the HypreParMatrix should
   /// not be destroyed before the PetscParMatrix; assembled is not referenced.
   PetscParMatrix(const HypreParMatrix* a, bool wrap=false, bool assembled=true);

   /// If wrap is true, it converts any mfem::Operator op implementing Mult and
   /// MultTranspose into a PetscParMatrix. The Operator destructor should not
   /// be called before the one of the PetscParMatrix.
   /// If wrap is false, it tries to convert the operator in PETSc's classes.
   PetscParMatrix(MPI_Comm comm, const Operator* op, bool wrap = true);

   /** Creates block-diagonal square parallel matrix. Diagonal is given by diag
       which must be in CSR format (finalized). The new PetscParMatrix does not
       take ownership of any of the input arrays.
       If assembled is false, a MATIS object is constructed.
       Otherwise, a MATAIJ (parallel distributed CSR) is used */
   PetscParMatrix(MPI_Comm comm, PetscInt glob_size, PetscInt *row_starts,
                  SparseMatrix *diag, bool assembled = true);

   /** Creates block-diagonal rectangular parallel matrix. Diagonal is given by
       diag which must be in CSR format (finalized). The new PetscParMatrix does
       not take ownership of any of the input arrays.
       If assembled is false, a MATIS object is constructed.
       Otherwise, a MATAIJ (parallel distributed CSR) is used */
   PetscParMatrix(MPI_Comm comm, PetscInt global_num_rows,
                  PetscInt global_num_cols, PetscInt *row_starts,
                  PetscInt *col_starts, SparseMatrix *diag, bool assembled = true);

   // MatMultAdd operations
   void Mult(double a, const Vector &x, double b, Vector &y) const;

   // MatMultTransoseAdd operations
   void MultTranspose(double a, const Vector &x, double b, Vector &y) const;

   virtual void Mult(const Vector &x, Vector &y) const
   { Mult(1.0, x, 0.0, y); }

   virtual void MultTranspose(const Vector &x, Vector &y) const
   { MultTranspose(1.0, x, 0.0, y); }

   /// MPI communicator
   MPI_Comm GetComm() const { return PetscObjectComm((PetscObject)A); }

   /// Typecasting to PETSc's Mat
   operator Mat() { return A; }

   /// Returns the local number of rows
   PetscInt GetNumRows();

   /// Returns the local number of columns
   PetscInt GetNumCols();

   /// Returns the global number of rows
   PetscInt M();

   /// Returns the global number of columns
   PetscInt N();

   /// Returns the global number of rows
   PetscInt GetGlobalNumRows() { return M(); };

   /// Returns the global number of columns
   PetscInt GetGlobalNumCols() { return N(); };

   /// Returns the number of nonzeros
   /// Differently from HYPRE, this call is collective on the communicator,
   /// as this number is not stored inside PETSc, but needs to be computed
   PetscInt NNZ();

   /// Returns the inner vector in the domain of A (it creates it if needed)
   PetscParVector* GetX() const;

   /// Returns the inner vector in the range of A (it creates it if needed)
   PetscParVector* GetY() const;

   /// Returns the transpose of *this if action is false
   /// If action is true, then the matrix is not actually transposed.
   /// Instead, an object that behaves like the transpose is returned.
   PetscParMatrix* Transpose(bool action = false);

   /// Prints the matrix (to stdout if fname is NULL)
   void Print(const char *fname) const;

   /// Scale all entries by s: A_scaled = s*A.
   void operator*=(double s);

   /** Eliminate rows and columns from the matrix, and rows from the vector B.
       Modify B with the BC values in X. */
   void EliminateRowsCols(const Array<int> &rows_cols, const HypreParVector &X,
                          HypreParVector &B);

   /** Eliminate rows and columns from the matrix and store the eliminated
      elements in a new matrix Ae (returned), so that the modified matrix and
      Ae sum to the original matrix. */
   PetscParMatrix* EliminateRowsCols(const Array<int> &rows_cols);

   /// Takes a reference to another PetscParMatrix
   void MakeRef(const PetscParMatrix &master);

};

/// Returns the matrix Rt^t * A * P
PetscParMatrix * RAP(PetscParMatrix *Rt, PetscParMatrix *A, PetscParMatrix *P);

/// Returns the matrix P^t * A * P
PetscParMatrix * RAP(PetscParMatrix *A, PetscParMatrix *P);

/** Eliminate essential BC specified by 'ess_dof_list' from the solution X to
    the r.h.s. B. Here A is a matrix with eliminated BC, while Ae is such that
    (A+Ae) is the original (Neumann) matrix before elimination. */
void EliminateBC(PetscParMatrix &A, PetscParMatrix &Ae,
                 const Array<int> &ess_dof_list, const Vector &X, Vector &B);

/// Abstract class for PETSc's solvers
class PetscSolver : public Solver
{
protected:
   /// The actual PETSc object (KSP, SNES or TS)
   PetscObject obj;

   /// The class is of the actual PETSc object (KSP, SNES or TS)
   PetscClassId cid;

   /// The string prefix
   std::string _prefix;

   /// booleans to handle SetFromOptions calls
   mutable bool clcustom;
   mutable bool _prset;

   /// Right-hand side and solution vector
   mutable PetscParVector *B, *X;

   /// Set prefix for PETSc's options database
   void SetPrefix(std::string prefix);

   /// Call SetFromOptions
   void Customize() const;

private:
   void Init();

public:
   /// Initialize protected objects to NULL
   PetscSolver();

   /// Destroy the PetscParVectors allocated (if any)
   virtual ~PetscSolver();

   // virtual methods for base class
   virtual void SetOperator(const Operator &op)
   {
      MFEM_ABORT("Set Operator not implemented!")
   };
   virtual void Mult(const Vector &b, Vector &x) const;

   void SetTol(double tol);
   void SetRelTol(double tol);
   void SetAbsTol(double tol);
   void SetMaxIter(int max_iter);
   void SetPrintLevel(int plev);
   void Mult(const PetscParVector &b, PetscParVector &x) const;
   int GetConverged();
   int GetNumIterations();
   double GetFinalNorm();

   /// Typecasting to PETScObject
   operator PetscObject() const { return obj; }
};

/// Abstract class for PETSc's linear solvers
/// It inherits from Solver and not from IterativeSolver because we let PETSc
/// handle the various SetOperator/SetPreconditioner methods.
class PetscLinearSolver : public PetscSolver
{
private:
   bool wrap; // internal flag to handle HypreParMatrix conversion or not
   void Init();

public:
   PetscLinearSolver(MPI_Comm comm, std::string prefix = std::string());

   PetscLinearSolver(PetscParMatrix &_A, std::string prefix = std::string());

   /// Constructs a solver using a HypreParMatrix.
   /// If wrap is true, then the MatMult ops of HypreParMatrix are wrapped.
   /// No preconditioner can be automatically constructed from PETSc.
   /// If wrap is false, the HypreParMatrix is converted into PETSc format.
   PetscLinearSolver(HypreParMatrix &_A,bool wrap = true,
                     std::string prefix = std::string());

   /// virtual methods for base classes
   virtual void SetOperator(const Operator &op);
   virtual ~PetscLinearSolver();

   /// Sets the solver to be used as a preconditioner
   void SetPreconditioner(Solver &precond);

   /// Typecasting to PETSc's KSP
   operator KSP() const { return (KSP)obj; }
};

// iterative solvers

class PetscPCGSolver : public PetscLinearSolver
{
public:
   PetscPCGSolver(MPI_Comm comm, std::string prefix = std::string());
   PetscPCGSolver(PetscParMatrix &_A, std::string prefix = std::string());
   PetscPCGSolver(HypreParMatrix &_A,bool wrap=true,
                  std::string prefix = std::string());
};

/// Abstract class for PETSc's preconditioners
class PetscPreconditioner : public PetscSolver
{
private:
   void Init();

public:
   PetscPreconditioner(MPI_Comm comm, std::string prefix = std::string());

   PetscPreconditioner(PetscParMatrix &_A, std::string prefix = std::string());

   PetscPreconditioner(MPI_Comm comm, Operator &op,
                       std::string prefix = std::string());

   /// virtual methods for base classes
   virtual void SetOperator(const Operator &op);
   //virtual void SetPrintLevel(int plev);
   //virtual void Mult(const PetscParVector &b, PetscParVector &x) const;
   virtual ~PetscPreconditioner();
   /// Typecasting to PETSc's PC
   operator PC() const { return (PC)obj; }
};

// preconditioners

// auxiliary class for BDDC customization
class PetscBDDCSolverParams
{
protected:
   ParFiniteElementSpace *fespace;
   Array<int>            *ess_dof;
   bool                  ess_dof_local;
   Array<int>            *nat_dof;
   bool                  nat_dof_local;
   friend class PetscBDDCSolver;

public:
   PetscBDDCSolverParams() : fespace(NULL),
      ess_dof(NULL), ess_dof_local(false),
      nat_dof(NULL), nat_dof_local(false) {};
   void SetSpace(ParFiniteElementSpace *fe) { fespace = fe; };
   /// Specify dofs on the essential boundary
   /// If loc is false, it is a list of true dofs in local ordering
   /// If loc is true, it is a marker for Vdofs in local ordering
   void SetEssBdrDofs(Array<int> *essdofs, bool loc = false) { ess_dof = essdofs; ess_dof_local = loc; };
   /// Specify dofs on the natural boundary
   /// If loc is false, it is a list of true dofs in local ordering
   /// If loc is true, it is a marker for Vdofs in local ordering
   void SetNatBdrDofs(Array<int> *natdofs, bool loc = false) { nat_dof = natdofs; nat_dof_local = loc; };
   ~PetscBDDCSolverParams() {};
};

class PetscBDDCSolver : public PetscPreconditioner
{
private:
   void BDDCSolverConstructor(PetscBDDCSolverParams opts);

public:
   PetscBDDCSolver(MPI_Comm comm, Operator &op,
                   PetscBDDCSolverParams opts = PetscBDDCSolverParams(),
                   std::string prefix = std::string());
   PetscBDDCSolver(PetscParMatrix &op,
                   PetscBDDCSolverParams opts = PetscBDDCSolverParams(),
                   std::string prefix = std::string());
};

class PetscFieldSplitSolver : public PetscPreconditioner
{
public:
   PetscFieldSplitSolver(MPI_Comm comm, Operator &op,
                         std::string prefix = std::string());
};


}

#endif // MFEM_USE_MPI
#endif // MFEM_USE_PETSC

#endif
