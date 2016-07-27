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

#include "../config/config.hpp"

#ifdef MFEM_USE_MPI
#ifdef MFEM_USE_PETSC

#include "linalg.hpp"
#include "../fem/fem.hpp"

#include <fstream>
#include <iomanip>
#include <cmath>
#include <cstdlib>

// Error handling
// Prints PETSc's stacktrace and then calls MFEM_ABORT
// We cannot use PETSc's CHKERRQ since it returns a PetscErrorCode
// TODO this can be improved
#define PCHKERRQ(obj,err)                                                  \
  if ((err)) {                                                             \
    PetscError(PetscObjectComm((PetscObject)obj),__LINE__,_MFEM_FUNC_NAME, \
               __FILE__,err,PETSC_ERROR_REPEAT,NULL);                      \
    MFEM_ABORT("Error in PETSc. See stacktrace above.");                   \
  }

#define CCHKERRQ(comm,err)                               \
  if ((err)) {                                           \
    PetscError(comm,__LINE__,_MFEM_FUNC_NAME,            \
               __FILE__,err,PETSC_ERROR_REPEAT,NULL);    \
    MFEM_ABORT("Error in PETSc. See stacktrace above."); \
  }

// prototype for auxiliary functions
static PetscErrorCode MatConvert_hypreParCSR_AIJ(hypre_ParCSRMatrix*,Mat*);
static PetscErrorCode MatConvert_hypreParCSR_IS(hypre_ParCSRMatrix*,Mat*);

// use global scope ierr to check PETSc errors inside mfem calls
PetscErrorCode ierr;

using namespace std;

namespace mfem
{

// PetscParVector methods

void PetscParVector::_SetDataAndSize_()
{
   const PetscScalar *array;
   PetscInt           n;

   ierr = VecGetArrayRead(x,&array); PCHKERRQ(x,ierr);
   ierr = VecGetLocalSize(x,&n); PCHKERRQ(x,ierr);
   SetDataAndSize((PetscScalar*)array,n);
   ierr = VecRestoreArrayRead(x,&array); PCHKERRQ(x,ierr);
}

PetscInt PetscParVector::GlobalSize()
{
   PetscInt N;
   ierr = VecGetSize(x,&N); PCHKERRQ(x,ierr);
   return N;
}

PetscParVector::PetscParVector(MPI_Comm comm, PetscInt glob_size,
                               PetscInt *col) : Vector()
{
   ierr = VecCreate(comm,&x); CCHKERRQ(comm,ierr);
   if (col)
   {
      PetscMPIInt myid;
      MPI_Comm_rank(comm, &myid);
      ierr = VecSetSizes(x,col[myid+1]-col[myid],PETSC_DECIDE); PCHKERRQ(x,ierr);
   }
   else
   {
      ierr = VecSetSizes(x,PETSC_DECIDE,glob_size); PCHKERRQ(x,ierr);
   }
   ierr = VecSetType(x,VECSTANDARD); PCHKERRQ(x,ierr);
   _SetDataAndSize_();
}

PetscParVector::~PetscParVector()
{
   MPI_Comm comm = PetscObjectComm((PetscObject)x);
   ierr = VecDestroy(&x); CCHKERRQ(comm,ierr);
}

PetscParVector::PetscParVector(MPI_Comm comm, PetscInt glob_size,
                               PetscScalar *_data, PetscInt *col) : Vector()
{
   if (col)
   {
      PetscMPIInt myid;
      MPI_Comm_rank(comm, &myid);
      ierr = VecCreateMPIWithArray(comm,1,col[myid+1]-col[myid],PETSC_DECIDE,_data,
                                   &x); CCHKERRQ(comm,ierr)
   }
   else
   {
      ierr = VecCreateMPIWithArray(comm,1,PETSC_DECIDE,glob_size,_data,&x);
      CCHKERRQ(comm,ierr)
   }
   _SetDataAndSize_();
}

PetscParVector::PetscParVector(const PetscParVector &y) : Vector()
{
   ierr = VecDuplicate(y.x,&x); PCHKERRQ(x,ierr);
}

PetscParVector::PetscParVector(const PetscParMatrix &A,
                               int transpose) : Vector()
{
   MPI_Comm comm = PETSC_COMM_WORLD;
   if (!transpose)
   {
      ierr = MatCreateVecs(const_cast<PetscParMatrix&>(A),&x,NULL);
      CCHKERRQ(comm,ierr);
   }
   else
   {
      ierr = MatCreateVecs(const_cast<PetscParMatrix&>(A),NULL,&x);
      CCHKERRQ(comm,ierr);
   }
   _SetDataAndSize_();
}

PetscParVector::PetscParVector(Vec y, bool ref) : Vector()
{
   if (ref)
   {
      ierr = PetscObjectReference((PetscObject)y); PCHKERRQ(y,ierr);
   }
   x = y;
   _SetDataAndSize_();
}

PetscParVector::PetscParVector(ParFiniteElementSpace *pfes) : Vector()
{

   HYPRE_Int* offsets = pfes->GetTrueDofOffsets();
   MPI_Comm  comm = pfes->GetComm();
   ierr = VecCreate(comm,&x); CCHKERRQ(comm,ierr);

   PetscMPIInt myid = 0;
   if (!HYPRE_AssumedPartitionCheck())
   {
      MPI_Comm_rank(comm,&myid);
   }
   ierr = VecSetSizes(x,offsets[myid+1]-offsets[myid],PETSC_DECIDE);
   PCHKERRQ(x,ierr);
   ierr = VecSetType(x,VECSTANDARD); PCHKERRQ(x,ierr);
   _SetDataAndSize_();
}

HypreParVector* GetHypreParVector(Vec y)
{
   const PetscScalar *array;
   const PetscInt    *cols;
   PetscInt           N;
   HypreParVector    *out;

   ierr = VecGetSize(y,&N); PCHKERRQ(y,ierr);
   ierr = VecGetArrayRead(y,&array); PCHKERRQ(y,ierr);
   ierr = VecGetOwnershipRanges(y,&cols); PCHKERRQ(y,ierr);
   // TODO ASK
   if (!HYPRE_AssumedPartitionCheck())
   {
      out = new HypreParVector(PetscObjectComm((PetscObject)y),N,(double*)array,
                               (HYPRE_Int*)cols);
   }
   else
   {
      PetscMPIInt myid;
      HYPRE_Int   range[2];

      MPI_Comm comm = PetscObjectComm((PetscObject)y);
      MPI_Comm_rank(comm,&myid);
      range[0] = cols[myid];
      range[1] = cols[myid+1];
      out = new HypreParVector(comm,N,(double*)array,range);
   }
   ierr = VecRestoreArrayRead(y,&array); PCHKERRQ(y,ierr);
   return out;
}

HypreParVector* GetHypreParVector(Mat B,bool transpose)
{
   const PetscInt    *cols;
   PetscInt           N;
   HypreParVector    *out;

   if (transpose)
   {
      ierr = MatGetSize(B,NULL,&N); PCHKERRQ(B,ierr);
      ierr = MatGetOwnershipRangesColumn(B,&cols); PCHKERRQ(B,ierr);
   }
   else
   {
      ierr = MatGetSize(B,&N,NULL); PCHKERRQ(B,ierr);
      ierr = MatGetOwnershipRanges(B,&cols); PCHKERRQ(B,ierr);
   }
   // TODO ASK
   if (!HYPRE_AssumedPartitionCheck())
   {
      out = new HypreParVector(PetscObjectComm((PetscObject)B),N,NULL,
                               (HYPRE_Int*)cols);
   }
   else
   {
      PetscMPIInt myid;
      HYPRE_Int   range[2];

      MPI_Comm comm = PetscObjectComm((PetscObject)B);
      MPI_Comm_rank(comm,&myid);
      range[0] = cols[myid];
      range[1] = cols[myid+1];
      out = new HypreParVector(comm,N,NULL,range);
   }
   return out;
}

Vector * PetscParVector::GlobalVector() const
{
   VecScatter   scctx;
   Vec          vout;
   PetscScalar *array;
   PetscInt     size;

   ierr = VecScatterCreateToAll(x,&scctx,&vout); PCHKERRQ(x,ierr);
   ierr = VecScatterBegin(scctx,x,vout,INSERT_VALUES,SCATTER_FORWARD);
   PCHKERRQ(x,ierr);
   ierr = VecScatterEnd(scctx,x,vout,INSERT_VALUES,SCATTER_FORWARD);
   PCHKERRQ(x,ierr);
   ierr = VecScatterDestroy(&scctx); PCHKERRQ(x,ierr);
   ierr = VecGetArray(vout,&array); PCHKERRQ(x,ierr);
   ierr = VecGetLocalSize(vout,&size); PCHKERRQ(x,ierr);
   Array<PetscScalar> data(size);
   data.Assign(array);
   ierr = VecRestoreArray(vout,&array); PCHKERRQ(x,ierr);
   ierr = VecDestroy(&vout); PCHKERRQ(x,ierr);
   Vector *v = new Vector(data, internal::to_int(size));
   v->MakeDataOwner();
   data.LoseData();
   return v;
}

PetscParVector& PetscParVector::operator=(PetscScalar d)
{
   ierr = VecSet(x,d); PCHKERRQ(x,ierr);
   return *this;
}

PetscParVector& PetscParVector::operator=(const PetscParVector &y)
{
   ierr = VecCopy(y.x,x); PCHKERRQ(x,ierr);
   return *this;
}

void PetscParVector::SetData(PetscScalar *_data)
{
   ierr = VecPlaceArray(x,_data); PCHKERRQ(x,ierr);
}

void PetscParVector::ResetData()
{
   ierr = VecResetArray(x); PCHKERRQ(x,ierr);
}

void PetscParVector::Randomize(PetscInt seed)
{
   PetscRandom rctx;

   ierr = PetscRandomCreate(PetscObjectComm((PetscObject)x),&rctx);
   PCHKERRQ(x,ierr);
   ierr = PetscRandomSetSeed(rctx,(unsigned long)seed); PCHKERRQ(x,ierr);
   ierr = PetscRandomSeed(rctx); PCHKERRQ(x,ierr);
   ierr = VecSetRandom(x,rctx); PCHKERRQ(x,ierr);
   ierr = PetscRandomDestroy(&rctx); PCHKERRQ(x,ierr);
}

void PetscParVector::Print(const char *fname) const
{
   if (fname)
   {
      PetscViewer view;

      ierr = PetscViewerASCIIOpen(PetscObjectComm((PetscObject)x),fname,&view);
      PCHKERRQ(x,ierr);
      ierr = VecView(x,view); PCHKERRQ(x,ierr);
      ierr = PetscViewerDestroy(&view); PCHKERRQ(x,ierr);
   }
   else
   {
      ierr = VecView(x,NULL); PCHKERRQ(x,ierr);
   }
}

// PetscParMatrix methods

PetscInt PetscParMatrix::GetNumRows()
{
   PetscInt N;
   ierr = MatGetLocalSize(A,&N,NULL); PCHKERRQ(A,ierr);
   return N;
}

PetscInt PetscParMatrix::GetNumCols()
{
   PetscInt N;
   ierr = MatGetLocalSize(A,NULL,&N); PCHKERRQ(A,ierr);
   return N;
}

PetscInt PetscParMatrix::M()
{
   PetscInt N;
   ierr = MatGetSize(A,&N,NULL); PCHKERRQ(A,ierr);
   return N;
}

PetscInt PetscParMatrix::N()
{
   PetscInt N;
   ierr = MatGetSize(A,NULL,&N); PCHKERRQ(A,ierr);
   return N;
}

void PetscParMatrix::Init()
{
   A = NULL;
   X = Y = NULL;
   height = width = 0;
}

PetscParMatrix::PetscParMatrix()
{
   Init();
}

PetscParMatrix::PetscParMatrix(const HypreParMatrix *hmat, bool wrap,
                               bool assembled)
{
   Init();
   height = hmat->Height();
   width  = hmat->Width();
   if (wrap)
   {
      MFEM_VERIFY(assembled,
                  "PetscParMatrix::PetscParMatrix(const HypreParMatrix*,bool,bool)" <<
                  "Cannot wrap in PETSc's unassembled format")
      MakeWrapper(hmat->GetComm(),hmat,&A);
   }
   else
   {
      if (assembled)
      {
         ierr = MatConvert_hypreParCSR_AIJ(const_cast<HypreParMatrix&>(*hmat),&A);
         CCHKERRQ(hmat->GetComm(),ierr);
      }
      else
      {
         ierr = MatConvert_hypreParCSR_IS(const_cast<HypreParMatrix&>(*hmat),&A);
         CCHKERRQ(hmat->GetComm(),ierr);
      }
   }
}

PetscParMatrix::PetscParMatrix(MPI_Comm comm, const Operator *op)
{
   Init();
   height = op->Height();
   width  = op->Width();
   MakeWrapper(comm,op,&A);
}

PetscParMatrix::PetscParMatrix(MPI_Comm comm, PetscInt glob_size,
                               PetscInt *row_starts, SparseMatrix *diag, bool assembled)
{
   Init();
   PetscInt lsize,start;
   // TODO ASK
   PetscMPIInt myid = 0;
   if (!HYPRE_AssumedPartitionCheck())
   {
      ierr = MPI_Comm_rank(comm,&myid); CCHKERRQ(comm,ierr);
   }
   lsize = row_starts[myid+1]-row_starts[myid];
   start = row_starts[myid];
   IS is;
   ierr = ISCreateStride(comm,diag->Height(),start,1,&is); CCHKERRQ(comm,ierr);
   ISLocalToGlobalMapping l2g;
   ierr = ISLocalToGlobalMappingCreateIS(is,&l2g); PCHKERRQ(is,ierr);
   ierr = ISDestroy(&is); CCHKERRQ(comm,ierr);

   // Create the PETSc object (MATIS format)
   ierr = MatCreate(comm,&A); CCHKERRQ(comm,ierr);
   ierr = MatSetSizes(A,lsize,lsize,PETSC_DECIDE,PETSC_DECIDE); PCHKERRQ(A,ierr);
   ierr = MatSetType(A,MATIS); PCHKERRQ(A,ierr);
   ierr = MatSetLocalToGlobalMapping(A,l2g,l2g); PCHKERRQ(A,ierr);
   ierr = ISLocalToGlobalMappingDestroy(&l2g); PCHKERRQ(A,ierr)

   // Copy SparseMatrix into PETSc SeqAIJ format
   Mat lA;
   ierr = MatISGetLocalMat(A,&lA); PCHKERRQ(A,ierr);
   ierr = MatSeqAIJSetPreallocationCSR(lA,diag->GetI(),diag->GetJ(),
                                       diag->GetData()); PCHKERRQ(lA,ierr);

   // Assemble into MPIAIJ format if requested
   if (assembled)
   {
      Mat B;

      ierr = MatISGetMPIXAIJ(A,MAT_INITIAL_MATRIX,&B); PCHKERRQ(A,ierr);
      ierr = MatDestroy(&A); PCHKERRQ(B,ierr);
      A = B;
   }

   // Tell PETSc the matrix is ready to be used
   ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY); PCHKERRQ(A,ierr);
   ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY); PCHKERRQ(A,ierr);

   // update base class
   height = lsize;
   width  = lsize;
}

PetscParMatrix::PetscParMatrix(MPI_Comm comm, PetscInt global_num_rows,
                               PetscInt global_num_cols, PetscInt *row_starts,
                               PetscInt *col_starts, SparseMatrix *diag, bool assembled)
{
   Init();
   PetscInt lrsize,lcsize,rstart,cstart;
   // TODO ASK
   PetscMPIInt myid = 0;
   if (!HYPRE_AssumedPartitionCheck())
   {
      ierr = MPI_Comm_rank(comm,&myid); CCHKERRQ(comm,ierr);
   }
   lrsize = row_starts[myid+1]-row_starts[myid];
   rstart = row_starts[myid];
   lcsize = col_starts[myid+1]-col_starts[myid];
   cstart = col_starts[myid];

   IS is;
   ierr = ISCreateStride(comm,diag->Height(),rstart,1,&is); CCHKERRQ(comm,ierr);
   ISLocalToGlobalMapping rl2g,cl2g;
   ierr = ISLocalToGlobalMappingCreateIS(is,&rl2g); PCHKERRQ(is,ierr);
   ierr = ISDestroy(&is); CCHKERRQ(comm,ierr);
   if (row_starts != col_starts)
   {
      ierr = ISCreateStride(comm,diag->Width(),cstart,1,&is); CCHKERRQ(comm,ierr);
      ierr = ISLocalToGlobalMappingCreateIS(is,&cl2g); PCHKERRQ(is,ierr);
      ierr = ISDestroy(&is); CCHKERRQ(comm,ierr);
   }
   else
   {
      ierr = PetscObjectReference((PetscObject)rl2g); PCHKERRQ(rl2g,ierr);
      cl2g = rl2g;
   }

   // Create the PETSc object (MATIS format)
   ierr = MatCreate(comm,&A); CCHKERRQ(comm,ierr);
   ierr = MatSetSizes(A,lrsize,lcsize,PETSC_DECIDE,PETSC_DECIDE); PCHKERRQ(A,ierr);
   ierr = MatSetType(A,MATIS); PCHKERRQ(A,ierr);
   ierr = MatSetLocalToGlobalMapping(A,rl2g,cl2g); PCHKERRQ(A,ierr);
   ierr = ISLocalToGlobalMappingDestroy(&rl2g); PCHKERRQ(A,ierr)
   ierr = ISLocalToGlobalMappingDestroy(&cl2g); PCHKERRQ(A,ierr)

   // Copy SparseMatrix into PETSc SeqAIJ format
   Mat lA;
   ierr = MatISGetLocalMat(A,&lA); PCHKERRQ(A,ierr);
   ierr = MatSeqAIJSetPreallocationCSR(lA,diag->GetI(),diag->GetJ(),
                                       diag->GetData()); PCHKERRQ(lA,ierr);

   // Assemble into MPIAIJ format if requested
   if (assembled)
   {
      Mat B;

      ierr = MatISGetMPIXAIJ(A,MAT_INITIAL_MATRIX,&B); PCHKERRQ(A,ierr);
      ierr = MatDestroy(&A); PCHKERRQ(B,ierr);
      A = B;
   }

   // Tell PETSc the matrix is ready to be used
   ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY); PCHKERRQ(A,ierr);
   ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY); PCHKERRQ(A,ierr);

   // update base class
   height = lrsize;
   width  = lcsize;
}

// TODO ADD THIS CONSTRUCTOR
//PetscParMatrix::PetscParMatrix(MPI_Comm comm, int nrows, PetscInt glob_nrows,
//                  PetscInt glob_ncols, int *I, PetscInt *J,
//                  double *data, PetscInt *rows, PetscInt *cols)
//{
//}

typedef struct
{
   Operator *op;
} mat_shell_ctx;

#undef __FUNCT__
#define __FUNCT__ "mat_shell_apply"
static PetscErrorCode mat_shell_apply(Mat A, Vec x, Vec y)
{
   mat_shell_ctx     *ctx;
   PetscErrorCode     ierr;

   PetscFunctionBeginUser;
   ierr = MatShellGetContext(A,(void **)&ctx); PCHKERRQ(A,ierr);
   PetscParVector xx(x,true);
   PetscParVector yy(y,true);
   ctx->op->Mult(xx,yy);
   PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "mat_shell_apply_transpose"
static PetscErrorCode mat_shell_apply_transpose(Mat A, Vec x, Vec y)
{
   mat_shell_ctx     *ctx;
   PetscErrorCode     ierr;

   PetscFunctionBeginUser;
   ierr = MatShellGetContext(A,(void **)&ctx); PCHKERRQ(A,ierr);
   PetscParVector xx(x,true);
   PetscParVector yy(y,true);
   ctx->op->MultTranspose(xx,yy);
   PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "mat_shell_destroy"
static PetscErrorCode mat_shell_destroy(Mat A)
{
   mat_shell_ctx  *ctx;

   PetscFunctionBeginUser;
   ierr = MatShellGetContext(A,(void **)&ctx); PCHKERRQ(A,ierr);
   delete ctx;
   PetscFunctionReturn(0);
}
#undef __FUNCT__

// TODO This should take a reference but how?
void PetscParMatrix::MakeWrapper(MPI_Comm comm, const Operator* op, Mat *A)
{
   mat_shell_ctx *ctx = new mat_shell_ctx;
   ierr = MatCreate(comm,A); CCHKERRQ(comm,ierr);
   ierr = MatSetSizes(*A,op->Height(),op->Width(),
                      PETSC_DECIDE,PETSC_DECIDE); PCHKERRQ(A,ierr);
   ierr = MatSetType(*A,MATSHELL); PCHKERRQ(A,ierr);
   ierr = MatShellSetContext(*A,(void *)ctx); PCHKERRQ(A,ierr);
   ierr = MatShellSetOperation(*A,MATOP_MULT,
                               (void (*)())mat_shell_apply); PCHKERRQ(A,ierr);
   ierr = MatShellSetOperation(*A,MATOP_MULT_TRANSPOSE,
                               (void (*)())mat_shell_apply_transpose); PCHKERRQ(A,ierr);
   ierr = MatShellSetOperation(*A,MATOP_DESTROY,
                               (void (*)())mat_shell_destroy); PCHKERRQ(A,ierr);
   ierr = MatSetUp(*A); PCHKERRQ(*A,ierr);
   ctx->op = const_cast<Operator *>(op);
}

void PetscParMatrix::Destroy()
{
   MPI_Comm comm = MPI_COMM_NULL;
   if (A != NULL)
   {
      ierr = PetscObjectGetComm((PetscObject)A,&comm); PCHKERRQ(A,ierr);
      ierr = MatDestroy(&A); CCHKERRQ(comm,ierr);
   }
   if (X) { delete X; }
   if (Y) { delete Y; }
}

PetscParMatrix::PetscParMatrix(Mat a, bool ref)
{
   if (ref)
   {
      ierr = PetscObjectReference((PetscObject)a); PCHKERRQ(a,ierr);
   }
   Init();
   A = a;
   height = GetNumRows();
   width = GetNumCols();
}

// Computes y = alpha * A  * x + beta * y
//       or y = alpha * A^T* x + beta * y
static void MatMultKernel(Mat A,PetscScalar a,Vec X,PetscScalar b,Vec Y,
                          bool transpose)
{
   PetscErrorCode (*f)(Mat,Vec,Vec);
   PetscErrorCode (*fadd)(Mat,Vec,Vec,Vec);
   if (transpose)
   {
      f = MatMultTranspose;
      fadd = MatMultTransposeAdd;
   }
   else
   {
      f = MatMult;
      fadd = MatMultAdd;
   }
   if (a != 0.)
   {
      if (b == 1.)
      {
         ierr = VecScale(X,a); PCHKERRQ(A,ierr);
         ierr = (*fadd)(A,X,Y,Y); PCHKERRQ(A,ierr);
         ierr = VecScale(X,1./a); PCHKERRQ(A,ierr);
      }
      else if (b != 0.)
      {
         ierr = VecScale(X,a); PCHKERRQ(A,ierr);
         ierr = VecScale(Y,b); PCHKERRQ(A,ierr);
         ierr = (*fadd)(A,X,Y,Y); PCHKERRQ(A,ierr);
         ierr = VecScale(X,1./a); PCHKERRQ(A,ierr);
      }
      else
      {
         ierr = (*f)(A,X,Y); PCHKERRQ(A,ierr);
         if (a != 1.)
         {
            ierr = VecScale(Y,a); PCHKERRQ(A,ierr);
         }
      }
   }
   else
   {
      if (b == 1.)
      {
         // do nothing
      }
      else if (b != 0.)
      {
         ierr = VecScale(Y,b); PCHKERRQ(A,ierr);
      }
      else
      {
         ierr = VecSet(Y,0.); PCHKERRQ(A,ierr);
      }
   }
}

void PetscParMatrix::MakeRef(const PetscParMatrix &master)
{
   ierr = PetscObjectReference((PetscObject)master.A); PCHKERRQ(master.A,ierr);
   Destroy();
   Init();
   A = master.A;
   height = master.height;
   width = master.width;
}

PetscParVector * PetscParMatrix::GetX() const
{
   if (!X)
   {
      MFEM_VERIFY(A,"Mat not present");
      X = new PetscParVector(*this,false); PCHKERRQ(A,ierr);
   }
   return X;
}

PetscParVector * PetscParMatrix::GetY() const
{
   if (!Y)
   {
      MFEM_VERIFY(A,"Mat not present");
      Y = new PetscParVector(*this,true); PCHKERRQ(A,ierr);
   }
   return Y;
}

void PetscParMatrix::Mult(double a, const Vector &x, double b, Vector &y) const
{
   MFEM_ASSERT(x.Size() == Width(), "invalid x.Size() = " << x.Size()
               << ", expected size = " << Width());
   MFEM_ASSERT(y.Size() == Height(), "invalid y.Size() = " << y.Size()
               << ", expected size = " << Height());

   PetscParVector *XX = GetX();
   PetscParVector *YY = GetY();
   XX->SetData(x.GetData());
   YY->SetData(y.GetData());
   MatMultKernel(A,a,XX->x,b,YY->x,false);
   XX->ResetData();
   YY->ResetData();
}

void PetscParMatrix::MultTranspose(double a, const Vector &x, double b,
                                   Vector &y) const
{
   MFEM_ASSERT(x.Size() == Height(), "invalid x.Size() = " << x.Size()
               << ", expected size = " << Height());
   MFEM_ASSERT(y.Size() == Width(), "invalid y.Size() = " << y.Size()
               << ", expected size = " << Width());

   PetscParVector *XX = GetX();
   PetscParVector *YY = GetY();
   YY->SetData(x.GetData());
   XX->SetData(y.GetData());
   MatMultKernel(A,a,YY->x,b,XX->x,true);
   XX->ResetData();
   YY->ResetData();
}

PetscParMatrix * RAP(PetscParMatrix *A, PetscParMatrix *P)
{
   Mat       pA = *A,pP = *P;
   Mat       B;
   PetscBool Aismatis,Aisaij,Pismatis,Pisaij;

   MFEM_VERIFY(A->Width() == P->Height(),
               "Petsc RAP: Number of local cols of A " << A->Width() <<
               " differs from number of local rows of P " << P->Height());
   MFEM_VERIFY(A->Height() == P->Height(),
               "Petsc RAP: Number of local rows of A " << A->Height() <<
               " differs from number of local rows of P " << P->Height());
   ierr = PetscObjectTypeCompare((PetscObject)pA,MATIS,&Aismatis);
   PCHKERRQ(pA,ierr);
   ierr = PetscObjectTypeCompare((PetscObject)pA,MATAIJ,&Aisaij);
   PCHKERRQ(pA,ierr);
   ierr = PetscObjectTypeCompare((PetscObject)pP,MATIS,&Pismatis);
   PCHKERRQ(pA,ierr);
   ierr = PetscObjectTypeCompare((PetscObject)pP,MATAIJ,&Pisaij);
   PCHKERRQ(pA,ierr);
   if (Aismatis &&
       Pismatis) // handle special case (this code will eventually go into PETSc)
   {
      Mat                    lA,lP,lB;
      ISLocalToGlobalMapping cl2gP;
      PetscInt               lsize;

      ierr = MatGetLocalToGlobalMapping(pP,NULL,&cl2gP); PCHKERRQ(pA,ierr);
      ierr = MatGetLocalSize(pP,NULL,&lsize); PCHKERRQ(pA,ierr);
      ierr = MatCreate(A->GetComm(),&B); PCHKERRQ(pA,ierr);
      ierr = MatSetSizes(B,lsize,lsize,PETSC_DECIDE,PETSC_DECIDE); PCHKERRQ(B,ierr);
      ierr = MatSetType(B,MATIS); PCHKERRQ(B,ierr);
      ierr = MatSetLocalToGlobalMapping(B,cl2gP,cl2gP); PCHKERRQ(B,ierr);
      ierr = MatISGetLocalMat(pA,&lA); PCHKERRQ(pA,ierr);
      ierr = MatISGetLocalMat(pP,&lP); PCHKERRQ(pA,ierr);
      ierr = MatPtAP(lA,lP,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&lB); PCHKERRQ(lA,ierr);
      ierr = MatISSetLocalMat(B,lB); PCHKERRQ(lB,ierr);
      ierr = MatDestroy(&lB); PCHKERRQ(lA,ierr);
      ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY); PCHKERRQ(B,ierr);
      ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY); PCHKERRQ(B,ierr);
   }
   else // it raises an error if the PtAP is not supported in PETSc
   {
      ierr = MatPtAP(pA,pP,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&B); PCHKERRQ(pA,ierr);
   }
   return new PetscParMatrix(B);
}

PetscParMatrix* PetscParMatrix::EliminateRowsCols(const Array<int> &rows_cols)
{
   Mat             Ae;
   const PetscInt *data;
   PetscInt        M,N,i,n,*idxs,rst;

   ierr = MatGetSize(A,&M,&N); PCHKERRQ(A,ierr);
   MFEM_VERIFY(M == N,"Rectangular case unsupported");
   ierr = MatGetOwnershipRange(A,&rst,NULL); PCHKERRQ(A,ierr);
   ierr = MatDuplicate(A,MAT_COPY_VALUES,&Ae); PCHKERRQ(A,ierr);
   ierr = MatSetOption(A,MAT_NO_OFF_PROC_ZERO_ROWS,PETSC_TRUE); PCHKERRQ(A,ierr);
   // rows need to be in global numbering
   n = rows_cols.Size();
   data = rows_cols.GetData();
   ierr = PetscMalloc1(n,&idxs); PCHKERRQ(A,ierr);
   for (i=0; i<n; i++) { idxs[i] = data[i] + rst; }
   ierr = MatZeroRowsColumns(A,n,idxs,1.,NULL,NULL); PCHKERRQ(A,ierr);
   ierr = PetscFree(idxs); PCHKERRQ(A,ierr);
   ierr = MatAXPY(Ae,-1.,A,SAME_NONZERO_PATTERN); PCHKERRQ(A,ierr);
   return new PetscParMatrix(Ae);
}

void PetscParMatrix::EliminateRowsCols(const Array<int> &rows_cols,
                                       const HypreParVector &X,
                                       HypreParVector &B)
{
   MFEM_ABORT("To be implemented");
}

void EliminateBC(PetscParMatrix &A, PetscParMatrix &Ae,
                 const Array<int> &ess_dof_list,
                 const Vector &X, Vector &B)
{
   const PetscScalar *array;
   Mat pA = const_cast<PetscParMatrix&>(A);

   // B -= Ae*X
   Ae.Mult(-1.0, X, 1.0, B);

   Vec diag = const_cast<PetscParVector&>((*A.GetX()));
   ierr = MatGetDiagonal(pA,diag); PCHKERRQ(pA,ierr);
   ierr = VecGetArrayRead(diag,&array); PCHKERRQ(diag,ierr);
   for (int i = 0; i < ess_dof_list.Size(); i++)
   {
      int r = ess_dof_list[i];
      B(r) = array[r] * X(r);
   }
   ierr = VecRestoreArrayRead(diag,&array); PCHKERRQ(diag,ierr);
}

// PetscSolver methods

void PetscSolver::Init()
{
   obj = NULL;
   B = X = NULL;
}

PetscSolver::PetscSolver()
{
   Init();
}

PetscSolver::~PetscSolver()
{
   if (B) { delete B; }
   if (X) { delete X; }
}

// PetscLinearSolver methods

void PetscLinearSolver::Init()
{
   wrap = false;
}

PetscLinearSolver::PetscLinearSolver(MPI_Comm comm) : PetscSolver()
{
   Init();
   KSP ksp;
   ierr = KSPCreate(comm,&ksp); CCHKERRQ(comm,ierr);
   obj = (PetscObject)ksp;
}

PetscLinearSolver::PetscLinearSolver(PetscParMatrix &_A) : PetscSolver()
{
   Init();
   KSP ksp;
   ierr = KSPCreate(_A.GetComm(),&ksp); CCHKERRQ(_A.GetComm(),ierr);
   obj = (PetscObject)ksp;
   SetOperator(_A);
}

PetscLinearSolver::PetscLinearSolver(HypreParMatrix &_A,bool wrapin) : PetscSolver()
{
   Init();
   wrap = wrapin;
   KSP ksp;
   ierr = KSPCreate(_A.GetComm(),&ksp); CCHKERRQ(_A.GetComm(),ierr);
   obj = (PetscObject)ksp;
   SetOperator(_A);
}

void PetscLinearSolver::SetOperator(const Operator &op)
{
   HypreParMatrix *hA = const_cast<HypreParMatrix *>
                        (dynamic_cast<const HypreParMatrix *>(&op));
   PetscParMatrix *pA = const_cast<PetscParMatrix *>
                        (dynamic_cast<const PetscParMatrix *>(&op));
   Operator       *oA = const_cast<Operator *>
                        (dynamic_cast<const Operator *>(&op));
   // update base classes: Operator, Solver, PetscLinearSolver
   bool delete_pA = false;
   if (!pA)
   {
      if (hA)
      {
         // Create MATSHELL object or convert
         // into PETSc AIJ format depending on wrap
         pA = new PetscParMatrix(hA,wrap);
         delete_pA = true;

      }
      else if (oA) // fallback to general operator
      {
         // Create MATSHELL object
         pA = new PetscParMatrix(PetscObjectComm(obj),oA);
         delete_pA = true;
      }
   }
   if (!pA)
   {
      MFEM_ABORT("PetscLinearSolver::SetOperator : Unsupported operation!");
   }

   KSP ksp = (KSP)obj;
   Mat A = pA->A;
   PetscInt nheight,nwidth;
   ierr = MatGetSize(A,&nheight,&nwidth); PCHKERRQ(A,ierr);
   if (ksp)
   {
      if (nheight != height || nwidth != width)
      {
         // reinit without destroying the KSP
         // communicator remains the same
         ierr = KSPReset(ksp); PCHKERRQ(ksp,ierr);
         if (X) { delete X; }
         if (B) { delete B; }
         B = X = NULL;
         wrap = false;
      }
   }
   ierr = KSPSetOperators(ksp,A,A); PCHKERRQ(ksp,ierr);
   // allow user customization
   ierr = KSPSetFromOptions(ksp); PCHKERRQ(ksp,ierr);
   if (delete_pA) { delete pA; }
   height = nheight;
   width  = nwidth;
}

typedef struct
{
   Solver *op;
} solver_shell_ctx;


#undef __FUNCT__
#define __FUNCT__ "pc_shell_apply"
static PetscErrorCode pc_shell_apply(PC pc, Vec x, Vec y)
{
   solver_shell_ctx  *ctx;
   PetscErrorCode     ierr;

   PetscFunctionBeginUser;
   ierr = PCShellGetContext(pc,(void **)&ctx); PCHKERRQ(pc,ierr);
   PetscParVector xx(x,true);
   PetscParVector yy(y,true);
   ctx->op->Mult(xx,yy);
   PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "pc_shell_apply_transpose"
static PetscErrorCode pc_shell_apply_transpose(PC pc, Vec x, Vec y)
{
   solver_shell_ctx  *ctx;
   PetscErrorCode     ierr;

   PetscFunctionBeginUser;
   ierr = PCShellGetContext(pc,(void **)&ctx); PCHKERRQ(pc,ierr);
   PetscParVector xx(x,true);
   PetscParVector yy(y,true);
   ctx->op->MultTranspose(xx,yy);
   PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "pc_shell_setup"
static PetscErrorCode pc_shell_setup(PC pc)
{
   //solver_shell_ctx *ctx;
   //Mat              A;
   //PetscErrorCode   ierr;

   PetscFunctionBeginUser;
   // TODO ask: is there a way to trigger the setup of ctx->op?
   PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "pc_shell_destroy"
static PetscErrorCode pc_shell_destroy(PC pc)
{
   solver_shell_ctx *ctx;

   PetscFunctionBeginUser;
   ierr = PCShellGetContext(pc,(void **)&ctx); PCHKERRQ(pc,ierr);
   delete ctx;
   PetscFunctionReturn(0);
}

#undef __FUNCT__

void PetscLinearSolver::SetPreconditioner(Solver &precond)
{
   PC pc;

   KSP ksp = (KSP)obj;
   ierr = KSPGetPC(ksp,&pc); PCHKERRQ(ksp,ierr);
   ierr = PCSetType(pc,PCSHELL); PCHKERRQ(pc,ierr);
   solver_shell_ctx *ctx = new solver_shell_ctx;
   ctx->op = &precond;
   ierr = PCShellSetContext(pc,(void *)ctx); PCHKERRQ(pc,ierr);
   ierr = PCShellSetApply(pc,pc_shell_apply); PCHKERRQ(pc,ierr);
   ierr = PCShellSetApplyTranspose(pc,pc_shell_apply_transpose); PCHKERRQ(pc,ierr);
   ierr = PCShellSetSetUp(pc,pc_shell_setup); PCHKERRQ(pc,ierr);
   ierr = PCShellSetDestroy(pc,pc_shell_destroy); PCHKERRQ(pc,ierr);
}

void PetscLinearSolver::Mult(const PetscParVector &b, PetscParVector &x) const
{
   KSP ksp = (KSP)obj;
   ierr = KSPSetInitialGuessNonzero(ksp,PetscBool(iterative_mode));
   PCHKERRQ(ksp,ierr);
   ierr = KSPSolve(ksp,b.x,x.x); PCHKERRQ(ksp,ierr);
}

void PetscLinearSolver::Mult(const Vector &b, Vector &x) const
{
   KSP ksp = (KSP)obj;
   if (!B)
   {
      Mat pA;
      ierr = KSPGetOperators(ksp,&pA,NULL); PCHKERRQ(ksp,ierr);
      PetscParMatrix A = PetscParMatrix(pA,true);
      B = new PetscParVector(A,true);
   }
   if (!X)
   {
      Mat pA;
      ierr = KSPGetOperators(ksp,&pA,NULL); PCHKERRQ(ksp,ierr);
      PetscParMatrix A = PetscParMatrix(pA,true);
      X = new PetscParVector(A,false);
   }
   B -> SetData(b.GetData());
   X -> SetData(x.GetData());

   Mult(*B, *X);

   B -> ResetData();
   X -> ResetData();
}

PetscLinearSolver::~PetscLinearSolver()
{
   MPI_Comm comm;
   KSP ksp = (KSP)obj;
   ierr = PetscObjectGetComm((PetscObject)ksp,&comm); PCHKERRQ(ksp,ierr);
   ierr = KSPDestroy(&ksp); CCHKERRQ(comm,ierr);
}

void PetscLinearSolver::SetTol(double tol)
{
   KSP ksp = (KSP)obj;
   // PETSC_DEFAULT does not change any other
   // customization previously set.
   ierr = KSPSetTolerances(ksp,tol,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);
   PCHKERRQ(ksp,ierr)
}

void PetscLinearSolver::SetMaxIter(int max_iter)
{
   KSP ksp = (KSP)obj;
   // PETSC_DEFAULT does not change any other
   // customization previously set.
   ierr = KSPSetTolerances(ksp,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT,max_iter);
   PCHKERRQ(ksp,ierr)
}

void PetscLinearSolver::SetPrintLevel(int plev)
{
   // TODO
}

// PetscPCGSolver methods

PetscPCGSolver::PetscPCGSolver(PetscParMatrix& _A) : PetscLinearSolver(_A)
{
   KSP ksp = (KSP)obj;
   ierr = KSPSetType(ksp,KSPCG); PCHKERRQ(ksp,ierr);
   // this is to obtain a textbook PCG
   ierr = KSPSetNormType(ksp,KSP_NORM_NATURAL); PCHKERRQ(ksp,ierr);
}

PetscPCGSolver::PetscPCGSolver(HypreParMatrix& _A,
                               bool wrap) : PetscLinearSolver(_A,wrap)
{
   KSP ksp = (KSP)obj;
   ierr = KSPSetType(ksp,KSPCG); PCHKERRQ(ksp,ierr);
   // this is to obtain a textbook PCG
   ierr = KSPSetNormType(ksp,KSP_NORM_NATURAL); PCHKERRQ(ksp,ierr);
}

}

#undef __FUNCT__
#define __FUNCT__ "array_container_destroy"
static PetscErrorCode array_container_destroy(void *ptr)
{
   PetscErrorCode ierr;

   PetscFunctionBeginUser;
   ierr = PetscFree(ptr); CHKERRQ(ierr);
   PetscFunctionReturn(0);
}

// TODO: These functions can be eventually moved to PETSc source code.
#include "_hypre_parcsr_mv.h"
#undef __FUNCT__
#define __FUNCT__ "MatConvert_hypreParCSR_AIJ"
static PetscErrorCode MatConvert_hypreParCSR_AIJ(hypre_ParCSRMatrix* hA,Mat* pA)
{
   MPI_Comm        comm = hypre_ParCSRMatrixComm(hA);
   hypre_CSRMatrix *hdiag,*hoffd;
   PetscScalar     *da,*oa,*aptr;
   PetscInt        *dii,*djj,*oii,*ojj,*iptr;
   PetscInt        i,dnnz,onnz,m,n;
   PetscMPIInt     size;
   PetscErrorCode  ierr;

   PetscFunctionBeginUser;
   hdiag = hypre_ParCSRMatrixDiag(hA);
   hoffd = hypre_ParCSRMatrixOffd(hA);
   m     = hypre_CSRMatrixNumRows(hdiag);
   n     = hypre_CSRMatrixNumCols(hdiag);
   dnnz  = hypre_CSRMatrixNumNonzeros(hdiag);
   onnz  = hypre_CSRMatrixNumNonzeros(hoffd);
   ierr  = PetscMalloc1(m+1,&dii); CHKERRQ(ierr);
   ierr  = PetscMalloc1(dnnz,&djj); CHKERRQ(ierr);
   ierr  = PetscMalloc1(dnnz,&da); CHKERRQ(ierr);
   ierr  = PetscMemcpy(dii,hypre_CSRMatrixI(hdiag),(m+1)*sizeof(PetscInt));
   CHKERRQ(ierr);
   ierr  = PetscMemcpy(djj,hypre_CSRMatrixJ(hdiag),dnnz*sizeof(PetscInt));
   CHKERRQ(ierr);
   ierr  = PetscMemcpy(da,hypre_CSRMatrixData(hdiag),dnnz*sizeof(PetscScalar));
   CHKERRQ(ierr);
   /* TODO should we add a case when the J pointer is already sorted? */
   iptr = djj;
   aptr = da;
   for (i=0; i<m; i++)
   {
      PetscInt nc = dii[i+1]-dii[i];
      ierr = PetscSortIntWithScalarArray(nc,iptr,aptr); CHKERRQ(ierr);
      iptr += nc;
      aptr += nc;
   }
   ierr = MPI_Comm_size(comm,&size); CHKERRQ(ierr);
   if (size > 1)
   {
      PetscInt *offdj,*coffd;

      ierr  = PetscMalloc1(m+1,&oii); CHKERRQ(ierr);
      ierr  = PetscMalloc1(onnz,&ojj); CHKERRQ(ierr);
      ierr  = PetscMalloc1(onnz,&oa); CHKERRQ(ierr);
      ierr  = PetscMemcpy(oii,hypre_CSRMatrixI(hoffd),(m+1)*sizeof(PetscInt));
      CHKERRQ(ierr);
      offdj = hypre_CSRMatrixJ(hoffd);
      coffd = hypre_ParCSRMatrixColMapOffd(hA);
      for (i=0; i<onnz; i++) { ojj[i] = coffd[offdj[i]]; }
      ierr  = PetscMemcpy(oa,hypre_CSRMatrixData(hoffd),onnz*sizeof(PetscScalar));
      CHKERRQ(ierr);
      /* TODO should we add a case when the J pointer is already sorted? */
      iptr = ojj;
      aptr = oa;
      for (i=0; i<m; i++)
      {
         PetscInt nc = oii[i+1]-oii[i];
         ierr = PetscSortIntWithScalarArray(nc,iptr,aptr); CHKERRQ(ierr);
         iptr += nc;
         aptr += nc;
      }
      ierr = MatCreateMPIAIJWithSplitArrays(comm,m,n,PETSC_DECIDE,PETSC_DECIDE,dii,
                                            djj,da,oii,ojj,oa,pA); CHKERRQ(ierr);
   }
   else
   {
      oii = ojj = NULL;
      oa = NULL;
      ierr = MatCreateSeqAIJWithArrays(comm,m,n,dii,djj,da,pA); CHKERRQ(ierr);
   }
   /* We are responsible to free the CSR arrays.
      However, since we can take references of a PetscParMatrix
      but we cannot take reference of PETSc arrays,
      we need to create a PetscContainer object
      to take reference of these arrays in
      reference objects */
   void *ptrs[6] = {dii,djj,da,oii,ojj,oa};
   const char *names[6] = {"_mfem_csr_dii",
                           "_mfem_csr_djj",
                           "_mfem_csr_da",
                           "_mfem_csr_oii",
                           "_mfem_csr_ojj",
                           "_mfem_csr_oa"
                          };
   for (i=0; i<6; i++)
   {
      PetscContainer c;

      ierr = PetscContainerCreate(comm,&c); CHKERRQ(ierr);
      ierr = PetscContainerSetPointer(c,ptrs[i]); CHKERRQ(ierr);
      ierr = PetscContainerSetUserDestroy(c,array_container_destroy);
      CHKERRQ(ierr);
      ierr = PetscObjectCompose((PetscObject)(*pA),names[i],(PetscObject)c);
      CHKERRQ(ierr);
      ierr = PetscContainerDestroy(&c); CHKERRQ(ierr);
   }
   PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatConvert_hypreParCSR_IS"
static PetscErrorCode MatConvert_hypreParCSR_IS(hypre_ParCSRMatrix* hA,Mat* pA)
{
   Mat                    lA;
   ISLocalToGlobalMapping rl2g,cl2g;
   IS                     is;
   hypre_CSRMatrix        *hdiag,*hoffd;
   MPI_Comm               comm = hypre_ParCSRMatrixComm(hA);
   PetscScalar            *hdd,*hod,*aa,*data;
   PetscInt               *col_map_offd,*hdi,*hdj,*hoi,*hoj;
   PetscInt               *aux,*ii,*jj;
   PetscInt               dr,dc,oc,str,stc,nnz,i,jd,jo;
   PetscErrorCode         ierr;

   PetscFunctionBeginUser;
   /* access relevant information in ParCSR */
   str   = hypre_ParCSRMatrixFirstRowIndex(hA);
   stc   = hypre_ParCSRMatrixFirstColDiag(hA);
   hdiag = hypre_ParCSRMatrixDiag(hA);
   hoffd = hypre_ParCSRMatrixOffd(hA);
   dr    = hypre_CSRMatrixNumRows(hdiag);
   dc    = hypre_CSRMatrixNumCols(hdiag);
   nnz   = hypre_CSRMatrixNumNonzeros(hdiag);
   hdi   = hypre_CSRMatrixI(hdiag);
   hdj   = hypre_CSRMatrixJ(hdiag);
   hdd   = hypre_CSRMatrixData(hdiag);
   oc    = hypre_CSRMatrixNumCols(hoffd);
   nnz  += hypre_CSRMatrixNumNonzeros(hoffd);
   hoi   = hypre_CSRMatrixI(hoffd);
   hoj   = hypre_CSRMatrixJ(hoffd);
   hod   = hypre_CSRMatrixData(hoffd);

   /* generate l2g maps for rows and cols */
   ierr = ISCreateStride(comm,dr,str,1,&is); CHKERRQ(ierr);
   ierr = ISLocalToGlobalMappingCreateIS(is,&rl2g); CHKERRQ(ierr);
   ierr = ISDestroy(&is); CHKERRQ(ierr);
   col_map_offd = hypre_ParCSRMatrixColMapOffd(hA);
   ierr = PetscMalloc1(dc+oc,&aux); CHKERRQ(ierr);
   for (i=0; i<dc; i++) aux[i]    = i+stc;
   for (i=0; i<oc; i++) aux[i+dc] = col_map_offd[i];
   ierr = ISCreateGeneral(comm,dc+oc,aux,PETSC_OWN_POINTER,&is); CHKERRQ(ierr);
   ierr = ISLocalToGlobalMappingCreateIS(is,&cl2g); CHKERRQ(ierr);
   ierr = ISDestroy(&is); CHKERRQ(ierr);

   /* create MATIS object */
   ierr = MatCreate(comm,pA); CHKERRQ(ierr);
   ierr = MatSetSizes(*pA,dr,dc,PETSC_DECIDE,PETSC_DECIDE); CHKERRQ(ierr);
   ierr = MatSetType(*pA,MATIS); CHKERRQ(ierr);
   ierr = MatSetLocalToGlobalMapping(*pA,rl2g,cl2g); CHKERRQ(ierr);
   ierr = ISLocalToGlobalMappingDestroy(&rl2g); CHKERRQ(ierr);
   ierr = ISLocalToGlobalMappingDestroy(&cl2g); CHKERRQ(ierr);

   /* merge local matrices */
   ierr = PetscMalloc2(nnz+dr+1,&aux,nnz,&data); CHKERRQ(ierr);
   ii   = aux;
   jj   = aux+dr+1;
   aa   = data;
   hdi++;
   hoi++;
   for (i=0,jd=0,jo=0;i<dr;i++)
   {
      *ii = jd + jo; ii++;
      for (;jd<hdi[i];jd++) { *jj = hdj[jd];    jj++; *aa = hdd[jd]; aa++; }
      for (;jo<hoi[i];jo++) { *jj = hoj[jo]+dc; jj++; *aa = hod[jo]; aa++; }
   }
   *ii = jd + jo;
   ii   = aux;
   jj   = aux+dr+1;
   aa   = data;
   ierr = MatISGetLocalMat(*pA,&lA); CHKERRQ(ierr);
   ierr = MatSeqAIJSetPreallocationCSR(lA,ii,jj,aa); CHKERRQ(ierr);
   ierr = PetscFree2(aux,data); CHKERRQ(ierr);
   ierr = MatAssemblyBegin(*pA,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
   ierr = MatAssemblyEnd(*pA,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
   PetscFunctionReturn(0);
}

#endif
#endif
