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

#include "../config/config.hpp"

#ifdef MFEM_USE_SUPERLU
#ifdef MFEM_USE_MPI

#include "superlu.hpp"

// SuperLU headers
#include <superlu_defs.h>
#include <superlu_ddefs.h>

#include "hypre.hpp"

using namespace std;

namespace mfem
{
unsigned int superlu_internal::sqrti(const unsigned int & a)
{
   unsigned int a_ = a;
   unsigned int rem = 0;
   unsigned int root = 0;
   unsigned short len   = sizeof(int); len <<= 2;
   unsigned short shift = (unsigned short)((len<<1) - 2);

   for (int i=0; i<len; i++)
   {
      root <<= 1;
      rem = ((rem << 2) + (a_ >> shift));
      a_ <<= 2;
      root ++;
      if (root <= rem)
      {
         rem -= root;
         root++;
      }
      else
      {
         root--;
      }
   }
   return (root >> 1);
}

SuperLURowLocMatrix::SuperLURowLocMatrix(MPI_Comm comm,
					 int num_loc_rows, int first_loc_row,
					 int glob_nrows, int glob_ncols,
					 int *I, int *J, double *data)
  : comm_(comm),
    rowLocPtr_(NULL)
{
   // Set mfem::Operator member data
   height = glob_nrows;
   width = glob_ncols;

   // Allocate SuperLU's SuperMatrix struct
   rowLocPtr_      = new SuperMatrix;
   SuperMatrix * A = (SuperMatrix*)rowLocPtr_;

   A->Store = NULL;

   int m       = glob_nrows;
   int n       = glob_ncols;
   int nnz_loc = I[num_loc_rows];
   int m_loc   = num_loc_rows;
   int fst_row = first_loc_row;

   double * nzval  = NULL;
   int    * colind = NULL;
   int    * rowptr = NULL;

   if ( !(nzval  = doubleMalloc_dist(nnz_loc)) )
   {
      ABORT("Malloc fails for nzval[].");
   }
   for (int i=0; i<nnz_loc; i++)
   {
     nzval[i] = data[i];
   }

   if ( !(colind = intMalloc_dist(nnz_loc)) )
   {
      ABORT("Malloc fails for colind[].");
   }
   for (int i=0; i<nnz_loc; i++)
   {
     colind[i] = J[i];
   }

   if ( !(rowptr = intMalloc_dist(m_loc+1)) )
   {
     ABORT("Malloc fails for rowptr[].");
   }
   for (int i=0; i<=m_loc; i++)
   {
     rowptr[i] = I[i];
   }

   // Assign he matrix data to SuperLU's SuperMatrix structure
   dCreate_CompRowLoc_Matrix_dist(A, m, n, nnz_loc, m_loc, fst_row,
				  nzval, colind, rowptr,
				  SLU_NR_loc, SLU_D, SLU_GE);
}

SuperLURowLocMatrix::SuperLURowLocMatrix(const HypreParMatrix & hypParMat)
  : comm_(hypParMat.GetComm()),
    rowLocPtr_(NULL)
{
   rowLocPtr_      = new SuperMatrix;
   SuperMatrix * A = (SuperMatrix*)rowLocPtr_;

   A->Store = NULL;

   // First cast the parameter to a hypre_ParCSRMatrix
   hypre_ParCSRMatrix * parcsr_op =
     (hypre_ParCSRMatrix *)const_cast<HypreParMatrix&>(hypParMat);

   MFEM_ASSERT(parcsr_op != NULL,"SuperLU: const_cast failed in SetOperator");

   // Create the SuperMatrix A by borrowing the internal data from a
   // hypre_CSRMatrix.
   hypre_CSRMatrix * csr_op = hypre_MergeDiagAndOffd(parcsr_op);
   hypre_CSRMatrixSetDataOwner(csr_op,0);

   int m         = parcsr_op->global_num_rows;
   int n         = parcsr_op->global_num_cols;
   int fst_row   = parcsr_op->first_row_index;
   int nnz_loc   = csr_op->num_nonzeros;
   int m_loc     = csr_op->num_rows;

   double * nzval  = csr_op->data;
   int    * colind = csr_op->j;
   int    * rowptr = NULL;

   // The "i" array cannot be stolen from the hypre_CSRMatrix so we'll copy it
   if ( !(rowptr = intMalloc_dist(m_loc+1)) )
   {
      ABORT("Malloc fails for rowptr[].");
   }
   for (int i=0; i<=m_loc; i++)
   {
      rowptr[i] = (csr_op->i)[i];
   }

   // Everything has been copied or abducted so delete the structure
   hypre_CSRMatrixDestroy(csr_op);

   // Assign he matrix data to SuperLU's SuperMatrix structure
   dCreate_CompRowLoc_Matrix_dist(A, m, n, nnz_loc, m_loc, fst_row,
				  nzval, colind, rowptr,
				  SLU_NR_loc, SLU_D, SLU_GE);
}

SuperLURowLocMatrix::~SuperLURowLocMatrix()
{
  SuperMatrix * A = (SuperMatrix*)rowLocPtr_;

  // Delete the internal data
  Destroy_CompRowLoc_Matrix_dist(A);

  // Delete the struct
  if ( A != NULL ) { delete A; }
}

SuperLUSolver::SuperLUSolver( MPI_Comm comm )
   : comm_(comm),
     APtr_(NULL),
     optionsPtr_(NULL),
     statPtr_(NULL),
     ScalePermstructPtr_(NULL),
     LUstructPtr_(NULL),
     SOLVEstructPtr_(NULL),
     gridPtr_(NULL),
     berr_(NULL),
     nrhs_(1),
     nprow_(0),
     npcol_(0),
     ownsA_(false),
     firstSolveWithThisA_(false),
     gridInitialized_(false),
     LUStructInitialized_(false)
{
  this->Init();
}

SuperLUSolver::SuperLUSolver( SuperLURowLocMatrix & A)
   : comm_(A.GetComm()),
     APtr_(&A),
     optionsPtr_(NULL),
     statPtr_(NULL),
     ScalePermstructPtr_(NULL),
     LUstructPtr_(NULL),
     SOLVEstructPtr_(NULL),
     gridPtr_(NULL),
     berr_(NULL),
     nrhs_(1),
     nprow_(0),
     npcol_(0),
     ownsA_(false),
     firstSolveWithThisA_(true),
     gridInitialized_(false),
     LUStructInitialized_(false)
{
  height = A.Height();
  width  = A.Width();

  this->Init();
}

SuperLUSolver::~SuperLUSolver()
{
   superlu_dist_options_t * options = (superlu_dist_options_t*)optionsPtr_;
   SuperLUStat_t     * stat         = (SuperLUStat_t*)statPtr_;
   ScalePermstruct_t * SPstruct     = (ScalePermstruct_t*)ScalePermstructPtr_;
   LUstruct_t        * LUstruct     = (LUstruct_t*)LUstructPtr_;
   SOLVEstruct_t     * SOLVEstruct  = (SOLVEstruct_t*)SOLVEstructPtr_;
   gridinfo_t        * grid         = (gridinfo_t*)gridPtr_;

   SUPERLU_FREE(berr_);
   PStatFree(stat);

   if ( LUStructInitialized_ )
   {
      ScalePermstructFree(SPstruct);
      Destroy_LU(width, grid, LUstruct);
      LUstructFree(LUstruct);
   }
   if ( options->SolveInitialized )
   {
      dSolveFinalize(options, SOLVEstruct);
   }

   if ( gridInitialized_ )
   {
      superlu_gridexit(grid);
   }

   if (     options != NULL ) { delete options; }
   if (        stat != NULL ) { delete stat; }
   if (    SPstruct != NULL ) { delete SPstruct; }
   if (    LUstruct != NULL ) { delete LUstruct; }
   if ( SOLVEstruct != NULL ) { delete SOLVEstruct; }
   if (        grid != NULL ) { delete grid; }
   if (       APtr_ != NULL && ownsA_ ) { delete APtr_; }
}

void
SuperLUSolver::Init()
{
   MPI_Comm_size(comm_, &numProcs_);
   MPI_Comm_rank(comm_, &myid_);

   optionsPtr_         = new superlu_dist_options_t;
   statPtr_            = new SuperLUStat_t;
   ScalePermstructPtr_ = new ScalePermstruct_t;
   LUstructPtr_        = new LUstruct_t;
   SOLVEstructPtr_     = new SOLVEstruct_t;
   gridPtr_            = new gridinfo_t;

   superlu_dist_options_t * options = (superlu_dist_options_t*)optionsPtr_;
   SuperLUStat_t          *    stat = (SuperLUStat_t*)statPtr_;

   if ( !(berr_ = doubleMalloc_dist(nrhs_)) )
   {
      ABORT("Malloc fails for berr[].");
   }

   // Set default options
   set_default_options_dist(options);

   options->ParSymbFact = YES;
   options->ColPerm     = PARMETIS;

   // Choose nprow and npcol so that the process grid is as square as possible.
   // If the processes cannot be divided evenly, keep the row dimension
   // smaller than the column dimension.

   nprow_ = (int)superlu_internal::sqrti((unsigned int)numProcs_);
   while (numProcs_ % nprow_ != 0 && nprow_ > 0)
   {
      nprow_--;
   }

   npcol_ = (int)(numProcs_ / nprow_);
   assert(nprow_ * npcol_ == numProcs_);

   PStatInit(stat); // Initialize the statistics variables.
}

void
SuperLUSolver::Setup()
{
   gridinfo_t * grid = (gridinfo_t*)gridPtr_;

   // Make sure the values of nprow and npcol are reasonable
   if ( ((nprow_ * npcol_) > numProcs_) || ((nprow_ * npcol_) < 1) )
   {
      if ( myid_ == 0 )
      {
	 cerr << "Warning: User specified nprow and npcol are such that "
	      << "(nprow * npcol) > numProcs or (nprow * npcol) < 1.  "
	      << "Using default values for nprow and npcol instead." << endl;
      }

      nprow_ = (int)superlu_internal::sqrti((unsigned int)numProcs_);
      while (numProcs_ % nprow_ != 0 && nprow_ > 0)
      {
         nprow_--;
      }

      npcol_ = (int)(numProcs_ / nprow_);
      assert(nprow_ * npcol_ == numProcs_);
   }

   superlu_gridinit(comm_, nprow_, npcol_, grid);

   gridInitialized_ = true;
}

void
SuperLUSolver::Mult( const Vector & x, Vector & y ) const
{
   MFEM_ASSERT(APtr_ != NULL,
               "SuperLU Error: The operator must be set before the system can be solved.");

   superlu_dist_options_t * options = (superlu_dist_options_t*)optionsPtr_;
   SuperLUStat_t     * stat         = (SuperLUStat_t*)statPtr_;
   SuperMatrix       * A            = (SuperMatrix*)APtr_->InternalData();

   ScalePermstruct_t * SPstruct     = (ScalePermstruct_t*)ScalePermstructPtr_;
   LUstruct_t        * LUstruct     = (LUstruct_t*)LUstructPtr_;
   SOLVEstruct_t     * SOLVEstruct  = (SOLVEstruct_t*)SOLVEstructPtr_;
   gridinfo_t        * grid         = (gridinfo_t*)gridPtr_;

   if (!firstSolveWithThisA_)
   {
      options->Fact = FACTORED; // Indicate the factored form of A is supplied.
   }
   else // This is the first sovle with this A
   {
      firstSolveWithThisA_ = false;

      // Make sure that the parameters have been initialized
      // The only parameter we might have to worry about is ScalePermstruct,
      // if the user is supplying a row or column permutation.

      // Initialize ScalePermstruct and LUstruct.
      ScalePermstructInit(A->nrow, A->ncol, SPstruct);
      LUstructInit(A->ncol, LUstruct);

      LUStructInitialized_ = true;
   }

   // SuperLU overwrites x with y, so copy x to y and pass that
   // to the solve routine.

   y = x;

   double*  yPtr = (double*)y;
   int      info = -1, locSize = y.Size();

   // Solve the system
   pdgssvx(options, A, SPstruct, yPtr, locSize, nrhs_, grid,
           LUstruct, SOLVEstruct, berr_, stat, &info);

   if ( info != 0 )
   {
      if ( info <= A->ncol )
      {
         MFEM_ABORT("SuperLU:  Found a singular matrix, U("
                    << info << "," << info << ") is exactly zero.");
      }
      else if ( info > A->ncol )
      {
         MFEM_ABORT("SuperLU:  Memory allocation error with "
                    << info - A->ncol << " bytes already allocated,");
      }
      else
      {
         MFEM_ABORT("Unknown SuperLU Error");
      }
   }
}

void
SuperLUSolver::SetOperator( const Operator & op )
{
   // Verify that we have a compatible operator
   APtr_ = dynamic_cast<const SuperLURowLocMatrix*>(&op);
   if ( APtr_ == NULL )
   {
      const HypreParMatrix * hypParMat =
        dynamic_cast<const HypreParMatrix *>(&op);

      MFEM_ASSERT(hypParMat != NULL,"SuperLU: unrecognized Operator type");

      APtr_  = new SuperLURowLocMatrix(*hypParMat);
      ownsA_ = true;
   }

   // Everything is ok so finish setting the operator
   firstSolveWithThisA_ = true;

   // Set mfem::Operator member data
   height = op.Height();
   width  = op.Width();

   // Initialize the processor grid if necessary
   if (!gridInitialized_)
   {
      this->Setup();
   }
}

} // mfem namespace

#endif // MFEM_USE_MPI
#endif // MFEM_USE_SUPERLU
