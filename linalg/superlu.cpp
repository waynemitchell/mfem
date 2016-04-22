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

#include "superlu.hpp"

// SuperLU headers
#include <superlu_defs.h>
#include <superlu_ddefs.h>

#ifdef MFEM_USE_MPI
#include "hypre.hpp"
#endif

using namespace std;

namespace mfem
{
using superlu_internal::sqrti;

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


SuperLUSolver::SuperLUSolver( MPI_Comm comm )
   : commPtr(&comm),
     oper(NULL),
     optionsPtr(NULL),
     statPtr(NULL),
     APtr(NULL),
     ScalePermstructPtr(NULL),
     LUstructPtr(NULL),
     SOLVEstructPtr(NULL),
     gridPtr(NULL)
{
   firstSolveWithThisA = true;
   gridInitialized     = false;
   LUStructInitialized = false;

   optionsPtr         = new superlu_dist_options_t;
   statPtr            = new SuperLUStat_t;
   APtr               = new SuperMatrix;
   ScalePermstructPtr = new ScalePermstruct_t;
   LUstructPtr        = new LUstruct_t;
   SOLVEstructPtr     = new SOLVEstruct_t;
   gridPtr            = new gridinfo_t;

   superlu_dist_options_t * options = (superlu_dist_options_t*)optionsPtr;
   SuperLUStat_t          *    stat = (SuperLUStat_t*)statPtr;
   SuperMatrix            *       A = (SuperMatrix*)APtr;

   A->Store = NULL;

   nrhs = 1;

   if ( !(berr = doubleMalloc_dist(nrhs)) )
   {
      ABORT("Malloc fails for berr[].");
   }

   /* Set default options */
   set_default_options_dist(options);

   options->ParSymbFact = YES;
   options->ColPerm     = PARMETIS;

   /* Choose nprow and npcol so that the process grid is as square as possible.
     If the processes cannot be divided evenly, keep the row dimension
     smaller than the column dimension. */
   int_t numProcs;
   MPI_Comm_size(*commPtr, &numProcs);

   nprow = (int)sqrti((unsigned int)numProcs);
   while (numProcs % nprow != 0 && nprow > 0)
   {
      nprow--;
   }

   npcol = (int)(numProcs / nprow);
   assert(nprow * npcol == numProcs);

   PStatInit(stat); /* Initialize the statistics variables. */
}

SuperLUSolver::~SuperLUSolver()
{
   superlu_dist_options_t * options = (superlu_dist_options_t*)optionsPtr;
   SuperLUStat_t     * stat         = (SuperLUStat_t*)statPtr;
   SuperMatrix       * A            = (SuperMatrix*)APtr;
   ScalePermstruct_t * SPstruct     = (ScalePermstruct_t*)ScalePermstructPtr;
   LUstruct_t        * LUstruct     = (LUstruct_t*)LUstructPtr;
   SOLVEstruct_t     * SOLVEstruct  = (SOLVEstruct_t*)SOLVEstructPtr;
   gridinfo_t        * grid         = (gridinfo_t*)gridPtr;

   SUPERLU_FREE(berr);
   PStatFree(stat);
   Destroy_CompRowLoc_Matrix_dist(A);

   if ( LUStructInitialized )
   {
      ScalePermstructFree(SPstruct);
      Destroy_LU(A->nrow, grid, LUstruct);
      LUstructFree(LUstruct);
   }
   if ( options->SolveInitialized )
   {
      dSolveFinalize(options, SOLVEstruct);
   }

   if ( gridInitialized )
   {
      superlu_gridexit(grid);
   }

   if (     options != NULL ) { delete options; }
   if (        stat != NULL ) { delete stat; }
   if (           A != NULL ) { delete A; }
   if (    SPstruct != NULL ) { delete SPstruct; }
   if (    LUstruct != NULL ) { delete LUstruct; }
   if ( SOLVEstruct != NULL ) { delete SOLVEstruct; }
   if (        grid != NULL ) { delete grid; }
}

void
SuperLUSolver::Setup()
{
   gridinfo_t * grid = (gridinfo_t*)gridPtr;

   int numProcs;
   MPI_Comm_size(*commPtr, &numProcs);

   // Make sure the values of nprow and npcol are reasonable
   if ( ((nprow * npcol) > numProcs) || ((nprow * npcol) < 1) )
   {
      cerr << "Warning: User specified nprow and npcol are such that "
           << "(nprow * npcol) > numProcs or (nprow * npcol) < 1.  "
           << "Using default values for nprow and npcol instead." << endl;

      // nprow = floor(sqrt((float)numProcs));
      nprow = (int)sqrti((unsigned int)numProcs);
      while (numProcs % nprow != 0 && nprow > 0)
      {
         nprow--;
      }

      npcol = (int)(numProcs / nprow);
      assert(nprow * npcol == numProcs);
   }

   superlu_gridinit(*commPtr, nprow, npcol, grid);

   gridInitialized = true;
}

void
SuperLUSolver::Mult( const Vector & x, Vector & y ) const
{
   superlu_dist_options_t * options = (superlu_dist_options_t*)optionsPtr;
   SuperLUStat_t     * stat         = (SuperLUStat_t*)statPtr;
   SuperMatrix       * A            = (SuperMatrix*)APtr;

   ScalePermstruct_t * SPstruct     = (ScalePermstruct_t*)ScalePermstructPtr;
   LUstruct_t        * LUstruct     = (LUstruct_t*)LUstructPtr;
   SOLVEstruct_t     * SOLVEstruct  = (SOLVEstruct_t*)SOLVEstructPtr;
   gridinfo_t        * grid         = (gridinfo_t*)gridPtr;

   MFEM_ASSERT(A->Store != NULL,
               "SuperLU Error: The operator must be set before the system can be solved.");

   if (!firstSolveWithThisA)
   {
      options->Fact = FACTORED; /* Indicate the factored form of A is supplied.*/
   }
   else // This is the first sovle with this A
   {
      firstSolveWithThisA = false;

      // Make sure that the parameters have been initialized
      // The only parameter we might have to worry about is ScalePermstruct,
      // if the user is supplying a row or column permutation.

      /* Initialize ScalePermstruct and LUstruct. */
      ScalePermstructInit(A->nrow, A->ncol, SPstruct);
      LUstructInit(A->ncol, LUstruct);

      LUStructInitialized = true;
   }

   // SuperLU overwrites x with y, so copy x to y and pass that
   // to the solve routine.

   y = x;

   double*  yPtr = (double*)y;
   int      info = -1, locSize = y.Size();

   // Solve the system
   pdgssvx(options, A, SPstruct, yPtr, locSize, nrhs, grid,
           LUstruct, SOLVEstruct, berr, stat, &info);

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
   firstSolveWithThisA = true;

   oper   = &op;
   height = op.Height();
   width  = op.Width();

   SuperMatrix * A = (SuperMatrix*)APtr;

   int_t    m, n, nnz_loc, m_loc, fst_row;
   double * nzval_loc;
   int_t  * colind;
   int_t  * rowptr;

   if (!gridInitialized)
   {
      this->Setup();
   }

   // First cast the parameter to a hypre_ParCSRMatrix
   const HypreParMatrix * hypParMat =
      dynamic_cast<const HypreParMatrix *>(&op);
   MFEM_ASSERT(hypParMat != NULL,"SuperLU requires a HypreParMatrix operator");

   hypre_ParCSRMatrix * parcsr_op =
      (hypre_ParCSRMatrix *)const_cast<HypreParMatrix&>(*hypParMat);

   MFEM_ASSERT(parcsr_op != NULL,"SuperLU: const_cast failed in SetOperator");

   // Create the SuperLUMatrix A by borrowing the internal data from a
   // hypre_CSRMatrix.
   hypre_CSRMatrix * csr_op = hypre_MergeDiagAndOffd(parcsr_op);
   hypre_CSRMatrixSetDataOwner(csr_op,0);

   m         = parcsr_op->global_num_rows;
   n         = parcsr_op->global_num_cols;
   fst_row   = parcsr_op->first_row_index;
   nnz_loc   = csr_op->num_nonzeros;
   m_loc     = csr_op->num_rows;
   nzval_loc = csr_op->data;
   colind    = csr_op->j;

   // The "i" array cannot be stolen from the hypre_CSRMatrix so we'll copy it
   if ( !(rowptr = intMalloc_dist(m_loc+1)) )
   {
      ABORT("Malloc fails for rowptr[].");
   }
   for (int i=0; i<=m_loc; i++)
   {
      rowptr[i] = (csr_op->i)[i];
   }
   hypre_CSRMatrixDestroy(csr_op);

   dCreate_CompRowLoc_Matrix_dist(A, m, n, nnz_loc, m_loc, fst_row,
                                  nzval_loc, colind, rowptr,
                                  SLU_NR_loc, SLU_D, SLU_GE);
}

}

#endif
