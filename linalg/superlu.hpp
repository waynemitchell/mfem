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

#ifndef MFEM_SUPERLU
#define MFEM_SUPERLU

#include "../config/config.hpp"
#include "operator.hpp"

#ifdef MFEM_USE_SUPERLU
#ifdef MFEM_USE_MPI

#include <mpi.h>

namespace mfem
{

namespace superlu_internal
{
unsigned int sqrti(const unsigned int & a);
}

/** The MFEM SuperLU Direct Solver class.

    The mfem::SuperLUSolver class is a solver capable of handling
    double precision types.  This solver uses the SuperLU_DIST library to
    perform LU factorization of sparse matrices.  SuperLU_DIST is
    currently maintained by Xiaoye Sherry Li at NERSC.

*/
class SuperLUSolver :
   public virtual mfem::Solver
{
public:
   /** Constructor with MPI_Comm paramter. */
   SuperLUSolver( MPI_Comm comm );

   /** Default destructor. */
   ~SuperLUSolver( void );

   void Mult( const Vector & x, Vector & y ) const;

   /** Set the operator. */
   void SetOperator( const Operator & op );

private:
   void Setup();

protected:

   MPI_Comm*                        commPtr;
   const Operator    * oper;
   // esi::MatrixData<int>       * A_md;

   /*
   superlu_options_t   options;
   SuperLUStat_t       stat;
   SuperMatrix         A;
   ScalePermstruct_t   ScalePermstruct;
   LUstruct_t          LUstruct;
   SOLVEstruct_t       SOLVEstruct;
   gridinfo_t          grid;
   */

   void*               optionsPtr;
   void*               statPtr;
   void*               APtr;
   void*               ScalePermstructPtr;
   void*               LUstructPtr;
   void*               SOLVEstructPtr;
   void*               gridPtr;
   double*             berr;
   int                 nrhs;
   int                 nprow;
   int                 npcol;
   mutable bool                firstSolveWithThisA;
   bool                gridInitialized;
   mutable bool                LUStructInitialized;
   /*
   // Debug flag
   int db;
   int dbg_level;
   int dbg_proc;
   */
};     // mfem::SuperLUSolver class

}

#endif // MFEM_USE_MPI
#endif // MFEM_USE_SUPERLU

#endif
