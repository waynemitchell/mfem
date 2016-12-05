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

#ifndef MFEM_PNONLINEARFORM
#define MFEM_PNONLINEARFORM

#include "../config/config.hpp"

#ifdef MFEM_USE_MPI

#include "pgridfunc.hpp"
#include "nonlinearform.hpp"

namespace mfem
{

/// Parallel non-linear operator on the true dofs
class ParNonlinearForm : public NonlinearForm
{
protected:
   mutable ParGridFunction X, Y;
   mutable HypreParMatrix *pGrad;
#ifdef MFEM_USE_PETSC
   mutable PetscParMatrix *ppGrad;
#else // unused: just to not pollute the hpp file with lots of ifdefs
   mutable HypreParMatrix *ppGrad;
#endif

   // Assemble Jacobian matrix with PETSc
   bool use_petsc;

   // assemble the Jacobian matrix in "unassembled format"
   // for non-overlapping DD
   // significant only with PETSc backend
   bool unassembled;

public:
   ParNonlinearForm(ParFiniteElementSpace *pf)
      : NonlinearForm(pf), X(pf), Y(pf)
   {
      height = width = pf->TrueVSize(); pGrad = NULL; ppGrad = NULL,
             use_petsc = false; unassembled = false;
   }

   ParFiniteElementSpace *ParFESpace() const
   { return (ParFiniteElementSpace *)fes; }

   // Here, rhs is a true dof vector
   virtual void SetEssentialBC(const Array<int> &bdr_attr_is_ess,
                               Vector *rhs = NULL);

   /// Compute the energy of a ParGridFunction
   virtual double GetEnergy(const ParGridFunction &x) const;

   /// Compute the energy of a true-dof vector 'x'
   virtual double GetEnergy(const Vector &x) const;

   virtual void Mult(const Vector &x, Vector &y) const;

   /// Return the local gradient matrix for the given true-dof vector x
   const SparseMatrix &GetLocalGradient(const Vector &x) const;

   virtual Operator &GetGradient(const Vector &x) const;

   /// Assemble the Jacobian matrix with PETSc
   void SetUsePetsc(bool use = true)
   {
#ifndef MFEM_USE_PETSC
      if (true) { MFEM_ABORT("You did not configure MFEM with PETSc support"); }
      use_petsc = false;
#else
      use_petsc = use;
#endif
   }

   bool GetUsePetsc() { return use_petsc; }

   /// Assemble the Jacobian matrix in "unassembled format"
   //  for non-overlapping DD
   /// Only significant with PETSc backend
   void SetUseNonoverlappingFormat(bool use = true)
   {
#ifndef MFEM_USE_PETSC
      if (true) { MFEM_ABORT("You did not configure MFEM with PETSc support"); }
      unassembled = false;
#else
      unassembled = use;
#endif
   }

   virtual ~ParNonlinearForm() { delete pGrad; delete ppGrad; }
};

}

#endif // MFEM_USE_MPI

#endif
