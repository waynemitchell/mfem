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

#ifndef MFEM_PFEM_EXTRAS
#define MFEM_PFEM_EXTRAS

#include "../../config/config.hpp"

#ifdef MFEM_USE_MPI

#include <cstddef>
#include "../../fem/pfespace.hpp"
#include "../../fem/pbilinearform.hpp"

namespace mfem
{

/** The H1_ParFESpace class is a ParFiniteElementSpace which automatically
    allocates and destroys its own FiniteElementCollection, in this
    case an H1_FECollection object.
*/
class H1_ParFESpace : public ParFiniteElementSpace
{
public:
   H1_ParFESpace(ParMesh *m,
                 const int p, const int space_dim = 3, const int type = 0,
                 int vdim = 1, int order = Ordering::byNODES);
   ~H1_ParFESpace();
private:
   const FiniteElementCollection *FEC_;
};

/** The ND_ParFESpace class is a ParFiniteElementSpace which automatically
    allocates and destroys its own FiniteElementCollection, in this
    case an ND_FECollection object.
*/
class ND_ParFESpace : public ParFiniteElementSpace
{
public:
   ND_ParFESpace(ParMesh *m, const int p, const int space_dim,
                 int vdim = 1, int order = Ordering::byNODES);
   ~ND_ParFESpace();
private:
   const FiniteElementCollection *FEC_;
};

/** The RT_ParFESpace class is a ParFiniteElementSpace which automatically
    allocates and destroys its own FiniteElementCollection, in this
    case an RT_FECollection object.
*/
class RT_ParFESpace : public ParFiniteElementSpace
{
public:
   RT_ParFESpace(ParMesh *m, const int p, const int space_dim,
                 int vdim = 1, int order = Ordering::byNODES);
   ~RT_ParFESpace();
private:
   const FiniteElementCollection *FEC_;
};

/** The L2_ParFESpace class is a ParFiniteElementSpace which automatically
    allocates and destroys its own FiniteElementCollection, in this
    case an L2_FECollection object.
*/
class L2_ParFESpace : public ParFiniteElementSpace
{
public:
   L2_ParFESpace(ParMesh *m, const int p, const int space_dim,
                 int vdim = 1, int order = Ordering::byNODES);
   ~L2_ParFESpace();
private:
   const FiniteElementCollection *FEC_;
};

class ParDiscreteInterpolationOperator
{
public:
   virtual ~ParDiscreteInterpolationOperator();

   /// Computes y = alpha * A * x + beta * y
   HYPRE_Int Mult(HypreParVector &x, HypreParVector &y,
                  double alpha = 1.0, double beta = 0.0);
   /// Computes y = alpha * A * x + beta * y
   HYPRE_Int Mult(HYPRE_ParVector x, HYPRE_ParVector y,
                  double alpha = 1.0, double beta = 0.0);

   /// Computes y = alpha * A^t * x + beta * y
   HYPRE_Int MultTranspose(HypreParVector &x, HypreParVector &y,
                           double alpha = 1.0, double beta = 0.0);

   /// Computes y = alpha * A * x + beta * y
   void Mult(double a, const Vector &x, double b, Vector &y) const;
   /// Computes y = alpha * A^t * x + beta * y
   void MultTranspose(double a, const Vector &x, double b, Vector &y) const;

   /// Computes y = A * x
   void Mult(const Vector &x, Vector &y) const;
   /// Computes y = A^t * x
   void MultTranspose(const Vector &x, Vector &y) const;

   HypreParMatrix * ParallelAssemble() { return mat_; }

protected:
   ParDiscreteInterpolationOperator() : pdlo_(NULL), mat_(NULL) {}

   ParDiscreteLinearOperator *pdlo_;
   HypreParMatrix            *mat_;
};

class ParDiscreteGradOperator : public ParDiscreteInterpolationOperator
{
public:
   ParDiscreteGradOperator(ParFiniteElementSpace *dfes,
                           ParFiniteElementSpace *rfes);
};

class ParDiscreteCurlOperator : public ParDiscreteInterpolationOperator
{
public:
   ParDiscreteCurlOperator(ParFiniteElementSpace *dfes,
                           ParFiniteElementSpace *rfes);
};

class ParDiscreteDivOperator : public ParDiscreteInterpolationOperator
{
public:
   ParDiscreteDivOperator(ParFiniteElementSpace *dfes,
                          ParFiniteElementSpace *rfes);
};

} // namespace mfem

#endif // MFEM_USE_MPI

#endif
