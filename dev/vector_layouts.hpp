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

#ifndef MFEM_TEMPLATE_VECTOR_LAYOUTS
#define MFEM_TEMPLATE_VECTOR_LAYOUTS

#include "config.hpp"
#include "fem/fespace.hpp"

namespace mfem
{

// Vector layout classes

class DynamicVectorLayout
{
public:
   static const int vec_dim = 0; // 0 - dynamic

protected:
   int scal_stride, comp_stride;
   int num_components;

   void Init(Ordering::Type ordering, int scalar_size, int num_comp)
   {
      num_components = num_comp;
      if (ordering == Ordering::byNODES)
      {
         scal_stride = 1;
         comp_stride = scalar_size;
      }
      else
      {
         scal_stride = num_comp;
         comp_stride = 1;
      }
   }

public:
   DynamicVectorLayout(Ordering::Type ordering, int scalar_size, int num_comp)
   {
      Init(ordering, scalar_size, num_comp);
   }
   DynamicVectorLayout(const FiniteElementSpace &fes)
   {
      Init(fes.GetOrdering(), fes.GetNDofs(), fes.GetVDim());
   }
   // default copy constructor

   int NumComponents() const { return num_components; }

   int ind(int scalar_idx, int comp_idx) const
   {
      return scal_stride * scalar_idx + comp_stride * comp_idx;
   }

   static bool Matches(const FiniteElementSpace &fes)
   {
      return true;
   }
};

// The default value (NumComp = 0) indicates that the number of components is
// dynamic, i.e. it will be specified at run-time.
template <Ordering::Type Ord, int NumComp = 0>
class VectorLayout
{
public:
   static const int vec_dim = NumComp;

protected:
   int num_components, scalar_size;

public:
   VectorLayout(int scalar_size_, int num_comp_ = NumComp)
      : num_components(num_comp_),
        scalar_size(scalar_size_)
   {
      MFEM_ASSERT(NumComp == 0 || num_components == NumComp,
                  "invalid number of components");
   }

   VectorLayout(const FiniteElementSpace &fes)
      : num_components(fes.GetVDim()),
        scalar_size(fes.GetNDofs())
   {
      MFEM_ASSERT(fes.GetOrdering() == Ord, "ordering mismatch");
      MFEM_ASSERT(NumComp == 0 || num_components == NumComp,
                  "invalid number of components");
   }
   // default copy constructor

   int NumComponents() const { return (NumComp ? NumComp : num_components); }

   int ind(int scalar_idx, int comp_idx) const
   {
      if (Ord == Ordering::byNODES)
      {
         return scalar_idx + comp_idx * scalar_size;
      }
      else
      {
         return comp_idx + (NumComp ? NumComp : num_components) * scalar_idx;
      }
   }

   static bool Matches(const FiniteElementSpace &fes)
   {
      return (Ord == fes.GetOrdering() &&
              (NumComp == 0 || NumComp == fes.GetVDim()));
   }
};

class ScalarLayout
{
public:
   static const int vec_dim = 1;

   ScalarLayout() { }

   ScalarLayout(const FiniteElementSpace &fes)
   {
      MFEM_ASSERT(fes.GetVDim() == 1, "invalid number of components");
   }

   int NumComponents() const { return 1; }

   int ind(int scalar_idx, int comp_idx) const { return scalar_idx; }

   static bool Matches(const FiniteElementSpace &fes)
   {
      return (fes.GetVDim() == 1);
   }
};

} // namespace mfem

#endif // MFEM_TEMPLATE_VECTOR_LAYOUTS
