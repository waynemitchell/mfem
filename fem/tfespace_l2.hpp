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

#ifndef MFEM_TEMPLATE_FESPACE_L2
#define MFEM_TEMPLATE_FESPACE_L2

#include "../config/tconfig.hpp"
#include "../linalg/tassign_ops.hpp"
#include "../linalg/ttensor_types.hpp"
#include "tfespace_h1.hpp" // for TFiniteElementSpace_simple
#include "fespace.hpp"

namespace mfem
{

template <typename FE>
class DGIndexer
{
protected:
   int offset;

public:
   typedef FE FE_type;

   DGIndexer(const FE &fe, const FiniteElementSpace &fes)
   {
      MFEM_ASSERT(fes.GetNDofs() == fes.GetNE() * FE::dofs,
                  "the FE space is not compatible with this FE!");
      offset = 0;
   }

   // default copy constructor

   inline MFEM_ALWAYS_INLINE
   void SetElement(int elem_idx)
   {
      offset = FE::dofs * elem_idx;
   }

   inline MFEM_ALWAYS_INLINE
   int map(int loc_dof_idx, int elem_offset) const
   {
      return offset + loc_dof_idx + elem_offset * FE::dofs;
   }
};


template <typename FE>
class L2_FiniteElementSpace
   : public TFiniteElementSpace_simple<FE,DGIndexer<FE> >
{
public:
   typedef FE FE_type;
   typedef TFiniteElementSpace_simple<FE,DGIndexer<FE> > base_class;

   L2_FiniteElementSpace(const FE &fe, const FiniteElementSpace &fes)
      : base_class(fe, fes)
   { }

   // default copy constructor

   static bool Matches(const FiniteElementSpace &fes)
   {
      const FiniteElementCollection *fec = fes.FEColl();
      const L2_FECollection *l2_fec =
         dynamic_cast<const L2_FECollection *>(fec);
      if (!l2_fec) { return false; }
      const FiniteElement *fe = l2_fec->FiniteElementForGeometry(FE_type::geom);
      if (fe->GetOrder() != FE_type::degree) { return false; }
      return true;
   }

   template <typename vec_layout_t>
   static bool VectorMatches(const FiniteElementSpace &fes)
   {
      return Matches(fes) && vec_layout_t::Matches(fes);
   }
};

} // namespace mfem

#endif // MFEM_TEMPLATE_FESPACE_L2
