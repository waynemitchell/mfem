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

#include "config.hpp"
#include "assign_ops.hpp"
#include "tensor_types.hpp"
#include "fem/fespace.hpp"

namespace mfem
{

template <typename FE>
class L2_FiniteElementSpace
{
protected:
   int offset;

public:
   typedef FE FE_type;

   L2_FiniteElementSpace(const FE &fe, const FiniteElementSpace &fes)
   {
      MFEM_ASSERT(fes.GetNDofs() == fes.GetNE() * FE::dofs,
                  "the FE space is not compatible with this FE!");
      offset = 0;
   }

   L2_FiniteElementSpace(const L2_FiniteElementSpace &orig)
      : offset(orig.offset) { }

   void SetElement(int elem_idx)
   {
      offset = FE::dofs * elem_idx;
   }

   // Extract dofs for multiple elements starting with the current element.
   // The number of elements to extract is given by the second dimension of
   // dof_layout_t: dof_layout is (DOFS x NumElems).
   template <AssignOp::Type Op, typename glob_dof_data_t,
             typename dof_layout_t, typename dof_data_t>
   void Extract(const glob_dof_data_t &glob_dof_data,
                const dof_layout_t &dof_layout,
                dof_data_t &dof_data) const
   {
      const int NE = dof_layout_t::dim_2;
      MFEM_STATIC_ASSERT(FE::dofs == dof_layout_t::dim_1,
                         "invalid number of dofs");
      TAssign<Op>(dof_layout, dof_data,
                  ColumnMajorLayout2D<FE::dofs,NE>(), &glob_dof_data[offset]);
   }

   template <typename glob_dof_data_t,
             typename dof_layout_t, typename dof_data_t>
   void Extract(const glob_dof_data_t &glob_dof_data,
                const dof_layout_t &dof_layout,
                dof_data_t &dof_data) const
   {
      Extract<AssignOp::Set>(glob_dof_data, dof_layout, dof_data);
   }

   // Multi-element assemble.
   template <AssignOp::Type Op, typename dof_layout_t, typename dof_data_t,
             typename glob_dof_data_t>
   void Assemble(const dof_layout_t &dof_layout,
                 const dof_data_t &dof_data,
                 glob_dof_data_t &glob_dof_data) const
   {
      const int NE = dof_layout_t::dim_2;
      MFEM_STATIC_ASSERT(FE::dofs == dof_layout_t::dim_1,
                         "invalid number of dofs");
      for (int j = 0; j < NE; j++)
      {
         for (int i = 0; i < FE::dofs; i++)
         {
            Assign<Op>(glob_dof_data[offset+i+FE::dofs*j],
                       dof_data[dof_layout.ind(i,j)]);
         }
      }
   }

   template <typename dof_layout_t, typename dof_data_t,
             typename glob_dof_data_t>
   void Assemble(const dof_layout_t &dof_layout,
                 const dof_data_t &dof_data,
                 glob_dof_data_t &glob_dof_data) const
   {
      Assemble<AssignOp::Add>(dof_layout, dof_data, glob_dof_data);
   }

   // Multi-element VectorExtract: vdof_layout is (DOFS x NumComp x NumElems).
   template <AssignOp::Type Op,
             typename vec_layout_t, typename glob_vdof_data_t,
             typename vdof_layout_t, typename vdof_data_t>
   void VectorExtract(const vec_layout_t &vl,
                      const glob_vdof_data_t &glob_vdof_data,
                      const vdof_layout_t &vdof_layout,
                      vdof_data_t &vdof_data) const
   {
      const int NC = vdof_layout_t::dim_2;
      const int NE = vdof_layout_t::dim_3;
      MFEM_STATIC_ASSERT(FE::dofs == vdof_layout_t::dim_1,
                         "invalid number of dofs");
      MFEM_ASSERT(NC == vl.NumComponents(), "invalid number of components");
      for (int k = 0; k < NC; k++)
      {
         for (int j = 0; j < NE; j++)
         {
            for (int i = 0; i < FE::dofs; i++)
            {
               Assign<Op>(vdof_data[vdof_layout.ind(i,k,j)],
                          glob_vdof_data[vl.ind(offset+i+FE::dofs*j, k)]);
            }
         }
      }
   }

   template <typename vec_layout_t, typename glob_vdof_data_t,
             typename vdof_layout_t, typename vdof_data_t>
   void VectorExtract(const vec_layout_t &vl,
                      const glob_vdof_data_t &glob_vdof_data,
                      const vdof_layout_t &vdof_layout,
                      vdof_data_t &vdof_data) const
   {
      VectorExtract<AssignOp::Set>(vl, glob_vdof_data, vdof_layout, vdof_data);
   }

   // Multi-element VectorAssemble: vdof_layout is (DOFS x NumComp x NumElems).
   template <AssignOp::Type Op,
             typename vdof_layout_t, typename vdof_data_t,
             typename vec_layout_t, typename glob_vdof_data_t>
   void VectorAssemble(const vdof_layout_t &vdof_layout,
                       const vdof_data_t &vdof_data,
                       const vec_layout_t &vl,
                       glob_vdof_data_t &glob_vdof_data) const
   {
      const int NC = vdof_layout_t::dim_2;
      const int NE = vdof_layout_t::dim_3;
      MFEM_STATIC_ASSERT(FE::dofs == vdof_layout_t::dim_1,
                         "invalid number of dofs");
      MFEM_ASSERT(NC == vl.NumComponents(), "invalid number of components");
      for (int k = 0; k < NC; k++)
      {
         for (int j = 0; j < NE; j++)
         {
            for (int i = 0; i < FE::dofs; i++)
            {
               Assign<Op>(glob_vdof_data[vl.ind(offset+i+FE::dofs*j, k)],
                          vdof_data[vdof_layout.ind(i,k,j)]);
            }
         }
      }
   }

   template <typename vdof_layout_t, typename vdof_data_t,
             typename vec_layout_t, typename glob_vdof_data_t>
   void VectorAssemble(const vdof_layout_t &vdof_layout,
                       const vdof_data_t &vdof_data,
                       const vec_layout_t &vl,
                       glob_vdof_data_t &glob_vdof_data) const
   {
      VectorAssemble<AssignOp::Add>(vdof_layout, vdof_data, vl, glob_vdof_data);
   }
};

} // namespace mfem

#endif // MFEM_TEMPLATE_FESPACE_L2
