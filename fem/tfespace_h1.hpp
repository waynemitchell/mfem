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

#ifndef MFEM_TEMPLATE_FESPACE_H1
#define MFEM_TEMPLATE_FESPACE_H1

#include "config/tconfig.hpp"
#include "linalg/tassign_ops.hpp"
#include "linalg/ttensor_types.hpp"
#include "tfinite_elements_h1.hpp"
#include "fespace.hpp"

namespace mfem
{

template <typename FE>
class H1_FiniteElementSpace
{
protected:
   const int *el_dof_list, *loc_dof_list;
   bool own_list;

public:
   typedef FE FE_type;

   H1_FiniteElementSpace(const FE &fe, const FiniteElementSpace &fes)
   {
      const Array<int> *loc_dof_map = fe.GetDofMap();
      fes.BuildElementToDofTable();
      const Table &el_dof = fes.GetElementToDofTable();
      MFEM_ASSERT(el_dof.Size_of_connections() == el_dof.Size() * FE::dofs,
                  "the element-to-dof Table is not compatible with this FE!");
      int num_dofs = el_dof.Size() * FE::dofs;
      if (!loc_dof_map)
      {
         // no local dof reordering
         el_dof_list = el_dof.GetJ();
         own_list = false;
      }
      else
      {
         // reorder the local dofs according to loc_dof_map
         int *el_dof_list_ = new int[num_dofs];
         const int *loc_dof_map_ = loc_dof_map->GetData();
         for (int i = 0; i < el_dof.Size(); i++)
         {
            MFEM_ASSERT(el_dof.RowSize(i) == FE::dofs,
                        "incompatible element-to-dof Table!");
            for (int j = 0; j < FE::dofs; j++)
            {
               el_dof_list_[j+FE::dofs*i] =
                  el_dof.GetJ()[loc_dof_map_[j]+FE::dofs*i];
            }
         }
         el_dof_list = el_dof_list_;
         own_list = true;
      }
      loc_dof_list = el_dof_list; // point to element 0
   }

   // Shallow copy constructor
   H1_FiniteElementSpace(const H1_FiniteElementSpace &orig)
      : el_dof_list(orig.el_dof_list),
        loc_dof_list(orig.loc_dof_list),
        own_list(false)
   { }

   ~H1_FiniteElementSpace() { if (own_list) { delete [] el_dof_list; } }

   void SetElement(int elem_idx)
   {
      loc_dof_list = el_dof_list + elem_idx * FE::dofs;
   }

   // Multi-element Extract:
   // Extract dofs for multiple elements starting with the current element.
   // The number of elements to extract is given by the second dimension of
   // dof_layout_t: dof_layout is (DOFS x NumElems).
   template <AssignOp::Type Op, typename glob_dof_data_t,
             typename dof_layout_t, typename dof_data_t>
   MFEM_ALWAYS_INLINE
   void Extract(const glob_dof_data_t &glob_dof_data,
                const dof_layout_t &dof_layout,
                dof_data_t &dof_data) const
   {
      const int NE = dof_layout_t::dim_2;
      MFEM_STATIC_ASSERT(FE::dofs == dof_layout_t::dim_1,
                         "invalid number of dofs");
      for (int j = 0; j < NE; j++)
      {
         for (int i = 0; i < FE::dofs; i++)
         {
            Assign<Op>(dof_data[dof_layout.ind(i,j)],
                       glob_dof_data[loc_dof_list[i+FE::dofs*j]]);
         }
      }
   }

   template <typename glob_dof_data_t,
             typename dof_layout_t, typename dof_data_t>
   MFEM_ALWAYS_INLINE
   void Extract(const glob_dof_data_t &glob_dof_data,
                const dof_layout_t &dof_layout,
                dof_data_t &dof_data) const
   {
      Extract<AssignOp::Set>(glob_dof_data, dof_layout, dof_data);
   }

   // Multi-element assemble.
   template <AssignOp::Type Op,
             typename dof_layout_t, typename dof_data_t,
             typename glob_dof_data_t>
   MFEM_ALWAYS_INLINE
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
            Assign<Op>(glob_dof_data[loc_dof_list[i+FE::dofs*j]],
                       dof_data[dof_layout.ind(i,j)]);
         }
      }
   }

   template <typename dof_layout_t, typename dof_data_t,
             typename glob_dof_data_t>
   MFEM_ALWAYS_INLINE
   void Assemble(const dof_layout_t &dof_layout,
                 const dof_data_t &dof_data,
                 glob_dof_data_t &glob_dof_data) const
   {
      Assemble<AssignOp::Add>(dof_layout, dof_data, glob_dof_data);
   }

   void Assemble(const TMatrix<FE::dofs,FE::dofs,double> &m,
                 SparseMatrix &M) const
   {
      MFEM_FLOPS_ADD(FE::dofs*FE::dofs);
      for (int i = 0; i < FE::dofs; i++)
      {
         M.SetColPtr(loc_dof_list[i]);
         for (int j = 0; j < FE::dofs; j++)
         {
            M._Add_(loc_dof_list[j], m(i,j));
         }
         M.ClearColPtr();
      }
   }

   // Multi-element VectorExtract: vdof_layout is (DOFS x NumComp x NumElems).
   template <AssignOp::Type Op,
             typename vec_layout_t, typename glob_vdof_data_t,
             typename vdof_layout_t, typename vdof_data_t>
   MFEM_ALWAYS_INLINE
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
               Assign<Op>(
                  vdof_data[vdof_layout.ind(i,k,j)],
                  glob_vdof_data[vl.ind(loc_dof_list[i+FE::dofs*j], k)]);
            }
         }
      }
   }

   template <typename vec_layout_t, typename glob_vdof_data_t,
             typename vdof_layout_t, typename vdof_data_t>
   MFEM_ALWAYS_INLINE
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
   MFEM_ALWAYS_INLINE
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
               Assign<Op>(
                  glob_vdof_data[vl.ind(loc_dof_list[i+FE::dofs*j], k)],
                  vdof_data[vdof_layout.ind(i,k,j)]);
            }
         }
      }
   }

   template <typename vdof_layout_t, typename vdof_data_t,
             typename vec_layout_t, typename glob_vdof_data_t>
   MFEM_ALWAYS_INLINE
   void VectorAssemble(const vdof_layout_t &vdof_layout,
                       const vdof_data_t &vdof_data,
                       const vec_layout_t &vl,
                       glob_vdof_data_t &glob_vdof_data) const
   {
      VectorAssemble<AssignOp::Add>(vdof_layout, vdof_data, vl, glob_vdof_data);
   }

   static bool Matches(const FiniteElementSpace &fes)
   {
      const FiniteElementCollection *fec = fes.FEColl();
      const H1_FECollection *h1_fec =
         dynamic_cast<const H1_FECollection *>(fec);
      if (!h1_fec) { return false; }
      const FiniteElement *fe = h1_fec->FiniteElementForGeometry(FE_type::geom);
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

#endif // MFEM_TEMPLATE_FESPACE_H1
