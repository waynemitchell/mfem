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

#ifndef MFEM_TEMPLATE_TENSOR_TYPES
#define MFEM_TEMPLATE_TENSOR_TYPES

#include "config/tconfig.hpp"
#include "tlayouts.hpp"
#include "ttensor_ops.hpp"
#include "tsmall_matrix_ops.hpp"

#define MFEM_ROUNDUP(val,base) ((((val)+(base)-1)/(base))*(base))
#define MFEM_ALIGN_SIZE(size,type) \
   MFEM_ROUNDUP(size,(MFEM_SIMD_SIZE)/sizeof(type))

namespace mfem
{

// classes TVector, TMatrix, TTensor3, TTensor4

template <int S, typename data_t = double, bool align = false>
struct TVector
{
public:
   static const int size = S;
   static const int aligned_size = align ? MFEM_ALIGN_SIZE(S,data_t) : size;
   typedef data_t data_type;
   data_t data[aligned_size>0?aligned_size:1];

   typedef StridedLayout1D<S,1> layout_type;
   static const layout_type layout;

   data_t &operator[](int i) { return data[i]; }
   const data_t &operator[](int i) const { return data[i]; }

   template <AssignOp::Type Op>
   void Assign(const data_t d)
   {
      TAssign<Op>(layout, data, d);
   }

   template <AssignOp::Type Op, typename src_data_t>
   void Assign(const src_data_t &src)
   {
      TAssign<Op>(layout, data, layout, src);
   }

   template <AssignOp::Type Op, typename dest_data_t>
   void AssignTo(dest_data_t &dest)
   {
      TAssign<Op>(layout, dest, layout, data);
   }

   void Set(const data_t d)
   {
      Assign<AssignOp::Set>(d);
   }

   template <typename src_data_t>
   void Set(const src_data_t &src)
   {
      Assign<AssignOp::Set>(src);
   }

   template <typename dest_data_t>
   void Assemble(dest_data_t &dest) const
   {
      AssignTo<AssignOp::Add>(dest);
   }

   void Scale(const data_t scale)
   {
      Assign<AssignOp::Mult>(scale);
   }
};

template <int S, typename data_t, bool align>
const typename TVector<S,data_t,align>::layout_type
TVector<S,data_t,align>::layout = layout_type();


template <int N1, int N2, typename data_t = double, bool align = false>
struct TMatrix : public TVector<N1*N2,data_t,align>
{
   typedef TVector<N1*N2,data_t,align> base_class;
   using base_class::size;
   using base_class::data;

   typedef ColumnMajorLayout2D<N1,N2> layout_type;
   static const layout_type layout;
   static inline int ind(int i1, int i2) { return layout.ind(i1,i2); }

   data_t &operator()(int i, int j) { return data[ind(i,j)]; }
   const data_t &operator()(int i, int j) const { return data[ind(i,j)]; }

   inline data_t Det() const
   {
      return TDet<data_t>(layout, data);
   }

   inline void Adjugate(TMatrix<N1,N2,data_t> &adj) const
   {
      TAdjugate<data_t>(layout, data, layout, adj.data);
   }

   // Compute the adjugate and the determinant of a (small) matrix.
   inline data_t AdjDet(TMatrix<N2,N1,data_t> &adj) const
   {
      return TAdjDet<data_t>(layout, data, layout, adj.data);
   }
};

template <int N1, int N2, typename data_t, bool align>
const typename TMatrix<N1,N2,data_t,align>::layout_type
TMatrix<N1,N2,data_t,align>::layout = layout_type();


template <int N1, int N2, int N3, typename data_t = double, bool align = false>
struct TTensor3 : TVector<N1*N2*N3,data_t,align>
{
   typedef TVector<N1*N2*N3,data_t,align> base_class;
   using base_class::size;
   using base_class::data;

   typedef ColumnMajorLayout3D<N1,N2,N3> layout_type;
   static const layout_type layout;
   static inline int ind(int i1, int i2, int i3)
   { return layout.ind(i1,i2,i3); }

   data_t &operator()(int i, int j, int k) { return data[ind(i,j,k)]; }
   const data_t &operator()(int i, int j, int k) const
   { return data[ind(i,j,k)]; }
};

template <int N1, int N2, int N3, typename data_t, bool align>
const typename TTensor3<N1,N2,N3,data_t,align>::layout_type
TTensor3<N1,N2,N3,data_t,align>::layout = layout_type();

template <int N1, int N2, int N3, int N4, typename data_t = double,
          bool align = false>
struct TTensor4 : TVector<N1*N2*N3*N4,data_t,align>
{
   typedef TVector<N1*N2*N3*N4,data_t,align> base_class;
   using base_class::size;
   using base_class::data;

   typedef ColumnMajorLayout4D<N1,N2,N3,N4> layout_type;
   static const layout_type layout;
   static inline int ind(int i1, int i2, int i3, int i4)
   { return layout.ind(i1,i2,i3,i4); }

   data_t &operator()(int i, int j, int k, int l)
   { return data[ind(i,j,k,l)]; }
   const data_t &operator()(int i, int j, int k, int l) const
   { return data[ind(i,j,k,l)]; }
};

template <int N1, int N2, int N3, int N4, typename data_t, bool align>
const typename TTensor4<N1,N2,N3,N4,data_t,align>::layout_type
TTensor4<N1,N2,N3,N4,data_t,align>::layout = layout_type();

} // namespace mfem

#endif // MFEM_TEMPLATE_TENSOR_TYPES
