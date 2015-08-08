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

#ifndef MFEM_TEMPLATE_TENSOR_OPS
#define MFEM_TEMPLATE_TENSOR_OPS

#include "config.hpp"
#include "assign_ops.hpp"

namespace mfem
{

// Element-wise tensor operations

namespace internal
{

template <int Rank>
struct TensorOps;

template <>
struct TensorOps<1> // rank = 1
{
   // Assign: A {=,+=,*=} scalar_value
   template <AssignOp::Type Op, typename A_layout_t, typename A_data_t,
             typename scalar_t>
   static void Assign(const A_layout_t &A_layout, A_data_t &A_data,
                      scalar_t value)
   {
      MFEM_STATIC_ASSERT(A_layout_t::rank == 1, "invalid rank");
      for (int i1 = 0; i1 < A_layout_t::dim_1; i1++)
      {
         mfem::Assign<Op>(A_data[A_layout.ind(i1)], value);
      }
   }

   // Assign: A {=,+=,*=} B
   template <AssignOp::Type Op,
             typename A_layout_t, typename A_data_t,
             typename B_layout_t, typename B_data_t>
   static void Assign(const A_layout_t &A_layout, A_data_t &A_data,
                      const B_layout_t &B_layout, const B_data_t &B_data)
   {
      MFEM_STATIC_ASSERT(A_layout_t::rank == 1 && B_layout_t::rank == 1,
                         "invalid ranks");
      MFEM_STATIC_ASSERT(A_layout_t::dim_1 == B_layout_t::dim_1,
                         "invalid dimensions");
      for (int i1 = 0; i1 < A_layout_t::dim_1; i1++)
      {
         mfem::Assign<Op>(A_data[A_layout.ind(i1)], B_data[B_layout.ind(i1)]);
      }
   }
};

template <>
struct TensorOps<2> // rank = 2
{
   // Assign: A {=,+=,*=} scalar_value
   template <AssignOp::Type Op, typename A_layout_t, typename A_data_t,
             typename scalar_t>
   static void Assign(const A_layout_t &A_layout, A_data_t &A_data,
                      scalar_t value)
   {
      MFEM_STATIC_ASSERT(A_layout_t::rank == 2, "invalid rank");
      for (int i2 = 0; i2 < A_layout_t::dim_2; i2++)
      {
         for (int i1 = 0; i1 < A_layout_t::dim_1; i1++)
         {
            mfem::Assign<Op>(A_data[A_layout.ind(i1,i2)], value);
         }
      }
   }

   // Assign: A {=,+=,*=} B
   template <AssignOp::Type Op,
             typename A_layout_t, typename A_data_t,
             typename B_layout_t, typename B_data_t>
   static void Assign(const A_layout_t &A_layout, A_data_t &A_data,
                      const B_layout_t &B_layout, const B_data_t &B_data)
   {
      MFEM_STATIC_ASSERT(A_layout_t::rank == 2 && B_layout_t::rank == 2,
                         "invalid ranks");
      MFEM_STATIC_ASSERT(A_layout_t::dim_1 == B_layout_t::dim_1 &&
                         A_layout_t::dim_2 == B_layout_t::dim_2,
                         "invalid dimensions");
      for (int i2 = 0; i2 < A_layout_t::dim_2; i2++)
      {
         for (int i1 = 0; i1 < A_layout_t::dim_1; i1++)
         {
            mfem::Assign<Op>(A_data[A_layout.ind(i1,i2)],
                             B_data[B_layout.ind(i1,i2)]);
         }
      }
   }
};

template <>
struct TensorOps<3> // rank = 3
{
   // Assign: A {=,+=,*=} scalar_value
   template <AssignOp::Type Op, typename A_layout_t, typename A_data_t,
             typename scalar_t>
   static void Assign(const A_layout_t &A_layout, A_data_t &A_data,
                      scalar_t value)
   {
      MFEM_STATIC_ASSERT(A_layout_t::rank == 3, "invalid rank");
      for (int i3 = 0; i3 < A_layout_t::dim_3; i3++)
      {
         for (int i2 = 0; i2 < A_layout_t::dim_2; i2++)
         {
            for (int i1 = 0; i1 < A_layout_t::dim_1; i1++)
            {
               mfem::Assign<Op>(A_data[A_layout.ind(i1,i2,i3)], value);
            }
         }
      }
   }

   // Assign: A {=,+=,*=} B
   template <AssignOp::Type Op,
             typename A_layout_t, typename A_data_t,
             typename B_layout_t, typename B_data_t>
   static void Assign(const A_layout_t &A_layout, A_data_t &A_data,
                      const B_layout_t &B_layout, const B_data_t &B_data)
   {
      MFEM_STATIC_ASSERT(A_layout_t::rank == 3 && B_layout_t::rank == 3,
                         "invalid ranks");
      MFEM_STATIC_ASSERT(A_layout_t::dim_1 == B_layout_t::dim_1 &&
                         A_layout_t::dim_2 == B_layout_t::dim_2 &&
                         A_layout_t::dim_3 == B_layout_t::dim_3,
                         "invalid dimensions");
      for (int i3 = 0; i3 < A_layout_t::dim_3; i3++)
      {
         for (int i2 = 0; i2 < A_layout_t::dim_2; i2++)
         {
            for (int i1 = 0; i1 < A_layout_t::dim_1; i1++)
            {
               mfem::Assign<Op>(A_data[A_layout.ind(i1,i2,i3)],
                                B_data[B_layout.ind(i1,i2,i3)]);
            }
         }
      }
   }
};

template <>
struct TensorOps<4> // rank = 4
{
   // Assign: A {=,+=,*=} scalar_value
   template <AssignOp::Type Op, typename A_layout_t, typename A_data_t,
             typename scalar_t>
   static void Assign(const A_layout_t &A_layout, A_data_t &A_data,
                      scalar_t value)
   {
      MFEM_STATIC_ASSERT(A_layout_t::rank == 4, "invalid rank");
      for (int i4 = 0; i4 < A_layout_t::dim_4; i4++)
      {
         for (int i3 = 0; i3 < A_layout_t::dim_3; i3++)
         {
            for (int i2 = 0; i2 < A_layout_t::dim_2; i2++)
            {
               for (int i1 = 0; i1 < A_layout_t::dim_1; i1++)
               {
                  mfem::Assign<Op>(A_data[A_layout.ind(i1,i2,i3,i4)], value);
               }
            }
         }
      }
   }

   // Assign: A {=,+=,*=} B
   template <AssignOp::Type Op,
             typename A_layout_t, typename A_data_t,
             typename B_layout_t, typename B_data_t>
   static void Assign(const A_layout_t &A_layout, A_data_t &A_data,
                      const B_layout_t &B_layout, const B_data_t &B_data)
   {
      MFEM_STATIC_ASSERT(A_layout_t::rank == 4 && B_layout_t::rank == 4,
                         "invalid ranks");
      MFEM_STATIC_ASSERT(A_layout_t::dim_1 == B_layout_t::dim_1 &&
                         A_layout_t::dim_2 == B_layout_t::dim_2 &&
                         A_layout_t::dim_3 == B_layout_t::dim_3 &&
                         A_layout_t::dim_4 == B_layout_t::dim_4,
                         "invalid dimensions");
      for (int i4 = 0; i4 < A_layout_t::dim_4; i4++)
      {
         for (int i3 = 0; i3 < A_layout_t::dim_3; i3++)
         {
            for (int i2 = 0; i2 < A_layout_t::dim_2; i2++)
            {
               for (int i1 = 0; i1 < A_layout_t::dim_1; i1++)
               {
                  mfem::Assign<Op>(A_data[A_layout.ind(i1,i2,i3,i4)],
                                   B_data[B_layout.ind(i1,i2,i3,i4)]);
               }
            }
         }
      }
   }
};

} // namespace mfem::internal

// Tensor or sub-tensor assign function: A {=,+=,*=} scalar_value.
template <AssignOp::Type Op, typename A_layout_t, typename A_data_t,
          typename scalar_t>
inline void TAssign(const A_layout_t &A_layout, A_data_t &A_data,
                    scalar_t value)
{
   internal::TensorOps<A_layout_t::rank>::
   template Assign<Op>(A_layout, A_data, value);
}

// Tensor assign function: A {=,+=,*=} B that allows different input and output
// layouts. With suitable layouts this function can be used to permute
// (transpose) tensors, extract sub-tensors, etc.
template <AssignOp::Type Op,
          typename A_layout_t, typename A_data_t,
          typename B_layout_t, typename B_data_t>
inline void TAssign(const A_layout_t &A_layout, A_data_t &A_data,
                    const B_layout_t &B_layout, const B_data_t &B_data)
{
   internal::TensorOps<A_layout_t::rank>::
   template Assign<Op>(A_layout, A_data, B_layout, B_data);
}

} // namespace mfem

#endif // MFEM_TEMPLATE_TENSOR_OPS
