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

#ifndef MFEM_TEMPLATE_SMALL_MATRIX_OPS
#define MFEM_TEMPLATE_SMALL_MATRIX_OPS

#include "config.hpp"
#include "assign_ops.hpp"

namespace mfem
{

// Small matrix operations, defined by specializations: determinant, adjugate,
// etc.

namespace internal
{

template <int N1, int N2>
struct MatrixOps { };

template <>
struct MatrixOps<1,1>
{
   // Compute det(A).
   template <typename scalar_t, typename layout_t, typename data_t>
   static inline scalar_t Det(const layout_t &a, const data_t &A)
   {
      return A[a.ind(0,0)];
   }

   // Compute det(A). Batched version: D[i] {=,+=,*=} det(A[i,*,*])
   template <AssignOp::Type Op, typename A_layout_t, typename A_data_t,
             typename D_data_t>
   static inline void Det(const A_layout_t &a, const A_data_t &A, D_data_t &D)
   {
      const int M = A_layout_t::dim_1;
      for (int i = 0; i < M; i++)
      {
         Assign<Op>(D[i], A[a.ind(i,0,0)]);
      }
   }

   // Compute B = adj(A).
   template <typename scalar_t,
             typename A_layout_t, typename A_data_t,
             typename B_layout_t, typename B_data_t>
   static inline void Adjugate(const A_layout_t &a, const A_data_t &A,
                               const B_layout_t &b, B_data_t &B)
   {
      B[b.ind(0,0)] = scalar_t(1);
   }

   // Compute adj(A) and det(A).
   template <typename scalar_t,
             typename A_layout_t, typename A_data_t,
             typename B_layout_t, typename B_data_t>
   static inline scalar_t AdjDet(const A_layout_t &a, const A_data_t &A,
                                 const B_layout_t &b, B_data_t &B)
   {
      Adjugate<scalar_t>(a, A, b, B);
      return Det<scalar_t>(a, A);
   }
};

template <>
struct MatrixOps<2,2>
{
   // Compute det(A).
   template <typename scalar_t, typename layout_t, typename data_t>
   static inline scalar_t Det(const layout_t &a, const data_t &A)
   {
      return (A[a.ind(0,0)]*A[a.ind(1,1)] -
              A[a.ind(1,0)]*A[a.ind(0,1)]);
   }

   // Compute det(A). Batched version: D[i] {=,+=,*=} det(A[i,*,*])
   template <AssignOp::Type Op, typename A_layout_t, typename A_data_t,
             typename D_data_t>
   static inline void Det(const A_layout_t &a, const A_data_t &A, D_data_t &D)
   {
      const int M = A_layout_t::dim_1;
      for (int i = 0; i < M; i++)
      {
         Assign<Op>(D[i], (A[a.ind(i,0,0)]*A[a.ind(i,1,1)] -
                           A[a.ind(i,1,0)]*A[a.ind(i,0,1)]));
      }
   }

   // Compute B = adj(A).
   template <typename scalar_t,
             typename A_layout_t, typename A_data_t,
             typename B_layout_t, typename B_data_t>
   static inline void Adjugate(const A_layout_t &a, const A_data_t &A,
                               const B_layout_t &b, B_data_t &B)
   {
      B[b.ind(0,0)] =  A[a.ind(1,1)];
      B[b.ind(0,1)] = -A[a.ind(0,1)];
      B[b.ind(1,0)] = -A[a.ind(1,0)];
      B[b.ind(1,1)] =  A[a.ind(0,0)];
   }

   // Compute adj(A) and det(A).
   template <typename scalar_t,
             typename A_layout_t, typename A_data_t,
             typename B_layout_t, typename B_data_t>
   static inline scalar_t AdjDet(const A_layout_t &a, const A_data_t &A,
                                 const B_layout_t &b, B_data_t &B)
   {
      Adjugate<scalar_t>(a, A, b, B);
      return Det<scalar_t>(a, A);
   }
};

template <>
struct MatrixOps<3,3>
{
   // Compute det(A).
   template <typename scalar_t, typename layout_t, typename data_t>
   static inline scalar_t Det(const layout_t &a, const data_t &A)
   {
      return (A[a.ind(0,0)]*(A[a.ind(1,1)]*A[a.ind(2,2)] -
                             A[a.ind(2,1)]*A[a.ind(1,2)]) -
              A[a.ind(1,0)]*(A[a.ind(0,1)]*A[a.ind(2,2)] -
                             A[a.ind(2,1)]*A[a.ind(0,2)]) +
              A[a.ind(2,0)]*(A[a.ind(0,1)]*A[a.ind(1,2)] -
                             A[a.ind(1,1)]*A[a.ind(0,2)]));
   }

   // Compute det(A). Batched version: D[i] {=,+=,*=} det(A[i,*,*])
   template <AssignOp::Type Op, typename A_layout_t, typename A_data_t,
             typename D_data_t>
   static inline void Det(const A_layout_t &a, const A_data_t &A, D_data_t &D)
   {
      const int M = A_layout_t::dim_1;
      for (int i = 0; i < M; i++)
      {
         Assign<Op>(
            D[i],
            A[a.ind(i,0,0)]*(A[a.ind(i,1,1)]*A[a.ind(i,2,2)] -
                             A[a.ind(i,2,1)]*A[a.ind(i,1,2)]) -
            A[a.ind(i,1,0)]*(A[a.ind(i,0,1)]*A[a.ind(i,2,2)] -
                             A[a.ind(i,2,1)]*A[a.ind(i,0,2)]) +
            A[a.ind(i,2,0)]*(A[a.ind(i,0,1)]*A[a.ind(i,1,2)] -
                             A[a.ind(i,1,1)]*A[a.ind(i,0,2)]));
      }
   }

   // Compute B = adj(A).
   template <typename scalar_t,
             typename A_layout_t, typename A_data_t,
             typename B_layout_t, typename B_data_t>
   static inline void Adjugate(const A_layout_t &a, const A_data_t &A,
                               const B_layout_t &b, B_data_t &B)
   {
      B[b.ind(0,0)] = A[a.ind(1,1)]*A[a.ind(2,2)] - A[a.ind(1,2)]*A[a.ind(2,1)];
      B[b.ind(0,1)] = A[a.ind(0,2)]*A[a.ind(2,1)] - A[a.ind(0,1)]*A[a.ind(2,2)];
      B[b.ind(0,2)] = A[a.ind(0,1)]*A[a.ind(1,2)] - A[a.ind(0,2)]*A[a.ind(1,1)];
      B[b.ind(1,0)] = A[a.ind(1,2)]*A[a.ind(2,0)] - A[a.ind(1,0)]*A[a.ind(2,2)];
      B[b.ind(1,1)] = A[a.ind(0,0)]*A[a.ind(2,2)] - A[a.ind(0,2)]*A[a.ind(2,0)];
      B[b.ind(1,2)] = A[a.ind(0,2)]*A[a.ind(1,0)] - A[a.ind(0,0)]*A[a.ind(1,2)];
      B[b.ind(2,0)] = A[a.ind(1,0)]*A[a.ind(2,1)] - A[a.ind(1,1)]*A[a.ind(2,0)];
      B[b.ind(2,1)] = A[a.ind(0,1)]*A[a.ind(2,0)] - A[a.ind(0,0)]*A[a.ind(2,1)];
      B[b.ind(2,2)] = A[a.ind(0,0)]*A[a.ind(1,1)] - A[a.ind(0,1)]*A[a.ind(1,0)];
   }

   // Compute adj(A) and det(A).
   template <typename scalar_t,
             typename A_layout_t, typename A_data_t,
             typename B_layout_t, typename B_data_t>
   static inline scalar_t AdjDet(const A_layout_t &a, const A_data_t &A,
                                 const B_layout_t &b, B_data_t &B)
   {
      Adjugate<scalar_t>(a, A, b, B);
      return (A[a.ind(0,0)]*B[b.ind(0,0)] +
              A[a.ind(1,0)]*B[b.ind(0,1)] +
              A[a.ind(2,0)]*B[b.ind(0,2)]);
   }
};

} // namespace mfem::internal

// Compute the determinant of a (small) matrix: det(A).
template <typename scalar_t, typename layout_t, typename data_t>
inline scalar_t TDet(const layout_t &a, const data_t &A)
{
   MFEM_STATIC_ASSERT(layout_t::rank == 2, "invalid rank");
   return internal::MatrixOps<layout_t::dim_1,layout_t::dim_2>::
          template Det<scalar_t>(a, A);
}

// Compute the determinants of a set of (small) matrices: D[i] = det(A[i,*,*]).
// The layout of A is (M x N1 x N2) and the size of D is M.
template <AssignOp::Type Op, typename A_layout_t, typename A_data_t,
          typename D_data_t>
inline void TDet(const A_layout_t &a, const A_data_t &A, D_data_t &D)
{
   MFEM_STATIC_ASSERT(A_layout_t::rank == 3, "invalid rank");
   internal::MatrixOps<A_layout_t::dim_2,A_layout_t::dim_3>::
   template Det<Op>(a, A, D);
}

// Compute the adjugate matrix of a (small) matrix: B = adj(A).
template <typename scalar_t,
          typename A_layout_t, typename A_data_t,
          typename B_layout_t, typename B_data_t>
inline void TAdjugate(const A_layout_t &a, const A_data_t &A,
                      const B_layout_t &b, B_data_t &B)
{
   MFEM_STATIC_ASSERT(A_layout_t::rank == 2 && B_layout_t::rank == 2,
                      "invalid ranks");
   internal::MatrixOps<A_layout_t::dim_1,A_layout_t::dim_2>::
   template Adjugate<scalar_t>(a, A, b, B);
}

// Compute the adjugate and the determinant of a (small) matrix: B = adj(A),
// return det(A).
template <typename scalar_t,
          typename A_layout_t, typename A_data_t,
          typename B_layout_t, typename B_data_t>
inline scalar_t TAdjDet(const A_layout_t &a, const A_data_t &A,
                        const B_layout_t &b, B_data_t &B)
{
   MFEM_STATIC_ASSERT(A_layout_t::rank == 2 && B_layout_t::rank == 2,
                      "invalid ranks");
   return internal::MatrixOps<A_layout_t::dim_1,A_layout_t::dim_2>::
          template AdjDet<scalar_t>(a, A, b, B);
}

} // namespace mfem

#endif // MFEM_TEMPLATE_SMALL_MATRIX_OPS
