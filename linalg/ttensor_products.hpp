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

#ifndef MFEM_TEMPLATE_TENSOR_PRODUCTS
#define MFEM_TEMPLATE_TENSOR_PRODUCTS

#include "config/tconfig.hpp"
#include "ttensor_ops.hpp"
#include "tmatrix_products.hpp"
#include "ttensor_types.hpp"

namespace mfem
{

// C_{i,j,k}  {=|+=}  \sum_s A_{s,j} B_{i,s,k}
template <bool Add,
          typename A_layout_t, typename A_data_t,
          typename B_layout_t, typename B_data_t,
          typename C_layout_t, typename C_data_t>
MFEM_ALWAYS_INLINE inline
void Mult_1_2(const A_layout_t &A_layout, const A_data_t &A_data,
              const B_layout_t &B_layout, const B_data_t &B_data,
              const C_layout_t &C_layout, C_data_t &C_data)
{
   MFEM_STATIC_ASSERT(A_layout_t::rank == 2 && B_layout_t::rank == 3 &&
                      C_layout_t::rank == 3, "invalid ranks");
   const int B3 = B_layout_t::dim_3;
   const int C3 = C_layout_t::dim_3;
   MFEM_STATIC_ASSERT(B3 == C3, "invalid dimentions");
   for (int k = 0; k < B3; k++)
   {
      Mult_AB<Add>(B_layout.ind3(k), B_data,
                   A_layout, A_data,
                   C_layout.ind3(k), C_data);
   }
}

// C_{i,j,k}  {=|+=}  \sum_s A_{i,s} B_{s,j,k}
template <bool Add,
          typename A_layout_t, typename A_data_t,
          typename B_layout_t, typename B_data_t,
          typename C_layout_t, typename C_data_t>
MFEM_ALWAYS_INLINE inline
void Mult_2_1(const A_layout_t &A_layout, const A_data_t &A_data,
              const B_layout_t &B_layout, const B_data_t &B_data,
              const C_layout_t &C_layout, C_data_t &C_data)
{
   MFEM_STATIC_ASSERT(A_layout_t::rank == 2 && B_layout_t::rank == 3 &&
                      C_layout_t::rank == 3, "invalid ranks");
   Mult_AB<Add>(A_layout, A_data,
                B_layout.merge_23(), B_data,
                C_layout.merge_23(), C_data);
}

// C_{i,k,j,l}  {=|+=}  \sum_s A_{s,i} A_{s,j} B_{k,s,l}
template <bool Add,
          typename A_layout_t, typename A_data_t,
          typename B_layout_t, typename B_data_t,
          typename C_layout_t, typename C_data_t>
MFEM_ALWAYS_INLINE inline
void TensorAssemble(const A_layout_t &A_layout, const A_data_t &A_data,
                    const B_layout_t &B_layout, const B_data_t &B_data,
                    const C_layout_t &C_layout, C_data_t &C_data)
{
   MFEM_STATIC_ASSERT(A_layout_t::rank == 2 && B_layout_t::rank == 3 &&
                      C_layout_t::rank == 4, "invalid ranks");
   const int A1 = A_layout_t::dim_1;
   const int A2 = A_layout_t::dim_2;
   const int B1 = B_layout_t::dim_1;
   const int B2 = B_layout_t::dim_2;
   const int B3 = B_layout_t::dim_3;
   const int C1 = C_layout_t::dim_1;
   const int C2 = C_layout_t::dim_2;
   const int C3 = C_layout_t::dim_3;
   const int C4 = C_layout_t::dim_4;
   MFEM_STATIC_ASSERT(A1 == B2 && A2 == C1 && A2 == C3 && B1 == C2 && B3 == C4,
                      "invalid dimensions");

#if 1
   // Impl == 3
   MFEM_FLOPS_ADD(3*A1*A2*A2*B1*B3);
   if (!Add) { TAssign<AssignOp::Set>(C_layout, C_data, 0.0); }
   for (int j = 0; j < A2; j++)
   {
      for (int i = 0; i < A2; i++)
      {
         for (int l = 0; l < B3; l++)
         {
            for (int k = 0; k < B1; k++)
            {
               for (int s = 0; s < A1; s++)
               {
                  // C(i,k,j,l) += A(s,i) * A(s,j) * B(k,s,l);
                  C_data[C_layout.ind(i,k,j,l)] +=
                     A_data[A_layout.ind(s,i)] *
                     A_data[A_layout.ind(s,j)] *
                     B_data[B_layout.ind(k,s,l)];
               }
            }
         }
      }
   }
#else
   // Impl == 1
   if (!Add) { TAssign<AssignOp::Set>(C_layout, C_data, 0.0); }
   for (int s = 0; s < A1; s++)
   {
      for (int i = 0; i < A2; i++)
      {
         for (int k = 0; k < B1; k++)
         {
            for (int j = 0; j < A2; j++)
            {
               for (int l = 0; l < B3; l++)
               {
                  // C(i,k,j,l) += A(s,i) * A(s,j) * B(k,s,l);
                  C_data[C_layout.ind(i,k,j,l)] +=
                     A_data[A_layout.ind(s,i)] *
                     A_data[A_layout.ind(s,j)] *
                     B_data[B_layout.ind(k,s,l)];
               }
            }
         }
      }
   }
#endif
}

// D_{i,k,j,l}  {=|+=}  \sum_s A_{i,s} B_{s,j} C_{k,s,l}
template <bool Add,
          typename A_layout_t, typename A_data_t,
          typename B_layout_t, typename B_data_t,
          typename C_layout_t, typename C_data_t,
          typename D_layout_t, typename D_data_t>
MFEM_ALWAYS_INLINE inline
void TensorAssemble(const A_layout_t &A_layout, const A_data_t &A_data,
                    const B_layout_t &B_layout, const B_data_t &B_data,
                    const C_layout_t &C_layout, const C_data_t &C_data,
                    const D_layout_t &D_layout, D_data_t &D_data)
{
   MFEM_STATIC_ASSERT(A_layout_t::rank == 2 && B_layout_t::rank == 2 &&
                      C_layout_t::rank == 3 && D_layout_t::rank == 4,
                      "invalid ranks");
   const int A1 = A_layout_t::dim_1;
   const int A2 = A_layout_t::dim_2;
   const int B1 = B_layout_t::dim_1;
   const int B2 = B_layout_t::dim_2;
   const int C1 = C_layout_t::dim_1;
   const int C2 = C_layout_t::dim_2;
   const int C3 = C_layout_t::dim_3;
   const int D1 = D_layout_t::dim_1;
   const int D2 = D_layout_t::dim_2;
   const int D3 = D_layout_t::dim_3;
   const int D4 = D_layout_t::dim_4;
   MFEM_STATIC_ASSERT(A2 == B1 && A2 == C2 && A1 == D1 && B2 == D3 &&
                      C1 == D2 && C3 == D4, "invalid dimensions");

#if 0
   TTensor4<A1,C1,A2,C3> H;
   // H_{i,k,s,l} = A_{i,s} C_{k,s,l}
   for (int l = 0; l < C3; l++)
   {
      for (int s = 0; s < B1; s++)
      {
         for (int k = 0; k < C1; k++)
         {
            for (int i = 0; i < A1; i++)
            {
               H(i,k,s,l) = A_data[A_layout.ind(i,s)]*
                            C_data[C_layout.ind(k,s,l)];
            }
         }
      }
   }
   // D_{(i,k),j,l} = \sum_s B_{s,j} H_{(i,k),s,l}
   Mult_1_2<Add>(B_layout, B_data, H.layout.merge_12(), H,
                 D_layout.merge_12(), D_data);
#elif 1
   MFEM_FLOPS_ADD(A1*B1*C1*C3); // computation of H(l)
   for (int l = 0; l < C3; l++)
   {
      TTensor3<A1,C1,A2,typename C_data_t::data_type> H;
      // H(l)_{i,k,s} = A_{i,s} C_{k,s,l}
      for (int s = 0; s < B1; s++)
      {
         for (int k = 0; k < C1; k++)
         {
            for (int i = 0; i < A1; i++)
            {
               H(i,k,s) = A_data[A_layout.ind(i,s)]*
                          C_data[C_layout.ind(k,s,l)];
            }
         }
      }
      // D_{(i,k),j,l} = \sum_s H(l)_{(i,k),s} B_{s,j}
      Mult_AB<Add>(H.layout.merge_12(), H, B_layout, B_data,
                   D_layout.merge_12().ind3(l), D_data);
   }
#else
   TTensor4<B1,C1,B2,C3> F;
   for (int l = 0; l < C3; l++)
   {
      for (int j = 0; j < B2; j++)
      {
         for (int k = 0; k < C1; k++)
         {
            for (int s = 0; s < B1; s++)
            {
               F(s,k,j,l) = B_data[B_layout.ind(s,j)]*
                            C_data[C_layout.ind(k,s,l)];
            }
         }
      }
   }
   Mult_AB<Add>(A_layout, A_data, F.layout.merge_34().merge_23(), F,
                D_layout.merge_34().merge_23(), D_data);
#endif
}


// C_{i,j,k,l}  {=|+=}  A_{i,j,k} B_{j,l}
template <AssignOp::Type Op,
          typename A_layout_t, typename A_data_t,
          typename B_layout_t, typename B_data_t,
          typename C_layout_t, typename C_data_t>
MFEM_ALWAYS_INLINE inline
void TensorProduct(const A_layout_t &a, const A_data_t &A,
                   const B_layout_t &b, const B_data_t &B,
                   const C_layout_t &c, C_data_t &C)
{
   const int A1 = A_layout_t::dim_1;
   const int A2 = A_layout_t::dim_2;
   const int A3 = A_layout_t::dim_3;
   const int B1 = B_layout_t::dim_1;
   const int B2 = B_layout_t::dim_2;
   const int C1 = C_layout_t::dim_1;
   const int C2 = C_layout_t::dim_2;
   const int C3 = C_layout_t::dim_3;
   const int C4 = C_layout_t::dim_4;
   MFEM_STATIC_ASSERT(A1 == C1 && A2 == B1 && A2 == C2 && A3 == C3 && B2 == C4,
                      "invalid dimensions");

   MFEM_FLOPS_ADD(A1*A2*A3*B2);
   for (int l = 0; l < B2; l++)
   {
      for (int k = 0; k < A3; k++)
      {
         for (int j = 0; j < A2; j++)
         {
            for (int i = 0; i < A1; i++)
            {
               mfem::Assign<Op>(C[c.ind(i,j,k,l)],
                                A[a.ind(i,j,k)]*B[b.ind(j,l)]);
            }
         }
      }
   }
}

} // namespace mfem

#endif // MFEM_TEMPLATE_TENSOR_PRODUCTS
