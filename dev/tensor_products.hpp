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

#include "config.hpp"
#include "tensor_ops.hpp"
#include "matrix_products.hpp"

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

// C_{i,k,j,l}  {=|+=}  A_{s,i} A_{s,j} B_{k,s,l}
template <bool Add,
          typename A_layout_t, typename A_data_t,
          typename B_layout_t, typename B_data_t,
          typename C_layout_t, typename C_data_t>
inline
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
}

} // namespace mfem

#endif // MFEM_TEMPLATE_TENSOR_PRODUCTS
