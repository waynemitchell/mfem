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

#ifndef MFEM_TEMPLATE_MATRIX_PRODUCTS
#define MFEM_TEMPLATE_MATRIX_PRODUCTS

#include "../config/tconfig.hpp"

namespace mfem
{

// Matrix-matrix products

// C  {=|+=}  A.B -- simple version (no blocks)
template <bool Add,
          typename A_layout_t, typename A_data_t,
          typename B_layout_t, typename B_data_t,
          typename C_layout_t, typename C_data_t>
MFEM_ALWAYS_INLINE inline
void sMult_AB(const A_layout_t &A_layout, const A_data_t &A_data,
              const B_layout_t &B_layout, const B_data_t &B_data,
              const C_layout_t &C_layout, C_data_t &C_data)
{
   MFEM_STATIC_ASSERT(A_layout_t::rank == 2 && B_layout_t::rank == 2 &&
                      C_layout_t::rank == 2, "invalid ranks");
   const int A1 = A_layout_t::dim_1;
   const int A2 = A_layout_t::dim_2;
   const int B1 = B_layout_t::dim_1;
   const int B2 = B_layout_t::dim_2;
   const int C1 = C_layout_t::dim_1;
   const int C2 = C_layout_t::dim_2;
   MFEM_STATIC_ASSERT(A2 == B1 && A1 == C1 && B2 == C2,
                      "invalid dimensions");

   MFEM_FLOPS_ADD(Add ? 2*A1*A2*B2 : 2*A1*A2*B2-A1*B2);
   for (int b2 = 0; b2 < B2; b2++)
   {
      for (int s = 0; s < A2; s++)
      {
         for (int a1 = 0; a1 < A1; a1++)
         {
            if (!Add && s == 0)
            {
               // C(a1,b2) = A(a1,s) * B(s,b2);
               C_data[C_layout.ind(a1,b2)] =
                  A_data[A_layout.ind(a1,s)] * B_data[B_layout.ind(s,b2)];
            }
            else
            {
               // C(a1,b2) += A(a1,s) * B(s,b2);
               C_data[C_layout.ind(a1,b2)] +=
                  A_data[A_layout.ind(a1,s)] * B_data[B_layout.ind(s,b2)];
            }
         }
      }
   }
}

// C  {=|+=}  A.B  -- block version
template <int bA1, int bA2, int bB2, // block sizes
          bool Add,
          typename A_layout_t, typename A_data_t,
          typename B_layout_t, typename B_data_t,
          typename C_layout_t, typename C_data_t>
MFEM_ALWAYS_INLINE inline
void bMult_AB(const A_layout_t &A_layout, const A_data_t &A_data,
              const B_layout_t &B_layout, const B_data_t &B_data,
              const C_layout_t &C_layout, C_data_t &C_data)
{
   MFEM_STATIC_ASSERT(A_layout_t::rank == 2 && B_layout_t::rank == 2 &&
                      C_layout_t::rank == 2, "invalid ranks");
   const int A1 = A_layout_t::dim_1;
   const int A2 = A_layout_t::dim_2;
   const int B1 = B_layout_t::dim_1;
   const int B2 = B_layout_t::dim_2;
   const int C1 = C_layout_t::dim_1;
   const int C2 = C_layout_t::dim_2;
   MFEM_STATIC_ASSERT(A2 == B1 && A1 == C1 && B2 == C2,
                      "invalid dimensions");

   const int rA1 = A1%bA1;
   const int rA2 = A2%bA2;
   const int rB2 = B2%bB2;

   for (int b2_b = 0; b2_b < B2/bB2; b2_b++)
   {
      if (A2/bA2 > 0)
      {
         // s_b == 0
         for (int a1_b = 0; a1_b < A1/bA1; a1_b++)
         {
            sMult_AB<Add>(
               A_layout.template sub<bA1,bA2>(a1_b*bA1,0), A_data,
               B_layout.template sub<bA2,bB2>(0,b2_b*bB2), B_data,
               C_layout.template sub<bA1,bB2>(a1_b*bA1,b2_b*bB2), C_data);
         }
         if (rA1)
         {
            sMult_AB<Add>(
               A_layout.template sub<rA1,bA2>(A1-rA1,0), A_data,
               B_layout.template sub<bA2,bB2>(0,b2_b*bB2), B_data,
               C_layout.template sub<rA1,bB2>(A1-rA1,b2_b*bB2), C_data);
         }
         for (int s_b = 1; s_b < A2/bA2; s_b++)
         {
            for (int a1_b = 0; a1_b < A1/bA1; a1_b++)
            {
               sMult_AB<true>(
                  A_layout.template sub<bA1,bA2>(a1_b*bA1,s_b*bA2), A_data,
                  B_layout.template sub<bA2,bB2>(s_b*bA2,b2_b*bB2), B_data,
                  C_layout.template sub<bA1,bB2>(a1_b*bA1,b2_b*bB2), C_data);
            }
            if (rA1)
            {
               sMult_AB<true>(
                  A_layout.template sub<rA1,bA2>(A1-rA1,s_b*bA2), A_data,
                  B_layout.template sub<bA2,bB2>(s_b*bA2,b2_b*bB2), B_data,
                  C_layout.template sub<rA1,bB2>(A1-rA1,b2_b*bB2), C_data);
            }
         }
      }
      if (rA2)
      {
         const bool rAdd = Add || (A2/bA2 > 0);
         for (int a1_b = 0; a1_b < A1/bA1; a1_b++)
         {
            sMult_AB<rAdd>(
               A_layout.template sub<bA1,rA2>(a1_b*bA1,A2-rA2), A_data,
               B_layout.template sub<rA2,bB2>(A2-rA2,b2_b*bB2), B_data,
               C_layout.template sub<bA1,bB2>(a1_b*bA1,b2_b*bB2), C_data);
         }
         if (rA1)
         {
            sMult_AB<rAdd>(
               A_layout.template sub<rA1,rA2>(A1-rA1,A2-rA2), A_data,
               B_layout.template sub<rA2,bB2>(A2-rA2,b2_b*bB2), B_data,
               C_layout.template sub<rA1,bB2>(A1-rA1,b2_b*bB2), C_data);
         }
      }
   }
   if (rB2)
   {
      if (A2/bA2 > 0)
      {
         // s_b == 0
         for (int a1_b = 0; a1_b < A1/bA1; a1_b++)
         {
            sMult_AB<Add>(
               A_layout.template sub<bA1,bA2>(a1_b*bA1,0), A_data,
               B_layout.template sub<bA2,rB2>(0,B2-rB2), B_data,
               C_layout.template sub<bA1,rB2>(a1_b*bA1,B2-rB2), C_data);
         }
         if (rA1)
         {
            sMult_AB<Add>(
               A_layout.template sub<rA1,bA2>(A1-rA1,0), A_data,
               B_layout.template sub<bA2,rB2>(0,B2-rB2), B_data,
               C_layout.template sub<rA1,rB2>(A1-rA1,B2-rB2), C_data);
         }
      }
      if (A2/bA2 > 1)
      {
         for (int s_b = 1; s_b < A2/bA2; s_b++)
         {
            for (int a1_b = 0; a1_b < A1/bA1; a1_b++)
            {
               sMult_AB<true>(
                  A_layout.template sub<bA1,bA2>(a1_b*bA1,s_b*bA2), A_data,
                  B_layout.template sub<bA2,rB2>(s_b*bA2,B2-rB2), B_data,
                  C_layout.template sub<bA1,rB2>(a1_b*bA1,B2-rB2), C_data);
            }
            if (rA1)
            {
               sMult_AB<true>(
                  A_layout.template sub<rA1,bA2>(A1-rA1,s_b*bA2), A_data,
                  B_layout.template sub<bA2,rB2>(s_b*bA2,B2-rB2), B_data,
                  C_layout.template sub<rA1,rB2>(A1-rA1,B2-rB2), C_data);
            }
         }
      }
      if (rA2)
      {
         const bool rAdd = Add || (A2/bA2 > 0);
         for (int a1_b = 0; a1_b < A1/bA1; a1_b++)
         {
            sMult_AB<rAdd>(
               A_layout.template sub<bA1,rA2>(a1_b*bA1,A2-rA2), A_data,
               B_layout.template sub<rA2,rB2>(A2-rA2,B2-rB2), B_data,
               C_layout.template sub<bA1,rB2>(a1_b*bA1,B2-rB2), C_data);
         }
         if (rA1)
         {
            sMult_AB<rAdd>(
               A_layout.template sub<rA1,rA2>(A1-rA1,A2-rA2), A_data,
               B_layout.template sub<rA2,rB2>(A2-rA2,B2-rB2), B_data,
               C_layout.template sub<rA1,rB2>(A1-rA1,B2-rB2), C_data);
         }
      }
   }
}

template <bool Add,
          typename A_layout_t, typename A_data_t,
          typename B_layout_t, typename B_data_t,
          typename C_layout_t, typename C_data_t>
MFEM_ALWAYS_INLINE inline
void Mult_AB(const A_layout_t &A_layout, const A_data_t &A_data,
             const B_layout_t &B_layout, const B_data_t &B_data,
             const C_layout_t &C_layout, C_data_t &C_data)
{
   const int b = MFEM_TEMPLATE_BLOCK_SIZE;
   bMult_AB<b,b,b,Add>(A_layout, A_data, B_layout, B_data, C_layout, C_data);
}

} // namespace mfem

#endif // MFEM_TEMPLATE_MATRIX_PRODUCTS
