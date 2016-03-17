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

#ifndef MFEM_TEMPLATE_MASS_KERNEL
#define MFEM_TEMPLATE_MASS_KERNEL

#include "config/tconfig.hpp"
#include "tintegrator.hpp"
#include "tcoefficient.hpp"
#include "tbilinearform.hpp"

namespace mfem
{

template <int SDim, int Dim, typename complex_t>
struct TMassKernel
{
   typedef complex_t complex_type;

   // needed for the TElementTransformation::Result class
   static const bool uses_Jacobians = true;

   // needed for the FieldEvaluator::Data class
   static const bool in_values     = true;
   static const bool in_gradients  = false;
   static const bool out_values    = true;
   static const bool out_gradients = false;

   // Partially assembled data type for one element with the given number of
   // quadrature points. This type is used in partial assembly, and partially
   // assembled action.
   template <int qpts>
   struct p_asm_data { typedef TVector<qpts,complex_t> type; };

   // Partially assembled data type for one element with the given number of
   // quadrature points. This type is used in full element matrix assembly.
   template <int qpts>
   struct f_asm_data { typedef TVector<qpts,complex_t> type; };

   template <typename IR, typename coeff_t, int NE>
   struct CoefficientEval
   {
      typedef typename IntRuleCoefficient<IR,coeff_t,NE>::Type Type;
   };

   // Method used for un-assembled (matrix free) action.
   // Jt       [M x Dim x SDim x NE] - Jacobian transposed, data member in F
   // Q                              - CoefficientEval<>::Type
   // q                              - CoefficientEval<>::Type::result_t
   // val_qpts [M x NC x NE]         - in/out data member in R
   //
   // val_qpts *= w det(J)
   template <typename T_result_t, typename Q_t, typename q_t,
             typename S_data_t>
   static inline MFEM_ALWAYS_INLINE
   void Action(const int k, const T_result_t &F,
               const Q_t &Q, const q_t &q, S_data_t &R)
   {
      typedef typename T_result_t::Jt_type::data_type real_t;
      const int M = R.val_qpts.layout.dim_1;
      MFEM_STATIC_ASSERT(F.Jt.layout.dim_1 == M, "incompatible dimensions");
      MFEM_FLOPS_ADD(2*M);
      for (int i = 0; i < M; i++)
      {
         R.val_qpts(i,0,k) *=
            Q.get(q,i,k) * TDet<real_t>(F.Jt.layout.ind14(i,k), F.Jt);
      }
   }

   // Method defining partial assembly.
   // Jt   [M x Dim x SDim x NE] - Jacobian transposed, data member in F
   // Q                          - CoefficientEval<>::Type
   // q                          - CoefficientEval<>::Type::result_t
   // A    [M]                   - partially assembled scalars
   //
   // A = w det(J)
   template <typename T_result_t, typename Q_t, typename q_t, int qpts>
   static inline MFEM_ALWAYS_INLINE
   void Assemble(const int k, const T_result_t &F,
                 const Q_t &Q, const q_t &q, TVector<qpts,complex_t> &A)
   {
      typedef typename T_result_t::Jt_type::data_type real_t;
      const int M = F.Jt.layout.dim_1;
      MFEM_STATIC_ASSERT(qpts == M, "incompatible dimensions");
      MFEM_FLOPS_ADD(M);
      for (int i = 0; i < M; i++)
      {
         A[i] = Q.get(q,i,k) * TDet<real_t>(F.Jt.layout.ind14(i,k), F.Jt);
      }
   }

   // Method for partially assembled action.
   // A        [M]           - partially assembled scalars
   // val_qpts [M x NC x NE] - in/out data member in R
   //
   // val_qpts *= A
   template <int qpts, typename S_data_t>
   static inline MFEM_ALWAYS_INLINE
   void MultAssembled(const int k, const TVector<qpts,complex_t> &A, S_data_t &R)
   {
      const int M = R.val_qpts.layout.dim_1;
      MFEM_STATIC_ASSERT(qpts == M, "incompatible dimensions");
      MFEM_FLOPS_ADD(M);
      for (int i = 0; i < M; i++)
      {
         R.val_qpts(i,0,k) *= A[i];
      }
   }
};

} // namespace mfem

#endif // MFEM_TEMPLATE_MASS_KERNEL
