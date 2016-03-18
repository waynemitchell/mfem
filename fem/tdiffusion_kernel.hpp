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

#ifndef MFEM_TEMPLATE_DIFFUSION_KERNEL
#define MFEM_TEMPLATE_DIFFUSION_KERNEL

#include "../config/tconfig.hpp"
#include "tintegrator.hpp"
#include "tcoefficient.hpp"
#include "tbilinearform.hpp"

namespace mfem
{

// complex_t - type for the assembled data
template <int SDim, int Dim, typename complex_t>
struct TDiffusionKernel;

template <typename complex_t>
struct TDiffusionKernel<1,1,complex_t>
{
   typedef complex_t complex_type;

   // needed for the TElementTransformation::Result class
   static const bool uses_Jacobians = true;

   // needed for the FieldEvaluator::Data class
   static const bool in_values     = false;
   static const bool in_gradients  = true;
   static const bool out_values    = false;
   static const bool out_gradients = true;

   // Partially assembled data type for one element with the given number of
   // quadrature points. This type is used in partial assembly, and partially
   // assembled action.
   template <int qpts>
   struct p_asm_data { typedef TMatrix<qpts,1,complex_t> type; };

   // Partially assembled data type for one element with the given number of
   // quadrature points. This type is used in full element matrix assembly.
   template <int qpts>
   struct f_asm_data { typedef TTensor3<qpts,1,1,complex_t> type; };

   template <typename IR, typename coeff_t, int NE>
   struct CoefficientEval
   {
      typedef typename IntRuleCoefficient<IR,coeff_t,NE>::Type Type;
   };

   // Method used for un-assembled (matrix free) action.
   // Jt        [M x Dim x SDim x NE] - Jacobian transposed, data member in F
   // Q                               - CoefficientEval<>::Type
   // q                               - CoefficientEval<>::Type::result_t
   // grad_qpts [M x SDim x NC x NE]  - in/out data member in R
   //
   // grad_qpts = (w/det(J)) adj(J) adj(J)^t grad_qpts
   template <typename T_result_t, typename Q_t, typename q_t,
             typename S_data_t>
   static inline MFEM_ALWAYS_INLINE
   void Action(const int k, const T_result_t &F,
               const Q_t &Q, const q_t &q, S_data_t &R)
   {
      const int M = S_data_t::eval_type::qpts;
      MFEM_STATIC_ASSERT(T_result_t::Jt_type::layout_type::dim_1 == M,
                         "incompatible dimensions");
      MFEM_FLOPS_ADD(2*M);
      for (int i = 0; i < M; i++)
      {
         R.grad_qpts(i,0,0,k) *= (Q.get(q,i,k) / F.Jt(i,0,0,k));
      }
   }

   // Method defining partial assembly. The pointwise Dim x Dim matrices are
   // stored as symmetric (when asm_type == p_asm_data, i.e. A.layout.rank == 2)
   // or non-symmetric (when asm_type == f_asm_data, i.e. A.layout.rank == 3)
   // matrices.
   // Jt   [M x Dim x SDim x NE] - Jacobian transposed, data member in F
   // Q                          - CoefficientEval<>::Type
   // q                          - CoefficientEval<>::Type::result_t
   // A    [M x Dim*(Dim+1)/2]   - partially assembled Dim x Dim symm. matrices
   // A    [M x Dim x Dim]       - partially assembled Dim x Dim matrices
   //
   // A = (w/det(J)) adj(J) adj(J)^t
   template <typename T_result_t, typename Q_t, typename q_t, typename asm_type>
   static inline MFEM_ALWAYS_INLINE
   void Assemble(const int k, const T_result_t &F,
                 const Q_t &Q, const q_t &q, asm_type &A)
   {
      const int M = T_result_t::Jt_type::layout_type::dim_1;
      MFEM_STATIC_ASSERT(asm_type::layout_type::dim_1 == M,
                         "incompatible dimensions");
      MFEM_FLOPS_ADD(M);
      for (int i = 0; i < M; i++)
      {
         // A[i] is A(i,0) or A(i,0,0)
         A[i] = Q.get(q,i,k) / F.Jt(i,0,0,k);
      }
   }

   // Method for partially assembled action.
   // A         [M x Dim*(Dim+1)/2]  - partially assembled Dim x Dim symmetric
   //                                  matrices
   // grad_qpts [M x SDim x NC x NE] - in/out data member in R
   //
   // grad_qpts = A grad_qpts
   template <int qpts, typename S_data_t>
   static inline MFEM_ALWAYS_INLINE
   void MultAssembled(const int k, const TMatrix<qpts,1,complex_t> &A,
                      S_data_t &R)
   {
      const int M = S_data_t::eval_type::qpts;
      MFEM_STATIC_ASSERT(qpts == M, "incompatible dimensions");
      MFEM_FLOPS_ADD(M);
      for (int i = 0; i < M; i++)
      {
         R.grad_qpts(i,0,0,k) *= A(i,0);
      }
   }
};

template <typename complex_t>
struct TDiffusionKernel<2,2,complex_t>
{
   typedef complex_t complex_type;

   // needed for the TElementTransformation::Result class
   static const bool uses_Jacobians = true;

   // needed for the FieldEvaluator::Data class
   static const bool in_values     = false;
   static const bool in_gradients  = true;
   static const bool out_values    = false;
   static const bool out_gradients = true;

   // Partially assembled data type for one element with the given number of
   // quadrature points. This type is used in partial assembly, and partially
   // assembled action. Stores one symmetric 2 x 2 matrix per point.
   template <int qpts>
   struct p_asm_data { typedef TMatrix<qpts,3,complex_t> type; };

   // Partially assembled data type for one element with the given number of
   // quadrature points. This type is used in full element matrix assembly.
   // Stores one general (non-symmetric) 2 x 2 matrix per point.
   template <int qpts>
   struct f_asm_data { typedef TTensor3<qpts,2,2,complex_t> type; };

   template <typename IR, typename coeff_t, int NE>
   struct CoefficientEval
   {
      typedef typename IntRuleCoefficient<IR,coeff_t,NE>::Type Type;
   };

   // Method used for un-assembled (matrix free) action.
   // Jt        [M x Dim x SDim x NE] - Jacobian transposed, data member in F
   // Q                               - CoefficientEval<>::Type
   // q                               - CoefficientEval<>::Type::result_t
   // grad_qpts [M x SDim x NC x NE]  - in/out data member in R
   //
   // grad_qpts = (w/det(J)) adj(J) adj(J)^t grad_qpts
   template <typename T_result_t, typename Q_t, typename q_t,
             typename S_data_t>
   static inline MFEM_ALWAYS_INLINE
   void Action(const int k, const T_result_t &F,
               const Q_t &Q, const q_t &q, S_data_t &R)
   {
      const int M = S_data_t::eval_type::qpts;
      MFEM_STATIC_ASSERT(T_result_t::Jt_type::layout_type::dim_1 == M,
                         "incompatible dimensions");
      MFEM_FLOPS_ADD(18*M);
      for (int i = 0; i < M; i++)
      {
         typedef typename T_result_t::Jt_type::data_type real_t;
         const real_t J11 = F.Jt(i,0,0,k);
         const real_t J12 = F.Jt(i,1,0,k);
         const real_t J21 = F.Jt(i,0,1,k);
         const real_t J22 = F.Jt(i,1,1,k);
         const complex_t x1 = R.grad_qpts(i,0,0,k);
         const complex_t x2 = R.grad_qpts(i,1,0,k);
         // z = adj(J)^t x
         const complex_t z1 = J22 * x1 - J21 * x2;
         const complex_t z2 = J11 * x2 - J12 * x1;
         const complex_t w_det_J = Q.get(q,i,k) / (J11 * J22 - J21 * J12);
         R.grad_qpts(i,0,0,k) = w_det_J * (J22 * z1 - J12 * z2);
         R.grad_qpts(i,1,0,k) = w_det_J * (J11 * z2 - J21 * z1);
      }
   }

   // Method defining partial assembly. The pointwise Dim x Dim matrices are
   // stored as symmetric (when asm_type == p_asm_data, i.e. A.layout.rank == 2)
   // or non-symmetric (when asm_type == f_asm_data, i.e. A.layout.rank == 3)
   // matrices.
   // Jt   [M x Dim x SDim x NE] - Jacobian transposed, data member in F
   // Q                          - CoefficientEval<>::Type
   // q                          - CoefficientEval<>::Type::result_t
   // A    [M x Dim*(Dim+1)/2]   - partially assembled Dim x Dim symm. matrices
   // A    [M x Dim x Dim]       - partially assembled Dim x Dim matrices
   //
   // A = (w/det(J)) adj(J) adj(J)^t
   template <typename T_result_t, typename Q_t, typename q_t, typename asm_type>
   static inline MFEM_ALWAYS_INLINE
   void Assemble(const int k, const T_result_t &F,
                 const Q_t &Q, const q_t &q, asm_type &A)
   {
      typedef typename T_result_t::Jt_type::data_type real_t;
      const int M = T_result_t::Jt_type::layout_type::dim_1;
      MFEM_STATIC_ASSERT(asm_type::layout_type::dim_1 == M,
                         "incompatible dimensions");
      MFEM_FLOPS_ADD(16*M);
      const bool Symm = (asm_type::layout_type::rank == 2);
      for (int i = 0; i < M; i++)
      {
         const real_t J11 = F.Jt(i,0,0,k);
         const real_t J12 = F.Jt(i,1,0,k);
         const real_t J21 = F.Jt(i,0,1,k);
         const real_t J22 = F.Jt(i,1,1,k);
         const complex_t w_det_J = Q.get(q,i,k) / (J11 * J22 - J21 * J12);
         internal::MatrixOps<2,2>::Symm<Symm>::Set(
            A.layout.ind1(i), A,
            + w_det_J * (J12*J12 + J22*J22), // (1,1)
            - w_det_J * (J11*J12 + J21*J22), // (2,1)
            + w_det_J * (J11*J11 + J21*J21)  // (2,2)
         );
      }
   }

   // Method for partially assembled action.
   // A         [M x Dim*(Dim+1)/2]  - partially assembled Dim x Dim symmetric
   //                                  matrices
   // grad_qpts [M x SDim x NC x NE] - in/out data member in R
   //
   // grad_qpts = A grad_qpts
   template <int qpts, typename S_data_t>
   static inline MFEM_ALWAYS_INLINE
   void MultAssembled(const int k, const TMatrix<qpts,3,complex_t> &A,
                      S_data_t &R)
   {
      const int M = S_data_t::eval_type::qpts;
      MFEM_STATIC_ASSERT(qpts == M, "incompatible dimensions");
      MFEM_FLOPS_ADD(6*M);
      for (int i = 0; i < M; i++)
      {
         const complex_t A21 = A(i,1);
         const complex_t x1 = R.grad_qpts(i,0,0,k);
         const complex_t x2 = R.grad_qpts(i,1,0,k);
         R.grad_qpts(i,0,0,k) = A(i,0) * x1 +    A21 * x2;
         R.grad_qpts(i,1,0,k) =    A21 * x1 + A(i,2) * x2;
      }
   }
};

template <typename complex_t>
struct TDiffusionKernel<3,3,complex_t>
{
   typedef complex_t complex_type;

   // needed for the TElementTransformation::Result class
   static const bool uses_Jacobians = true;

   // needed for the FieldEvaluator::Data class
   static const bool in_values     = false;
   static const bool in_gradients  = true;
   static const bool out_values    = false;
   static const bool out_gradients = true;

   // Partially assembled data type for one element with the given number of
   // quadrature points. This type is used in partial assembly, and partially
   // assembled action. Stores one symmetric 3 x 3 matrix per point.
   template <int qpts>
   struct p_asm_data { typedef TMatrix<qpts,6,complex_t> type; };

   // Partially assembled data type for one element with the given number of
   // quadrature points. This type is used in full element matrix assembly.
   // Stores one general (non-symmetric) 3 x 3 matrix per point.
   template <int qpts>
   struct f_asm_data { typedef TTensor3<qpts,3,3,complex_t> type; };

   template <typename IR, typename coeff_t, int NE>
   struct CoefficientEval
   {
      typedef typename IntRuleCoefficient<IR,coeff_t,NE>::Type Type;
   };

   // Method used for un-assembled (matrix free) action.
   // Jt        [M x Dim x SDim x NE] - Jacobian transposed, data member in F
   // Q                               - CoefficientEval<>::Type
   // q                               - CoefficientEval<>::Type::result_t
   // grad_qpts [M x SDim x NC x NE]  - in/out data member in R
   //
   // grad_qpts = (w/det(J)) adj(J) adj(J)^t grad_qpts
   template <typename T_result_t, typename Q_t, typename q_t,
             typename S_data_t>
   static inline MFEM_ALWAYS_INLINE
   void Action(const int k, const T_result_t &F,
               const Q_t &Q, const q_t &q, S_data_t &R)
   {
      const int M = S_data_t::eval_type::qpts;
      MFEM_STATIC_ASSERT(T_result_t::Jt_type::layout_type::dim_1 == M,
                         "incompatible dimensions");
      MFEM_FLOPS_ADD(M);
      for (int i = 0; i < M; i++)
      {
         typedef typename T_result_t::Jt_type::data_type real_t;
         TMatrix<3,3,real_t> adj_J;
         const complex_t w_det_J =
            (Q.get(q,i,k) /
             TAdjDet<real_t>(F.Jt.layout.ind14(i,k).transpose_12(), F.Jt,
                             adj_J.layout, adj_J));
         TMatrix<3,1,complex_t> z; // z = adj(J)^t x
         sMult_AB<false>(adj_J.layout.transpose_12(), adj_J,
                         R.grad_qpts.layout.ind14(i,k), R.grad_qpts,
                         z.layout, z);
         z.Scale(w_det_J);
         sMult_AB<false>(adj_J.layout, adj_J,
                         z.layout, z,
                         R.grad_qpts.layout.ind14(i,k), R.grad_qpts);
      }
   }

   // Method defining partial assembly. The pointwise Dim x Dim matrices are
   // stored as symmetric (when asm_type == p_asm_data, i.e. A.layout.rank == 2)
   // or non-symmetric (when asm_type == f_asm_data, i.e. A.layout.rank == 3)
   // matrices.
   // Jt   [M x Dim x SDim x NE] - Jacobian transposed, data member in F
   // Q                          - CoefficientEval<>::Type
   // q                          - CoefficientEval<>::Type::result_t
   // A    [M x Dim*(Dim+1)/2]   - partially assembled Dim x Dim symm. matrices
   // A    [M x Dim x Dim]       - partially assembled Dim x Dim matrices
   //
   // A = (w/det(J)) adj(J) adj(J)^t
   template <typename T_result_t, typename Q_t, typename q_t, typename asm_type>
   static inline MFEM_ALWAYS_INLINE
   void Assemble(const int k, const T_result_t &F,
                 const Q_t &Q, const q_t &q, asm_type &A)
   {
      typedef typename T_result_t::Jt_type::data_type real_t;
      const int M = T_result_t::Jt_type::layout_type::dim_1;
      MFEM_STATIC_ASSERT(asm_type::layout_type::dim_1 == M,
                         "incompatible dimensions");
      MFEM_FLOPS_ADD(37*M);
      const bool Symm = (asm_type::layout_type::rank == 2);
      for (int i = 0; i < M; i++)
      {
         TMatrix<3,3,real_t> B; // = adj(J)
         const complex_t u =
            (Q.get(q,i,k) /
             TAdjDet<real_t>(F.Jt.layout.ind14(i,k).transpose_12(), F.Jt,
                             B.layout, B));
         internal::MatrixOps<3,3>::Symm<Symm>::Set(
            A.layout.ind1(i), A,
            u*(B(0,0)*B(0,0)+B(0,1)*B(0,1)+B(0,2)*B(0,2)), // 1,1
            u*(B(0,0)*B(1,0)+B(0,1)*B(1,1)+B(0,2)*B(1,2)), // 2,1
            u*(B(0,0)*B(2,0)+B(0,1)*B(2,1)+B(0,2)*B(2,2)), // 3,1
            u*(B(1,0)*B(1,0)+B(1,1)*B(1,1)+B(1,2)*B(1,2)), // 2,2
            u*(B(1,0)*B(2,0)+B(1,1)*B(2,1)+B(1,2)*B(2,2)), // 3,2
            u*(B(2,0)*B(2,0)+B(2,1)*B(2,1)+B(2,2)*B(2,2))  // 3,3
         );
      }
   }

   // Method for partially assembled action.
   // A         [M x Dim*(Dim+1)/2]  - partially assembled Dim x Dim symmetric
   //                                  matrices
   // grad_qpts [M x SDim x NC x NE] - in/out data member in R
   //
   // grad_qpts = A grad_qpts
   template <int qpts, typename S_data_t>
   static inline MFEM_ALWAYS_INLINE
   void MultAssembled(const int k, const TMatrix<qpts,6,complex_t> &A,
                      S_data_t &R)
   {
      const int M = S_data_t::eval_type::qpts;
      MFEM_STATIC_ASSERT(qpts == M, "incompatible dimensions");
      MFEM_FLOPS_ADD(15*M);
      for (int i = 0; i < M; i++)
      {
         const complex_t A11 = A(i,0);
         const complex_t A21 = A(i,1);
         const complex_t A31 = A(i,2);
         const complex_t A22 = A(i,3);
         const complex_t A32 = A(i,4);
         const complex_t A33 = A(i,5);
         const complex_t x1 = R.grad_qpts(i,0,0,k);
         const complex_t x2 = R.grad_qpts(i,1,0,k);
         const complex_t x3 = R.grad_qpts(i,2,0,k);
         R.grad_qpts(i,0,0,k) = A11*x1 + A21*x2 + A31*x3;
         R.grad_qpts(i,1,0,k) = A21*x1 + A22*x2 + A32*x3;
         R.grad_qpts(i,2,0,k) = A31*x1 + A32*x2 + A33*x3;
      }
   }
};

} // namespace mfem

#endif // MFEM_TEMPLATE_DIFFUSION_KERNEL
