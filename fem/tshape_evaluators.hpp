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

#ifndef MFEM_TEMPLATE_SHAPE_EVALUATORS
#define MFEM_TEMPLATE_SHAPE_EVALUATORS

#include "../config/tconfig.hpp"
#include "../linalg/ttensor_types.hpp"
#include "../linalg/ttensor_products.hpp"

namespace mfem
{

// Shape evaluators

template <class FE, class IR, bool TP, typename real_t>
class ShapeEvaluator_base;

template <class FE, class IR, typename real_t>
class ShapeEvaluator_base<FE, IR, false, real_t>
{
public:
   static const int DOF = FE::dofs;
   static const int NIP = IR::qpts;
   static const int DIM = FE::dim;

protected:
   TMatrix<NIP,DOF,real_t,true> B;
   TMatrix<DOF,NIP,real_t,true> Bt;
   TTensor3<NIP,DIM,DOF,real_t,true> G;
   TTensor3<DOF,NIP,DIM,real_t> Gt;

public:
   ShapeEvaluator_base(const FE &fe)
   {
      fe.CalcShapes(IR::GetIntRule(), B.data, G.data);
      TAssign<AssignOp::Set>(Bt.layout, Bt, B.layout.transpose_12(), B);
      TAssign<AssignOp::Set>(Gt.layout.merge_23(), Gt,
                             G.layout.merge_12().transpose_12(), G);
   }

   // default copy constructor

   // Multi-component shape evaluation from DOFs to quadrature points.
   // dof_layout is (DOF x NumComp) and qpt_layout is (NIP x NumComp).
   template <typename dof_layout_t, typename dof_data_t,
             typename qpt_layout_t, typename qpt_data_t>
   MFEM_ALWAYS_INLINE
   void Calc(const dof_layout_t &dof_layout, const dof_data_t &dof_data,
             const qpt_layout_t &qpt_layout, qpt_data_t &qpt_data) const
   {
      MFEM_STATIC_ASSERT(dof_layout_t::rank  == 2 &&
                         dof_layout_t::dim_1 == DOF,
                         "invalid dof_layout_t.");
      MFEM_STATIC_ASSERT(qpt_layout_t::rank  == 2 &&
                         qpt_layout_t::dim_1 == NIP,
                         "invalid qpt_layout_t.");
      MFEM_STATIC_ASSERT(dof_layout_t::dim_2 == qpt_layout_t::dim_2,
                         "incompatible dof- and qpt- layouts.");

      Mult_AB<false>(B.layout, B,
                     dof_layout, dof_data,
                     qpt_layout, qpt_data);
   }

   // Multi-component shape evaluation transpose from quadrature points to DOFs.
   // qpt_layout is (NIP x NumComp) and dof_layout is (DOF x NumComp).
   template <bool Add,
             typename qpt_layout_t, typename qpt_data_t,
             typename dof_layout_t, typename dof_data_t>
   MFEM_ALWAYS_INLINE
   void CalcT(const qpt_layout_t &qpt_layout, const qpt_data_t &qpt_data,
              const dof_layout_t &dof_layout, dof_data_t &dof_data) const
   {
      MFEM_STATIC_ASSERT(dof_layout_t::rank  == 2 &&
                         dof_layout_t::dim_1 == DOF,
                         "invalid dof_layout_t.");
      MFEM_STATIC_ASSERT(qpt_layout_t::rank  == 2 &&
                         qpt_layout_t::dim_1 == NIP,
                         "invalid qpt_layout_t.");
      MFEM_STATIC_ASSERT(dof_layout_t::dim_2 == qpt_layout_t::dim_2,
                         "incompatible dof- and qpt- layouts.");

      Mult_AB<Add>(Bt.layout, Bt,
                   qpt_layout, qpt_data,
                   dof_layout, dof_data);
   }

   // Multi-component gradient evaluation from DOFs to quadrature points.
   // dof_layout is (DOF x NumComp) and grad_layout is (NIP x DIM x NumComp).
   template <typename dof_layout_t, typename dof_data_t,
             typename grad_layout_t, typename grad_data_t>
   MFEM_ALWAYS_INLINE
   void CalcGrad(const dof_layout_t  &dof_layout,
                 const dof_data_t    &dof_data,
                 const grad_layout_t &grad_layout,
                 grad_data_t         &grad_data) const
   {
      MFEM_STATIC_ASSERT(dof_layout_t::rank  == 2 &&
                         dof_layout_t::dim_1 == DOF,
                         "invalid dof_layout_t.");
      MFEM_STATIC_ASSERT(grad_layout_t::rank  == 3 &&
                         grad_layout_t::dim_1 == NIP &&
                         grad_layout_t::dim_2 == DIM,
                         "invalid grad_layout_t.");
      MFEM_STATIC_ASSERT(dof_layout_t::dim_2 == grad_layout_t::dim_3,
                         "incompatible dof- and grad- layouts.");

      Mult_AB<false>(G.layout.merge_12(), G,
                     dof_layout, dof_data,
                     grad_layout.merge_12(), grad_data);
   }

   // Multi-component gradient evaluation transpose from quadrature points to
   // DOFs. grad_layout is (NIP x DIM x NumComp), dof_layout is (DOF x NumComp).
   template <bool Add,
             typename grad_layout_t, typename grad_data_t,
             typename dof_layout_t, typename dof_data_t>
   MFEM_ALWAYS_INLINE
   void CalcGradT(const grad_layout_t &grad_layout,
                  const grad_data_t   &grad_data,
                  const dof_layout_t  &dof_layout,
                  dof_data_t          &dof_data) const
   {
      MFEM_STATIC_ASSERT(dof_layout_t::rank  == 2 &&
                         dof_layout_t::dim_1 == DOF,
                         "invalid dof_layout_t.");
      MFEM_STATIC_ASSERT(grad_layout_t::rank  == 3 &&
                         grad_layout_t::dim_1 == NIP &&
                         grad_layout_t::dim_2 == DIM,
                         "invalid grad_layout_t.");
      MFEM_STATIC_ASSERT(dof_layout_t::dim_2 == grad_layout_t::dim_3,
                         "incompatible dof- and grad- layouts.");

      Mult_AB<Add>(Gt.layout.merge_23(), Gt,
                   grad_layout.merge_12(), grad_data,
                   dof_layout, dof_data);
   }

   // Multi-component assemble.
   // qpt_layout is (NIP x NumComp), M_layout is (DOF x DOF x NumComp)
   template <typename qpt_layout_t, typename qpt_data_t,
             typename M_layout_t, typename M_data_t>
   MFEM_ALWAYS_INLINE
   void Assemble(const qpt_layout_t &qpt_layout, const qpt_data_t &qpt_data,
                 const M_layout_t &M_layout, M_data_t &M_data) const
   {
      // M_{i,j,k} = \sum_{s} B_{s,i} B_{s,j} qpt_data_{s,k}
      // Using TensorAssemble: <1,NIP,NC> --> <DOF,1,DOF,NC>
#if 0
      TensorAssemble<false>(
         B.layout, B,
         qpt_layout.template split_1<1,NIP>(), qpt_data,
         M_layout.template split_1<DOF,1>(), M_data);
#else
      TensorAssemble<false>(
         Bt.layout, Bt, B.layout, B,
         qpt_layout.template split_1<1,NIP>(), qpt_data,
         M_layout.template split_1<DOF,1>(), M_data);
#endif
   }

   // Multi-component assemble of grad-grad element matrices.
   // qpt_layout is (NIP x DIM x DIM x NumComp), and
   // D_layout is (DOF x DOF x NumComp).
   template <typename qpt_layout_t, typename qpt_data_t,
             typename D_layout_t, typename D_data_t>
   MFEM_ALWAYS_INLINE
   void AssembleGradGrad(const qpt_layout_t &qpt_layout,
                         const qpt_data_t   &qpt_data,
                         const D_layout_t   &D_layout,
                         D_data_t           &D_data) const
   {
      const int NC = qpt_layout_t::dim_4;
      TTensor4<NIP,DIM,DOF,NC> F;
      for (int k = 0; k < NC; k++)
      {
         // Next loop performs a batch of matrix-matrix products of size
         // (DIM x DIM) x (DIM x DOF) --> (DIM x DOF)
         for (int j = 0; j < NIP; j++)
         {
            Mult_AB<false>(qpt_layout.ind14(j,k), qpt_data,
                           G.layout.ind1(j), G,
                           F.layout.ind14(j,k), F);
         }
      }
      // (DOF x (NIP x DIM)) x ((NIP x DIM) x DOF x NC) --> (DOF x DOF x NC)
      Mult_2_1<false>(Gt.layout.merge_23(), Gt,
                      F.layout.merge_12(), F,
                      D_layout, D_data);
   }
};

template <int Dim, int DOF, int NIP, typename real_t>
class TProductShapeEvaluator;

template <int DOF, int NIP, typename real_t>
class TProductShapeEvaluator<1, DOF, NIP, real_t>
{
protected:
   static const int TDOF = DOF; // total dofs

   TMatrix<NIP,DOF,real_t,true> B_1d, G_1d;
   TMatrix<DOF,NIP,real_t,true> Bt_1d, Gt_1d;

public:
   TProductShapeEvaluator() { }

   // Multi-component shape evaluation from DOFs to quadrature points.
   // dof_layout is (DOF x NumComp) and qpt_layout is (NIP x NumComp).
   template <typename dof_layout_t, typename dof_data_t,
             typename qpt_layout_t, typename qpt_data_t>
   MFEM_ALWAYS_INLINE
   void Calc(const dof_layout_t &dof_layout, const dof_data_t &dof_data,
             const qpt_layout_t &qpt_layout, qpt_data_t &qpt_data) const
   {
      Mult_AB<false>(B_1d.layout, B_1d,
                     dof_layout, dof_data,
                     qpt_layout, qpt_data);
   }

   // Multi-component shape evaluation transpose from quadrature points to DOFs.
   // qpt_layout is (NIP x NumComp) and dof_layout is (DOF x NumComp).
   template <bool Add,
             typename qpt_layout_t, typename qpt_data_t,
             typename dof_layout_t, typename dof_data_t>
   MFEM_ALWAYS_INLINE
   void CalcT(const qpt_layout_t &qpt_layout, const qpt_data_t &qpt_data,
              const dof_layout_t &dof_layout, dof_data_t &dof_data) const
   {
      Mult_AB<Add>(Bt_1d.layout, Bt_1d,
                   qpt_layout, qpt_data,
                   dof_layout, dof_data);
   }

   // Multi-component gradient evaluation from DOFs to quadrature points.
   // dof_layout is (DOF x NumComp) and grad_layout is (NIP x DIM x NumComp).
   template <typename dof_layout_t, typename dof_data_t,
             typename grad_layout_t, typename grad_data_t>
   MFEM_ALWAYS_INLINE
   void CalcGrad(const dof_layout_t  &dof_layout,
                 const dof_data_t    &dof_data,
                 const grad_layout_t &grad_layout,
                 grad_data_t         &grad_data) const
   {
      // grad_data(nip,dim,comp) = sum(dof) G(nip,dim,dof) * dof_data(dof,comp)
      Mult_AB<false>(G_1d.layout, G_1d,
                     dof_layout, dof_data,
                     grad_layout.merge_12(), grad_data);
   }

   // Multi-component gradient evaluation transpose from quadrature points to
   // DOFs. grad_layout is (NIP x DIM x NumComp), dof_layout is (DOF x NumComp).
   template <bool Add,
             typename grad_layout_t, typename grad_data_t,
             typename dof_layout_t, typename dof_data_t>
   MFEM_ALWAYS_INLINE
   void CalcGradT(const grad_layout_t &grad_layout,
                  const grad_data_t   &grad_data,
                  const dof_layout_t  &dof_layout,
                  dof_data_t          &dof_data) const
   {
      // dof_data(dof,comp) +=
      //    sum(nip,dim) G(nip,dim,dof) * grad_data(nip,dim,comp)
      Mult_AB<Add>(Gt_1d.layout, Gt_1d,
                   grad_layout.merge_12(), grad_data,
                   dof_layout, dof_data);
   }

   // Multi-component assemble.
   // qpt_layout is (NIP x NumComp), M_layout is (DOF x DOF x NumComp)
   template <typename qpt_layout_t, typename qpt_data_t,
             typename M_layout_t, typename M_data_t>
   MFEM_ALWAYS_INLINE
   void Assemble(const qpt_layout_t &qpt_layout, const qpt_data_t &qpt_data,
                 const M_layout_t &M_layout, M_data_t &M_data) const
   {
      // M_{i,j,k} = \sum_{s} B_1d_{s,i} B_{s,j} qpt_data_{s,k}
      // Using TensorAssemble: <1,NIP,NC> --> <DOF,1,DOF,NC>
#if 0
      TensorAssemble<false>(
         B_1d.layout, B_1d,
         qpt_layout.template split_1<1,NIP>(), qpt_data,
         M_layout.template split_1<DOF,1>(), M_data);
#else
      TensorAssemble<false>(
         Bt_1d.layout, Bt_1d, B_1d.layout, B_1d,
         qpt_layout.template split_1<1,NIP>(), qpt_data,
         M_layout.template split_1<DOF,1>(), M_data);
#endif
   }

   // Multi-component assemble of grad-grad element matrices.
   // qpt_layout is (NIP x DIM x DIM x NumComp), and
   // D_layout is (DOF x DOF x NumComp).
   template <typename qpt_layout_t, typename qpt_data_t,
             typename D_layout_t, typename D_data_t>
   MFEM_ALWAYS_INLINE
   void AssembleGradGrad(const qpt_layout_t &qpt_layout,
                         const qpt_data_t   &qpt_data,
                         const D_layout_t   &D_layout,
                         D_data_t           &D_data) const
   {
      // D_{i,j,k} = \sum_{s} G_1d_{s,i} G_{s,j} qpt_data_{s,k}
      // Using TensorAssemble: <1,NIP,NC> --> <DOF,1,DOF,NC>
#if 0
      TensorAssemble<false>(
         G_1d.layout, G_1d,
         qpt_layout.merge_12().merge_23().template split_1<1,NIP>(), qpt_data,
         D_layout.template split_1<DOF,1>(), D_data);
#else
      TensorAssemble<false>(
         Gt_1d.layout, Gt_1d, G_1d.layout, G_1d,
         qpt_layout.merge_12().merge_23().template split_1<1,NIP>(), qpt_data,
         D_layout.template split_1<DOF,1>(), D_data);
#endif
   }
};

template <int DOF, int NIP, typename real_t>
class TProductShapeEvaluator<2, DOF, NIP, real_t>
{
protected:
   TMatrix<NIP,DOF,real_t,true> B_1d, G_1d;
   TMatrix<DOF,NIP,real_t,true> Bt_1d, Gt_1d;

public:
   static const int TDOF = DOF*DOF; // total dofs
   static const int TNIP = NIP*NIP; // total qpts

   TProductShapeEvaluator() { }

   template <bool Dx, bool Dy,
             typename dof_layout_t, typename dof_data_t,
             typename qpt_layout_t, typename qpt_data_t>
   MFEM_ALWAYS_INLINE
   void Calc(const dof_layout_t &dof_layout, const dof_data_t &dof_data,
             const qpt_layout_t &qpt_layout, qpt_data_t &qpt_data) const
   {
      const int NC = dof_layout_t::dim_2;
      // DOF x DOF x NC --> NIP x DOF x NC --> NIP x NIP x NC
      TTensor3<NIP,DOF,NC> A;

      // (1) A_{i,j,k} = \sum_s B_1d_{i,s} dof_data_{s,j,k}
      Mult_2_1<false>(B_1d.layout, Dx ? G_1d : B_1d,
                      dof_layout. template split_1<DOF,DOF>(), dof_data,
                      A.layout, A);
      // (2) qpt_data_{i,j,k} = \sum_s B_1d_{j,s} A_{i,s,k}
      Mult_1_2<false>(Bt_1d.layout, Dy ? Gt_1d : Bt_1d,
                      A.layout, A,
                      qpt_layout.template split_1<NIP,NIP>(), qpt_data);
   }

   // Multi-component shape evaluation from DOFs to quadrature points.
   // dof_layout is (TDOF x NumComp) and qpt_layout is (TNIP x NumComp).
   template <typename dof_layout_t, typename dof_data_t,
             typename qpt_layout_t, typename qpt_data_t>
   MFEM_ALWAYS_INLINE
   void Calc(const dof_layout_t &dof_layout, const dof_data_t &dof_data,
             const qpt_layout_t &qpt_layout, qpt_data_t &qpt_data) const
   {
      Calc<false,false>(dof_layout, dof_data, qpt_layout, qpt_data);
   }

   template <bool Dx, bool Dy, bool Add,
             typename qpt_layout_t, typename qpt_data_t,
             typename dof_layout_t, typename dof_data_t>
   MFEM_ALWAYS_INLINE
   void CalcT(const qpt_layout_t &qpt_layout, const qpt_data_t &qpt_data,
              const dof_layout_t &dof_layout, dof_data_t &dof_data) const
   {
      const int NC = dof_layout_t::dim_2;
      // NIP x NIP X NC --> NIP x DOF x NC --> DOF x DOF x NC
      TTensor3<NIP,DOF,NC> A;

      // (1) A_{i,j,k} = \sum_s B_1d_{s,j} qpt_data_{i,s,k}
      Mult_1_2<false>(B_1d.layout, Dy ? G_1d : B_1d,
                      qpt_layout.template split_1<NIP,NIP>(), qpt_data,
                      A.layout, A);
      // (2) dof_data_{i,j,k} = \sum_s B_1d_{s,i} A_{s,j,k}
      Mult_2_1<Add>(Bt_1d.layout, Dx ? Gt_1d : Bt_1d,
                    A.layout, A,
                    dof_layout.template split_1<DOF,DOF>(), dof_data);
   }

   // Multi-component shape evaluation transpose from quadrature points to DOFs.
   // qpt_layout is (TNIP x NumComp) and dof_layout is (TDOF x NumComp).
   template <bool Add,
             typename qpt_layout_t, typename qpt_data_t,
             typename dof_layout_t, typename dof_data_t>
   MFEM_ALWAYS_INLINE
   void CalcT(const qpt_layout_t &qpt_layout, const qpt_data_t &qpt_data,
              const dof_layout_t &dof_layout, dof_data_t &dof_data) const
   {
      CalcT<false,false,Add>(qpt_layout, qpt_data, dof_layout, dof_data);
   }

   // Multi-component gradient evaluation from DOFs to quadrature points.
   // dof_layout is (TDOF x NumComp) and grad_layout is (TNIP x DIM x NumComp).
   template <typename dof_layout_t, typename dof_data_t,
             typename grad_layout_t, typename grad_data_t>
   MFEM_ALWAYS_INLINE
   void CalcGrad(const dof_layout_t  &dof_layout,
                 const dof_data_t    &dof_data,
                 const grad_layout_t &grad_layout,
                 grad_data_t &grad_data) const
   {
      Calc<true,false>(dof_layout, dof_data,
                       grad_layout.ind2(0), grad_data);
      Calc<false,true>(dof_layout, dof_data,
                       grad_layout.ind2(1), grad_data);
   }

   // Multi-component gradient evaluation transpose from quadrature points to
   // DOFs. grad_layout is (TNIP x DIM x NumComp), dof_layout is
   // (TDOF x NumComp).
   template <bool Add,
             typename grad_layout_t, typename grad_data_t,
             typename dof_layout_t, typename dof_data_t>
   MFEM_ALWAYS_INLINE
   void CalcGradT(const grad_layout_t &grad_layout,
                  const grad_data_t   &grad_data,
                  const dof_layout_t  &dof_layout,
                  dof_data_t          &dof_data) const
   {
      CalcT<true,false, Add>(grad_layout.ind2(0), grad_data,
                             dof_layout, dof_data);
      CalcT<false,true,true>(grad_layout.ind2(1), grad_data,
                             dof_layout, dof_data);
   }

   // Multi-component assemble.
   // qpt_layout is (TNIP x NumComp), M_layout is (TDOF x TDOF x NumComp)
   template <typename qpt_layout_t, typename qpt_data_t,
             typename M_layout_t, typename M_data_t>
   MFEM_ALWAYS_INLINE
   void Assemble(const qpt_layout_t &qpt_layout, const qpt_data_t &qpt_data,
                 const M_layout_t &M_layout, M_data_t &M_data) const
   {
      const int NC = qpt_layout_t::dim_2;

      // Using TensorAssemble: <I,NIP,J> --> <DOF,I,DOF,J>

#if 0
      TTensor4<DOF,NIP,DOF,NC> A;
      // qpt_data<NIP1,NIP2,NC> --> A<DOF2,NIP1,DOF2,NC>
      TensorAssemble<false>(
         B_1d.layout, B_1d,
         qpt_layout.template split_1<NIP,NIP>(), qpt_data,
         A.layout, A);
      // A<DOF2,NIP1,DOF2*NC> --> M<DOF1,DOF2,DOF1,DOF2*NC>
      TensorAssemble<false>(
         B_1d.layout, B_1d,
         TTensor3<DOF,NIP,DOF*NC>::layout, A,
         M_layout.merge_23().template split_12<DOF,DOF,DOF,DOF*NC>(), M_data);
#elif 1
      TTensor4<DOF,NIP,DOF,NC> A;
      // qpt_data<NIP1,NIP2,NC> --> A<DOF2,NIP1,DOF2,NC>
      TensorAssemble<false>(
         Bt_1d.layout, Bt_1d, B_1d.layout, B_1d,
         qpt_layout.template split_1<NIP,NIP>(), qpt_data,
         A.layout, A);
      // A<DOF2,NIP1,DOF2*NC> --> M<DOF1,DOF2,DOF1,DOF2*NC>
      TensorAssemble<false>(
         Bt_1d.layout, Bt_1d, B_1d.layout, B_1d,
         A.layout.merge_34(), A,
         M_layout.merge_23().template split_12<DOF,DOF,DOF,DOF*NC>(), M_data);
#else
      TTensor3<NIP,NIP,DOF> F3;
      TTensor4<NIP,NIP,DOF,DOF> F4;
      TTensor3<NIP,DOF,DOF*DOF> H3;
      for (int k = 0; k < NC; k++)
      {
         // <1,NIP1,NIP2> --> <1,NIP1,NIP2,DOF1>
         TensorProduct<AssignOp::Set>(
            qpt_layout.ind2(k).template split_1<NIP,NIP>().
            template split_1<1,NIP>(), qpt_data,
            B_1d.layout, B_1d, F3.layout.template split_1<1,NIP>(), F3);
         // <NIP1,NIP2,DOF1> --> <NIP1,NIP2,DOF1,DOF2>
         TensorProduct<AssignOp::Set>(
            F3.layout, F3, B_1d.layout, B_1d, F4.layout, F4);
         // <NIP1,NIP2,DOF1,DOF2> --> <NIP1,DOF2,DOF1,DOF2>
         Mult_1_2<false>(B_1d.layout, B_1d,
                         F4.layout.merge_34(), F4,
                         H3.layout, H3);
         // <NIP1,DOF2,DOF1,DOF2> --> <DOF1,DOF2,DOF1,DOF2>
         Mult_2_1<false>(Bt_1d.layout, Bt_1d,
                         H3.layout, H3,
                         M_layout.ind3(k).template split_1<DOF,DOF>(),
                         M_data);
      }
#endif
   }

   template <int D1, int D2, bool Add,
             typename qpt_layout_t, typename qpt_data_t,
             typename D_layout_t, typename D_data_t>
   MFEM_ALWAYS_INLINE
   void Assemble(const qpt_layout_t &qpt_layout,
                 const qpt_data_t   &qpt_data,
                 const D_layout_t   &D_layout,
                 D_data_t           &D_data) const
   {
      const int NC = qpt_layout_t::dim_2;
      TTensor4<DOF,NIP,DOF,NC> A;

      // Using TensorAssemble: <I,NIP,J> --> <DOF,I,DOF,J>

      // qpt_data<NIP1,NIP2,NC> --> A<DOF2,NIP1,DOF2,NC>
      TensorAssemble<false>(
         Bt_1d.layout, D1 == 0 ? Bt_1d : Gt_1d,
         B_1d.layout, D2 == 0 ? B_1d : G_1d,
         qpt_layout.template split_1<NIP,NIP>(), qpt_data,
         A.layout, A);
      // A<DOF2,NIP1,DOF2*NC> --> M<DOF1,DOF2,DOF1,DOF2*NC>
      TensorAssemble<Add>(
         Bt_1d.layout, D1 == 1 ? Bt_1d : Gt_1d,
         B_1d.layout, D2 == 1 ? B_1d : G_1d,
         TTensor3<DOF,NIP,DOF*NC>::layout, A,
         D_layout.merge_23().template split_12<DOF,DOF,DOF,DOF*NC>(), D_data);
   }

   // Multi-component assemble of grad-grad element matrices.
   // qpt_layout is (TNIP x DIM x DIM x NumComp), and
   // D_layout is (TDOF x TDOF x NumComp).
   template <typename qpt_layout_t, typename qpt_data_t,
             typename D_layout_t, typename D_data_t>
   MFEM_ALWAYS_INLINE
   void AssembleGradGrad(const qpt_layout_t &qpt_layout,
                         const qpt_data_t   &qpt_data,
                         const D_layout_t   &D_layout,
                         D_data_t           &D_data) const
   {
#if 1
      Assemble<0,0,false>(qpt_layout.ind23(0,0), qpt_data, D_layout, D_data);
      Assemble<1,0,true >(qpt_layout.ind23(1,0), qpt_data, D_layout, D_data);
      Assemble<0,1,true >(qpt_layout.ind23(0,1), qpt_data, D_layout, D_data);
      Assemble<1,1,true >(qpt_layout.ind23(1,1), qpt_data, D_layout, D_data);
#else
      const int NC = qpt_layout_t::dim_4;
      TTensor3<NIP,NIP,DOF> F3;
      TTensor4<NIP,NIP,DOF,DOF> F4;
      TTensor3<NIP,DOF,DOF*DOF> H3;

      for (int k = 0; k < NC; k++)
      {
         for (int d1 = 0; d1 < 2; d1++)
         {
            const AssignOp::Type Set = AssignOp::Set;
            const AssignOp::Type Add = AssignOp::Add;
            // <1,NIP1,NIP2> --> <1,NIP1,NIP2,DOF1>
            TensorProduct<Set>(qpt_layout.ind23(d1,0).ind2(k).
                               template split_1<NIP,NIP>().
                               template split_1<1,NIP>(), qpt_data,
                               G_1d.layout, G_1d,
                               F3.layout.template split_1<1,NIP>(), F3);
            // <NIP1,NIP2,DOF1> --> <NIP1,NIP2,DOF1,DOF2>
            TensorProduct<Set>(F3.layout, F3,
                               B_1d.layout, B_1d,
                               F4.layout, F4);
            // <1,NIP1,NIP2> --> <1,NIP1,NIP2,DOF1>
            TensorProduct<Set>(qpt_layout.ind23(d1,1).ind2(k).
                               template split_1<NIP,NIP>().
                               template split_1<1,NIP>(), qpt_data,
                               B_1d.layout, B_1d,
                               F3.layout.template split_1<1,NIP>(), F3);
            // <NIP1,NIP2,DOF1> --> <NIP1,NIP2,DOF1,DOF2>
            TensorProduct<Add>(F3.layout, F3,
                               G_1d.layout, G_1d,
                               F4.layout, F4);

            Mult_1_2<false>(B_1d.layout, d1 == 0 ? B_1d : G_1d,
                            F4.layout.merge_34(), F4,
                            H3.layout, H3);
            if (d1 == 0)
            {
               Mult_2_1<false>(Bt_1d.layout, Gt_1d,
                               H3.layout, H3,
                               D_layout.ind3(k).template split_1<DOF,DOF>(),
                               D_data);
            }
            else
            {
               Mult_2_1<true>(Bt_1d.layout, Bt_1d,
                              H3.layout, H3,
                              D_layout.ind3(k).template split_1<DOF,DOF>(),
                              D_data);
            }
         }
      }
#endif
   }
};

template <int DOF, int NIP, typename real_t>
class TProductShapeEvaluator<3, DOF, NIP, real_t>
{
protected:
   TMatrix<NIP,DOF,real_t,true> B_1d, G_1d;
   TMatrix<DOF,NIP,real_t,true> Bt_1d, Gt_1d;

public:
   static const int TDOF = DOF*DOF*DOF; // total dofs
   static const int TNIP = NIP*NIP*NIP; // total qpts

   TProductShapeEvaluator() { }

   template <bool Dx, bool Dy, bool Dz,
             typename dof_layout_t, typename dof_data_t,
             typename qpt_layout_t, typename qpt_data_t>
   MFEM_ALWAYS_INLINE
   void Calc(const dof_layout_t &dof_layout, const dof_data_t &dof_data,
             const qpt_layout_t &qpt_layout, qpt_data_t &qpt_data) const
   {
      const int NC = dof_layout_t::dim_2;
      TVector<NIP*DOF*DOF*NC> QDD;
      TVector<NIP*NIP*DOF*NC> QQD;

      // QDD_{i,jj,k} = \sum_s B_1d_{i,s} dof_data_{s,jj,k}
      Mult_2_1<false>(B_1d.layout, Dx ? G_1d : B_1d,
                      dof_layout.template split_1<DOF,DOF*DOF>(), dof_data,
                      TTensor3<NIP,DOF*DOF,NC>::layout, QDD);
      // QQD_{i,j,kk} = \sum_s B_1d_{j,s} QDD_{i,s,kk}
      Mult_1_2<false>(Bt_1d.layout, Dy ? Gt_1d : Bt_1d,
                      TTensor3<NIP,DOF,DOF*NC>::layout, QDD,
                      TTensor3<NIP,NIP,DOF*NC>::layout, QQD);
      // qpt_data_{ii,j,k} = \sum_s B_1d_{j,s} QQD_{ii,s,k}
      Mult_1_2<false>(Bt_1d.layout, Dz ? Gt_1d : Bt_1d,
                      TTensor3<NIP*NIP,DOF,NC>::layout, QQD,
                      qpt_layout.template split_1<NIP*NIP,NIP>(), qpt_data);
   }

   // Multi-component shape evaluation from DOFs to quadrature points.
   // dof_layout is (TDOF x NumComp) and qpt_layout is (TNIP x NumComp).
   template <typename dof_layout_t, typename dof_data_t,
             typename qpt_layout_t, typename qpt_data_t>
   MFEM_ALWAYS_INLINE
   void Calc(const dof_layout_t &dof_layout, const dof_data_t &dof_data,
             const qpt_layout_t &qpt_layout, qpt_data_t &qpt_data) const
   {
      Calc<false,false,false>(dof_layout, dof_data, qpt_layout, qpt_data);
   }

   template <bool Dx, bool Dy, bool Dz, bool Add,
             typename qpt_layout_t, typename qpt_data_t,
             typename dof_layout_t, typename dof_data_t>
   MFEM_ALWAYS_INLINE
   void CalcT(const qpt_layout_t &qpt_layout, const qpt_data_t &qpt_data,
              const dof_layout_t &dof_layout, dof_data_t &dof_data) const
   {
      const int NC = dof_layout_t::dim_2;
      TVector<NIP*DOF*DOF*NC> QDD;
      TVector<NIP*NIP*DOF*NC> QQD;

      // QQD_{ii,j,k} = \sum_s B_1d_{s,j} qpt_data_{ii,s,k}
      Mult_1_2<false>(B_1d.layout, Dz ? G_1d : B_1d,
                      qpt_layout.template split_1<NIP*NIP,NIP>(), qpt_data,
                      TTensor3<NIP*NIP,DOF,NC>::layout, QQD);
      // QDD_{i,j,kk} = \sum_s B_1d_{s,j} QQD_{i,s,kk}
      Mult_1_2<false>(B_1d.layout, Dy ? G_1d : B_1d,
                      TTensor3<NIP,NIP,DOF*NC>::layout, QQD,
                      TTensor3<NIP,DOF,DOF*NC>::layout, QDD);
      // dof_data_{i,jj,k} = \sum_s B_1d_{s,i} QDD_{s,jj,k}
      Mult_2_1<Add>(Bt_1d.layout, Dx ? Gt_1d : Bt_1d,
                    TTensor3<NIP,DOF*DOF,NC>::layout, QDD,
                    dof_layout.template split_1<DOF,DOF*DOF>(), dof_data);
   }

   // Multi-component shape evaluation transpose from quadrature points to DOFs.
   // qpt_layout is (TNIP x NumComp) and dof_layout is (TDOF x NumComp).
   template <bool Add,
             typename qpt_layout_t, typename qpt_data_t,
             typename dof_layout_t, typename dof_data_t>
   MFEM_ALWAYS_INLINE
   void CalcT(const qpt_layout_t &qpt_layout, const qpt_data_t &qpt_data,
              const dof_layout_t &dof_layout, dof_data_t &dof_data) const
   {
      CalcT<false,false,false,Add>(qpt_layout, qpt_data, dof_layout, dof_data);
   }

   // Multi-component gradient evaluation from DOFs to quadrature points.
   // dof_layout is (TDOF x NumComp) and grad_layout is (TNIP x DIM x NumComp).
   template <typename dof_layout_t, typename dof_data_t,
             typename grad_layout_t, typename grad_data_t>
   MFEM_ALWAYS_INLINE
   void CalcGrad(const dof_layout_t  &dof_layout,
                 const dof_data_t    &dof_data,
                 const grad_layout_t &grad_layout,
                 grad_data_t         &grad_data) const
   {
      Calc<true,false,false>(dof_layout, dof_data,
                             grad_layout.ind2(0), grad_data);
      Calc<false,true,false>(dof_layout, dof_data,
                             grad_layout.ind2(1), grad_data);
      Calc<false,false,true>(dof_layout, dof_data,
                             grad_layout.ind2(2), grad_data);
      // optimization: the x-transition (dof->nip) is done twice -- once for the
      // y-derivatives and second time for the z-derivatives.
   }

   // Multi-component gradient evaluation transpose from quadrature points to
   // DOFs. grad_layout is (TNIP x DIM x NumComp), dof_layout is
   // (TDOF x NumComp).
   template <bool Add,
             typename grad_layout_t, typename grad_data_t,
             typename dof_layout_t, typename dof_data_t>
   MFEM_ALWAYS_INLINE
   void CalcGradT(const grad_layout_t &grad_layout,
                  const grad_data_t   &grad_data,
                  const dof_layout_t  &dof_layout,
                  dof_data_t          &dof_data) const
   {
      CalcT<true,false,false, Add>(grad_layout.ind2(0), grad_data,
                                   dof_layout, dof_data);
      CalcT<false,true,false,true>(grad_layout.ind2(1), grad_data,
                                   dof_layout, dof_data);
      CalcT<false,false,true,true>(grad_layout.ind2(2), grad_data,
                                   dof_layout, dof_data);
   }

   // Multi-component assemble.
   // qpt_layout is (TNIP x NumComp), M_layout is (TDOF x TDOF x NumComp)
   template <typename qpt_layout_t, typename qpt_data_t,
             typename M_layout_t, typename M_data_t>
   MFEM_ALWAYS_INLINE
   void Assemble(const qpt_layout_t &qpt_layout, const qpt_data_t &qpt_data,
                 const M_layout_t &M_layout, M_data_t &M_data) const
   {
      const int NC = qpt_layout_t::dim_2;
      TTensor4<DOF,NIP*NIP,DOF,NC> A1;
      TTensor4<DOF,DOF*NIP,DOF,DOF*NC> A2;

      // Using TensorAssemble: <I,NIP,J> --> <DOF,I,DOF,J>

#if 0
      // qpt_data<NIP1*NIP2,NIP3,NC> --> A1<DOF3,NIP1*NIP2,DOF3,NC>
      TensorAssemble<false>(
         B_1d.layout, B_1d,
         qpt_layout.template split_1<NIP*NIP,NIP>(), qpt_data,
         A1.layout, A1);
      // A1<DOF3*NIP1,NIP2,DOF3*NC> --> A2<DOF2,DOF3*NIP1,DOF2,DOF3*NC>
      TensorAssemble<false>(
         B_1d.layout, B_1d,
         TTensor3<DOF*NIP,NIP,DOF*NC>::layout, A1,
         A2.layout, A2);
      // A2<DOF2*DOF3,NIP1,DOF2*DOF3*NC> --> M<DOF1,DOF2*DOF3,DOF1,DOF2*DOF3*NC>
      TensorAssemble<false>(
         B_1d.layout, B_1d,
         TTensor3<DOF*DOF,NIP,DOF*DOF*NC>::layout, A2,
         M_layout.merge_23().template split_12<DOF,DOF*DOF,DOF,DOF*DOF*NC>(),
         M_data);
#else
      // qpt_data<NIP1*NIP2,NIP3,NC> --> A1<DOF3,NIP1*NIP2,DOF3,NC>
      TensorAssemble<false>(
         Bt_1d.layout, Bt_1d, B_1d.layout, B_1d,
         qpt_layout.template split_1<NIP*NIP,NIP>(), qpt_data,
         A1.layout, A1);
      // A1<DOF3*NIP1,NIP2,DOF3*NC> --> A2<DOF2,DOF3*NIP1,DOF2,DOF3*NC>
      TensorAssemble<false>(
         Bt_1d.layout, Bt_1d, B_1d.layout, B_1d,
         TTensor3<DOF*NIP,NIP,DOF*NC>::layout, A1,
         A2.layout, A2);
      // A2<DOF2*DOF3,NIP1,DOF2*DOF3*NC> --> M<DOF1,DOF2*DOF3,DOF1,DOF2*DOF3*NC>
      TensorAssemble<false>(
         Bt_1d.layout, Bt_1d, B_1d.layout, B_1d,
         TTensor3<DOF*DOF,NIP,DOF*DOF*NC>::layout, A2,
         M_layout.merge_23().template split_12<DOF,DOF*DOF,DOF,DOF*DOF*NC>(),
         M_data);
#endif
   }

   template <int D1, int D2, bool Add,
             typename qpt_layout_t, typename qpt_data_t,
             typename D_layout_t, typename D_data_t>
   MFEM_ALWAYS_INLINE
   void Assemble(const qpt_layout_t &qpt_layout,
                 const qpt_data_t   &qpt_data,
                 const D_layout_t   &D_layout,
                 D_data_t           &D_data) const
   {
      const int NC = qpt_layout_t::dim_2;
      TTensor4<DOF,NIP*NIP,DOF,NC> A1;
      TTensor4<DOF,DOF*NIP,DOF,DOF*NC> A2;

      // Using TensorAssemble: <I,NIP,J> --> <DOF,I,DOF,J>

      // qpt_data<NIP1*NIP2,NIP3,NC> --> A1<DOF3,NIP1*NIP2,DOF3,NC>
      TensorAssemble<false>(
         Bt_1d.layout, D1 != 2 ? Bt_1d : Gt_1d,
         B_1d.layout, D2 != 2 ? B_1d : G_1d,
         qpt_layout.template split_1<NIP*NIP,NIP>(), qpt_data,
         A1.layout, A1);
      // A1<DOF3*NIP1,NIP2,DOF3*NC> --> A2<DOF2,DOF3*NIP1,DOF2,DOF3*NC>
      TensorAssemble<false>(
         Bt_1d.layout, D1 != 1 ? Bt_1d : Gt_1d,
         B_1d.layout, D2 != 1 ? B_1d : G_1d,
         TTensor3<DOF*NIP,NIP,DOF*NC>::layout, A1,
         A2.layout, A2);
      // A2<DOF2*DOF3,NIP1,DOF2*DOF3*NC> --> M<DOF1,DOF2*DOF3,DOF1,DOF2*DOF3*NC>
      TensorAssemble<Add>(
         Bt_1d.layout, D1 != 0 ? Bt_1d : Gt_1d,
         B_1d.layout, D2 != 0 ? B_1d : G_1d,
         TTensor3<DOF*DOF,NIP,DOF*DOF*NC>::layout, A2,
         D_layout.merge_23().template split_12<DOF,DOF*DOF,DOF,DOF*DOF*NC>(),
         D_data);
   }

#if 0
   template <typename qpt_layout_t, typename qpt_data_t,
             typename D_layout_t, typename D_data_t>
   MFEM_ALWAYS_INLINE
   void Assemble(int D1, int D2,
                 const qpt_layout_t &qpt_layout,
                 const qpt_data_t   &qpt_data,
                 const D_layout_t   &D_layout,
                 D_data_t           &D_data) const
   {
      const int NC = qpt_layout_t::dim_2;
      TTensor4<DOF,NIP*NIP,DOF,NC> A1;
      TTensor4<DOF,DOF*NIP,DOF,DOF*NC> A2;

      // Using TensorAssemble: <I,NIP,J> --> <DOF,I,DOF,J>

      // qpt_data<NIP1*NIP2,NIP3,NC> --> A1<DOF3,NIP1*NIP2,DOF3,NC>
      TensorAssemble<false>(
         Bt_1d.layout, D1 != 2 ? Bt_1d : Gt_1d,
         B_1d.layout, D2 != 2 ? B_1d : G_1d,
         qpt_layout.template split_1<NIP*NIP,NIP>(), qpt_data,
         A1.layout, A1);
      // A1<DOF3*NIP1,NIP2,DOF3*NC> --> A2<DOF2,DOF3*NIP1,DOF2,DOF3*NC>
      TensorAssemble<false>(
         Bt_1d.layout, D1 != 1 ? Bt_1d : Gt_1d,
         B_1d.layout, D2 != 1 ? B_1d : G_1d,
         TTensor3<DOF*NIP,NIP,DOF*NC>::layout, A1,
         A2.layout, A2);
      // A2<DOF2*DOF3,NIP1,DOF2*DOF3*NC> --> M<DOF1,DOF2*DOF3,DOF1,DOF2*DOF3*NC>
      TensorAssemble<true>(
         Bt_1d.layout, D1 != 0 ? Bt_1d : Gt_1d,
         B_1d.layout, D2 != 0 ? B_1d : G_1d,
         TTensor3<DOF*DOF,NIP,DOF*DOF*NC>::layout, A2,
         D_layout.merge_23().template split_12<DOF,DOF*DOF,DOF,DOF*DOF*NC>(),
         D_data);
   }
#endif

   // Multi-component assemble of grad-grad element matrices.
   // qpt_layout is (TNIP x DIM x DIM x NumComp), and
   // D_layout is (TDOF x TDOF x NumComp).
   template <typename qpt_layout_t, typename qpt_data_t,
             typename D_layout_t, typename D_data_t>
   MFEM_ALWAYS_INLINE
   void AssembleGradGrad(const qpt_layout_t &qpt_layout,
                         const qpt_data_t   &qpt_data,
                         const D_layout_t   &D_layout,
                         D_data_t           &D_data) const
   {
#if 1
      // NOTE: This function compiles into a large chunk of machine code
      Assemble<0,0,false>(qpt_layout.ind23(0,0), qpt_data, D_layout, D_data);
      Assemble<1,0,true >(qpt_layout.ind23(1,0), qpt_data, D_layout, D_data);
      Assemble<2,0,true >(qpt_layout.ind23(2,0), qpt_data, D_layout, D_data);
      Assemble<0,1,true >(qpt_layout.ind23(0,1), qpt_data, D_layout, D_data);
      Assemble<1,1,true >(qpt_layout.ind23(1,1), qpt_data, D_layout, D_data);
      Assemble<2,1,true >(qpt_layout.ind23(2,1), qpt_data, D_layout, D_data);
      Assemble<0,2,true >(qpt_layout.ind23(0,2), qpt_data, D_layout, D_data);
      Assemble<1,2,true >(qpt_layout.ind23(1,2), qpt_data, D_layout, D_data);
      Assemble<2,2,true >(qpt_layout.ind23(2,2), qpt_data, D_layout, D_data);
#else
      TAssign<AssignOp::Set>(D_layout, D_data, 0.0);
      for (int d2 = 0; d2 < 3; d2++)
      {
         for (int d1 = 0; d1 < 3; d1++)
         {
            Assemble(d1, d2, qpt_layout.ind23(d1,d2), qpt_data,
                     D_layout, D_data);
         }
      }
#endif
   }
};

template <class FE, class IR, typename real_t>
class ShapeEvaluator_base<FE, IR, true, real_t>
   : public TProductShapeEvaluator<FE::dim, FE::dofs_1d, IR::qpts_1d, real_t>
{
protected:
   typedef TProductShapeEvaluator<FE::dim,FE::dofs_1d,
           IR::qpts_1d,real_t> base_class;
   using base_class::B_1d;
   using base_class::Bt_1d;
   using base_class::G_1d;
   using base_class::Gt_1d;

public:
   ShapeEvaluator_base(const FE &fe)
   {
      fe.Calc1DShapes(IR::Get1DIntRule(), B_1d.data, G_1d.data);
      TAssign<AssignOp::Set>(Bt_1d.layout, Bt_1d,
                             B_1d.layout.transpose_12(), B_1d);
      TAssign<AssignOp::Set>(Gt_1d.layout, Gt_1d,
                             G_1d.layout.transpose_12(), G_1d);
   }

   // default copy constructor
};

template <class FE, class IR, typename real_t>
class ShapeEvaluator
   : public ShapeEvaluator_base<FE,IR,FE::tensor_prod && IR::tensor_prod,real_t>
{
public:
   typedef real_t real_type;
   static const int dim  = FE::dim;
   static const int qpts = IR::qpts;
   static const bool tensor_prod = FE::tensor_prod && IR::tensor_prod;
   typedef FE FE_type;
   typedef IR IR_type;
   typedef ShapeEvaluator_base<FE,IR,tensor_prod,real_t> base_class;

   using base_class::Calc;
   using base_class::CalcT;
   using base_class::CalcGrad;
   using base_class::CalcGradT;

   ShapeEvaluator(const FE &fe) : base_class(fe) { }

   // default copy constructor
};

} // namespace mfem

#endif // MFEM_TEMPLATE_SHAPE_EVALUATORS
