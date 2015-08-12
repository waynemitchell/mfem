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

#include "config.hpp"
#include "tensor_types.hpp"
#include "tensor_products.hpp"

namespace mfem
{

// Shape evaluators

template <class FE, class IR, bool TP>
class ShapeEvaluator_base;

template <class FE, class IR>
class ShapeEvaluator_base<FE, IR, false>
{
public:
   static const int DOF = FE::dofs;
   static const int NIP = IR::qpts;
   static const int DIM = FE::dim;

protected:
   TMatrix<NIP,DOF> B;
   TMatrix<DOF,NIP> Bt;
   TTensor3<NIP,DIM,DOF> G;
   TTensor3<DOF,NIP,DIM> Gt;

public:
   ShapeEvaluator_base(const FE &fe)
   {
      fe.CalcShapes(IR::GetIntRule(), B.data, G.data);
      TAssign<AssignOp::Set>(Bt.layout, Bt, B.layout.transpose_12(), B);
      TAssign<AssignOp::Set>(Gt.layout.merge_23(), Gt,
                             G.layout.merge_12().transpose_12(), G);
   }

   ShapeEvaluator_base(const ShapeEvaluator_base &se)
   {
      B.Set(se.B);
      Bt.Set(se.Bt);
      G.Set(se.G);
      Gt.Set(se.Gt);
   }

   // Multi-component shape evaluation from DOFs to quadrature points.
   // dof_layout is (DOF x NumComp) and qpt_layout is (NIP x NumComp).
   template <typename dof_layout_t, typename dof_data_t,
             typename qpt_layout_t, typename qpt_data_t>
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
   void Assemble(const qpt_layout_t &qpt_layout, const qpt_data_t &qpt_data,
                 const M_layout_t &M_layout, M_data_t &M_data) const
   {
      // M_{i,j,k} = \sum_{s} B_{s,i} B_{s,j} qpt_data_{s,k}
      // Using TensorAssemble: <1,NIP,NC> --> <DOF,1,DOF,NC>
      TensorAssemble<false>(
         B.layout, B,
         qpt_layout.template split_1<1,NIP>(), qpt_data,
         M_layout.template split_1<DOF,1>(), M_data);
   }
};

template <int Dim, int DOF, int NIP>
class TProductShapeEvaluator;

template <int DOF, int NIP>
class TProductShapeEvaluator<1, DOF, NIP>
{
protected:
   static const int TDOF = DOF; // total dofs

   TMatrix<NIP,DOF> B_1d, G_1d;
   TMatrix<DOF,NIP> Bt_1d, Gt_1d;

public:
   TProductShapeEvaluator() { }

   // Multi-component shape evaluation from DOFs to quadrature points.
   // dof_layout is (DOF x NumComp) and qpt_layout is (NIP x NumComp).
   template <typename dof_layout_t, typename dof_data_t,
             typename qpt_layout_t, typename qpt_data_t>
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
   void Assemble(const qpt_layout_t &qpt_layout, const qpt_data_t &qpt_data,
                 const M_layout_t &M_layout, M_data_t &M_data) const
   {
      // M_{i,j,k} = \sum_{s} B_1d_{s,i} B_{s,j} qpt_data_{s,k}
      // Using TensorAssemble: <1,NIP,NC> --> <DOF,1,DOF,NC>
      TensorAssemble<false>(
         B_1d.layout, B_1d,
         qpt_layout.template split_1<1,NIP>(), qpt_data,
         M_layout.template split_1<DOF,1>(), M_data);
   }
};

template <int DOF, int NIP>
class TProductShapeEvaluator<2, DOF, NIP>
{
protected:
   TMatrix<NIP,DOF> B_1d, G_1d;
   TMatrix<DOF,NIP> Bt_1d, Gt_1d;

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
   void Assemble(const qpt_layout_t &qpt_layout, const qpt_data_t &qpt_data,
                 const M_layout_t &M_layout, M_data_t &M_data) const
   {
      const int NC = qpt_layout_t::dim_2;
      TTensor4<DOF,NIP,DOF,NC> A;

      // Using TensorAssemble: <I,NIP,J> --> <DOF,I,DOF,J>

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
   }
};

template <int DOF, int NIP>
class TProductShapeEvaluator<3, DOF, NIP>
{
protected:
   TMatrix<NIP,DOF> B_1d, G_1d;
   TMatrix<DOF,NIP> Bt_1d, Gt_1d;

public:
   static const int TDOF = DOF*DOF*DOF; // total dofs
   static const int TNIP = NIP*NIP*NIP; // total qpts

   TProductShapeEvaluator() { }

   template <bool Dx, bool Dy, bool Dz,
             typename dof_layout_t, typename dof_data_t,
             typename qpt_layout_t, typename qpt_data_t>
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
   void Calc(const dof_layout_t &dof_layout, const dof_data_t &dof_data,
             const qpt_layout_t &qpt_layout, qpt_data_t &qpt_data) const
   {
      Calc<false,false,false>(dof_layout, dof_data, qpt_layout, qpt_data);
   }

   template <bool Dx, bool Dy, bool Dz, bool Add,
             typename qpt_layout_t, typename qpt_data_t,
             typename dof_layout_t, typename dof_data_t>
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
   void CalcT(const qpt_layout_t &qpt_layout, const qpt_data_t &qpt_data,
              const dof_layout_t &dof_layout, dof_data_t &dof_data) const
   {
      CalcT<false,false,false,Add>(qpt_layout, qpt_data, dof_layout, dof_data);
   }

   // Multi-component gradient evaluation from DOFs to quadrature points.
   // dof_layout is (TDOF x NumComp) and grad_layout is (TNIP x DIM x NumComp).
   template <typename dof_layout_t, typename dof_data_t,
             typename grad_layout_t, typename grad_data_t>
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
   void Assemble(const qpt_layout_t &qpt_layout, const qpt_data_t &qpt_data,
                 const M_layout_t &M_layout, M_data_t &M_data) const
   {
      const int NC = qpt_layout_t::dim_2;
      TTensor4<DOF,NIP*NIP,DOF,NC> A1;
      TTensor4<DOF,DOF*NIP,DOF,DOF*NC> A2;

      // Using TensorAssemble: <I,NIP,J> --> <DOF,I,DOF,J>

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
   }
};

template <class FE, class IR>
class ShapeEvaluator_base<FE, IR, true>
   : public TProductShapeEvaluator<FE::dim, FE::dofs_1d, IR::qpts_1d>
{
protected:
   typedef TProductShapeEvaluator<FE::dim,FE::dofs_1d,IR::qpts_1d> base_class;
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

   ShapeEvaluator_base(const ShapeEvaluator_base &se)
   {
      B_1d.Set(se.B_1d);
      Bt_1d.Set(se.Bt_1d);
      G_1d.Set(se.G_1d);
      Gt_1d.Set(se.Gt_1d);
   }
};

template <class FE, class IR>
class ShapeEvaluator
   : public ShapeEvaluator_base<FE, IR, FE::tensor_prod && IR::tensor_prod>
{
public:
   static const int dim  = FE::dim;
   static const int qpts = IR::qpts;
   static const bool tensor_prod = FE::tensor_prod && IR::tensor_prod;
   typedef FE FE_type;
   typedef IR IR_type;
   typedef ShapeEvaluator_base<FE, IR, tensor_prod> base_class;

   using base_class::Calc;
   using base_class::CalcT;
   using base_class::CalcGrad;
   using base_class::CalcGradT;

   ShapeEvaluator(const FE &fe) : base_class(fe) { }

   ShapeEvaluator(const ShapeEvaluator &se) : base_class(se) { }
};

} // namespace mfem

#endif // MFEM_TEMPLATE_SHAPE_EVALUATORS
