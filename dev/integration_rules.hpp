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

#ifndef MFEM_TEMPLATE_INTEGRATION_RULES
#define MFEM_TEMPLATE_INTEGRATION_RULES

#include "config.hpp"
#include "tensor_types.hpp"
#include "fem/geom.hpp"

namespace mfem
{

// Integration rules

template <Geometry::Type G, int Q, int Order>
class GenericIntegrationRule
{
public:
   static const Geometry::Type geom = G;
   static const int dim = Geometry::Constants<geom>::Dimension;
   static const int qpts = Q;
   static const int order = Order;

   static const bool tensor_prod = false;

protected:
   TVector<qpts> weights;

public:
   GenericIntegrationRule()
   {
      const IntegrationRule &ir = GetIntRule();
      MFEM_ASSERT(ir.GetNPoints() == qpts, "quadrature rule mismatch");
      for (int j = 0; j < qpts; j++)
      {
         weights[j] = ir.IntPoint(j).weight;
      }
   }

   GenericIntegrationRule(const GenericIntegrationRule &ir)
   {
      weights.Set(ir.weights);
   }

   static const IntegrationRule &GetIntRule()
   {
      return IntRules.Get(geom, order);
   }

   // Multi-component weight assignment. qpt_layout_t must be (qpts x n1 x ...).
   template <AssignOp::Type Op, typename qpt_layout_t, typename qpt_data_t>
   void AssignWeights(const qpt_layout_t &qpt_layout,
                      qpt_data_t &qpt_data) const
   {
      MFEM_STATIC_ASSERT(qpt_layout_t::rank > 1, "invalid rank");
      MFEM_STATIC_ASSERT(qpt_layout_t::dim_1 == qpts, "invalid size");
      for (int j = 0; j < qpts; j++)
      {
         TAssign<Op>(qpt_layout.ind1(j), qpt_data, weights.data[j]);
      }
   }

   template <typename qpt_data_t>
   void ApplyWeights(qpt_data_t &qpt_data) const
   {
      AssignWeights<AssignOp::Mult>(ColumnMajorLayout2D<qpts,1>(), qpt_data);
   }
};

template <int Dim, int Q>
class TProductIntegrationRule_base;

template <int Q>
class TProductIntegrationRule_base<1, Q>
{
protected:
   TVector<Q> weights_1d;

public:
   // Multi-component weight assignment. qpt_layout_t must be (qpts x n1 x ...).
   template <AssignOp::Type Op, typename qpt_layout_t, typename qpt_data_t>
   void AssignWeights(const qpt_layout_t &qpt_layout,
                      qpt_data_t &qpt_data) const
   {
      MFEM_STATIC_ASSERT(qpt_layout_t::rank > 1, "invalid rank");
      MFEM_STATIC_ASSERT(qpt_layout_t::dim_1 == Q, "invalid size");
      for (int j = 0; j < Q; j++)
      {
         TAssign<Op>(qpt_layout.ind1(j), qpt_data, weights_1d.data[j]);
      }
   }

   template <typename qpt_data_t>
   void ApplyWeights(qpt_data_t &qpt_data) const
   {
      AssignWeights<AssignOp::Mult>(ColumnMajorLayout2D<Q,1>(), qpt_data);
   }
};

template <int Q>
class TProductIntegrationRule_base<2, Q>
{
protected:
   TVector<Q> weights_1d;

public:
   // Multi-component weight assignment. qpt_layout_t must be (qpts x n1 x ...).
   template <AssignOp::Type Op, typename qpt_layout_t, typename qpt_data_t>
   void AssignWeights(const qpt_layout_t &qpt_layout,
                      qpt_data_t &qpt_data) const
   {
      MFEM_STATIC_ASSERT(qpt_layout_t::rank > 1, "invalid rank");
      MFEM_STATIC_ASSERT(qpt_layout_t::dim_1 == Q*Q, "invalid size");
      for (int j2 = 0; j2 < Q; j2++)
      {
         for (int j1 = 0; j1 < Q; j1++)
         {
            TAssign<Op>(
               qpt_layout.ind1(TMatrix<Q,Q>::layout.ind(j1,j2)), qpt_data,
               weights_1d.data[j1]*weights_1d.data[j2]);
         }
      }
   }

   template <typename qpt_data_t>
   void ApplyWeights(qpt_data_t &qpt_data) const
   {
      AssignWeights<AssignOp::Mult>(ColumnMajorLayout2D<Q*Q,1>(), qpt_data);
   }
};

template <int Q>
class TProductIntegrationRule_base<3, Q>
{
protected:
   TVector<Q> weights_1d;

public:
   // Multi-component weight assignment. qpt_layout_t must be (qpts x n1 x ...).
   template <AssignOp::Type Op, typename qpt_layout_t, typename qpt_data_t>
   void AssignWeights(const qpt_layout_t &qpt_layout,
                      qpt_data_t &qpt_data) const
   {
      MFEM_STATIC_ASSERT(qpt_layout_t::rank > 1, "invalid rank");
      MFEM_STATIC_ASSERT(qpt_layout_t::dim_1 == Q*Q*Q, "invalid size");
      for (int j3 = 0; j3 < Q; j3++)
      {
         for (int j2 = 0; j2 < Q; j2++)
         {
            for (int j1 = 0; j1 < Q; j1++)
            {
               TAssign<Op>(
                  qpt_layout.ind1(TTensor3<Q,Q,Q>::layout.ind(j1,j2,j3)),
                  qpt_data,
                  weights_1d.data[j1]*weights_1d.data[j2]*weights_1d.data[j3]);
            }
         }
      }
   }

   template <typename qpt_data_t>
   void ApplyWeights(qpt_data_t &qpt_data) const
   {
      AssignWeights<AssignOp::Mult>(ColumnMajorLayout2D<Q*Q*Q,1>(), qpt_data);
   }
};

template <int Dim, int Q, int Order>
class TProductIntegrationRule : public TProductIntegrationRule_base<Dim, Q>
{
public:
   static const Geometry::Type geom =
      ((Dim == 1) ? Geometry::SEGMENT :
       ((Dim == 2) ? Geometry::SQUARE : Geometry::CUBE));
   static const int dim = Dim;
   static const int qpts_1d = Q;
   static const int qpts = (Dim == 1) ? Q : ((Dim == 2) ? (Q*Q) : (Q*Q*Q));
   static const int order = Order;

   static const bool tensor_prod = true;

protected:
   using TProductIntegrationRule_base<Dim, Q>::weights_1d;

public:
   TProductIntegrationRule() { }

   TProductIntegrationRule(const TProductIntegrationRule &ir)
   {
      weights_1d.Set(ir.weights_1d);
   }
};

template <int Dim, int Q>
class GaussIntegrationRule
   : public TProductIntegrationRule<Dim, Q, 2*Q-1>
{
public:
   typedef TProductIntegrationRule<Dim, Q, 2*Q-1> base_class;

   using base_class::geom;
   using base_class::order;
   using base_class::qpts_1d;

protected:
   using base_class::weights_1d;

public:
   GaussIntegrationRule()
   {
      const IntegrationRule &ir_1d = Get1DIntRule();
      MFEM_ASSERT(ir_1d.GetNPoints() == qpts_1d, "quadrature rule mismatch");
      for (int j = 0; j < qpts_1d; j++)
      {
         weights_1d.data[j] = ir_1d.IntPoint(j).weight;
      }
   }

   static const IntegrationRule &Get1DIntRule()
   {
      return IntRules.Get(Geometry::SEGMENT, order);
   }
   static const IntegrationRule &GetIntRule()
   {
      return IntRules.Get(geom, order);
   }
};

template <Geometry::Type G, int Order>
class TIntegrationRule;

template <int Order>
class TIntegrationRule<Geometry::SEGMENT, Order>
   : public GaussIntegrationRule<1, Order/2+1> { };

template <int Order>
class TIntegrationRule<Geometry::SQUARE, Order>
   : public GaussIntegrationRule<2, Order/2+1> { };

template <int Order>
class TIntegrationRule<Geometry::CUBE, Order>
   : public GaussIntegrationRule<3, Order/2+1> { };

// Triangle integration rules (based on intrules.cpp)
// These specializations define the number of quadrature points for each rule
// as a compile-time constant.
// TODO: add higher order rules
template <> class TIntegrationRule<Geometry::TRIANGLE, 0>
   : public GenericIntegrationRule<Geometry::TRIANGLE, 1, 0> { };
template <> class TIntegrationRule<Geometry::TRIANGLE, 1>
   : public GenericIntegrationRule<Geometry::TRIANGLE, 1, 1> { };
template <> class TIntegrationRule<Geometry::TRIANGLE, 2>
   : public GenericIntegrationRule<Geometry::TRIANGLE, 3, 2> { };
template <> class TIntegrationRule<Geometry::TRIANGLE, 3>
   : public GenericIntegrationRule<Geometry::TRIANGLE, 4, 3> { };
template <> class TIntegrationRule<Geometry::TRIANGLE, 4>
   : public GenericIntegrationRule<Geometry::TRIANGLE, 6, 4> { };
template <> class TIntegrationRule<Geometry::TRIANGLE, 5>
   : public GenericIntegrationRule<Geometry::TRIANGLE, 7, 5> { };
template <> class TIntegrationRule<Geometry::TRIANGLE, 6>
   : public GenericIntegrationRule<Geometry::TRIANGLE, 12, 6> { };
template <> class TIntegrationRule<Geometry::TRIANGLE, 7>
   : public GenericIntegrationRule<Geometry::TRIANGLE, 12, 7> { };

// Tetrahedron integration rules (based on intrules.cpp)
// These specializations define the number of quadrature points for each rule
// as a compile-time constant.
// TODO: add higher order rules
template <> class TIntegrationRule<Geometry::TETRAHEDRON, 0>
   : public GenericIntegrationRule<Geometry::TETRAHEDRON, 1, 0> { };
template <> class TIntegrationRule<Geometry::TETRAHEDRON, 1>
   : public GenericIntegrationRule<Geometry::TETRAHEDRON, 1, 1> { };
template <> class TIntegrationRule<Geometry::TETRAHEDRON, 2>
   : public GenericIntegrationRule<Geometry::TETRAHEDRON, 4, 2> { };
template <> class TIntegrationRule<Geometry::TETRAHEDRON, 3>
   : public GenericIntegrationRule<Geometry::TETRAHEDRON, 5, 3> { };
template <> class TIntegrationRule<Geometry::TETRAHEDRON, 4>
   : public GenericIntegrationRule<Geometry::TETRAHEDRON, 11, 4> { };
template <> class TIntegrationRule<Geometry::TETRAHEDRON, 5>
   : public GenericIntegrationRule<Geometry::TETRAHEDRON, 14, 5> { };
template <> class TIntegrationRule<Geometry::TETRAHEDRON, 6>
   : public GenericIntegrationRule<Geometry::TETRAHEDRON, 24, 6> { };
template <> class TIntegrationRule<Geometry::TETRAHEDRON, 7>
   : public GenericIntegrationRule<Geometry::TETRAHEDRON, 31, 7> { };

} // namespace mfem

#endif // MFEM_TEMPLATE_INTEGRATION_RULES
