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

#ifndef MFEM_TEMPLATE_FINITE_ELEMENTS_L2
#define MFEM_TEMPLATE_FINITE_ELEMENTS_L2

#include "../config/tconfig.hpp"
#include "tfinite_elements_h1.hpp" // mfem::CalcShapes
#include "fe_coll.hpp"

namespace mfem
{

// L2 finite elements

template <Geometry::Type G, int P, typename L2_FE_type, typename L2Pos_FE_type,
          int DOFS, bool TP>
class L2_FiniteElement_base
{
public:
   static const Geometry::Type geom = G;
   static const int dim    = Geometry::Constants<G>::Dimension;
   static const int degree = P;
   static const int dofs   = DOFS;

   static const bool tensor_prod = TP;
   static const int  dofs_1d     = P+1;

   // Type for run-time parameter for the constructor
   typedef L2_FECollection::BasisType parameter_type;

protected:
   const FiniteElement *my_fe, *my_fe_1d;
   parameter_type type; // run-time specified basis type

   void Init(const parameter_type type_)
   {
      type = type_;
      switch (type)
      {
         case L2_FECollection::GaussLegendre:
         case L2_FECollection::GaussLobatto:
            my_fe = new L2_FE_type(P, type);
            my_fe_1d = (TP && dim != 1) ? new L2_SegmentElement(P, type) : NULL;
            break;
         case L2_FECollection::Positive:
            my_fe = new L2Pos_FE_type(P);
            my_fe_1d = (TP && dim != 1) ? new L2Pos_SegmentElement(P) : NULL;
            break;
         default:
            MFEM_ABORT("invalid basis type");
      }
   }

   L2_FiniteElement_base(const parameter_type type)
   { Init(type); }

   L2_FiniteElement_base(const FiniteElementCollection &fec)
   {
      const L2_FECollection *l2_fec =
         dynamic_cast<const L2_FECollection *>(&fec);
      MFEM_ASSERT(l2_fec, "invalid FiniteElementCollection");
      Init(l2_fec->GetBasisType());
   }

   ~L2_FiniteElement_base() { delete my_fe; delete my_fe_1d; }

public:
   template <typename real_t>
   void CalcShapes(const IntegrationRule &ir, real_t *B, real_t *Grad) const
   {
      mfem::CalcShapes(*my_fe, ir, B, Grad, NULL);
   }
   template <typename real_t>
   void Calc1DShapes(const IntegrationRule &ir, real_t *B, real_t *Grad) const
   {
      mfem::CalcShapes(dim == 1 ? *my_fe : *my_fe_1d, ir, B, Grad, NULL);
   }
   const Array<int> *GetDofMap() const { return NULL; }
};


template <Geometry::Type G, int P>
class L2_FiniteElement;


template <int P>
class L2_FiniteElement<Geometry::SEGMENT, P>
   : public L2_FiniteElement_base<
     Geometry::SEGMENT,P,L2_SegmentElement,L2Pos_SegmentElement,P+1,true>
{
protected:
   typedef L2_FiniteElement_base<Geometry::SEGMENT,P,L2_SegmentElement,
           L2Pos_SegmentElement,P+1,true> base_class;
public:
   typedef typename base_class::parameter_type parameter_type;
   L2_FiniteElement(const parameter_type type_ = L2_FECollection::GaussLegendre)
      : base_class(type_) { }
   L2_FiniteElement(const FiniteElementCollection &fec)
      : base_class(fec) { }
};


template <int P>
class L2_FiniteElement<Geometry::TRIANGLE, P>
   : public L2_FiniteElement_base<Geometry::TRIANGLE,P,L2_TriangleElement,
     L2Pos_TriangleElement,((P+1)*(P+2))/2,false>
{
protected:
   typedef L2_FiniteElement_base<Geometry::TRIANGLE,P,L2_TriangleElement,
           L2Pos_TriangleElement,((P+1)*(P+2))/2,false> base_class;
public:
   typedef typename base_class::parameter_type parameter_type;
   L2_FiniteElement(const parameter_type type_ = L2_FECollection::GaussLegendre)
      : base_class(type_) { }
   L2_FiniteElement(const FiniteElementCollection &fec)
      : base_class(fec) { }
};


template <int P>
class L2_FiniteElement<Geometry::SQUARE, P>
   : public L2_FiniteElement_base<Geometry::SQUARE,P,L2_QuadrilateralElement,
     L2Pos_QuadrilateralElement,(P+1)*(P+1),true>
{
protected:
   typedef L2_FiniteElement_base<Geometry::SQUARE,P,L2_QuadrilateralElement,
           L2Pos_QuadrilateralElement,(P+1)*(P+1),true> base_class;
public:
   typedef typename base_class::parameter_type parameter_type;
   L2_FiniteElement(const parameter_type type_ = L2_FECollection::GaussLegendre)
      : base_class(type_) { }
   L2_FiniteElement(const FiniteElementCollection &fec)
      : base_class(fec) { }
};


template <int P>
class L2_FiniteElement<Geometry::TETRAHEDRON, P>
   : public L2_FiniteElement_base<Geometry::TETRAHEDRON,P,L2_TetrahedronElement,
     L2Pos_TetrahedronElement,((P+1)*(P+2)*(P+3))/6,false>
{
protected:
   typedef L2_FiniteElement_base<Geometry::TETRAHEDRON,P,L2_TetrahedronElement,
           L2Pos_TetrahedronElement,((P+1)*(P+2)*(P+3))/6,false> base_class;
public:
   typedef typename base_class::parameter_type parameter_type;
   L2_FiniteElement(const parameter_type type_ = L2_FECollection::GaussLegendre)
      : base_class(type_) { }
   L2_FiniteElement(const FiniteElementCollection &fec)
      : base_class(fec) { }
};


template <int P>
class L2_FiniteElement<Geometry::CUBE, P>
   : public L2_FiniteElement_base<Geometry::CUBE,P,L2_HexahedronElement,
     L2Pos_HexahedronElement,(P+1)*(P+1)*(P+1),true>
{
protected:
   typedef L2_FiniteElement_base<Geometry::CUBE,P,L2_HexahedronElement,
           L2Pos_HexahedronElement,(P+1)*(P+1)*(P+1),true> base_class;
public:
   typedef typename base_class::parameter_type parameter_type;
   L2_FiniteElement(const parameter_type type_ = L2_FECollection::GaussLegendre)
      : base_class(type_) { }
   L2_FiniteElement(const FiniteElementCollection &fec)
      : base_class(fec) { }
};

} // namespace mfem

#endif // MFEM_TEMPLATE_FINITE_ELEMENTS_L2
