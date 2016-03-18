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

#ifndef MFEM_TEMPLATE_FIELD_EVALUATOR
#define MFEM_TEMPLATE_FIELD_EVALUATOR

#include "config/tconfig.hpp"
#include "linalg/ttensor_types.hpp"
#include "tshape_evaluators.hpp"
#include "general/error.hpp"
#include "fespace.hpp"

namespace mfem
{

template <typename FESpace_t, typename VecLayout_t, typename IR,
          typename complex_t, typename real_t>
class FieldEvaluator_base
{
protected:
   typedef typename FESpace_t::FE_type       FE_type;
   typedef ShapeEvaluator<FE_type,IR,real_t> ShapeEval_type;

   FESpace_t       fespace;
   ShapeEval_type  shapeEval;
   VecLayout_t     vec_layout;

   // With this constructor, fespace is a shallow copy.
   inline MFEM_ALWAYS_INLINE
   FieldEvaluator_base(const FESpace_t &tfes, const ShapeEval_type &shape_eval,
                       const VecLayout_t &vec_layout)
      : fespace(tfes),
        shapeEval(shape_eval),
        vec_layout(vec_layout)
   { }

   // This constructor creates new fespace, not a shallow copy.
   inline MFEM_ALWAYS_INLINE
   FieldEvaluator_base(const FE_type &fe, const FiniteElementSpace &fes)
      : fespace(fe, fes), shapeEval(fe), vec_layout(fes)
   { }
};

// complex_t - dof/qpt data type, real_t - ShapeEvaluator (FE basis) data type
template <typename FESpace_t, typename VecLayout_t, typename IR,
          typename complex_t = double, typename real_t = double>
class FieldEvaluator
   : public FieldEvaluator_base<FESpace_t,VecLayout_t,IR,complex_t,real_t>
{
public:
   typedef complex_t                         complex_type;
   typedef FESpace_t                         FESpace_type;
   typedef typename FESpace_t::FE_type       FE_type;
   typedef ShapeEvaluator<FE_type,IR,real_t> ShapeEval_type;
   typedef VecLayout_t                       VecLayout_type;

   // this type
   typedef FieldEvaluator<FESpace_t,VecLayout_t,IR,complex_t,real_t> T_type;

   static const int dofs = FE_type::dofs;
   static const int dim  = FE_type::dim;
   static const int qpts = IR::qpts;
   static const int vdim = VecLayout_t::vec_dim;

protected:

   typedef FieldEvaluator_base<FESpace_t,VecLayout_t,IR,complex_t,real_t>
   base_class;

   using base_class::fespace;
   using base_class::shapeEval;
   using base_class::vec_layout;
   const complex_t *data_in;
   complex_t       *data_out;

public:
   // With this constructor, fespace is a shallow copy of tfes.
   inline MFEM_ALWAYS_INLINE
   FieldEvaluator(const FESpace_t &tfes, const ShapeEval_type &shape_eval,
                  const VecLayout_type &vec_layout,
                  const complex_t *global_data_in, complex_t *global_data_out)
      : base_class(tfes, shape_eval, vec_layout),
        data_in(global_data_in),
        data_out(global_data_out)
   { }

   // With this constructor, fespace is a shallow copy of f.fespace.
   inline MFEM_ALWAYS_INLINE
   FieldEvaluator(const FieldEvaluator &f,
                  const complex_t *global_data_in, complex_t *global_data_out)
      : base_class(f.fespace, f.shapeEval, f.vec_layout),
        data_in(global_data_in),
        data_out(global_data_out)
   { }

   // This constructor creates a new fespace, not a shallow copy.
   inline MFEM_ALWAYS_INLINE
   FieldEvaluator(const FiniteElementSpace &fes,
                  const complex_t *global_data_in, complex_t *global_data_out)
      : base_class(FE_type(*fes.FEColl()), fes),
        data_in(global_data_in),
        data_out(global_data_out)
   { }

   // Default copy constructor

   inline MFEM_ALWAYS_INLINE FESpace_type &FESpace() { return fespace; }
   inline MFEM_ALWAYS_INLINE ShapeEval_type &ShapeEval() { return shapeEval; }
   inline MFEM_ALWAYS_INLINE VecLayout_type &VecLayout() { return vec_layout; }

   inline MFEM_ALWAYS_INLINE
   void SetElement(int el)
   {
      fespace.SetElement(el);
   }

   // val_layout_t is (qpts x vdim x NE)
   template <typename val_layout_t, typename val_data_t>
   inline MFEM_ALWAYS_INLINE
   void GetValues(int el, const val_layout_t &l, val_data_t &vals)
   {
      const int ne = val_layout_t::dim_3;
      TTensor3<dofs,vdim,ne,complex_type> val_dofs;
      SetElement(el);
      fespace.VectorExtract(vec_layout, data_in, val_dofs.layout, val_dofs);
      shapeEval.Calc(val_dofs.layout.merge_23(), val_dofs, l.merge_23(), vals);
   }

   // grad_layout_t is (qpts x dim x vdim x NE)
   template <typename grad_layout_t, typename grad_data_t>
   inline MFEM_ALWAYS_INLINE
   void GetGradients(int el, const grad_layout_t &l, grad_data_t &grad)
   {
      const int ne = grad_layout_t::dim_4;
      TTensor3<dofs,vdim,ne,complex_type> val_dofs;
      SetElement(el);
      fespace.VectorExtract(vec_layout, data_in, val_dofs.layout, val_dofs);
      shapeEval.CalcGrad(val_dofs.layout.merge_23(), val_dofs,
                         l.merge_34(), grad);
   }

   // TODO: add method GetValuesAndGradients()

   template <typename DataType>
   inline MFEM_ALWAYS_INLINE
   void Eval(DataType &F)
   {
      // T.SetElement() must be called outside
      Action<DataType::InData,true>::Eval(vec_layout, *this, F);
   }

   template <typename DataType>
   inline MFEM_ALWAYS_INLINE
   void Eval(int el, DataType &F)
   {
      SetElement(el);
      Eval(F);
   }

   template <bool Add, typename DataType>
   inline MFEM_ALWAYS_INLINE
   void Assemble(DataType &F)
   {
      // T.SetElement() must be called outside
      Action<DataType::OutData,true>::
      template Assemble<Add>(vec_layout, *this, F);
   }

   template <bool Add, typename DataType>
   inline MFEM_ALWAYS_INLINE
   void Assemble(int el, DataType &F)
   {
      SetElement(el);
      Assemble<Add>(F);
   }

#ifdef MFEM_TEMPLATE_ENABLE_SERIALIZE
   template <typename DataType>
   inline MFEM_ALWAYS_INLINE
   void EvalSerialized(const complex_t *loc_dofs, DataType &F)
   {
      Action<DataType::InData,true>::EvalSerialized(*this, loc_dofs, F);
   }

   template <bool Add, typename DataType>
   inline MFEM_ALWAYS_INLINE
   void AssembleSerialized(const DataType &F, complex_t *loc_dofs)
   {
      Action<DataType::OutData,true>::
      template AssembleSerialized<Add>(*this, F, loc_dofs);
   }
#endif

   // Enumeration for the data type used by the Eval() and Assemble() methods.
   // The types can obtained by summing constants from this enumeration and used
   // as a template parameter in struct Data.
   enum InOutData
   {
      None      = 0,
      Values    = 1,
      Gradients = 2
   };

   // Auxiliary templated struct AData, used by the Eval() and Assemble()
   // methods. The template parameter IOData is "bitwise or" of constants from
   // the enum InOutData. The parameter NE is the number of elements to be
   // processed in the Eval() and Assemble() methods.
   template<int IOData, int NE> struct AData;

   template <int NE> struct AData<0,NE> // 0 = None
   {
      // Do we need this?
   };

   template <int NE> struct AData<1,NE> // 1 = Values
   {
#ifdef MFEM_TEMPLATE_FIELD_EVAL_DATA_HAS_DOFS
      typedef TTensor3<dofs,vdim,NE,complex_t,true> val_dofs_t;
      val_dofs_t val_dofs;
#else
      typedef TTensor3<dofs,vdim,NE,complex_t> val_dofs_t;
#endif
      TTensor3<qpts,vdim,NE,complex_t>      val_qpts;
   };

   template <int NE> struct AData<2,NE> // 2 = Gradients
   {
#ifdef MFEM_TEMPLATE_FIELD_EVAL_DATA_HAS_DOFS
      typedef TTensor3<dofs,vdim,NE,complex_t,true> val_dofs_t;
      val_dofs_t val_dofs;
#else
      typedef TTensor3<dofs,vdim,NE,complex_t> val_dofs_t;
#endif
      TTensor4<qpts,dim,vdim,NE,complex_t>      grad_qpts;
   };

   template <int NE> struct AData<3,NE> // 3 = Values+Gradients
   {
#ifdef MFEM_TEMPLATE_FIELD_EVAL_DATA_HAS_DOFS
      typedef TTensor3<dofs,vdim,NE,complex_t,true> val_dofs_t;
      val_dofs_t val_dofs;
#else
      typedef TTensor3<dofs,vdim,NE,complex_t> val_dofs_t;
#endif
      TTensor3<qpts,    vdim,NE,complex_t,true>  val_qpts;
      TTensor4<qpts,dim,vdim,NE,complex_t>      grad_qpts;
   };

   // This struct is similar to struct AData, adding separate static data
   // members for the input (InData) and output (OutData) data types.
   template <int IData, int OData, int NE>
   struct BData : public AData<IData|OData,NE>
   {
      typedef T_type eval_type;
      static const int ne = NE;
      static const int InData = IData;
      static const int OutData = OData;
   };

   // This struct implements the input (Eval, EvalSerialized) and output
   // (Assemble, AssembleSerialized) operations for the given Ops.
   // Ops is "bitwise or" of constants from the enum InOutData.
   template <int Ops, bool dummy> struct Action;

   template <bool dummy> struct Action<0,dummy> // 0 = None
   {
      // Do we need this?
   };

   template <bool dummy> struct Action<1,dummy> // 1 = Values
   {
      template <typename vec_layout_t, typename AData_t>
      static inline MFEM_ALWAYS_INLINE
      void Eval(const vec_layout_t &l, T_type &T, AData_t &D)
      {
#ifdef MFEM_TEMPLATE_FIELD_EVAL_DATA_HAS_DOFS
         typename AData_t::val_dofs_t &val_dofs = D.val_dofs;
#else
         typename AData_t::val_dofs_t val_dofs;
#endif
         T.fespace.VectorExtract(l, T.data_in, val_dofs.layout, val_dofs);
         T.shapeEval.Calc(val_dofs.layout.merge_23(), val_dofs,
                          D.val_qpts.layout.merge_23(), D.val_qpts);
      }

      template <bool Add, typename vec_layout_t, typename AData_t>
      static inline MFEM_ALWAYS_INLINE
      void Assemble(const vec_layout_t &l, T_type &T, AData_t &D)
      {
         const AssignOp::Type Op = Add ? AssignOp::Add : AssignOp::Set;
#ifdef MFEM_TEMPLATE_FIELD_EVAL_DATA_HAS_DOFS
         typename AData_t::val_dofs_t &val_dofs = D.val_dofs;
#else
         typename AData_t::val_dofs_t val_dofs;
#endif
         T.shapeEval.template CalcT<false>(
            D.val_qpts.layout.merge_23(), D.val_qpts,
            val_dofs.layout.merge_23(), val_dofs);
         T.fespace.template VectorAssemble<Op>(
            val_dofs.layout, val_dofs, l, T.data_out);
      }

#ifdef MFEM_TEMPLATE_ENABLE_SERIALIZE
      template <typename AData_t>
      static inline MFEM_ALWAYS_INLINE
      void EvalSerialized(T_type &T, const complex_t *loc_dofs, AData_t &D)
      {
         T.shapeEval.Calc(AData_t::val_dofs_t::layout.merge_23(), loc_dofs,
                          D.val_qpts.layout.merge_23(), D.val_qpts);
      }

      template <bool Add, typename AData_t>
      static inline MFEM_ALWAYS_INLINE
      void AssembleSerialized(T_type &T, const AData_t &D, complex_t *loc_dofs)
      {
         T.shapeEval.template CalcT<Add>(
            D.val_qpts.layout.merge_23(), D.val_qpts,
            AData_t::val_dofs_t::layout.merge_23(), loc_dofs);
      }
#endif
   };

   template <bool dummy> struct Action<2,dummy> // 2 = Gradients
   {
      template <typename vec_layout_t, typename AData_t>
      static inline MFEM_ALWAYS_INLINE
      void Eval(const vec_layout_t &l, T_type &T, AData_t &D)
      {
#ifdef MFEM_TEMPLATE_FIELD_EVAL_DATA_HAS_DOFS
         typename AData_t::val_dofs_t &val_dofs = D.val_dofs;
#else
         typename AData_t::val_dofs_t val_dofs;
#endif
         T.fespace.VectorExtract(l, T.data_in, val_dofs.layout, val_dofs);
         T.shapeEval.CalcGrad(val_dofs.layout.merge_23(),  val_dofs,
                              D.grad_qpts.layout.merge_34(), D.grad_qpts);
      }

      template <bool Add, typename vec_layout_t, typename AData_t>
      static inline MFEM_ALWAYS_INLINE
      void Assemble(const vec_layout_t &l, T_type &T, AData_t &D)
      {
         const AssignOp::Type Op = Add ? AssignOp::Add : AssignOp::Set;
#ifdef MFEM_TEMPLATE_FIELD_EVAL_DATA_HAS_DOFS
         typename AData_t::val_dofs_t &val_dofs = D.val_dofs;
#else
         typename AData_t::val_dofs_t val_dofs;
#endif
         T.shapeEval.template CalcGradT<false>(
            D.grad_qpts.layout.merge_34(), D.grad_qpts,
            val_dofs.layout.merge_23(), val_dofs);
         T.fespace.template VectorAssemble<Op>(
            val_dofs.layout, val_dofs, l, T.data_out);
      }

#ifdef MFEM_TEMPLATE_ENABLE_SERIALIZE
      template <typename AData_t>
      static inline MFEM_ALWAYS_INLINE
      void EvalSerialized(T_type &T, const complex_t *loc_dofs, AData_t &D)
      {
         T.shapeEval.CalcGrad(AData_t::val_dofs_t::layout.merge_23(), loc_dofs,
                              D.grad_qpts.layout.merge_34(), D.grad_qpts);
      }

      template <bool Add, typename AData_t>
      static inline MFEM_ALWAYS_INLINE
      void AssembleSerialized(T_type &T, const AData_t &D, complex_t *loc_dofs)
      {
         T.shapeEval.template CalcGradT<Add>(
            D.grad_qpts.layout.merge_34(), D.grad_qpts,
            AData_t::val_dofs_t::layout.merge_23(), loc_dofs);
      }
#endif
   };

   template <bool dummy> struct Action<3,dummy> // 3 = Values+Gradients
   {
      template <typename vec_layout_t, typename AData_t>
      static inline MFEM_ALWAYS_INLINE
      void Eval(const vec_layout_t &l, T_type &T, AData_t &D)
      {
#ifdef MFEM_TEMPLATE_FIELD_EVAL_DATA_HAS_DOFS
         typename AData_t::val_dofs_t &val_dofs = D.val_dofs;
#else
         typename AData_t::val_dofs_t val_dofs;
#endif
         T.fespace.VectorExtract(l, T.data_in, val_dofs.layout, val_dofs);
         T.shapeEval.Calc(val_dofs.layout.merge_23(), val_dofs,
                          D.val_qpts.layout.merge_23(), D.val_qpts);
         T.shapeEval.CalcGrad(val_dofs.layout.merge_23(),  val_dofs,
                              D.grad_qpts.layout.merge_34(), D.grad_qpts);
      }

      template <bool Add, typename vec_layout_t, typename AData_t>
      static inline MFEM_ALWAYS_INLINE
      void Assemble(const vec_layout_t &l, T_type &T, AData_t &D)
      {
         const AssignOp::Type Op = Add ? AssignOp::Add : AssignOp::Set;
#ifdef MFEM_TEMPLATE_FIELD_EVAL_DATA_HAS_DOFS
         typename AData_t::val_dofs_t &val_dofs = D.val_dofs;
#else
         typename AData_t::val_dofs_t val_dofs;
#endif
         T.shapeEval.template CalcT<false>(
            D.val_qpts.layout.merge_23(), D.val_qpts,
            val_dofs.layout.merge_23(), val_dofs);
         T.shapeEval.template CalcGradT<true>(
            D.grad_qpts.layout.merge_34(), D.grad_qpts,
            val_dofs.layout.merge_23(),  val_dofs);
         T.fespace.template VectorAssemble<Op>(
            val_dofs.layout, val_dofs, l, T.data_out);
      }

#ifdef MFEM_TEMPLATE_ENABLE_SERIALIZE
      template <typename AData_t>
      static inline MFEM_ALWAYS_INLINE
      void EvalSerialized(T_type &T, const complex_t *loc_dofs, AData_t &D)
      {
         T.shapeEval.Calc(AData_t::val_dofs_t::layout.merge_23(), loc_dofs,
                          D.val_qpts.layout.merge_23(), D.val_qpts);
         T.shapeEval.CalcGrad(AData_t::val_dofs_t::layout.merge_23(), loc_dofs,
                              D.grad_qpts.layout.merge_34(), D.grad_qpts);
      }

      template <bool Add, typename AData_t>
      static inline MFEM_ALWAYS_INLINE
      void AssembleSerialized(T_type &T, const AData_t &D, complex_t *loc_dofs)
      {
         T.shapeEval.template CalcT<Add>(
            D.val_qpts.layout.merge_23(), D.val_qpts,
            AData_t::val_dofs_t::layout.merge_23(), loc_dofs);
         T.shapeEval.template CalcGradT<true>(
            D.grad_qpts.layout.merge_34(), D.grad_qpts,
            AData_t::val_dofs_t::layout.merge_23(), loc_dofs);
      }
#endif
   };

   // This struct implements element matrix computation for some combinations
   // of input (InOps) and output (OutOps) operations.
   template <int InOps, int OutOps, int NE> struct TElementMatrix;

   template <int NE> struct TElementMatrix<1,1,NE> // 1,1 = Values,Values
   {
      // qpt_layout_t is (nip), M_layout_t is (dof x dof)
      // NE = 1 is assumed
      template <typename qpt_layout_t, typename qpt_data_t,
                typename M_layout_t, typename M_data_t>
      static inline MFEM_ALWAYS_INLINE
      void Compute(const qpt_layout_t &a, const qpt_data_t &A,
                   const M_layout_t &m, M_data_t &M, ShapeEval_type &ev)
      {
         ev.Assemble(a.template split_1<qpts,1>(), A,
                     m.template split_2<dofs,1>(), M);
      }
   };

   template <int NE> struct TElementMatrix<2,2,NE> // 2,2 = Gradients,Gradients
   {
      // qpt_layout_t is (nip x dim x dim), M_layout_t is (dof x dof)
      // NE = 1 is assumed
      template <typename qpt_layout_t, typename qpt_data_t,
                typename M_layout_t, typename M_data_t>
      static inline MFEM_ALWAYS_INLINE
      void Compute(const qpt_layout_t &a, const qpt_data_t &A,
                   const M_layout_t &m, M_data_t &M, ShapeEval_type &ev)
      {
         ev.AssembleGradGrad(a.template split_3<dim,1>(), A,
                             m.template split_2<dofs,1>(), M);
      }
   };

   template <typename kernel_t, int NE> struct Spec
   {
      static const int InData =
         Values*kernel_t::in_values + Gradients*kernel_t::in_gradients;
      static const int OutData =
         Values*kernel_t::out_values + Gradients*kernel_t::out_gradients;

      typedef BData<InData,OutData,NE>          DataType;
      typedef TElementMatrix<InData,OutData,NE> ElementMatrix;
   };
};

} // namespace mfem

#endif // MFEM_TEMPLATE_FIELD_EVALUATOR
