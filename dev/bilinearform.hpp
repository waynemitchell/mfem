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

#ifndef MFEM_TEMPLATE_BILINEAR_FORM
#define MFEM_TEMPLATE_BILINEAR_FORM

#include "config.hpp"
#include "tensor_types.hpp"
#include "shape_evaluators.hpp"
#include "eltrans.hpp"
#include "coefficient.hpp"
#include "field_evaluator.hpp"
#include "vector_layouts.hpp"
#include "fem/fespace.hpp"

namespace mfem
{

// complex_t - sol dof data type
// real_t - mesh nodes, sol basis, mesh basis data type
template <typename meshType, typename solFESpace, typename solVecLayout_t,
          typename IR, typename IntegratorType,
          typename complex_t, typename real_t>
class TBilinearForm : public Operator
{
protected:
   typedef complex_t complex_type;
   typedef real_t    real_type;

   typedef typename meshType::FE_type            meshFE_type;
   typedef ShapeEvaluator<meshFE_type,IR,real_t> meshShapeEval;
   typedef typename solFESpace::FE_type          solFE_type;
   typedef ShapeEvaluator<solFE_type,IR,real_t>  solShapeEval;
   typedef solVecLayout_t                        solVecLayout_type;

   static const int dim  = meshType::dim;
   static const int sdim = meshType::space_dim;
   static const int dofs = solFE_type::dofs;
   static const int vdim = solVecLayout_t::vec_dim;
   static const int qpts = IR::qpts;

   typedef IntegratorType integ_t;
   typedef typename integ_t::coefficient_type coeff_t;
   typedef typename integ_t::template kernel<sdim,dim,complex_t>::type kernel_t;
   typedef typename kernel_t::template p_asm_data<qpts>::type p_assembled_t;
   typedef typename kernel_t::template f_asm_data<qpts>::type f_assembled_t;

   typedef TElementTransformation<meshType,IR,real_t> Trans_t;
   template <int NE> struct T_result
   {
      static const int EvalOps =
         Trans_t::template Get<coeff_t,kernel_t>::EvalOps;
      typedef typename Trans_t::template Result<EvalOps,NE> Type;
   };

   typedef FieldEvaluator<solFESpace,solVecLayout_t,IR,
           complex_t,real_t> solFieldEval;
   template <int BE> struct S_spec
   {
      typedef typename solFieldEval::template Spec<kernel_t,BE> Spec;
      typedef typename Spec::DataType DataType;
      typedef typename Spec::ElementMatrix ElementMatrix;
   };

   // Data members

   meshType      mesh;
   meshShapeEval meshEval;

   solFE_type         sol_fe;
   solShapeEval       solEval;
   mutable solFESpace solFES;
   solVecLayout_t     solVecLayout;

   IR int_rule;

   coeff_t coeff;

   p_assembled_t *assembled_data;

public:
   TBilinearForm(const IntegratorType &integ, const FiniteElementSpace &sol_fes)
      : Operator(sol_fes.GetNDofs()*vdim),
        mesh(*sol_fes.GetMesh()),
        meshEval(mesh.fe),
        sol_fe(*sol_fes.FEColl()),
        solEval(sol_fe),
        solFES(sol_fe, sol_fes),
        solVecLayout(sol_fes),
        int_rule(),
        coeff(integ.coeff),
        assembled_data(NULL)
   { }

   virtual ~TBilinearForm()
   {
      delete [] assembled_data;
   }

   virtual void Mult(const Vector &x, Vector &y) const
   {
      if (assembled_data)
      {
         const int num_elem = 1;
         MultAssembled<num_elem>(x, y);
      }
      else
      {
         MultUnassembled(x, y);
      }
   }

   // complex_t = double
   void MultUnassembled(const Vector &x, Vector &y) const
   {
      y = 0.0;

      const int BE = 1; // batch-size of elements
      typedef typename kernel_t::template
      CoefficientEval<IR,coeff_t,BE>::Type coeff_eval_t;

      // For better performance, create stack copies of solFES, and solEval
      // inside 'solFEval'. The element-transformation 'T' also copies the
      // meshFES, meshEval, etc internally.
      // Is performance actually better with this implementation?
      Trans_t T(mesh, meshEval);
      solFieldEval solFEval(solFES, solEval, solVecLayout,
                            x.GetData(), y.GetData());
      coeff_eval_t wQ(int_rule, coeff);

      const int NE = mesh.GetNE();
      for (int el = 0; el < NE; el++)
      {
#if 0
         typename S_spec<BE>::DataType R;
         solFEval.Eval(el, R);

         typename T_result<BE>::Type F;
         T.Eval(el, F);
#else
         typename T_result<BE>::Type F;
         T.Eval(el, F);

         typename S_spec<BE>::DataType R;
         solFEval.Eval(el, R);
#endif

         typename coeff_eval_t::result_t res;
         wQ.Eval(F, res);

         kernel_t::Action(0, F, wQ, res, R);

         solFEval.template Assemble<true>(R);
      }
   }

   // Partial assembly of quadrature point data
   void Assemble()
   {
      const int BE = 1; // batch-size of elements
      typedef typename kernel_t::template
      CoefficientEval<IR,coeff_t,BE>::Type coeff_eval_t;

      Trans_t T(mesh, meshEval);
      coeff_eval_t wQ(int_rule, coeff);

      const int NE = mesh.GetNE();
      if (!assembled_data)
      {
         assembled_data = new p_assembled_t[NE];
      }
      for (int el = 0; el < NE; el++) // BE == 1
      {
         typename T_result<BE>::Type F;
         T.Eval(el, F);

         typename coeff_eval_t::result_t res;
         wQ.Eval(F, res);

         for (int k = 0; k < BE; k++)
         {
            kernel_t::Assemble(k, F, wQ, res, assembled_data[el+k]);
         }
      }
   }

   template <int num_elem>
   inline MFEM_ALWAYS_INLINE
   void ElementAddMultAssembled(int el, solFieldEval &solFEval) const
   {
      typename S_spec<num_elem>::DataType R;
      solFEval.Eval(el, R);

      for (int k = 0; k < num_elem; k++)
      {
         kernel_t::MultAssembled(k, assembled_data[el+k], R);
      }

      solFEval.template Assemble<true>(R);
   }

   // complex_t = double
   template <int num_elem>
   void MultAssembled(const Vector &x, Vector &y) const
   {
      y = 0.0;

      solFieldEval solFEval(solFES, solEval, solVecLayout,
                            x.GetData(), y.GetData());

      const int NE = mesh.GetNE();
      const int bNE = NE-NE%num_elem;
      for (int el = 0; el < bNE; el += num_elem)
      {
         ElementAddMultAssembled<num_elem>(el, solFEval);
      }
      for (int el = bNE; el < NE; el++)
      {
         ElementAddMultAssembled<1>(el, solFEval);
      }
   }

#ifdef MFEM_TEMPLATE_ENABLE_SERIALIZE
   // complex_t = double
   void ElementwiseExtractAssembleTest(const Vector &x, Vector &y) const
   {
      y = 0.0;

      solFESpace solFES(this->solFES);

      TMatrix<dofs,1,complex_t> xy_dof;

      const int NE = mesh.GetNE();
      for (int el = 0; el < NE; el++)
      {
         solFES.SetElement(el);

         solFES.Extract(x, xy_dof.layout, xy_dof);
         solFES.Assemble(xy_dof.layout, xy_dof, y);
      }
   }

   // real_t = double
   void SerializeNodesTest(Vector &sNodes) const
   {
      typedef typename meshType::FESpace_type meshFESpace;
      meshFESpace meshFES(mesh.t_fes);
      typedef TTensor3<meshFE_type::dofs,sdim,1,real_t> lnodes_t;

      const int NE = mesh.GetNE();
      sNodes.SetSize(lnodes_t::size*NE);
      real_t *lNodes = sNodes.GetData();
      for (int el = 0; el < NE; el++)
      {
         meshFES.SetElement(el);
         meshFES.VectorExtract(mesh.node_layout, mesh.Nodes,
                               lnodes_t::layout, lNodes);
         lNodes += lnodes_t::size;
      }
   }

   // partial assembly from "serialized" nodes
   // real_t = double
   void AssembleFromSerializedNodesTest(const Vector &sNodes)
   {
      const int  BE = 1; // batch-size of elements
      typedef typename kernel_t::template
      CoefficientEval<IR,coeff_t,BE>::Type coeff_eval_t;

      Trans_t T(this->mesh, this->meshEval);
      coeff_eval_t wQ(int_rule, coeff);

      const int NE = mesh.GetNE();
      if (!assembled_data)
      {
         assembled_data = new p_assembled_t[NE];
      }
      for (int el = 0; el < NE; el++)
      {
         typename T_result<BE>::Type F;
         T.EvalSerialized(el, sNodes.GetData(), F);

         typename coeff_eval_t::result_t res;
         wQ.Eval(F, res);

         kernel_t::Assemble(0, F, wQ, res, assembled_data[el]);
      }
   }

   // complex_t = double
   void Serialize(const Vector &x, Vector &sx) const
   {
      solVecLayout_t solVecLayout(this->solVecLayout);
      typedef TTensor3<dofs,vdim,1,complex_t> vdof_data_t;
      solFESpace solFES(this->solFES);

      const int NE = mesh.GetNE();
      sx.SetSize(vdim*dofs*NE);
      complex_t *loc_sx = sx.GetData();
      for (int el = 0; el < NE; el++)
      {
         solFES.SetElement(el);
         solFES.VectorExtract(solVecLayout, x, vdof_data_t::layout, loc_sx);
         loc_sx += vdim*dofs;
      }
   }

   // serialized vector sx --> serialized vector 'sy'
   // complex_t = double
   void MultAssembledSerialized(const Vector &sx, Vector &sy) const
   {
      solFieldEval solFEval(solFES, solEval, solVecLayout, NULL, NULL);

      const int NE = mesh.GetNE();
      const complex_t *loc_sx = sx.GetData();
      complex_t *loc_sy = sy.GetData();
      for (int el = 0; el < NE; el++)
      {
         typename S_spec<1>::DataType R;
         solFEval.EvalSerialized(loc_sx, R);

         kernel_t::MultAssembled(0, assembled_data[el], R);

         solFEval.template AssembleSerialized<false>(R, loc_sy);

         loc_sx += vdim*dofs;
         loc_sy += vdim*dofs;
      }
   }
#endif // MFEM_TEMPLATE_ENABLE_SERIALIZE

   // Assemble the operator in a SparseMatrix.
   // complex_t = double
   void AssembleMatrix(SparseMatrix &M) const
   {
      const int BE = 1; // batch-size of elements
      typedef typename kernel_t::template
      CoefficientEval<IR,coeff_t,BE>::Type coeff_eval_t;

      Trans_t T(mesh, meshEval);
      solFESpace solFES(this->solFES);
      solShapeEval solEval(this->solEval);
      coeff_eval_t wQ(int_rule, coeff);

      const int NE = mesh.GetNE();
      for (int el = 0; el < NE; el++)
      {
         f_assembled_t asm_qpt_data;
         {
            typename T_result<BE>::Type F;
            T.Eval(el, F);

            typename coeff_eval_t::result_t res;
            wQ.Eval(F, res);

            kernel_t::Assemble(0, F, wQ, res, asm_qpt_data);
         }

         TMatrix<dofs,dofs> M_loc;
         S_spec<BE>::ElementMatrix::Compute(
            asm_qpt_data.layout, asm_qpt_data, M_loc.layout, M_loc, solEval);

         solFES.SetElement(el);
         solFES.Assemble(M_loc, M);
      }
   }

   // Assemble element matrices and store them as DenseTensor.
   // complex_t = double
   void AssembleMatrix(DenseTensor &M) const
   {
      const int BE = 1; // batch-size of elements
      typedef typename kernel_t::template
      CoefficientEval<IR,coeff_t,BE>::Type coeff_eval_t;

      Trans_t T(mesh, meshEval);
      solShapeEval solEval(this->solEval);
      coeff_eval_t wQ(int_rule, coeff);

      const int NE = mesh.GetNE();
      for (int el = 0; el < NE; el++)
      {
         f_assembled_t asm_qpt_data;
         {
            typename T_result<BE>::Type F;
            T.Eval(el, F);

            typename coeff_eval_t::result_t res;
            wQ.Eval(F, res);

            kernel_t::Assemble(0, F, wQ, res, asm_qpt_data);
         }

         TMatrix<dofs,dofs> M_loc;
         S_spec<BE>::ElementMatrix::Compute(
            asm_qpt_data.layout, asm_qpt_data, M_loc.layout, M_loc, solEval);

         complex_t *M_data = M.GetData(el);
         M_loc.template AssignTo<AssignOp::Set>(M_data);
      }
   }

   // Multiplication using assembled element matrices stored as a DenseTensor.
   // complex_t = double
   void AddMult(DenseTensor &M, const Vector &x, Vector &y) const
   {
      const int NE = mesh.GetNE();
      for (int el = 0; el < NE; el++)
      {
         TMatrix<dofs,1,complex_t> x_dof, y_dof;

         solFES.SetElement(el);
         solFES.Extract(x, x_dof.layout, x_dof);
         Mult_AB<false>(TMatrix<dofs,dofs>::layout,
                        M(el).Data(),
                        x_dof.layout, x_dof,
                        y_dof.layout, y_dof);
         solFES.Assemble(y_dof.layout, y_dof, y);
      }
   }
};

} // namespace mfem

#endif // MFEM_TEMPLATE_BILINEAR_FORM
