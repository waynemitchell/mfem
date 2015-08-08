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

#ifndef MFEM_TEMPLATE_MASS_OPERATOR
#define MFEM_TEMPLATE_MASS_OPERATOR

#include "config.hpp"
#include "tensor_types.hpp"
#include "shape_evaluators.hpp"
#include "fem/fespace.hpp"

namespace mfem
{

template <typename meshFESpace, typename nodeLayout,
          typename spaceFESpace, typename IR>
class TMassOperator : public Operator
{
protected:
   typedef typename meshFESpace::FE_type meshFE;
   typedef typename spaceFESpace::FE_type spaceFE;

   typedef ShapeEvaluator<meshFE, IR> meshShapeEval;
   typedef ShapeEvaluator<spaceFE, IR> spaceShapeEval;

   static const int dim = meshFE::dim;
   static const int qpts = IR::qpts;

   meshFE mesh_fe;
   spaceFE space_fe;

   meshShapeEval meshEval;
   spaceShapeEval spaceEval;

   const FiniteElementSpace &mesh_fes;
   GridFunction &meshNodes;

   mutable meshFESpace meshFES;
   mutable spaceFESpace spaceFES;

   nodeLayout node_layout;

   IR int_rule;

   TVector<qpts> *assembled_data;

public:
   TMassOperator(const FiniteElementSpace &space_fes)
      : Operator(space_fes.GetNDofs()),
        mesh_fe(), space_fe(),
        meshEval(mesh_fe), spaceEval(space_fe),
        mesh_fes(*space_fes.GetMesh()->GetNodalFESpace()),
        meshNodes(*space_fes.GetMesh()->GetNodes()),
        meshFES(mesh_fe, mesh_fes), spaceFES(space_fe, space_fes),
        node_layout(mesh_fes), int_rule(), assembled_data(NULL)
   { }

   ~TMassOperator()
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

   void MultUnassembled(const Vector &x, Vector &y) const
   {
      y = 0.0;

      // For better performance, create stack copies of meshFES, spaceFES,
      // meshEval, spaceEval, and int_rule.
      // Is performance actually better with this implementation?
      meshFESpace meshFES(this->meshFES);
      meshShapeEval meshEval(this->meshEval);
      spaceFESpace spaceFES(this->spaceFES);
      spaceShapeEval spaceEval(this->spaceEval);
      IR int_rule(this->int_rule);

      TTensor3<meshFE::dofs,dim,1> nodes_dof;
      TTensor3<qpts,dim,dim> J_qpt;
      TMatrix<spaceFE::dofs,1> xy_dof;
      TMatrix<qpts,1> xy_qpt;

      const int NE = mesh_fes.GetNE();
      for (int el = 0; el < NE; el++)
      {
         meshFES.SetElement(el);
         spaceFES.SetElement(el);

         meshFES.VectorExtract(node_layout, meshNodes,
                               nodes_dof.layout, nodes_dof);
         meshEval.CalcGrad(nodes_dof.layout.merge_23(), nodes_dof,
                           J_qpt.layout, J_qpt);

         spaceFES.Extract(x, xy_dof.layout, xy_dof);
         spaceEval.Calc(xy_dof.layout, xy_dof, xy_qpt.layout, xy_qpt);

         TDet<AssignOp::Mult>(J_qpt.layout, J_qpt, xy_qpt);

         int_rule.ApplyWeights(xy_qpt);

         spaceEval.template CalcT<false>(xy_qpt.layout, xy_qpt,
                                         xy_dof.layout, xy_dof);
         spaceFES.Assemble(xy_dof.layout, xy_dof, y);
      }
   }

   // Partial assembly of quadrature point data
   void Assemble()
   {
      meshFESpace meshFES(this->meshFES);
      meshShapeEval meshEval(this->meshEval);
      IR int_rule(this->int_rule);

      TTensor3<meshFE::dofs,dim,1> nodes_dof;
      TTensor3<qpts,dim,dim> J_qpt;

      const int NE = mesh_fes.GetNE();
      if (!assembled_data)
      {
         assembled_data = new TVector<qpts>[NE];
      }
      for (int el = 0; el < NE; el++)
      {
         meshFES.SetElement(el);

         meshFES.VectorExtract(node_layout, meshNodes,
                               nodes_dof.layout, nodes_dof);
         meshEval.CalcGrad(nodes_dof.layout.merge_23(), nodes_dof,
                           J_qpt.layout, J_qpt);

         TDet<AssignOp::Set>(J_qpt.layout, J_qpt, assembled_data[el]);

         int_rule.ApplyWeights(assembled_data[el]);
      }
   }

   template <int num_elem>
   void ElementAddMultAssembled(int el, const Vector &x, Vector &y) const
   {
      TMatrix<spaceFE::dofs,num_elem> xy_dof;
      TMatrix<qpts,num_elem> xy_qpt;

      spaceFES.SetElement(el);

      spaceFES.Extract(x, xy_dof.layout, xy_dof);
      spaceEval.Calc(xy_dof.layout, xy_dof, xy_qpt.layout, xy_qpt);

      for (int k = 0; k < num_elem; k++)
      {
         for (int j = 0; j < qpts; j++)
         {
            xy_qpt(j,k) *= assembled_data[k+el].data[j];
         }
      }

      spaceEval.template CalcT<false>(xy_qpt.layout, xy_qpt,
                                      xy_dof.layout, xy_dof);
      spaceFES.Assemble(xy_dof.layout, xy_dof, y);
   }

   template <int num_elem>
   void MultAssembled(const Vector &x, Vector &y) const
   {
      y = 0.0;

      spaceFESpace spaceFES(this->spaceFES);
      spaceShapeEval spaceEval(this->spaceEval);

      TMatrix<spaceFE::dofs,num_elem> xy_dof;
      TMatrix<qpts,num_elem> xy_qpt;

      const int NE = mesh_fes.GetNE();
      for (int el = 0; el < NE; el += num_elem)
      {
         spaceFES.SetElement(el);

         spaceFES.Extract(x, xy_dof.layout, xy_dof);
         spaceEval.Calc(xy_dof.layout, xy_dof, xy_qpt.layout, xy_qpt);

         for (int k = 0; k < num_elem; k++)
         {
            for (int j = 0; j < qpts; j++)
            {
               xy_qpt(j,k) *= assembled_data[k+el].data[j];
            }
         }

         spaceEval.template CalcT<false>(xy_qpt.layout, xy_qpt,
                                         xy_dof.layout, xy_dof);
         spaceFES.Assemble(xy_dof.layout, xy_dof, y);
      }
      for (int el = NE-NE%num_elem; el < NE; el++)
      {
         spaceFES.SetElement(el);

         spaceFES.Extract(x, TMatrix<spaceFE::dofs,1>::layout, xy_dof);
         spaceEval.Calc(TMatrix<spaceFE::dofs,1>::layout, xy_dof,
                        TMatrix<qpts,1>::layout, xy_qpt);

         for (int j = 0; j < qpts; j++)
         {
            xy_qpt(j,0) *= assembled_data[el].data[j];
         }

         spaceEval.template CalcT<false>(
            TMatrix<qpts,1>::layout, xy_qpt,
            TMatrix<spaceFE::dofs,1>::layout, xy_dof);
         spaceFES.Assemble(TMatrix<spaceFE::dofs,1>::layout, xy_dof, y);
      }
   }

   void ElementwiseExtractAssembleTest(const Vector &x, Vector &y) const
   {
      y = 0.0;

      spaceFESpace spaceFES(this->spaceFES);

      TMatrix<spaceFE::dofs,1> xy_dof;

      const int NE = mesh_fes.GetNE();
      for (int el = 0; el < NE; el++)
      {
         spaceFES.SetElement(el);

         spaceFES.Extract(x, xy_dof.layout, xy_dof);
         spaceFES.Assemble(xy_dof.layout, xy_dof, y);
      }
   }

   void SerializeNodesTest(Vector &sNodes) const
   {
      meshFESpace meshFES(this->meshFES);
      typedef TTensor3<meshFE::dofs,dim,1> lnodes_t;

      const int NE = mesh_fes.GetNE();
      sNodes.SetSize(lnodes_t::size*NE);
      double *lNodes = sNodes.GetData();
      for (int el = 0; el < NE; el++)
      {
         meshFES.SetElement(el);
         meshFES.VectorExtract(node_layout, meshNodes,
                               lnodes_t::layout, lNodes);
         lNodes += lnodes_t::size;
      }
   }

   void AssembleFromSerializedNodesTest(const Vector &sNodes)
   {
      meshShapeEval meshEval(this->meshEval);
      IR int_rule(this->int_rule);

      TTensor3<meshFE::dofs,dim,1> nodes_dof;
      TTensor3<qpts,dim,dim> J_qpt;

      const int NE = mesh_fes.GetNE();
      if (!assembled_data)
      {
         assembled_data = new TVector<qpts>[NE];
      }
      for (int el = 0; el < NE; el++)
      {
         meshEval.CalcGrad(nodes_dof.layout.merge_23(),
                           &sNodes(nodes_dof.size*el),
                           J_qpt.layout, J_qpt);

         TDet<AssignOp::Set>(J_qpt.layout, J_qpt, assembled_data[el]);

         int_rule.ApplyWeights(assembled_data[el]);
      }
   }

   void Serialize(const Vector &x, Vector &sx) const
   {
      const int dofs = spaceFE::dofs;
      typedef TMatrix<dofs,1> dof_data_t;
      spaceFESpace spaceFES(this->spaceFES);

      const int NE = mesh_fes.GetNE();
      sx.SetSize(dofs*NE);
      double *loc_sx = sx.GetData();
      for (int el = 0; el < NE; el++)
      {
         spaceFES.SetElement(el);
         spaceFES.Extract(x, dof_data_t::layout, loc_sx);
         loc_sx += dofs;
      }
   }

   // serialized vector sx --> serialized vector 'sy'
   void MultAssembledSerialized(const Vector &sx, Vector &sy) const
   {
      const int dofs = spaceFE::dofs;
      typedef TMatrix<dofs,1> dof_data_t;
      spaceShapeEval spaceEval(this->spaceEval);

      TMatrix<qpts,1> xy_qpt;

      const int NE = mesh_fes.GetNE();
      const double *loc_sx = sx.GetData();
      double *loc_sy = sy.GetData();
      for (int el = 0; el < NE; el++)
      {
         spaceEval.Calc(dof_data_t::layout, loc_sx,
                        xy_qpt.layout, xy_qpt);

         for (int j = 0; j < qpts; j++)
         {
            xy_qpt(j,0) *= assembled_data[el].data[j];
         }

         spaceEval.template CalcT<false>(xy_qpt.layout, xy_qpt,
                                         dof_data_t::layout, loc_sy);
         loc_sx += dofs;
         loc_sy += dofs;
      }
   }

   // Assemble the operator in a SparseMatrix.
   void AssembleMatrix(SparseMatrix &M) const
   {
      meshFESpace meshFES(this->meshFES);
      meshShapeEval meshEval(this->meshEval);
      spaceFESpace spaceFES(this->spaceFES);
      spaceShapeEval spaceEval(this->spaceEval);
      IR int_rule(this->int_rule);

      TTensor3<meshFE::dofs,dim,1> nodes_dof;
      TTensor3<qpts,dim,dim> J_qpt;
      TVector<qpts> asm_qpt_data;
      const int dofs = spaceFE::dofs;
      TMatrix<dofs,dofs> M_loc;

      const int NE = mesh_fes.GetNE();
      for (int el = 0; el < NE; el++)
      {
         meshFES.SetElement(el);
         spaceFES.SetElement(el);

         meshFES.VectorExtract(node_layout, meshNodes,
                               nodes_dof.layout, nodes_dof);
         meshEval.CalcGrad(nodes_dof.layout.merge_23(), nodes_dof,
                           J_qpt.layout, J_qpt);

         TDet<AssignOp::Set>(J_qpt.layout, J_qpt, asm_qpt_data);

         int_rule.ApplyWeights(asm_qpt_data);

         spaceEval.Assemble(TMatrix<qpts,1>::layout, asm_qpt_data,
                            TTensor3<dofs,dofs,1>::layout, M_loc);
         spaceFES.Assemble(M_loc, M);
      }
   }

   // Assemble element matrices and store them as DenseTensor.
   void AssembleMatrix(DenseTensor &M) const
   {
      meshFESpace meshFES(this->meshFES);
      meshShapeEval meshEval(this->meshEval);
      spaceShapeEval spaceEval(this->spaceEval);
      IR int_rule(this->int_rule);

      TTensor3<meshFE::dofs,dim,1> nodes_dof;
      TTensor3<qpts,dim,dim> J_qpt;
      TVector<qpts> asm_qpt_data;
      const int dofs = spaceFE::dofs;
      TMatrix<dofs,dofs> M_loc;

      const int NE = mesh_fes.GetNE();
      for (int el = 0; el < NE; el++)
      {
         meshFES.SetElement(el);

         meshFES.VectorExtract(node_layout, meshNodes,
                               nodes_dof.layout, nodes_dof);
         meshEval.CalcGrad(nodes_dof.layout.merge_23(), nodes_dof,
                           J_qpt.layout, J_qpt);

         TDet<AssignOp::Set>(J_qpt.layout, J_qpt, asm_qpt_data);

         int_rule.ApplyWeights(asm_qpt_data);

         spaceEval.Assemble(TMatrix<qpts,1>::layout, asm_qpt_data,
                            TTensor3<dofs,dofs,1>::layout, M_loc);

         double *M_data = M.GetData(el);
         M_loc.template AssignTo<AssignOp::Set>(M_data);
      }
   }

   // Multiplication using assembled element matrices stored as a DenseTensor.
   void AddMult(DenseTensor &M, const Vector &x, Vector &y) const
   {
      TMatrix<spaceFE::dofs,1> x_dof, y_dof;

      const int NE = mesh_fes.GetNE();
      for (int el = 0; el < NE; el++)
      {
         spaceFES.SetElement(el);
         spaceFES.Extract(x, x_dof.layout, x_dof);
         Mult_AB<false>(TMatrix<spaceFE::dofs,spaceFE::dofs>::layout,
                        M(el).Data(),
                        x_dof.layout, x_dof,
                        y_dof.layout, y_dof);
         spaceFES.Assemble(y_dof.layout, y_dof, y);
      }
   }
};

} // namespace mfem

#endif // MFEM_TEMPLATE_MASS_OPERATOR
