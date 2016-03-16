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

#include "layouts.hpp"
#include "assign_ops.hpp"
#include "small_matrix_ops.hpp"
#include "tensor_ops.hpp"
#include "tensor_types.hpp"
#include "matrix_products.hpp"
#include "tensor_products.hpp"
#include "finite_elements_h1.hpp"
#include "integration_rules.hpp"
#include "shape_evaluators.hpp"
#include "vector_layouts.hpp"
#include "fespace_h1.hpp"
#include "fespace_l2.hpp"
#include "mesh.hpp"
#include "mass_kernel.hpp"
#include "diffusion_kernel.hpp"

#include "mfem.hpp"

#include <iostream>
#include <fstream>
#include <ctime>

using namespace mfem;

typedef double real_t;
typedef double complex_t;

template<int oper_type>
struct TOperator;

struct CoeffFunc
{
   static inline MFEM_ALWAYS_INLINE
   double Eval1D(double x) { return 1.0 + x; }

   static inline MFEM_ALWAYS_INLINE
   double Eval2D(double x, double y) { return 1.0 + x + y; }

   static inline MFEM_ALWAYS_INLINE
   double Eval3D(double x, double y, double z) { return 1.0 + x + y + z; }
};

template <Geometry::Type GEOM, int P, int IR_ORDER, int MESH_ORDER, int OPER>
void Test(int mesh_size, std::ostream &out = std::cout)
{
   out << "-----------------------------------------------\n"
       << "GEOMETRY   = " << Geometry::Name[GEOM] << "\n"
       << "ORDER      = " << P << "\n"
       << "IR_ORDER   = " << IR_ORDER << "\n"
       << "MESH_ORDER = " << MESH_ORDER << "\n"
       << "OPERATOR   = " << TOperator<OPER>::Name << "\n"
       << "-----------------------------------------------"
       << std::endl;
   typedef H1_FiniteElement<GEOM, P> sol_fe_t;

   typedef TIntegrationRule<GEOM, IR_ORDER, real_t> int_rule_1;
   // typedef GenericIntegrationRule<GEOM, int_rule_1::qpts, IR_ORDER,
   //         real_t> int_rule_2;

   typedef int_rule_1 int_rule_t;
   // typedef int_rule_2 int_rule_t;

   H1_FECollection::BasisType basis_type;
   basis_type = H1_FECollection::GaussLobatto;
   // basis_type = H1_FECollection::Positive;
   sol_fe_t fe(basis_type);

   int nx = 1, ny = 1, nz = 1;
   switch (sol_fe_t::dim)
   {
      case 1:
         switch (mesh_size)
         {
            case 0: nx = ( 120/P)*(  80); break;
            case 1: nx = ( 800/P)*( 600); break;
            case 2: nx = (3600/P)*(4800); break;
         }
         break;
      case 2:
         switch (mesh_size)
         {
            case 0: nx =  120/P; ny =   80/P; break;
            case 1: nx = 1200/P; ny =  800/P; break;
            // case 1: nx = 3200/P; ny = 3200/P; break;
            case 2: nx = 2400/P; ny = 3200/P; break;
         }
         break;
      case 3:
         switch (mesh_size)
         {
            case 0: nx =  20/P; ny =  12/P; nz =   8/P; break;
            // case 1: nx = 200/P; ny = 120/P; nz =  80/P; break;
            case 1: nx = 100/P; ny = 100/P; nz = 100/P; break;
            case 2: nx = 200/P; ny = 120/P; nz = 160/P; break;
         }
   }
   int generate_edges = 1;
   Mesh *mesh;
   switch (GEOM)
   {
      case Geometry::SEGMENT:
         mesh = new Mesh(nx);
         break;
      case Geometry::TRIANGLE:
         mesh = new Mesh(nx, ny, Element::TRIANGLE, generate_edges);
         break;
      case Geometry::SQUARE:
         // nx = ny = 1;
         mesh = new Mesh(nx, ny, Element::QUADRILATERAL, generate_edges);
         break;
      case Geometry::TETRAHEDRON:
         mesh =  new Mesh(nx, ny, nz, Element::TETRAHEDRON, generate_edges);
         break;
      case Geometry::CUBE:
         mesh =  new Mesh(nx, ny, nz, Element::HEXAHEDRON, generate_edges);
         break;
   }

   const int mesh_order = MESH_ORDER;
   H1_FECollection mesh_fec(mesh_order, sol_fe_t::dim, basis_type);
   FiniteElementSpace mesh_fes(mesh, &mesh_fec, sol_fe_t::dim);
   mesh->SetNodalFESpace(&mesh_fes);
   typedef H1_FiniteElement<GEOM, mesh_order> mesh_fe_t;

#if 1
   // Perturb the mesh nodes to transform the structured elements into more
   // general shapes.
   {
      GridFunction &nodes = *mesh->GetNodes();
      double h = 0.2*mesh->GetElementSize(0)/mesh_order;
      GridFunction node_pert(&mesh_fes);
      node_pert.Randomize();
      Array<int> bdr_vdofs;
      for (int i = 0; i < mesh->GetNBE(); i++)
      {
         mesh_fes.GetBdrElementVDofs(i, bdr_vdofs);
         for (int j = 0; j < bdr_vdofs.Size(); j++)
         {
            node_pert(bdr_vdofs[j]) = 0.5;
         }
      }
      for (int i = 0; i < node_pert.Size(); i++)
      {
         nodes(i) += h*(node_pert(i) - 0.5);
      }
   }
#endif
#if 0
   {
      std::ofstream mesh_file("test-"+std::string(Geometry::Name[GEOM])+
                              ".mesh");
      mesh_file << *mesh;
   }
#endif

   const double MiB = 1024.*1024.;
   out << "Number of mesh node dofs = " << mesh_fe_t::dim*mesh_fe_t::dofs
       << " per element" << std::endl;
   out << "Number of solution dofs  = " << sol_fe_t::dofs
       << " per element" << std::endl;
   out << "Number of quadr. points  = "
       << int_rule_t::qpts << " per element" << std::endl;
   out << "Number of mesh elements  = " << mesh->GetNE() << std::endl;
   out << "Number of mesh node dofs = " << mesh_fes.GetVSize()
       << " total (" << sizeof(double)/MiB*mesh_fes.GetVSize() << " MiB)"
       << std::endl;
   out << "Size of mesh el_dof J    = " << mesh_fe_t::dofs*mesh->GetNE()
       << "(" << sizeof(int)/MiB*mesh_fe_t::dofs*mesh->GetNE() << " MiB)\n";
   H1_FECollection fec(P, sol_fe_t::dim, basis_type);
   FiniteElementSpace fes(mesh, &fec);
   out << "Number of solution dofs  = " << fes.GetNDofs()
       << " total (" << sizeof(double)/MiB*fes.GetNDofs() << " MiB)"
       << std::endl;
   out << "Size of sol. el_dof J    = " << sol_fe_t::dofs*mesh->GetNE()
       << "(" << sizeof(int)/MiB*sol_fe_t::dofs*mesh->GetNE() << " MiB)\n";
   out << "Number of quadr. points  = "
       << mesh->GetNE()*int_rule_t::qpts << " total ("
       << sizeof(double)/MiB*mesh->GetNE()*int_rule_t::qpts << " MiB)"
       << std::endl;
   GridFunction x(&fes);

   typedef H1_FiniteElementSpace<sol_fe_t> sol_fes_t;

   typedef H1_FiniteElementSpace<mesh_fe_t> mesh_fes_t;
   typedef VectorLayout<Ordering::byNODES, mesh_fe_t::dim> node_layout_t;
   typedef TMesh<mesh_fes_t,node_layout_t> mesh_t;

#if 1
   typedef TConstantCoefficient<complex_t> coeff_t;
   coeff_t coeff(1.0);
#elif 1
   typedef TFunctionCoefficient<CoeffFunc,complex_t> coeff_t;
   coeff_t coeff;
#elif 1
   Vector Qs(mesh->attributes.Max());
   for (int i = 0; i < Qs.Size(); i++)
   {
      Qs(i) = i + 1.0;
   }
   typedef TPiecewiseConstCoefficient<complex_t> coeff_t;
   coeff_t coeff(Qs);
#elif 1
   GridFunction c(&fes); // complex_t = double
   c = 1.0;
   typedef FieldEvaluator<sol_fes_t,ScalarLayout,int_rule_t,
           complex_t,real_t> solFieldEval_t;
   typedef TGridFunctionCoefficient<solFieldEval_t> coeff_t;
   coeff_t coeff(c);
#endif

   typedef typename TOperator<OPER>::
   template Integrator<coeff_t>::type integ_type;
   integ_type t_integ(coeff);

   typedef TBilinearForm<mesh_t,sol_fes_t,ScalarLayout,int_rule_t,
           integ_type,complex_t,real_t>
           operator_type;
   operator_type templ_oper(t_integ, fes);
   int size = templ_oper.Height();
   Vector r1(size), r2(size), r3(size), r4(size);

   BilinearForm bilin_form(&fes);
   typedef typename TOperator<OPER>::Integrator_type integrator_type;
   integrator_type *integ = new integrator_type;
   integ->SetIntRule(&int_rule_t::GetIntRule());
   bilin_form.AddDomainIntegrator(integ);
#if 1
   bilin_form.UsePrecomputedSparsity();
   bilin_form.AllocateMatrix();
#endif

#if 1
   SparseMatrix A(bilin_form.SpMat());
   DenseTensor Ae(sol_fe_t::dofs, sol_fe_t::dofs, mesh->GetNE());
#endif

#if 1
   x.Randomize();
#elif 0
   x = 1.0;
#else
   struct myCoefficient : public Coefficient
   {
      virtual double Eval(ElementTransformation &Tr,
                          const IntegrationPoint &ip)
      {
         double x[3];
         Vector transip(x, 3);
         Tr.Transform(ip, transip);
         return x[0];
      }
   };
   myCoefficient my_coeff;
   x.ProjectCoefficient(my_coeff);
#endif

   double rmops, wmops;

   r1 = 0.0; // touch 'r1' to make sure the memory is actually allocated
   MFEM_FLOPS_RESET();
   tic_toc.Clear();
   tic_toc.Start();
   templ_oper.Mult(x, r1);
   tic_toc.Stop();
   out << "Unassembled Mult() flops         = "
       << MFEM_FLOPS_GET()*1e-9 << " GFlops" << std::endl;
   out << "Unassembled Mult() time          = "
       << tic_toc.RealTime() << " sec ("
       << tic_toc.UserTime() << " sec)" << std::endl;
   tic_toc.Clear();
   tic_toc.Start();
   templ_oper.Mult(x, r1);
   tic_toc.Stop();
   out << "Unassembled Mult() time          = "
       << tic_toc.RealTime() << " sec ("
       << tic_toc.UserTime() << " sec)" << std::endl;
#if 1
   tic_toc.Clear();
   tic_toc.Start();
   templ_oper.Mult(x, r1);
   tic_toc.Stop();
   out << "Unassembled Mult() time          = "
       << tic_toc.RealTime() << " sec ("
       << tic_toc.UserTime() << " sec)" << std::endl;
#endif

   MFEM_FLOPS_RESET();
   tic_toc.Clear();
   tic_toc.Start();
   templ_oper.Assemble();
   tic_toc.Stop();
   rmops = (sizeof(double)/MiB*mesh_fes.GetVSize() +
            sizeof(int)/MiB*mesh_fe_t::dofs*mesh->GetNE());
   wmops = sizeof(double)/MiB*mesh->GetNE()*int_rule_t::qpts;
   out << "Assemble() mem ops               = " << rmops+wmops << " MiB = "
       << rmops << " read + " << wmops << " write\n";
   out << "Assemble() flops                 = "
       << MFEM_FLOPS_GET()*1e-9 << " GFlops" << std::endl;
   out << "Assemble() time                  = "
       << tic_toc.RealTime() << " sec ("
       << tic_toc.UserTime() << " sec)" << std::endl;
#if 1
   tic_toc.Clear();
   tic_toc.Start();
   templ_oper.Assemble();
   tic_toc.Stop();
   out << "Assemble() time                  = "
       << tic_toc.RealTime() << " sec ("
       << tic_toc.UserTime() << " sec)" << std::endl;
#endif
#if 1
   tic_toc.Clear();
   tic_toc.Start();
   templ_oper.Assemble();
   tic_toc.Stop();
   out << "Assemble() time                  = "
       << tic_toc.RealTime() << " sec ("
       << tic_toc.UserTime() << " sec)" << std::endl;
#endif

#ifdef MFEM_TEMPLATE_ENABLE_SERIALIZE
   r2 = 0.0; // touch 'r2' to make sure the memory is actually allocated
   tic_toc.Clear();
   tic_toc.Start();
   templ_oper.ElementwiseExtractAssembleTest(x, r2);
   tic_toc.Stop();
   out << "ExtractAssembleTest() time       = "
       << tic_toc.RealTime() << " sec ("
       << tic_toc.UserTime() << " sec)" << std::endl;
   tic_toc.Clear();
   tic_toc.Start();
   templ_oper.ElementwiseExtractAssembleTest(x, r2);
   tic_toc.Stop();
   out << "ExtractAssembleTest() time       = "
       << tic_toc.RealTime() << " sec ("
       << tic_toc.UserTime() << " sec)" << std::endl;

   // Disable this section to test the correctness of templ_oper.Assemble()
   // if (0)
   {
      Vector s_nodes;
      tic_toc.Clear();
      tic_toc.Start();
      templ_oper.SerializeNodesTest(s_nodes);
      tic_toc.Stop();
      out << "SerializeNodesTest() time        = "
          << tic_toc.RealTime() << " sec ("
          << tic_toc.UserTime() << " sec)" << std::endl;
      tic_toc.Clear();
      tic_toc.Start();
      templ_oper.SerializeNodesTest(s_nodes);
      tic_toc.Stop();
      out << "SerializeNodesTest() time        = "
          << tic_toc.RealTime() << " sec ("
          << tic_toc.UserTime() << " sec)" << std::endl;
      out << "Size of serialized nodes         = "
          << sizeof(double)/MiB*s_nodes.Size() << " MiB ("
          << s_nodes.Size() << " doubles)" << std::endl;
      double s1 = sizeof(double)/MiB*mesh_fes.GetVSize();
      const Table &mesh_el_dof = mesh_fes.GetElementToDofTable();
      double s2 = sizeof(int)/MiB*mesh_el_dof.Size_of_connections();
      out << "Size of mesh nodes + mesh el_dof = "
          << s1+s2 << " MiB (" << s1 << " MiB nodes + "
          << s2 << " MiB el_dof)" << std::endl;

      MFEM_FLOPS_RESET();
      tic_toc.Clear();
      tic_toc.Start();
      templ_oper.AssembleFromSerializedNodesTest(s_nodes);
      tic_toc.Stop();
      out << "Assemble (SerializedNodes) flops = "
          << MFEM_FLOPS_GET()*1e-9 << " GFlops" << std::endl;
      out << "Assemble (SerializedNodes) time  = "
          << tic_toc.RealTime() << " sec ("
          << tic_toc.UserTime() << " sec)" << std::endl;
      tic_toc.Clear();
      tic_toc.Start();
      templ_oper.AssembleFromSerializedNodesTest(s_nodes);
      tic_toc.Stop();
      out << "Assemble (SerializedNodes) time  = "
          << tic_toc.RealTime() << " sec ("
          << tic_toc.UserTime() << " sec)" << std::endl;
   }
#endif // MFEM_TEMPLATE_ENABLE_SERIALIZE

   r2 = 0.0; // touch 'r2' to make sure the memory is actually allocated
   MFEM_FLOPS_RESET();
   tic_toc.Clear();
   tic_toc.Start();
   templ_oper.Mult(x, r2);
   tic_toc.Stop();
   rmops = (sizeof(double)/MiB*mesh->GetNE()*int_rule_t::qpts + // qpts data
            sizeof(double)/MiB*fes.GetNDofs() + // read input
            sizeof(int)/MiB*sol_fe_t::dofs*mesh->GetNE() + // sol. el_dof_J
            sizeof(double)/MiB*fes.GetNDofs() // read output for '+=' update
           );
   wmops = (sizeof(double)/MiB*fes.GetNDofs() + // output = 0.0
            sizeof(double)/MiB*fes.GetNDofs() // write output for '+=' update
           );
   out << "Assembled Mult() mem ops         = " << rmops+wmops << " MiB = "
       << rmops << " read + " << wmops << " write\n";
   out << "Assembled Mult() flops           = "
       << MFEM_FLOPS_GET()*1e-9 << " GFlops" << std::endl;
   out << "Assembled Mult() time            = "
       << tic_toc.RealTime() << " sec ("
       << tic_toc.UserTime() << " sec)" << std::endl;
   tic_toc.Clear();
   tic_toc.Start();
   templ_oper.Mult(x, r2);
   tic_toc.Stop();
   out << "Assembled Mult() time            = "
       << tic_toc.RealTime() << " sec ("
       << tic_toc.UserTime() << " sec)" << std::endl;
#if 1
   tic_toc.Clear();
   tic_toc.Start();
   templ_oper.Mult(x, r2);
   tic_toc.Stop();
   out << "Assembled Mult() time            = "
       << tic_toc.RealTime() << " sec ("
       << tic_toc.UserTime() << " sec)" << std::endl;
#endif

#if 1
   r2 = 0.0; // touch 'r2' to make sure the memory is actually allocated
   MFEM_FLOPS_RESET();
   const int batch = 4;
   tic_toc.Clear();
   tic_toc.Start();
   templ_oper.template MultAssembled<batch>(x, r2);
   tic_toc.Stop();
   out << "MultAssembled<" << std::setw(3) << batch
       << "> flops         = "
       << MFEM_FLOPS_GET()*1e-9 << " GFlops" << std::endl;
   out << "MultAssembled<" << std::setw(3) << batch
       << "> time          = "
       << tic_toc.RealTime() << " sec ("
       << tic_toc.UserTime() << " sec)" << std::endl;
   tic_toc.Clear();
   tic_toc.Start();
   templ_oper.template MultAssembled<batch>(x, r2);
   tic_toc.Stop();
   out << "MultAssembled<" << std::setw(3) << batch
       << "> time          = "
       << tic_toc.RealTime() << " sec ("
       << tic_toc.UserTime() << " sec)" << std::endl;
#endif

#ifdef MFEM_TEMPLATE_ENABLE_SERIALIZE
   if (1)
   {
      Vector sx, sy;
      templ_oper.Serialize(x, sx);
      out << "Serialized solution vector size  = "
          << sizeof(double)/MiB*sx.Size() << " MiB ("
          << sx.Size() << " doubles)" << std::endl;
      sy.SetSize(sx.Size());
      MFEM_FLOPS_RESET();
      tic_toc.Clear();
      tic_toc.Start();
      templ_oper.MultAssembledSerialized(sx, sy);
      tic_toc.Stop();
      out << "MultAssembled() (Seria...) flops = "
          << MFEM_FLOPS_GET()*1e-9 << " GFlops" << std::endl;
      out << "MultAssembled() (Seria...) time  = "
          << tic_toc.RealTime() << " sec ("
          << tic_toc.UserTime() << " sec)" << std::endl;
      tic_toc.Clear();
      tic_toc.Start();
      templ_oper.MultAssembledSerialized(sx, sy);
      tic_toc.Stop();
      out << "MultAssembled() (Seria...) time  = "
          << tic_toc.RealTime() << " sec ("
          << tic_toc.UserTime() << " sec)" << std::endl;
      tic_toc.Clear();
      tic_toc.Start();
      templ_oper.MultAssembledSerialized(sx, sy);
      tic_toc.Stop();
      out << "MultAssembled() (Seria...) time  = "
          << tic_toc.RealTime() << " sec ("
          << tic_toc.UserTime() << " sec)" << std::endl;
   }
#endif // MFEM_TEMPLATE_ENABLE_SERIALIZE

#if 1
   MFEM_FLOPS_RESET();
   tic_toc.Clear();
   tic_toc.Start();
   templ_oper.AssembleMatrix(A);
   A.Finalize();
   tic_toc.Stop();
   out << "AssembleMatrix() flops           = "
       << MFEM_FLOPS_GET()*1e-9 << " GFlops" << std::endl;
   out << "AssembleMatrix() + Finalize()    = "
       << tic_toc.RealTime() << " sec ("
       << tic_toc.UserTime() << " sec)" << std::endl;

   r3 = 0.0; // touch 'r3' to make sure the memory is actually allocated
   tic_toc.Clear();
   tic_toc.Start();
   A.Mult(x, r3);
   tic_toc.Stop();
   out << "Assembled Matrix Mult() time     = "
       << tic_toc.RealTime() << " sec ("
       << tic_toc.UserTime() << " sec)" << std::endl;
   out << "Avgerage number of nonzeros per row = "
       << double(A.NumNonZeroElems())/size << std::endl;
   double s1 = sizeof(double)/MiB*A.NumNonZeroElems();
   double s2 = sizeof(int)/MiB*A.NumNonZeroElems();
   double s3 = sizeof(int)/MiB*(A.Size()+1);
   out << "Size of all csr data             =    "
       << s1 << " (A) + " << s2 << " (J) + "
       << s3 << " (I) = " << s1+s2+s3 << " MiB" << std::endl;
#endif

#if 1
   MFEM_FLOPS_RESET();
   tic_toc.Clear();
   tic_toc.Start();
   templ_oper.AssembleMatrix(Ae);
   tic_toc.Stop();
   out << "AssembleMatrix() (Dens...) flops = "
       << MFEM_FLOPS_GET()*1e-9 << " GFlops" << std::endl;
   out << "AssembleMatrix() (DenseTensor)   = "
       << tic_toc.RealTime() << " sec ("
       << tic_toc.UserTime() << " sec)" << std::endl;

   r4 = 0.0; // touch 'r4' to make sure the memory is actually allocated
   MFEM_FLOPS_RESET();
   tic_toc.Clear();
   tic_toc.Start();
   r4 = 0.0;
   templ_oper.AddMult(Ae, x, r4);
   tic_toc.Stop();
   out << "DenseTensor Mult() flops         = "
       << MFEM_FLOPS_GET()*1e-9 << " GFlops" << std::endl;
   out << "DenseTensor Mult() time          = "
       << tic_toc.RealTime() << " sec ("
       << tic_toc.UserTime() << " sec)" << std::endl;
   s1 = sizeof(double)/MiB*Ae.SizeI()*Ae.SizeJ()*Ae.SizeK();
   s2 = sizeof(int)/MiB*Ae.SizeJ()*Ae.SizeK();
   out << "Size of all data                 =    "
       << s1 << " (Ae) + " << s2 << " (el_dof) = "
       << s1+s2 << " MiB" << std::endl;

   tic_toc.Clear();
   tic_toc.Start();
   bilin_form.Assemble();
   bilin_form.Finalize();
   tic_toc.Stop();
   out << "CSR Assemble() + Finalize() time = "
       << tic_toc.RealTime() << " sec ("
       << tic_toc.UserTime() << " sec)" << std::endl;
#endif

#if 1
   Vector r_mfem(bilin_form.Height());
   r_mfem = 0.0; // touch 'r_mfem' to make sure the memory is allocated

   tic_toc.Clear();
   tic_toc.Start();
   bilin_form.Mult(x, r_mfem);
   tic_toc.Stop();
   out << "CSR Mult() time                  = "
       << tic_toc.RealTime() << " sec ("
       << tic_toc.UserTime() << " sec)" << std::endl;

   int nnz = bilin_form.SpMat().NumNonZeroElems();
   out << "CSR avg. number of nonzeros per row = "
       << double(nnz)/r_mfem.Size() << std::endl;

   r1 -= r_mfem;
   r2 -= r_mfem;
   r3 -= r_mfem;
   r4 -= r_mfem;

   out << "\n max residual norm = " << r1.Normlinf()
       << " (un-assembled)\n";
   out << " max residual norm = " << r2.Normlinf()
       << " (partially assembled)\n";
   out << " max residual norm = " << r3.Normlinf()
       << " (assembled matrix)\n";
   out << " max residual norm = " << r4.Normlinf()
       << " (DenseTensor)\n" << std::endl;
#endif

   delete mesh;
}

template <>
struct TOperator<0>
{
   template <typename coeff_t>
   struct Integrator
   {
      typedef TIntegrator<coeff_t,TMassKernel> type;
   };

   typedef MassIntegrator Integrator_type;

   static const char Name[];
};

const char TOperator<0>::Name[] = "Mass";

template <>
struct TOperator<1>
{
   template <typename coeff_t>
   struct Integrator
   {
      typedef TIntegrator<coeff_t,TDiffusionKernel> type;
   };

   typedef DiffusionIntegrator Integrator_type;

   static const char Name[];
};

const char TOperator<1>::Name[] = "Diffusion";


int main(int argc, char *argv[])
{
#ifdef MFEM_USE_MPI
   int myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);
   std::ostringstream fname;
   fname << "test_output_" << std::setw(5) << std::setfill('0') << myid;
   std::ofstream out(fname.str().c_str());
#else
   std::ostream &out = std::cout;
#endif

#if 1
   const int p = 2;
   const int mesh_p = 1;
#else
   const int p = 2;
   const int mesh_p = 2;
#endif
   const int ir_order = 4*p-1;
   // const int ir_order = 2*p+2;
   // const int ir_order = 2*p;
   int mesh_size = 1; // 0 - small, 1 - medium, 2 - large

   const int oper = 1; // 0 - mass, 1 - diffusion

   std::srand(std::time(0));

   Test<Geometry::SEGMENT,     p, ir_order, mesh_p, oper>(mesh_size, out);
   Test<Geometry::SEGMENT,     p, ir_order, mesh_p, oper>(mesh_size, out);

   Test<Geometry::TRIANGLE,    p, ir_order, mesh_p, oper>(mesh_size, out);
   Test<Geometry::TRIANGLE,    p, ir_order, mesh_p, oper>(mesh_size, out);

   Test<Geometry::SQUARE,      p, ir_order, mesh_p, oper>(mesh_size, out);
   Test<Geometry::SQUARE,      p, ir_order, mesh_p, oper>(mesh_size, out);

   Test<Geometry::TETRAHEDRON, p, ir_order, mesh_p, oper>(mesh_size, out);
   Test<Geometry::TETRAHEDRON, p, ir_order, mesh_p, oper>(mesh_size, out);

   Test<Geometry::CUBE,        p, ir_order, mesh_p, oper>(mesh_size, out);
   Test<Geometry::CUBE,        p, ir_order, mesh_p, oper>(mesh_size, out);

#ifdef MFEM_USE_MPI
   MPI_Finalize();
#endif

   return 0;
}
