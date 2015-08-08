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
#include "mass_operator.hpp"

#include "mfem.hpp"

#include <iostream>
#include <ctime>

using namespace mfem;

template <Geometry::Type GEOM, int P, int IR_ORDER, int MESH_ORDER>
void Test(int mesh_size)
{
   std::cout << "-----------------------------------------------\n"
             << "GEOMETRY   = " << Geometry::Name[GEOM] << "\n"
             << "ORDER      = " << P << "\n"
             << "IR_ORDER   = " << IR_ORDER << "\n"
             << "MESH_ORDER = " << MESH_ORDER << "\n"
             << "-----------------------------------------------"
             << std::endl;
   typedef H1_FiniteElement<GEOM, P> h1_fe;

   typedef TIntegrationRule<GEOM, IR_ORDER> int_rule_1;
   // typedef GenericIntegrationRule<GEOM, int_rule_1::qpts, IR_ORDER> int_rule_2;

   typedef int_rule_1 int_rule;
   // typedef int_rule_2 int_rule;

   H1_FiniteElement_Basis::Type basis_type;
   basis_type = H1_FiniteElement_Basis::GaussLobatto;
   // basis_type = H1_FiniteElement_Basis::Positive;
   h1_fe fe(basis_type);

   int nx = 1, ny = 1, nz = 1;
   switch (h1_fe::dim)
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
            case 2: nx = 2400/P; ny = 3200/P; break;
         }
         break;
      case 3:
         switch (mesh_size)
         {
            case 0: nx =  20/P; ny =  12/P; nz =   8/P; break;
            case 1: nx = 200/P; ny = 120/P; nz =  80/P; break;
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
   H1_FECollection mesh_fec(mesh_order, h1_fe::dim);
   FiniteElementSpace mesh_fes(mesh, &mesh_fec, h1_fe::dim);
   mesh->SetNodalFESpace(&mesh_fes);
   typedef H1_FiniteElement<GEOM, mesh_order> mesh_fe;

   const double MiB = 1024.*1024.;
   std::cout << "Number of mesh node dofs = " << mesh_fe::dim*mesh_fe::dofs
             << " per element" << std::endl;
   std::cout << "Number of solution dofs  = " << h1_fe::dofs
             << " per element" << std::endl;
   std::cout << "Number of quadr. points  = "
             << int_rule::qpts << " per element" << std::endl;
   std::cout << "Number of mesh elements  = " << mesh->GetNE() << std::endl;
   std::cout << "Number of mesh node dofs = " << mesh_fes.GetVSize()
             << " total (" << sizeof(double)/MiB*mesh_fes.GetVSize() << " MiB)"
             << std::endl;
   H1_FECollection fec(P, h1_fe::dim);
   FiniteElementSpace fes(mesh, &fec);
   std::cout << "Number of solution dofs  = " << fes.GetNDofs()
             << " total (" << sizeof(double)/MiB*fes.GetNDofs() << " MiB)"
             << std::endl;
   std::cout << "Number of quadr. points  = "
             << mesh->GetNE()*int_rule::qpts << " total ("
             << sizeof(double)/MiB*mesh->GetNE()*int_rule::qpts << " MiB)"
             << std::endl;
   GridFunction x(&fes);

   typedef H1_FiniteElementSpace<h1_fe> spaceFES;

   typedef H1_FiniteElementSpace<mesh_fe> meshFES;
   typedef VectorLayout<Ordering::byNODES, mesh_fe::dim> node_layout;

   typedef TMassOperator<meshFES, node_layout, spaceFES, int_rule>
   mass_operator_type;

   mass_operator_type mass_oper(fes);
   int size = mass_oper.Height();
   Vector r1(size), r2(size), r3(size), r4(size);

   BilinearForm mass_form(&fes);
   MassIntegrator *mass_int = new MassIntegrator;
   mass_int->SetIntRule(&int_rule::GetIntRule());
   mass_form.AddDomainIntegrator(mass_int);
#if 1
   mass_form.UsePrecomputedSparsity();
   mass_form.AllocateMatrix();
#endif

   SparseMatrix A(mass_form.SpMat());
   DenseTensor Ae(h1_fe::dofs, h1_fe::dofs, mesh->GetNE());

   x.Randomize();

   r1 = 0.0; // touch 'r1' to make sure the memory is actually allocated
   tic_toc.Clear();
   tic_toc.Start();
   mass_oper.Mult(x, r1);
   tic_toc.Stop();
   std::cout << "Unassembled Mult() time          = "
             << tic_toc.RealTime() << " sec ("
             << tic_toc.UserTime() << " sec)" << std::endl;
   tic_toc.Clear();
   tic_toc.Start();
   mass_oper.Mult(x, r1);
   tic_toc.Stop();
   std::cout << "Unassembled Mult() time          = "
             << tic_toc.RealTime() << " sec ("
             << tic_toc.UserTime() << " sec)" << std::endl;
#if 1
   tic_toc.Clear();
   tic_toc.Start();
   mass_oper.Mult(x, r1);
   tic_toc.Stop();
   std::cout << "Unassembled Mult() time          = "
             << tic_toc.RealTime() << " sec ("
             << tic_toc.UserTime() << " sec)" << std::endl;
#endif

   tic_toc.Clear();
   tic_toc.Start();
   mass_oper.Assemble();
   tic_toc.Stop();
   std::cout << "Assemble() time                  = "
             << tic_toc.RealTime() << " sec ("
             << tic_toc.UserTime() << " sec)" << std::endl;
#if 1
   tic_toc.Clear();
   tic_toc.Start();
   mass_oper.Assemble();
   tic_toc.Stop();
   std::cout << "Assemble() time                  = "
             << tic_toc.RealTime() << " sec ("
             << tic_toc.UserTime() << " sec)" << std::endl;
#endif
#if 1
   tic_toc.Clear();
   tic_toc.Start();
   mass_oper.Assemble();
   tic_toc.Stop();
   std::cout << "Assemble() time                  = "
             << tic_toc.RealTime() << " sec ("
             << tic_toc.UserTime() << " sec)" << std::endl;
#endif

   r2 = 0.0; // touch 'r2' to make sure the memory is actually allocated
   tic_toc.Clear();
   tic_toc.Start();
   mass_oper.ElementwiseExtractAssembleTest(x, r2);
   tic_toc.Stop();
   std::cout << "ExtractAssembleTest() time       = "
             << tic_toc.RealTime() << " sec ("
             << tic_toc.UserTime() << " sec)" << std::endl;
   tic_toc.Clear();
   tic_toc.Start();
   mass_oper.ElementwiseExtractAssembleTest(x, r2);
   tic_toc.Stop();
   std::cout << "ExtractAssembleTest() time       = "
             << tic_toc.RealTime() << " sec ("
             << tic_toc.UserTime() << " sec)" << std::endl;

   {
      Vector s_nodes;
      tic_toc.Clear();
      tic_toc.Start();
      mass_oper.SerializeNodesTest(s_nodes);
      tic_toc.Stop();
      std::cout << "SerializeNodesTest() time        = "
                << tic_toc.RealTime() << " sec ("
                << tic_toc.UserTime() << " sec)" << std::endl;
      tic_toc.Clear();
      tic_toc.Start();
      mass_oper.SerializeNodesTest(s_nodes);
      tic_toc.Stop();
      std::cout << "SerializeNodesTest() time        = "
                << tic_toc.RealTime() << " sec ("
                << tic_toc.UserTime() << " sec)" << std::endl;
      std::cout << "Size of serialized nodes         = "
                << sizeof(double)/MiB*s_nodes.Size() << " MiB ("
                << s_nodes.Size() << " doubles)" << std::endl;
      double s1 = sizeof(double)/MiB*mesh_fes.GetVSize();
      const Table &mesh_el_dof = mesh_fes.GetElementToDofTable();
      double s2 = sizeof(int)/MiB*mesh_el_dof.Size_of_connections();
      std::cout << "Size of mesh nodes + mesh el_dof = "
                << s1+s2 << " MiB (" << s1 << " MiB nodes + "
                << s2 << " MiB el_dof)" << std::endl;

      tic_toc.Clear();
      tic_toc.Start();
      mass_oper.AssembleFromSerializedNodesTest(s_nodes);
      tic_toc.Stop();
      std::cout << "Assemble (SerializedNodes) time  = "
                << tic_toc.RealTime() << " sec ("
                << tic_toc.UserTime() << " sec)" << std::endl;
      tic_toc.Clear();
      tic_toc.Start();
      mass_oper.AssembleFromSerializedNodesTest(s_nodes);
      tic_toc.Stop();
      std::cout << "Assemble (SerializedNodes) time  = "
                << tic_toc.RealTime() << " sec ("
                << tic_toc.UserTime() << " sec)" << std::endl;
   }

   r2 = 0.0; // touch 'r2' to make sure the memory is actually allocated
   tic_toc.Clear();
   tic_toc.Start();
   mass_oper.Mult(x, r2);
   tic_toc.Stop();
   std::cout << "Assembled Mult() time            = "
             << tic_toc.RealTime() << " sec ("
             << tic_toc.UserTime() << " sec)" << std::endl;
   tic_toc.Clear();
   tic_toc.Start();
   mass_oper.Mult(x, r2);
   tic_toc.Stop();
   std::cout << "Assembled Mult() time            = "
             << tic_toc.RealTime() << " sec ("
             << tic_toc.UserTime() << " sec)" << std::endl;
#if 1
   tic_toc.Clear();
   tic_toc.Start();
   mass_oper.Mult(x, r2);
   tic_toc.Stop();
   std::cout << "Assembled Mult() time            = "
             << tic_toc.RealTime() << " sec ("
             << tic_toc.UserTime() << " sec)" << std::endl;
#endif

#if 1
   r2 = 0.0; // touch 'r2' to make sure the memory is actually allocated
   const int batch = 4;
   tic_toc.Clear();
   tic_toc.Start();
   mass_oper.template MultAssembled<batch>(x, r2);
   tic_toc.Stop();
   std::cout << "MultAssembled<" << std::setw(3) << batch
             << "> time          = "
             << tic_toc.RealTime() << " sec ("
             << tic_toc.UserTime() << " sec)" << std::endl;
   tic_toc.Clear();
   tic_toc.Start();
   mass_oper.template MultAssembled<batch>(x, r2);
   tic_toc.Stop();
   std::cout << "MultAssembled<" << std::setw(3) << batch
             << "> time          = "
             << tic_toc.RealTime() << " sec ("
             << tic_toc.UserTime() << " sec)" << std::endl;
#endif

   if (1)
   {
      Vector sx, sy;
      mass_oper.Serialize(x, sx);
      std::cout << "Serialized solution vector size = "
                << sizeof(double)/MiB*sx.Size() << " MiB ("
                << sx.Size() << " doubles)" << std::endl;
      sy.SetSize(sx.Size());
      tic_toc.Clear();
      tic_toc.Start();
      mass_oper.MultAssembledSerialized(sx, sy);
      tic_toc.Stop();
      std::cout << "MultAssembled() (Seria...) time = "
                << tic_toc.RealTime() << " sec ("
                << tic_toc.UserTime() << " sec)" << std::endl;
      tic_toc.Clear();
      tic_toc.Start();
      mass_oper.MultAssembledSerialized(sx, sy);
      tic_toc.Stop();
      std::cout << "MultAssembled() (Seria...) time = "
                << tic_toc.RealTime() << " sec ("
                << tic_toc.UserTime() << " sec)" << std::endl;
      tic_toc.Clear();
      tic_toc.Start();
      mass_oper.MultAssembledSerialized(sx, sy);
      tic_toc.Stop();
      std::cout << "MultAssembled() (Seria...) time = "
                << tic_toc.RealTime() << " sec ("
                << tic_toc.UserTime() << " sec)" << std::endl;
   }

   tic_toc.Clear();
   tic_toc.Start();
   mass_oper.AssembleMatrix(A);
   A.Finalize();
   tic_toc.Stop();
   std::cout << "AssembleMatrix() + Finalize()    = "
             << tic_toc.RealTime() << " sec ("
             << tic_toc.UserTime() << " sec)" << std::endl;

   r3 = 0.0; // touch 'r3' to make sure the memory is actually allocated
   tic_toc.Clear();
   tic_toc.Start();
   A.Mult(x, r3);
   tic_toc.Stop();
   std::cout << "Assembled Matrix Mult() time     = "
             << tic_toc.RealTime() << " sec ("
             << tic_toc.UserTime() << " sec)" << std::endl;
   std::cout << "Avgerage number of nonzeros per row = "
             << double(A.NumNonZeroElems())/size << std::endl;
   double s1 = sizeof(double)/MiB*A.NumNonZeroElems();
   double s2 = sizeof(int)/MiB*A.NumNonZeroElems();
   double s3 = sizeof(int)/MiB*(A.Size()+1);
   std::cout << "Size of all csr data             =    "
             << s1 << " (A) + " << s2 << " (J) + "
             << s3 << " (I) = " << s1+s2+s3 << " MiB" << std::endl;

   tic_toc.Clear();
   tic_toc.Start();
   mass_oper.AssembleMatrix(Ae);
   tic_toc.Stop();
   std::cout << "AssembleMatrix() (DenseTensor)   = "
             << tic_toc.RealTime() << " sec ("
             << tic_toc.UserTime() << " sec)" << std::endl;

   r4 = 0.0; // touch 'r4' to make sure the memory is actually allocated
   tic_toc.Clear();
   tic_toc.Start();
   r4 = 0.0;
   mass_oper.AddMult(Ae, x, r4);
   tic_toc.Stop();
   std::cout << "DenseTensor Mult() time          = "
             << tic_toc.RealTime() << " sec ("
             << tic_toc.UserTime() << " sec)" << std::endl;
   s1 = sizeof(double)/MiB*Ae.SizeI()*Ae.SizeJ()*Ae.SizeK();
   s2 = sizeof(int)/MiB*Ae.SizeJ()*Ae.SizeK();
   std::cout << "Size of all data                 =    "
             << s1 << " (Ae) + " << s2 << " (el_dof) = "
             << s1+s2 << " MiB" << std::endl;

   tic_toc.Clear();
   tic_toc.Start();
   mass_form.Assemble();
   mass_form.Finalize();
   tic_toc.Stop();
   std::cout << "CSR Assemble() + Finalize() time = "
             << tic_toc.RealTime() << " sec ("
             << tic_toc.UserTime() << " sec)" << std::endl;

   Vector r_mfem(mass_form.Height());
   r_mfem = 0.0; // touch 'r_mfem' to make sure the memory is allocated

   tic_toc.Clear();
   tic_toc.Start();
   mass_form.Mult(x, r_mfem);
   tic_toc.Stop();
   std::cout << "CSR Mult() time                  = "
             << tic_toc.RealTime() << " sec ("
             << tic_toc.UserTime() << " sec)" << std::endl;

   int nnz = mass_form.SpMat().NumNonZeroElems();
   std::cout << "CSR avg. number of nonzeros per row = "
             << double(nnz)/r_mfem.Size() << std::endl;

   r1 -= r_mfem;
   r2 -= r_mfem;
   r3 -= r_mfem;
   r4 -= r_mfem;

   std::cout << "\n max residual norm = " << r1.Normlinf()
             << " (un-assembled)\n";
   std::cout << " max residual norm = " << r2.Normlinf()
             << " (partially assembled)\n";
   std::cout << " max residual norm = " << r3.Normlinf()
             << " (assembled matrix)\n";
   std::cout << " max residual norm = " << r4.Normlinf()
             << " (DenseTensor)\n" << std::endl;

   delete mesh;
}

int main()
{
#if 1
   const int p = 2;
   const int mesh_p = 1;
#else
   const int p = 2;
   const int mesh_p = 2;
#endif
   const int ir_order = 4*p-1;
   int mesh_size = 1; // 0 - small, 1 - medium, 2 - large

   std::srand(std::time(0));

   // Test<Geometry::SEGMENT,     p, ir_order, mesh_p>(mesh_size);
   // Test<Geometry::SEGMENT,     p, ir_order, mesh_p>(mesh_size);

   // Test<Geometry::TRIANGLE,    p, ir_order, mesh_p>(mesh_size);

   Test<Geometry::SQUARE,      p, ir_order, mesh_p>(mesh_size);
   Test<Geometry::SQUARE,      p, ir_order, mesh_p>(mesh_size);

   // Test<Geometry::TETRAHEDRON, p, ir_order, mesh_p>(mesh_size);
   // Test<Geometry::TETRAHEDRON, p, ir_order, mesh_p>(mesh_size);

   // Test<Geometry::CUBE,        p, ir_order, mesh_p>(mesh_size);
   // Test<Geometry::CUBE,        p, ir_order, mesh_p>(mesh_size);

   return 0;
}
