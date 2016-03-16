
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

#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   // Define template parameters for optimized build.
   const Geometry::Type geom     = Geometry::CUBE;
   const int            mesh_p   = 3;
   const int            sol_p    = 3;
   const int            ir_order = 8;

   typedef H1_FiniteElement<geom,mesh_p>        mesh_fe_type;
   typedef H1_FiniteElementSpace<mesh_fe_type>  mesh_fes_type;
   typedef VectorLayout<Ordering::byNODES,
           mesh_fe_type::dim>                   node_layout;
   typedef TMesh<mesh_fes_type,node_layout>     mesh_type;
   typedef H1_FiniteElement<geom,sol_p>         sol_fe_type;
   typedef H1_FiniteElementSpace<sol_fe_type>   sol_fes_type;
#if 1
   typedef TIntegrationRule<geom,ir_order,
           double>                              int_rule_type;
#else
   typedef TIntegrationRule<geom,ir_order>      int_rule_1_type;
   typedef GenericIntegrationRule<geom,int_rule_1_type::qpts,
           ir_order> int_rule_type;
#endif
   typedef TConstantCoefficient<double> coeff_t;
   typedef TIntegrator<coeff_t,TDiffusionKernel> integ_t;
   typedef TBilinearForm<mesh_type,sol_fes_type,ScalarLayout,int_rule_type,
           integ_t,double,double> oper_type;

   // 1. Parse command-line options.
   const char *mesh_file = "../data/star.mesh";
   int order = 1;
   bool visualization = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   // 2. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral, hexahedral, surface and volume meshes with
   //    the same code.
   Mesh *mesh;
   ifstream imesh(mesh_file);
   if (!imesh)
   {
      cerr << "\nCan not open mesh file: " << mesh_file << '\n' << endl;
      return 2;
   }
   mesh = new Mesh(imesh, 1, 1);
   imesh.close();
   int dim = mesh->Dimension();
   if (mesh_type::MatchesGeometry(*mesh))
   {
      if (!mesh_type::MatchesNodes(*mesh))
      {
         cout << "Switching the mesh curvature to match the optimized values: ";
         const char *mesh_fec_name = "(none)";
         if (mesh->GetNodes())
         {
            mesh_fec_name = mesh->GetNodes()->FESpace()->FEColl()->Name();
         }
         cout << '\'' << mesh_fec_name << "' --> " << flush;
         H1_FECollection *new_mesh_fec = new H1_FECollection(mesh_p, dim);
         FiniteElementSpace *new_mesh_fes =
            new FiniteElementSpace(mesh, new_mesh_fec, dim);
         mesh->SetNodalFESpace(new_mesh_fes);
         mesh->GetNodes()->MakeOwner(new_mesh_fec);
         cout << '\'' << new_mesh_fec->Name() << '\'' << endl;
      }
   }
   else
   {
      cout << "The given mesh does not match the optimized 'geom' parameter.\n"
           << "Recompile with suitable 'geom' value." << endl;
      delete mesh;
      return 3;
   }

   // 3. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement. We choose 'ref_levels' to be the
   //    largest number that gives a final mesh with no more than 50,000
   //    elements.
   {
      int ref_levels =
         (int)floor(log(50000./mesh->GetNE())/log(2.)/dim);
      for (int l = 0; l < ref_levels; l++)
      {
         mesh->UniformRefinement();
      }
   }

   // 4. Define a finite element space on the mesh. Here we use continuous
   //    Lagrange finite elements of the specified order. If order < 1, we
   //    instead use an isoparametric/isogeometric space.
   FiniteElementCollection *fec;
   if (order > 0)
   {
      fec = new H1_FECollection(order, dim);
   }
   else if (mesh->GetNodes())
   {
      fec = mesh->GetNodes()->OwnFEC();
   }
   else
   {
      fec = new H1_FECollection(order = 1, dim);
   }
   FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec);
   cout << "Number of unknowns: " << fespace->GetVSize() << endl;
   if (!sol_fes_type::Matches(*fespace))
   {
      cout << "The given order does not match the optimized parameter.\n"
           << "Recompile with suitable 'sol_p' value." << endl;
      delete fespace;
      delete fec;
      delete mesh;
      return 4;
   }

   // 5. Set up the linear form b(.) which corresponds to the right-hand side of
   //    the FEM linear system, which in this case is (1,phi_i) where phi_i are
   //    the basis functions in the finite element fespace.
   LinearForm *b = new LinearForm(fespace);
   ConstantCoefficient one(1.0);
   b->AddDomainIntegrator(new DomainLFIntegrator(one));
   b->Assemble();

   // 6. Define the solution vector x as a finite element grid function
   //    corresponding to fespace. Initialize x with initial guess of zero,
   //    which satisfies the boundary conditions.
   GridFunction x(fespace);
   x = 0.0;

   // 7. Set up the bilinear form a(.,.) on the finite element space
   //    corresponding to the Laplacian operator -Delta, by adding the Diffusion
   //    domain integrator and imposing homogeneous Dirichlet boundary
   //    conditions. The boundary conditions are implemented by marking all the
   //    boundary attributes from the mesh as essential (Dirichlet). After
   //    assembly and finalizing we extract the corresponding sparse matrix A.
   BilinearForm *a = new BilinearForm(fespace);
   a->AddDomainIntegrator(new DiffusionIntegrator(one));
   if (0)
   {
      a->UsePrecomputedSparsity();
      a->AllocateMatrix();
      cout << "Assembling the matrix ..." << flush;
      tic_toc.Clear();
      tic_toc.Start();
      a->Assemble();
      tic_toc.Stop();
      cout << " done, " << tic_toc.RealTime() << " sec." << endl;
   }
   else
   {
      oper_type templ_oper(integ_t(coeff_t(1.0)), *fespace);
      a->UsePrecomputedSparsity();
      a->AllocateMatrix();
      cout << "Assembly using integration rule with " << int_rule_type::qpts
           << " points ..." << flush;
      tic_toc.Clear();
      tic_toc.Start();
      templ_oper.AssembleMatrix(a->SpMat());
      tic_toc.Stop();
      cout << " done, " << tic_toc.RealTime() << " sec." << endl;
   }
   Array<int> ess_bdr(mesh->bdr_attributes.Max());
   ess_bdr = 1;
   a->EliminateEssentialBC(ess_bdr, x, *b);
   a->Finalize();
   const SparseMatrix &A = a->SpMat();

#ifndef MFEM_USE_SUITESPARSE
   // 8. Define a simple symmetric Gauss-Seidel preconditioner and use it to
   //    solve the system Ax=b with PCG.
   GSSmoother M(A);
   PCG(A, M, *b, x, 1, 200, 1e-12, 0.0);
#else
   // 8. If MFEM was compiled with SuiteSparse, use UMFPACK to solve the system.
   UMFPackSolver umf_solver;
   umf_solver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
   umf_solver.SetOperator(A);
   umf_solver.Mult(*b, x);
#endif

#if 0
   // 9. Save the refined mesh and the solution. This output can be viewed later
   //    using GLVis: "glvis -m refined.mesh -g sol.gf".
   ofstream mesh_ofs("refined.mesh");
   mesh_ofs.precision(8);
   mesh->Print(mesh_ofs);
   ofstream sol_ofs("sol.gf");
   sol_ofs.precision(8);
   x.Save(sol_ofs);
#endif

   // 10. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "solution\n" << *mesh << x << flush;
   }

   // 11. Free the used memory.
   delete a;
   delete b;
   delete fespace;
   if (order > 0)
   {
      delete fec;
   }
   delete mesh;

   return 0;
}
