//                    MFEM Example 1 - High-Performance Version
//
// Compile with: make ex1
//
// Sample runs:  ex1 -perf -m ../../data/fichera.mesh
//               ex1 -perf -m ../../data/amr-hex.mesh -sc
//               ex1 -perf -m ../../data/ball-nurbs.mesh -sc
//               ex1 -perf -m ../../data/pipe-nurbs.mesh
//               ex1 -std -m ../../data/square-disc.mesh
//               ex1 -std -m ../../data/star.mesh
//               ex1 -std -m ../../data/escher.mesh
//               ex1 -std -m ../../data/square-disc-p2.vtk -o 2
//               ex1 -std -m ../../data/square-disc-p3.mesh -o 3
//               ex1 -std -m ../../data/square-disc-nurbs.mesh -o -1
//               ex1 -std -m ../../data/disc-nurbs.mesh -o -1
//               ex1 -std -m ../../data/pipe-nurbs.mesh -o -1
//               ex1 -std -m ../../data/star-surf.mesh
//               ex1 -std -m ../../data/square-disc-surf.mesh
//               ex1 -std -m ../../data/inline-segment.mesh
//               ex1 -std -m ../../data/amr-quad.mesh
//               ex1 -std -m ../../data/amr-hex.mesh
//               ex1 -std -m ../../data/fichera-amr.mesh
//               ex1 -std -m ../../data/mobius-strip.mesh
//               ex1 -std -m ../../data/mobius-strip.mesh -o -1 -sc
//
// Description:  This example code demonstrates the use of MFEM to define a
//               simple finite element discretization of the Laplace problem
//               -Delta u = 1 with homogeneous Dirichlet boundary conditions.
//               Specifically, we discretize using a FE space of the specified
//               order, or if order < 1 using an isoparametric/isogeometric
//               space (i.e. quadratic for quadratic curvilinear mesh, NURBS for
//               NURBS mesh, etc.)
//
//               The example highlights the use of mesh refinement, finite
//               element grid functions, as well as linear and bilinear forms
//               corresponding to the left-hand side and right-hand side of the
//               discrete linear system. We also cover the explicit elimination
//               of essential boundary conditions, static condensation, and the
//               optional connection to the GLVis tool for visualization.

#include "mfem-performance.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

// Define template parameters for optimized build.
const Geometry::Type geom         = Geometry::CUBE; // mesh elements  (default: hex)
const int            mesh_p       = 3;              // mesh curvature (default: 3)
const int            sol_p        = 3;              // solution order (default: 3)
const int            mesh_lor_p   = 1;              // mesh curvature (default: 3)
const int            sol_lor_p    = 1;              // solution order (default: 3)
const int            rdim         = Geometry::Constants<geom>::Dimension;
const int            ir_order     = 2*sol_p+rdim-1;
const int            ir_order_lor = 2*sol_lor_p+rdim-1;

// Static mesh type
typedef H1_FiniteElement<geom,mesh_p>         mesh_fe_t;
typedef H1_FiniteElementSpace<mesh_fe_t>      mesh_fes_t;
typedef TMesh<mesh_fes_t>                     mesh_t;

// Static solution finite element space type
typedef H1_FiniteElement<geom,sol_p>          sol_fe_t;
typedef H1_FiniteElementSpace<sol_fe_t>       sol_fes_t;

// Static quadrature, coefficient and integrator types
typedef TIntegrationRule<geom,ir_order>       int_rule_t;
typedef TConstantCoefficient<>                coeff_t;
typedef TIntegrator<coeff_t,TDiffusionKernel> integ_t;

// Static bilinear form type, combining the above types
typedef TBilinearForm<mesh_t,sol_fes_t,int_rule_t,integ_t> HPCBilinearForm;

// Low order refined types

// Static mesh type
typedef H1_FiniteElement<geom,mesh_lor_p>       mesh_lor_fe_t;
typedef H1_FiniteElementSpace<mesh_lor_fe_t>    mesh_lor_fes_t;
typedef TMesh<mesh_lor_fes_t>                   mesh_lor_t;

// Static solution finite element space type
typedef H1_FiniteElement<geom,sol_lor_p>        sol_fe_lor_t;
typedef H1_FiniteElementSpace<sol_fe_lor_t>     sol_fes_lor_t;

// Static quadrature, coefficient and integrator types
typedef TIntegrationRule<geom,ir_order_lor>     int_rule_lor_t;

// Static bilinear form type, combining the above types
typedef TBilinearForm<mesh_lor_t,sol_fes_lor_t,int_rule_lor_t,integ_t> HPCBilinearForm_lor;

double GaussianFunction(const Vector &x);

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../../data/fichera.mesh";
   const char *pc = "default";
   int order = sol_p;
   const char *basis_type = "G"; // Gauss-Lobatto
   bool static_cond = false;
   bool visualization = 1;
   bool perf = true;
   bool matrix_free = true;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&basis_type, "-b", "--basis-type",
                  "Basis: G - Gauss-Lobatto, P - Positive, U - Uniform");
   args.AddOption(&perf, "-perf", "--hpc-version", "-std", "--standard-version",
                  "Enable high-performance, tensor-based, assembly/evaluation.");
   args.AddOption(&matrix_free, "-mf", "--matrix-free", "-asm", "--assembly",
                  "Use matrix-free evaluation or efficient matrix assembly in "
                  "the high-performance version.");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&pc, "-pc", "--preconditioner",
                  "Preconditioner to use: `lor' for LOR BoomerAMG Prec.,"
                  "`dense' for Dense BoomerAMG Prec., `none'.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   if (static_cond && perf && matrix_free)
   {
      cout << "\nStatic condensation can not be used with matrix-free"
           " evaluation!\n" << endl;
      return 2;
   }
   if (!perf) { matrix_free = false; }
   args.PrintOptions(cout);

   enum PCType { NONE, LOR, DENSE };
   PCType pc_choice;
   if (!strcmp(pc, "dense")) { pc_choice = DENSE; }
   else if (!strcmp(pc, "lor")) { pc_choice = LOR; }
   else if (!strcmp(pc, "none")) { pc_choice = NONE; }
   else if (!strcmp(pc,"default"))
   {
      if (matrix_free)
      {
         pc_choice = NONE;
      }
      else
      {
         pc_choice = DENSE;
      }
   }
   else
   {
      mfem_error("Invalid Preconditioner specified");
      return 2;
   }

   // See class BasisType in fem/fe_coll.hpp for available basis types
   int basis = BasisType::GetType(basis_type[0]);
   cout << "Using " << BasisType::Name(basis) << " basis ..." << endl;

   // 2. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral, hexahedral, surface and volume meshes with
   //    the same code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   // 3. Check if the optimized version matches the given mesh
   if (perf)
   {
      cout << "High-performance version using integration rule with "
           << int_rule_t::qpts << " points ..." << endl;
      if (!mesh_t::MatchesGeometry(*mesh))
      {
         cout << "The given mesh does not match the optimized 'geom' parameter.\n"
              << "Recompile with suitable 'geom' value." << endl;
         delete mesh;
         return 3;
      }
      else if (!mesh_t::MatchesNodes(*mesh))
      {
         cout << "Switching the mesh curvature to match the "
              << "optimized value (order " << mesh_p << ") ..." << endl;
         mesh->SetCurvature(mesh_p, false, -1, Ordering::byNODES);
      }
   }

   // 4. Refine the mesh to increase the resolution. In this example we do
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

   // 5. Define a finite element space on the mesh. Here we use continuous
   //    Lagrange finite elements of the specified order. If order < 1, we
   //    instead use an isoparametric/isogeometric space.
   FiniteElementCollection *fec;
   if (order > 0)
   {
      fec = new H1_FECollection(order, dim, basis);
   }
   else if (mesh->GetNodes())
   {
      fec = mesh->GetNodes()->OwnFEC();
      cout << "Using isoparametric FEs: " << fec->Name() << endl;
   }
   else
   {
      fec = new H1_FECollection(order = 1, dim, basis);
   }
   FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec);

   FiniteElementSpace *fespace_lor = NULL;
   Operator *P = NULL, *R = NULL;
   if (pc_choice == LOR)
   {
      fespace->BuildDofToArrays();
      fespace_lor = fespace->LowOrderRefinement(sol_lor_p, P, R);
   }
   cout << "Number of finite element unknowns: "
        << fespace->GetTrueVSize() << endl;

   // 6. Check if the optimized version matches the given space
   if (perf && !sol_fes_t::Matches(*fespace))
   {
      cout << "The given order does not match the optimized parameter.\n"
           << "Recompile with suitable 'sol_p' value." << endl;
      delete fespace;
      delete fec;
      delete mesh;
      return 4;
   }

   // 7. Determine the list of true (i.e. conforming) essential boundary dofs.
   //    In this example, the boundary conditions are defined by marking all
   //    the boundary attributes from the mesh as essential (Dirichlet) and
   //    converting them to a list of true dofs.
   Array<int> ess_tdof_list;
   if (mesh->bdr_attributes.Size())
   {
      Array<int> ess_bdr(mesh->bdr_attributes.Max());
      ess_bdr = 1;
      fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   // 8. Set up the linear form b(.) which corresponds to the right-hand side of
   //    the FEM linear system, which in this case is (1,phi_i) where phi_i are
   //    the basis functions in the finite element fespace.
   LinearForm *b = new LinearForm(fespace);
   ConstantCoefficient one(1.0);
   b->AddDomainIntegrator(new DomainLFIntegrator(one));
   b->Assemble();

   // 9. Define the solution vector x as a finite element grid function
   //    corresponding to fespace. Initialize x with initial guess of zero,
   //    which satisfies the boundary conditions.
   GridFunction x(fespace);
   x = 0.0;

   // 10. Set up the bilinear form a(.,.) on the finite element space that will
   //     hold the matrix corresponding to the Laplacian operator -Delta.
   BilinearForm *a = new BilinearForm(fespace);
   BilinearForm *a_pc = NULL;
   if (pc_choice == LOR)
   {
      a_pc = new BilinearForm(fespace_lor);
   }
   else if (pc_choice != NONE)
   {
      a_pc = new BilinearForm(fespace);
   }

   // 11. Assemble the bilinear form and the corresponding linear system,
   //     applying any necessary transformations such as: eliminating boundary
   //     conditions, applying conforming constraints for non-conforming AMR,
   //     static condensation, etc.
   if (static_cond) { a->EnableStaticCondensation(); }

   cout << "Assembling the bilinear form ..." << flush;
   tic_toc.Clear();
   tic_toc.Start();
   // Pre-allocate sparsity assuming dense element matrices
   a->UsePrecomputedSparsity();

   HPCBilinearForm *a_hpc = NULL;
   HPCBilinearForm_lor *a_hpc_lor = NULL;
   Operator *a_oper = NULL;

   if (!perf)
   {
      // Standard assembly using a diffusion domain integrator
      a->AddDomainIntegrator(new DiffusionIntegrator(one));
      a->Assemble();
   }
   else
   {
      // High-performance assembly/evaluation using the templated operator type
      a_hpc = new HPCBilinearForm(integ_t(coeff_t(1.0)), *fespace);

      if (matrix_free)
      {
         a_hpc->Assemble(); // partial assembly
      }
      else
      {
         a_hpc->AssembleBilinearForm(*a); // full matrix assembly
      }
   }
   tic_toc.Stop();
   cout << " done, " << tic_toc.RealTime() << "s." << endl;

   // 12. Solve the system A X = B with CG. In the standard case, use a simple
   //     symmetric Gauss-Seidel preconditioner.

   // Setup the operator matrix
   SparseMatrix A;
   Vector B, X;
   Vector B_cp, X_cp;
   if (perf && matrix_free)
   {
      a_hpc->FormLinearSystem(ess_tdof_list, x, *b, a_oper, X, B);
      cout << "Size of linear system: " << a_hpc->Height() << endl;
   }
   else
   {
      a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);
      cout << "Size of linear system: " << A.Height() << endl;
   }

   // Setup the preconditioner
   SparseMatrix A_pc;
   if (pc_choice != NONE)
   {
      cout << "Assembling Linear System for the GS Preconditioner ... " << endl;
      tic_toc.Clear();
      tic_toc.Start();

      Vector X_tmp, B_tmp;
      if (pc_choice == LOR)
      {
         a_hpc_lor = new HPCBilinearForm_lor(integ_t(coeff_t(1.0)),
                                             *fespace_lor);
         a_hpc_lor->AssembleBilinearForm(*a_pc);
         a_pc->FormLinearSystem(ess_tdof_list, x, *b, A_pc, X_tmp, B_tmp);
      }
      else if (pc_choice == DENSE)
      {
         if (!perf)
         {
            A_pc.MakeRef(A);
         }
         else
         {
            a_hpc->AssembleBilinearForm(*a_pc);
            a_pc->FormLinearSystem(ess_tdof_list, x, *b, A_pc, X_tmp, B_tmp);
         }
      }
      tic_toc.Stop();
      cout << " done, " << tic_toc.RealTime() << "s." << endl;
   }

   if (perf && matrix_free)
   {
      // Cannot utilize Gauss-Seidel preconditioner with how matrix-free operator
      // handles essential BCs. See ConstrainedOperator.
      if (pc_choice == NONE)
      {
         CG(*a_oper, B, X, 1, 500, 1e-12, 0.0);
      }
      else
      {
         GSSmoother M(A_pc);
         PCG(*a_oper, M, B, X, 1, 500, 1e-12, 0.0);
      }
   }
   else
   {
#ifndef MFEM_USE_SUITESPARSE
      if (pc_choice != NONE)
      {
         GSSmoother M(A_pc);
         PCG(A, M, B, X, 1, 500, 1e-12, 0.0);
      }
      else
      {
         CG(A, B, X, 1, 500, 1e-12, 0.0);
      }
#else
      // If MFEM was compiled with SuiteSparse, use UMFPACK to solve the system.
      UMFPackSolver umf_solver;
      umf_solver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
      umf_solver.SetOperator(A);
      umf_solver.Mult(B, X);
#endif
   }

   // 13. Recover the solution as a finite element grid function.
   if (perf && matrix_free)
   {
      a_hpc->RecoverFEMSolution(X, *b, x);
   }
   else
   {
      a->RecoverFEMSolution(X, *b, x);
   }

   // 14. Save the refined mesh and the solution. This output can be viewed later
   //     using GLVis: "glvis -m refined.mesh -g sol.gf".
   ofstream mesh_ofs("refined.mesh");
   mesh_ofs.precision(8);
   mesh->Print(mesh_ofs);
   ofstream sol_ofs("sol.gf");
   sol_ofs.precision(8);
   x.Save(sol_ofs);

   // 15. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "solution\n" << *mesh << x << flush;
   }

   // 16. Free the used memory.
   delete a;
   delete a_hpc;
   delete a_oper;
   delete b;
   delete fespace;
   if (order > 0) { delete fec; }
   delete mesh;

   return 0;
}
