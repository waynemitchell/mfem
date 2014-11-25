//                                MFEM Example 5
//
//
// Description:  This example code demonstrates the use of MFEM to define a
//               triangulation of a unit sphere and a simple isoparametric
//               finite element discretization of the Laplace problem
//               -Delta u + u = f on a triangulated sphere using linear finite
//               elements.  The example highlights mesh generation, the use of
//               mesh refinement, finite element grid functions, as well as
//               linear and bilinear forms corresponding to the left-hand side
//               and right-hand side of the discrete linear system.

#include <fstream>
#include "mfem.hpp"
#include <cassert>

using namespace std;
using namespace mfem;

double analytic_solution (Vector & input)
{
   double l2 = (input[0]*input[0] + input[1]*input[1] +input[2]*input[2]);
   return input[0]*input[1]/l2;
}


double analytic_rhs (Vector & input)
{
   double l2 = (input[0]*input[0] + input[1]*input[1] +input[2]*input[2]);
   return 7*input[0]*input[1]/l2;
}

int main (int argc, char *argv[])
{
   if (argc != 3)
   {
      cout << "\nUsage: ex5 <poly_degree> <number_of_refinements>\n" << endl;
      return 1;
   }

   // Type of elements to use: 0 - triangles, 1 - quads
   int elem_type = 1;

   // If true, snap nodes to the sphere initilially and after each refinement;
   // otherwise, snap only after the last refinement.
   bool always_snap = false;

   int Nvert, NElem;
   if (elem_type == 0)
   {
      Nvert = 6;
      NElem = 8;
   }
   else
   {
      Nvert = 8;
      NElem = 6;
   }

   Mesh mesh(2, Nvert, NElem, 0, 3);

   if (elem_type == 0)
   {
      const double tri_v[6][3] =
         {{ 1,  0,  0}, { 0,  1,  0}, {-1,  0,  0},
          { 0, -1,  0}, { 0,  0,  1}, { 0,  0, -1}};
      const int tri_e[8][3] =
         {{0, 1, 4}, {1, 2, 4}, {2, 3, 4}, {3, 0, 4},
          {1, 0, 5}, {2, 1, 5}, {3, 2, 5}, {0, 3, 5}};

      for (int j = 0; j < Nvert; j++)
      {
         mesh.AddVertex(tri_v[j]);
      }
      for (int j = 0; j < NElem; j++)
      {
         int attribute = j + 1;
         mesh.AddTriangle(tri_e[j], attribute);
      }
      mesh.FinalizeTriMesh(1, 1, true);
   }
   else
   {
      const double quad_v[8][3] =
         {{-1, -1, -1}, {+1, -1, -1}, {+1, +1, -1}, {-1, +1, -1},
          {-1, -1, +1}, {+1, -1, +1}, {+1, +1, +1}, {-1, +1, +1}};
      const int quad_e[6][4] =
         {{3, 2, 1, 0}, {0, 1, 5, 4}, {1, 2, 6, 5},
          {2, 3, 7, 6}, {3, 0, 4, 7}, {4, 5, 6, 7}};

      for (int j = 0; j < Nvert; j++)
      {
         mesh.AddVertex(quad_v[j]);
      }
      for (int j = 0; j < NElem; j++)
      {
         int attribute = j + 1;
         mesh.AddQuad(quad_e[j], attribute);
      }
      mesh.FinalizeQuadMesh(1, 1, true);
   }

   // Order of the mesh and the solution space.
   int order = atoi(argv[1]);

   // Set the mesh nodes according to 'order'.
   H1_FECollection fec(order, mesh.Dimension());
   FiniteElementSpace nodal_fes(&mesh, &fec, mesh.spaceDimension());
   mesh.SetNodalFESpace(&nodal_fes);

   // Refine the mesh
   const int ref_levels = atoi(argv[2]);
   for (int l = 0; l <= ref_levels; l++)
   {
      if (l > 0) // for l == 0 just perform snapping
         mesh.UniformRefinement();

      // Snap the nodes of the refined mesh back to sphere surface.
      if (always_snap || l == ref_levels)
      {
         GridFunction &nodes = *mesh.GetNodes();
         Vector node(mesh.spaceDimension());
         for (int i = 0; i < nodes.FESpace()->GetNDofs(); i++)
         {
            for (int d = 0; d < mesh.spaceDimension(); d++)
               node(d) = nodes(nodes.FESpace()->DofToVDof(i, d));

            node /= node.Norml2();

            for (int d = 0; d < mesh.spaceDimension(); d++)
               nodes(nodes.FESpace()->DofToVDof(i, d)) = node(d);
         }
      }
   }

   // 3. Define a finite element space on the mesh. Here we use isoparametric
   //    finite elements -- the same as the mesh nodes.
   FiniteElementSpace *fespace = new FiniteElementSpace(&mesh, &fec);
   cout << "Number of unknowns: " << fespace->GetVSize() << endl;

   // 4. Set up the linear form b(.) which corresponds to the right-hand side of
   //    the FEM linear system, which in this case is (1,phi_i) where phi_i are
   //    the basis functions in the finite element fespace.
   LinearForm *b = new LinearForm(fespace);
   ConstantCoefficient one(1.0);
   FunctionCoefficient rhs_coef (analytic_rhs);
   FunctionCoefficient sol_coef (analytic_solution);

   b->AddDomainIntegrator(new DomainLFIntegrator(rhs_coef));
   b->Assemble();

   // 5. Define the solution vector x as a finite element grid function
   //    corresponding to fespace. Initialize x with initial guess of zero,
   //    which satisfies the boundary conditions.
   GridFunction x(fespace);
   x = 0.0;

   // 6. Set up the bilinear form a(.,.) on the finite element space
   //    corresponding to the Laplacian operator -Delta, by adding the Diffusion
   //    domain integrator and imposing homogeneous Dirichlet boundary
   //    conditions. The boundary conditions are implemented by marking all the
   //    boundary attributes from the mesh as essential (Dirichlet). After
   //    assembly and finalizing we extract the corresponding sparse matrix A.
   BilinearForm *a = new BilinearForm(fespace);
   a->AddDomainIntegrator(new DiffusionIntegrator(one));
   a->AddDomainIntegrator(new MassIntegrator(one));
   a->Assemble();
   a->Finalize();
   const SparseMatrix &A = a->SpMat();

#ifndef MFEM_USE_SUITESPARSE
   // 7. Define a simple symmetric Gauss-Seidel preconditioner and use it to
   //    solve the system Ax=b with PCG.
   GSSmoother M(A);
   PCG(A, M, *b, x, 0, 1000, 1e-28, 0.0);
#else
   // 7. If MFEM was compiled with SuiteSparse, use UMFPACK to solve the system.
   UMFPackSolver umf_solver;
   umf_solver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
   umf_solver.SetOperator(A);
   umf_solver.Mult(*b, x);
#endif

   // compare with exact solution
   cout<<"L2 norm of error: " << x.ComputeL2Error(sol_coef) << endl;

   // 8. Save the solution and the mesh.
   {
      ofstream mesh_ofs("sphere_refined.mesh");
      mesh_ofs.precision(8);
      mesh.Print(mesh_ofs);
      ofstream sol_ofs("sol.gf");
      sol_ofs.precision(8);
      x.Save(sol_ofs);
   }

   // 9. Visualization
   {
      const char vishost[] = "localhost";
      const int visport = 19916;
      socketstream out(vishost, visport);
      if (out.good())
      {
         out.precision(8);
         out << "solution\n";
         mesh.Print(out);
         x.Save(out);
         out << flush;
      }
      else
      {
         cout << "Unable to connect to " << vishost << ':' << visport << endl;
      }
   }

   // 10. Free the used memory.
   delete a;
   delete b;
   delete fespace;

   return 0;
}
