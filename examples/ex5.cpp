// TO BE REWRITTEN
//                                MFEM Example 5
//
//
// Description:  This example code demonstrates the use of MFEM to define a
//               simple isoparametric finite element discretization of the
//               Laplace problem -Delta u = 1 with homogeneous Dirichlet
//               boundary conditions. Specifically, we discretize with the
//               FE space coming from the mesh (linear by default, quadratic
//               for quadratic curvilinear mesh, NURBS for NURBS mesh, etc.)
//
//               The example highlights the use of mesh refinement, finite
//               element grid functions, as well as linear and bilinear forms
//               corresponding to the left-hand side and right-hand side of the
//               discrete linear system. We also cover the explicit elimination
//               of boundary conditions on all boundary edges, and the optional
//               connection to the GLVis tool for visualization.

#include <fstream>
#include "mfem.hpp"

int main (int argc, char *argv[])
{
  const int nx = 10;
  const int ny = 10;
  Mesh mesh(2, (nx+1) * (ny+1), 2 * nx * ny, 2 * nx + 2 * ny, 3);
  double c[3];
  int ind[3];
  double theta = 0*M_PI/2;

   // Sets vertices and the corresponding coordinates
   for (int j = 0; j < ny+1; j++)
   {
     c[1] = ((double) j / ny) ;
     for (int i = 0; i < nx+1; i++)
     {
       double tmp = ((double) i / nx);
       c[0] = tmp*cos(theta);
       c[2] = tmp*sin(theta);
       mesh.AddVertex(c);
     }
   }

   // Sets the elements and the corresponding indices of vertices
   for (int j = 0; j < ny; j++)
   {
     for (int i = 0; i < nx; i++)
     {
       ind[0] = i + j*(nx+1);
       ind[1] = i + 1 + (j+1)*(nx+1);
       ind[2] = i + (j+1)*(nx+1);
       mesh.AddTriangle(ind,1);
       ind[1] = i + 1 + j*(nx+1);
       ind[2] = i + 1 + (j+1)*(nx+1);
       mesh.AddTriangle(ind,1);
     }
   }

   // Sets boundary elements and the corresponding indices of vertices
   int m = (nx+1)*ny;
   for (int i = 0; i < nx; i++)
   {
     int vi[2] = {i,i+1};
     mesh.AddBdrSegment(vi,1);
     vi[0] = m+i;
     vi[1] = m+i+1;
     mesh.AddBdrSegment(vi,3);
   }
   m = nx+1;
   for (int j = 0; j < ny; j++)
   {
     int vi[2] = {j*m,(j+1)*m};
     mesh.AddBdrSegment(vi,4);
     vi[0] = j*m+nx;
     vi[1] = (j+1)*m+nx;
     mesh.AddBdrSegment(vi,2);
   }

   mesh.FinalizeTriMesh(1,1,1);

   // 3. Define a finite element space on the mesh. Here we use isoparametric
   //    finite elements coming from the mesh nodes (linear by default).
   FiniteElementCollection *fec;
   if (mesh.GetNodes())
      fec = mesh.GetNodes()->OwnFEC();
   else
      fec = new LinearFECollection;
   FiniteElementSpace *fespace = new FiniteElementSpace(&mesh, fec);
   cout << "Number of unknowns: " << fespace->GetVSize() << endl;

   // 4. Set up the linear form b(.) which corresponds to the right-hand side of
   //    the FEM linear system, which in this case is (1,phi_i) where phi_i are
   //    the basis functions in the finite element fespace.
   LinearForm *b = new LinearForm(fespace);
   ConstantCoefficient one(1.0);
   b->AddDomainIntegrator(new DomainLFIntegrator(one));
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
   a->Assemble();
   Array<int> ess_bdr(mesh.bdr_attributes.Max());
   ess_bdr = 1;
   a->EliminateEssentialBC(ess_bdr, x, *b);
   a->Finalize();
   const SparseMatrix &A = a->SpMat();

#ifndef MFEM_USE_SUITESPARSE
   // 7. Define a simple symmetric Gauss-Seidel preconditioner and use it to
   //    solve the system Ax=b with PCG.
   GSSmoother M(A);
   PCG(A, M, *b, x, 1, 200, 1e-12, 0.0);
#else
   // 7. If MFEM was compiled with SuiteSparse, use UMFPACK to solve the system.
   UMFPackSolver umf_solver;
   umf_solver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
   umf_solver.SetOperator(A);
   umf_solver.Mult(*b, x);
#endif

   // 8. Save the refined mesh and the solution. This output can be viewed later
   //    using GLVis: "glvis -m refined.mesh -g sol.gf".
   {
      ofstream mesh_ofs("refined.mesh");
      mesh_ofs.precision(8);
      mesh.Print(mesh_ofs);
      ofstream sol_ofs("sol.gf");
      sol_ofs.precision(8);
      x.Save(sol_ofs);
   }

   // 9. (Optional) Send the solution by socket to a GLVis server.
   char vishost[] = "localhost";
   int  visport   = 19916;
   osockstream sol_sock(visport, vishost);
   sol_sock << "solution\n";
   sol_sock.precision(8);
   mesh.Print(sol_sock);
   x.Save(sol_sock);
   sol_sock.send();

   // 10. Free the used memory.
   delete a;
   delete b;
   delete fespace;
   if (!mesh.GetNodes())
      delete fec;
   return 0;
}
