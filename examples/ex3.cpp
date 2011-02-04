//                                MFEM Example 3
//
// Compile with: make ex3
//
// Sample runs:  ex3 beam.mesh3d
//               ex3 fichera.mesh3d
//               ex3 escher.mesh3d
//
// Description:  This example code solves a simple 3D electromagnetic diffusion
//               problem corresponding to the second order definite Maxwell
//               equation curl curl E + E = f with perfectly conducting boundary
//               conditions E x n = 0. Here f = (0,0,1) and we discretize with
//               the lowest order Nedelec finite elements.
//
//               The example demonstrates the use of H(curl) finite element
//               spaces with the curl-curl and the (vector finite element) mass
//               bilinear form, the projection of grid functions between finite
//               element spaces and the extraction of scalar components of
//               vector fields.
//
//               We recommend viewing examples 1-2 before viewing this example.

#include <fstream>
#include "mfem.hpp"

int main (int argc, char *argv[])
{
   Mesh *mesh;

   if (argc == 1)
   {
      cout << "\nUsage: ex3 <mesh_file>\n" << endl;
      return 1;
   }

   // 1. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral or hexahedral elements with the same code.
   ifstream imesh(argv[1]);
   if (!imesh)
   {
      cerr << "\nCan not open mesh file: " << argv[1] << '\n' << endl;
      return 2;
   }
   mesh = new Mesh(imesh, 1, 1);
   imesh.close();
   if (mesh -> Dimension() != 3)
   {
      cerr << "\nThis example requires a 3D mesh\n" << endl;
      return 3;
   }

   // 2. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement. We choose 'ref_levels' to be the
   //    largest number that gives a final mesh with no more than 50,000
   //    elements.
   {
      int ref_levels =
         (int)floor(log(50000./mesh->GetNE())/log(2.)/mesh->Dimension());
      for (int l = 0; l < ref_levels; l++)
         mesh->UniformRefinement();
   }

   // 3. Define a finite element space on the mesh. Here we use the lowest order
   //    Nedelec finite elements.
   FiniteElementCollection *fec = new ND1_3DFECollection;
   FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec);

   // 4. Set up the linear form b(.) which corresponds to the right-hand side of
   //    the FEM linear system, which in this case is (f,phi_i) where f=(0,0,1)
   //    and phi_i are the basis functions in the finite element fespace.
   VectorArrayCoefficient f(3);
   f.Set(0, new ConstantCoefficient(0.0));
   f.Set(1, new ConstantCoefficient(0.0));
   f.Set(2, new ConstantCoefficient(1.0));
   LinearForm *b = new LinearForm(fespace);
   b->AddDomainIntegrator(new VectorFEDomainLFIntegrator(f));
   b->Assemble();

   // 5. Define the solution vector x as a finite element grid function
   //    corresponding to fespace. Initialize x with initial guess of zero,
   //    which satisfies the boundary conditions.
   GridFunction x(fespace);
   x = 0.0;

   // 6. Set up the bilinear form corresponding to the EM diffusion operator
   //    curl muinv curl + sigma I, by adding the curl-curl and the mass domain
   //    integrators and imposing homogeneous Dirichlet boundary conditions. The
   //    boundary conditions are implemented by marking all the boundary
   //    attributes from the mesh as essential (Dirichlet). After assembly and
   //    finalizing we extract the corresponding sparse matrix A.
   Coefficient *muinv = new ConstantCoefficient(1.0);
   Coefficient *sigma = new ConstantCoefficient(1.0);
   BilinearForm *a = new BilinearForm(fespace);
   a->AddDomainIntegrator(new CurlCurlIntegrator(*muinv));
   a->AddDomainIntegrator(new VectorFEMassIntegrator(sigma));
   a->Assemble();
   Array<int> ess_bdr(mesh->bdr_attributes.Max());
   ess_bdr = 1;
   a->EliminateEssentialBC(ess_bdr, x, *b);
   a->Finalize();
   const SparseMatrix &A = a->SpMat();

   // 7. Define a simple symmetric Gauss-Seidel preconditioner and use it to
   //    solve the system Ax=b with PCG.
   GSSmoother M(A);
   PCG(A, M, *b, x, 1, 500, 1e-12, 0.0);

   // 8. In order to visualize the solution, we first represent it in the space
   //    of linear discontinuous vector finite elements. The representation in
   //    this space is obtained by (exact) projection with ProjectVectorFieldOn.
   FiniteElementCollection *dfec = new LinearDiscont3DFECollection;
   FiniteElementSpace *dfespace = new FiniteElementSpace(mesh, dfec, 3);
   GridFunction dx(dfespace);
   x.ProjectVectorFieldOn(dx);

   // 9. Save the refined mesh and the solution. This output can be viewed later
   //    using GLVis: "glvis -m refined.mesh -g sol.gf".
   {
      ofstream mesh_ofs("refined.mesh");
      mesh->Print(mesh_ofs);
      ofstream sol_ofs("sol.gf");
      dx.Save(sol_ofs);
    }

   // 10. (Optional) Send the solution by socket to a GLVis server.
   char vishost[] = "localhost";
   int  visport   = 19916;
   osockstream sol_sock (visport, vishost);
   sol_sock << "vfem3d_gf_data\n";
   mesh->Print(sol_sock);
   dx.Save(sol_sock);
   sol_sock.send();

   // 11. Free the used memory.
   delete a;
   delete b;
   delete fespace;
   delete fec;
   delete mesh;
   delete dfespace;
   delete dfec;

   return 0;
}
