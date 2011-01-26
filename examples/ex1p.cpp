//                       MFEM Example 1 - Parallel Version
//
// Compile with: make ex1p
//
// Sample runs:  mpirun -np 4 ex1p square-disc.mesh2d
//               mpirun -np 4 ex1p star.mesh2d
//               mpirun -np 4 ex1p escher.mesh3d
//               mpirun -np 4 ex1p fichera.mesh3d
//
// Description:  This example code demonstrates the use of MFEM to define a
//               simple linear finite element discretization of the Laplace
//               problem -Delta u = 1 with homogeneous Dirichlet boundary
//               conditions.
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
   int num_procs, myid;

   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   Mesh *mesh;
   ParMesh *pmesh;

   if (argc == 1)
   {
      if (myid == 0)
         cout << "Usage: ex1 <mesh_file>" << endl;
      return 1;
   }

   // 1. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral or hexahedral elements with the same code.
   ifstream imesh(argv[1]);
   if (!imesh)
   {
      if (myid == 0)
         cerr << "can not open mesh file: " << argv[1] << endl;
      return 2;
   }
   mesh = new Mesh(imesh, 1, 1);
   MPI_Barrier(MPI_COMM_WORLD);
   imesh.close();

   // 2. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement. We choose 'ref_levels' to be the
   //    largest number that gives a final mesh with no more than 50,000
   //    elements.
   {
      int ref_levels =
         (int)floor(log(10000./mesh->GetNE())/log(2.)/mesh->Dimension());
      for (int l = 0; l < ref_levels; l++)
         mesh->UniformRefinement();
   }

   pmesh = new ParMesh(MPI_COMM_WORLD, *mesh, 0, 1);

   // DEBUG
   {
      ofstream test("pmesh.mesh");
      pmesh -> PrintAsOne(test);
   }

   // 3. Define a finite element space on the mesh. Here we use linear finite
   //    elements.
   FiniteElementCollection *fec = new LinearFECollection;
   ParFiniteElementSpace *fespace = new ParFiniteElementSpace(pmesh, fec);

   // 4. Set up the linear form b(.) which corresponds to the right-hand side of
   //    the FEM linear system, which in this case is (1,phi_i) where phi_i are
   //    the basis functions in the finite element fespace.
   ParLinearForm *b = new ParLinearForm(fespace);
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
   ParBilinearForm *a = new ParBilinearForm(fespace);
   a->AddDomainIntegrator(new DiffusionIntegrator(one));
   a->Assemble();
   Array<int> ess_bdr(mesh->bdr_attributes.Max());
   ess_bdr = 1;
   a->EliminateEssentialBC(ess_bdr, x, *b);
   a->Finalize();

   HypreParMatrix *A = a -> ParallelAssemble();
   HypreParVector *B = b -> ParallelAssemble();
   HypreParVector *X = new HypreParVector(*B);
   *X = 0.0;

   HypreSolver *solver;
   HypreSolver *prec = new HypreBoomerAMG(*A);
   HyprePCG *pcg_solver = new HyprePCG(*A);
   pcg_solver -> SetTol(1e-12);
   pcg_solver -> SetMaxIter(100);
   pcg_solver -> SetPrintLevel (2);
   pcg_solver -> SetPreconditioner(*prec);
   solver = pcg_solver;

   solver -> Mult(*B, *X);

   ParGridFunction xx(fespace, X);

   // 8. Save the refined mesh and the solution. This output can be viewed later
   //    using GLVis: "glvis -m refined.mesh -g sol.gf".
   {
      ofstream mesh_ofs("refined.mesh");
      pmesh->PrintAsOne(mesh_ofs);
      ofstream sol_ofs("sol.gf");
      xx.SaveAsOne(sol_ofs);
   }

   // 9. (Optional) Send the solution by socket to a GLVis server.
   char vishost[] = "localhost";
   int  visport   = 19916;
   osockstream *sol_sock;
   if (myid == 0)
   {
      sol_sock = new osockstream(visport, vishost);
      if (mesh->Dimension() == 2)
         *sol_sock << "fem2d_gf_data\n";
      else
         *sol_sock << "fem3d_gf_data\n";
   }
   pmesh->PrintAsOne(*sol_sock);
   xx.SaveAsOne(*sol_sock);
   if (myid == 0)
   {
      sol_sock->send();
      delete sol_sock;
   }

   // 10. Free the used memory.
   delete a;
   delete b;
   delete fespace;
   delete fec;
   delete mesh;
}
