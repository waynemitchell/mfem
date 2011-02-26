//                       MFEM Example 3 - Parallel Version
//
// Compile with: make ex3p
//
// Sample runs:  mpirun -np 4 ex3p beam.mesh3d
//               mpirun -np 4 ex3p fichera.mesh3d
//               mpirun -np 4 ex3p escher.mesh3d
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
   int num_procs, myid;

   // 1. Initialize MPI
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   Mesh *mesh;

   if (argc == 1)
   {
      if (myid == 0)
         cout << "\nUsage: ex3 <mesh_file>\n" << endl;
      MPI_Finalize();
      return 1;
   }

   // 2. Read the (serial) mesh from the given mesh file on all processors.
   //    We can handle triangular, quadrilateral, tetrahedral or hexahedral
   //    elements with the same code.
   ifstream imesh(argv[1]);
   if (!imesh)
   {
      if (myid == 0)
         cerr << "\nCan not open mesh file: " << argv[1] << '\n' << endl;
      MPI_Finalize();
      return 2;
   }
   mesh = new Mesh(imesh, 1, 1);
   imesh.close();
   if (mesh -> Dimension() != 3)
   {
      if (myid == 0)
         cerr << "\nThis example requires a 3D mesh\n" << endl;
      MPI_Finalize();
      return 3;
   }

   // 3. Refine the serial mesh on all processors to increase the resolution. In
   //    this example we do 'ref_levels' of uniform refinement. We choose
   //    'ref_levels' to be the largest number that gives a final mesh with no
   //    more than 1,000 elements.
   {
      int ref_levels =
         (int)floor(log(1000./mesh->GetNE())/log(2.)/mesh->Dimension());
      for (int l = 0; l < ref_levels; l++)
         mesh->UniformRefinement();
   }

   // 4. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted.
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   {
      int par_ref_levels = 2;
      for (int l = 0; l < par_ref_levels; l++)
         pmesh->UniformRefinement();
   }

   // 5. Define a parallel finite element space on the parallel mesh. Here we
   //    use the lowest order Nedelec finite elements.
   FiniteElementCollection *fec = new ND1_3DFECollection;
   ParFiniteElementSpace *fespace = new ParFiniteElementSpace(pmesh, fec);

   // 6. Set up the parallel linear form b(.) which corresponds to the
   //    right-hand side of the FEM linear system, which in this case is
   //    (f,phi_i) where f=(0,0,1) and phi_i are the basis functions in the
   //    finite element fespace.
   VectorArrayCoefficient f(3);
   f.Set(0, new ConstantCoefficient(0.0));
   f.Set(1, new ConstantCoefficient(0.0));
   f.Set(2, new ConstantCoefficient(1.0));
   ParLinearForm *b = new ParLinearForm(fespace);
   b->AddDomainIntegrator(new VectorFEDomainLFIntegrator(f));
   b->Assemble();

   // 7. Define the solution vector x as a parallel finite element grid function
   //    corresponding to fespace. Initialize x with initial guess of zero,
   //    which satisfies the boundary conditions.
   ParGridFunction x(fespace);
   (Vector &)x = 0.0;

   // 8. Set up the parallel bilinear form corresponding to the EM diffusion
   //    operator curl muinv curl + sigma I, by adding the curl-curl and the
   //    mass domain integrators and imposing homogeneous Dirichlet boundary
   //    conditions. The boundary conditions are implemented by marking all the
   //    boundary attributes from the mesh as essential (Dirichlet). After
   //    serial and parallel assembly we extract the parallel matrix A.
   Coefficient *muinv = new ConstantCoefficient(1.0);
   Coefficient *sigma = new ConstantCoefficient(1.0);
   ParBilinearForm *a = new ParBilinearForm(fespace);
   a->AddDomainIntegrator(new CurlCurlIntegrator(*muinv));
   a->AddDomainIntegrator(new VectorFEMassIntegrator(sigma));
   a->Assemble();
   {
      Array<int> ess_bdr(pmesh->bdr_attributes.Max());
      ess_bdr = 1;
      Array<int> ess_dofs;
      fespace->GetEssentialVDofs(ess_bdr, ess_dofs);
      a->EliminateEssentialBCFromDofs(ess_dofs, x, *b);
   }
   a->Finalize();

   // 9. Define the parallel (hypre) matrix and vectors representing a(.,.),
   //    b(.) and the finite element approximation.
   HypreParMatrix *A = a->ParallelAssemble();
   HypreParVector *B = b->ParallelAssemble();
   HypreParVector *X = x.ParallelAverage();

   delete a;
   delete b;

   // 10. Define and apply a parallel PCG solver for AX=B with the AMS
   //     preconditioner from hypre.
   HypreSolver *ams = new HypreAMS(*A, fespace);
   HyprePCG *pcg = new HyprePCG(*A);
   pcg->SetTol(1e-12);
   pcg->SetMaxIter(500);
   pcg->SetPrintLevel(2);
   pcg->SetPreconditioner(*ams);
   pcg->Mult(*B, *X);

   // 11. Extract the parallel grid function corresponding to the finite element
   //     approximation X. This is the local solution on each processor.
   x = *X;

   // 12. In order to visualize the solution, we first represent it in the space
   //     of linear discontinuous vector finite elements. The representation in
   //     this space is obtained by (exact) projection with ProjectVectorFieldOn.
   FiniteElementCollection *dfec = new LinearDiscont3DFECollection;
   ParFiniteElementSpace *dfespace = new ParFiniteElementSpace(pmesh, dfec, 3);
   ParGridFunction dx(dfespace);
   x.ProjectVectorFieldOn(dx);

   // 13. Save the refined mesh and the solution. This output can be viewed
   //     later using GLVis: "glvis -m refined.mesh -g sol.gf".
   {
      ofstream mesh_ofs;
      if (myid == 0)
         mesh_ofs.open("refined.mesh");
      pmesh->PrintAsOne(mesh_ofs);
      if (myid == 0)
         mesh_ofs.close();

      ofstream sol_ofs;
      if (myid == 0)
         sol_ofs.open("sol.gf");
      dx.SaveAsOne(sol_ofs);
      if (myid == 0)
         sol_ofs.close();
    }

   // 14. (Optional) Send the solution by socket to a GLVis server.
   char vishost[] = "localhost";
   int  visport   = 19916;
   osockstream *sol_sock;
   if (myid == 0)
   {
      sol_sock = new osockstream(visport, vishost);
      *sol_sock << "vfem3d_gf_data\n";
   }
   pmesh->PrintAsOne(*sol_sock);
   dx.SaveAsOne(*sol_sock);
   if (myid == 0)
   {
      sol_sock->send();
      delete sol_sock;
   }

   // 11. Free the used memory.
   delete pcg;
   delete ams;
   delete X;
   delete B;
   delete A;

   delete fespace;
   delete fec;
   delete pmesh;

   MPI_Finalize();

   return 0;
}
