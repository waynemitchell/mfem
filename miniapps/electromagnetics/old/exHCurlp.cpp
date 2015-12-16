//                       MFEM Example ?? - Parallel Version
//
// Compile with: make ex??p
//
// Sample runs:  mpirun -np 4 ex??p -m ../data/square-torus.mesh
//               mpirun -np 4 ex??p -m ../data/beam-tet.mesh
//               mpirun -np 4 ex??p -m ../data/beam-hex.mesh
//               mpirun -np 4 ex??p -m ../data/escher.mesh
//               mpirun -np 4 ex??p -m ../data/fichera.mesh
//               mpirun -np 4 ex??p -m ../data/fichera-q2.vtk
//               mpirun -np 4 ex??p -m ../data/fichera-q3.mesh
//               mpirun -np 4 ex??p -m ../data/beam-hex-nurbs.mesh
//
// Description:  This example code solves a simple 3D magnetostatic
//               problem corresponding to the second order
//               semi-definite Maxwell equation
//                  curl curl A = 0
//               with boundary condition
//                                / (0,0,1) on surface 1
//                  n x (A x n) = |
//                                \ (0,0,0) elsewhere
//               This is a perfect electrical conductor (PEC) boundary
//               condition which results in a magnetic field sasifying:
//                  n . B = 0 on all surfaces
//               We discretize with Nedelec finite elements.
//
//               The example demonstrates the use of H(curl) finite element
//               spaces with the curl-curl bilinear form.
//
//               The number and location of the numbered surfaces in
//               the mesh will greatly effect the character of the
//               solution.  For example if there is only one surface
//               the result should be a constant field.
//
//               We recommend viewing examples 1-2 before viewing this example.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   // 1. Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // 2. Parse command-line options.
   const char *mesh_file = "../data/square-torus-n1.mesh";
   int order = 1;
   bool visualization = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0)
      {
         args.PrintUsage(cout);
      }
      MPI_Finalize();
      return 1;
   }
   if (myid == 0)
   {
      args.PrintOptions(cout);
   }

   // 3. Read the (serial) mesh from the given mesh file on all processors.  We
   //    can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
   //    and volume meshes with the same code.
   Mesh *mesh;
   ifstream imesh(mesh_file);
   if (!imesh)
   {
      if (myid == 0)
      {
         cerr << "\nCan not open mesh file: " << mesh_file << '\n' << endl;
      }
      MPI_Finalize();
      return 2;
   }
   mesh = new Mesh(imesh, 1, 1);
   imesh.close();
   int dim = mesh->Dimension();
   if (dim != 3)
   {
      if (myid == 0)
      {
         cerr << "\nThis example requires a 3D mesh\n" << endl;
      }
      MPI_Finalize();
      return 3;
   }

   // 4. Refine the serial mesh on all processors to increase the resolution. In
   //    this example we do 'ref_levels' of uniform refinement. We choose
   //    'ref_levels' to be the largest number that gives a final mesh with no
   //    more than 100 elements.
   {
      int ref_levels =
         (int)floor(log(100./mesh->GetNE())/log(2.)/dim);
      for (int l = 0; l < ref_levels; l++)
      {
         mesh->UniformRefinement();
      }
   }

   // 5. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted. Tetrahedral
   //    meshes need to be reoriented before we can define high-order Nedelec
   //    spaces on them.
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   {
      int par_ref_levels = 2;
      for (int l = 0; l < par_ref_levels; l++)
      {
         pmesh->UniformRefinement();
      }
   }
   pmesh->ReorientTetMesh();

   // 6. Define a parallel finite element space on the parallel mesh. Here we
   //    use the lowest order Nedelec finite elements, but we can easily switch
   //    to higher-order spaces by changing the value of p.
   FiniteElementCollection *HCurlFEC     = new ND_FECollection(order, dim);
   FiniteElementCollection *HDivFEC      = new RT_FECollection(order, dim);
   ParFiniteElementSpace   *HCurlFESpace = new ParFiniteElementSpace(pmesh,
                                                                     HCurlFEC);
   ParFiniteElementSpace   *HDivFESpace  = new ParFiniteElementSpace(pmesh,
                                                                     HDivFEC);
   HYPRE_Int size = HCurlFESpace->GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of unknowns: " << size << endl;
   }

   // 7. Set up the parallel linear form b(.) which corresponds to the
   //    right-hand side of the FEM linear system, which in this case is
   //    zero.
   ParLinearForm *b = new ParLinearForm(HCurlFESpace);
   b->Assemble();

   // 8. Define the solution vector x as a parallel finite element
   // grid function corresponding to HCurlFESpace. Initialize x by
   // projecting the boundary conditions onto the appropriate edges.
   ParGridFunction x(HCurlFESpace);
   Vector zHat(3); zHat = 0.0; zHat(2) = 1.0;
   VectorConstantCoefficient UnitVecZ(zHat);

   Array<int> ess_bdr_1(pmesh->bdr_attributes.Max());
   ess_bdr_1 = 0; ess_bdr_1[0] = 1;

   // Set x to be the unit vector in the z direction on the first surface
   x = 0.0;
   x.ProjectBdrCoefficientTangent(UnitVecZ,ess_bdr_1);

   // 9. Set up the parallel bilinear form corresponding to the
   // magnetostatic operator curl muinv curl, by adding the curl-curl
   // domain integrator and finally imposing non-homogeneous Dirichlet
   // boundary conditions. The boundary conditions are implemented by
   // marking all the boundary attributes from the mesh as essential
   // (Dirichlet). After serial and parallel assembly we extract the
   // parallel matrix A.
   Coefficient *muinv = new ConstantCoefficient(1.0);
   ParBilinearForm *a = new ParBilinearForm(HCurlFESpace);
   a->AddDomainIntegrator(new CurlCurlIntegrator(*muinv));
   a->Assemble();

   // The entire outer surface of the mesh is held fixed at zero
   // except for the first surface which is set to he unit vector in
   // the z direction.
   Array<int> ess_bdr(pmesh->bdr_attributes.Max());
   ess_bdr = 1;
   a->EliminateEssentialBC(ess_bdr, x, *b);
   a->Finalize();

   // 10. Define the parallel (hypre) matrix and vectors representing a(.,.),
   //     b(.) and the finite element approximation.
   HypreParMatrix *A = a->ParallelAssemble();
   HypreParVector *B = b->ParallelAssemble();
   HypreParVector *X = x.ParallelAverage();
   *X = 0.0;

   delete a;
   delete muinv;
   delete b;

   // 11. Define and apply a parallel PCG solver for AX=B with the AMS
   //     preconditioner from hypre.

   HypreSolver *ams = new HypreAMS(*A, HCurlFESpace, 1);
   HyprePCG *pcg = new HyprePCG(*A);
   pcg->SetTol(1e-12);
   pcg->SetMaxIter(500);
   pcg->SetPrintLevel(2);
   pcg->SetPreconditioner(*ams);
   pcg->Mult(*B, *X);

   delete ams;
   delete pcg;

   // 12. Extract the parallel grid function corresponding to the finite element
   //     approximation X. This is the local solution on each processor.
   x = *X;

   // 13. Compute the Curl of the solution vector.  This is the
   //     magnetic field corresponding to the vector potential
   //     represented by x.
   ParDiscreteLinearOperator *curl =
      new ParDiscreteLinearOperator(HCurlFESpace, HDivFESpace);
   curl->AddDomainInterpolator(new CurlInterpolator);
   curl->Assemble();
   curl->Finalize();
   HypreParMatrix *Curl = curl->ParallelAssemble();
   HypreParVector *CurlX = new HypreParVector(HDivFESpace);
   Curl->Mult(*X,*CurlX);
   ParGridFunction curlx(HDivFESpace,CurlX);

   delete curl;

   // 14. Save the refined mesh and the solution in parallel. This output can
   //     be viewed later using GLVis: "glvis -np <np> -m mesh -g sol".
   {
      ostringstream mesh_name, sol_name, curl_name;
      mesh_name << "mesh." << setfill('0') << setw(6) << myid;
      sol_name  << "sol." << setfill('0') << setw(6) << myid;
      curl_name << "sol_curl." << setfill('0') << setw(6) << myid;

      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh->Print(mesh_ofs);

      ofstream sol_ofs(sol_name.str().c_str());
      sol_ofs.precision(8);
      x.Save(sol_ofs);

      ofstream curl_ofs(curl_name.str().c_str());
      curl_ofs.precision(8);
      curlx.Save(curl_ofs);
   }

   // 15. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;

      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << *pmesh << x
               << "window_title 'Vector Potential'" << flush;

      MPI_Barrier(pmesh->GetComm());

      socketstream curl_sock(vishost, visport);
      curl_sock << "parallel " << num_procs << " " << myid << "\n";
      curl_sock.precision(8);
      curl_sock << "solution\n" << *pmesh << curlx
                << "window_title 'Magnetic Field'" << flush;

   }

   // 16. Free the used memory.
   delete Curl;
   delete CurlX;
   delete X;
   delete B;
   delete A;
   delete HDivFESpace;
   delete HCurlFESpace;
   delete HDivFEC;
   delete HCurlFEC;
   delete pmesh;

   MPI_Finalize();

   return 0;
}
