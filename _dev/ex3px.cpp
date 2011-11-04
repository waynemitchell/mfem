//                       MFEM Example 3 - Parallel Version
//
// Compile with: make ex3p
//
// Sample runs:  mpirun -np 4 ex3p ../data/beam-tet.mesh
//               mpirun -np 4 ex3p ../data/beam-hex.mesh
//               mpirun -np 4 ex3p ../data/escher.mesh
//               mpirun -np 4 ex3p ../data/fichera.mesh
//               mpirun -np 4 ex3p ../data/fichera-q2.vtk
//               mpirun -np 4 ex3p ../data/fichera-q3.mesh
//               mpirun -np 4 ex3p ../data/beam-hex-nurbs.mesh
//
// Description:  This example code solves a simple 3D electromagnetic diffusion
//               problem corresponding to the second order definite Maxwell
//               equation curl curl E + E = f with boundary condition
//               E x n = <given tangential field>. Here, we use a given exact
//               solution E and compute the corresponding r.h.s. f.
//               We discretize with the lowest order Nedelec finite elements.
//
//               The example demonstrates the use of H(curl) finite element
//               spaces with the curl-curl and the (vector finite element) mass
//               bilinear form, as well as the computation of discretization
//               error when the exact solution is known.
//
//               We recommend viewing examples 1-2 before viewing this example.

#include <fstream>
#include "mfem.hpp"

// Exact solution, E, and r.h.s., f. See below for implementation.
void E_exact(const Vector &, Vector &);
void f_exact(const Vector &, Vector &);

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
         cout << "\nUsage: mpirun -np <np> " << argv[0] << " <mesh_file>\n"
              << endl;
      MPI_Finalize();
      return 1;
   }

   // 2. Read the (serial) mesh from the given mesh file on all processors.
   //    In this 3D example, we can handle tetrahedral or hexahedral meshes
   //    with the same code.
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

   // 3. Refine the serial mesh on all processors to increase the resolution.
   {
      int ref_levels;
      if (myid == 0)
      {
         mesh->PrintCharacteristics();
         cout << "enter serial ref. levels --> " << flush;
         cin >> ref_levels;
      }
      MPI_Bcast(&ref_levels, 1, MPI_INT, 0, MPI_COMM_WORLD);
      for (int l = 0; l < ref_levels; l++)
      {
         if (myid == 0)
            cout << "refinement level " << l + 1 << " ... " << flush;
         mesh->UniformRefinement();
         if (myid == 0)
            cout << "done." << endl;
      }
      if (myid == 0)
      {
         cout << endl;
         mesh->PrintCharacteristics();
      }
   }

   // 4. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted.
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   {
      int par_ref_levels;
      pmesh->PrintInfo();
      if (myid == 0)
      {
         cout << "\nenter parallel ref. levels --> " << flush;
         cin >> par_ref_levels;
      }
      MPI_Bcast(&par_ref_levels, 1, MPI_INT, 0, MPI_COMM_WORLD);
      for (int l = 0; l < par_ref_levels; l++)
      {
         if (myid == 0)
            cout << "parallel refinement level " << l + 1 << " ... " << flush;
         pmesh->UniformRefinement();
         if (myid == 0)
            cout << "done." << endl;
      }
   }

   if (pmesh->MeshGenerator() & 1)
   {
      if (myid == 0)
         cout << "\nReorienting tetrahedral mesh ... " << flush;
      pmesh->ReorientTetMesh();
      if (myid == 0)
         cout << "done.\n" << endl;
   }

   pmesh->PrintInfo();

   // 5. Define a parallel finite element space on the parallel mesh.
   int p;
   if (myid == 0)
   {
      cout << "\nenter Nedelec space order --> " << flush;
      cin >> p;
   }
   MPI_Bcast(&p, 1, MPI_INT, 0, MPI_COMM_WORLD);

   FiniteElementCollection *fec = new ND_FECollection(p, 3);
   ParFiniteElementSpace *fespace = new ParFiniteElementSpace(pmesh, fec);

   // 6. Set up the parallel linear form b(.) which corresponds to the
   //    right-hand side of the FEM linear system, which in this case is
   //    (f,phi_i) where f is given by the function f_exact and phi_i are the
   //    basis functions in the finite element fespace.
   VectorFunctionCoefficient f(3, f_exact);
   ParLinearForm *b = new ParLinearForm(fespace);
   b->AddDomainIntegrator(new VectorFEDomainLFIntegrator(f));
   b->Assemble();

   // 7. Define the solution vector x as a parallel finite element grid function
   //    corresponding to fespace. Initialize x by projecting the exact
   //    solution. Note that only values from the boundary edges will be used
   //    when eliminating the non-homogenious boundary condition to modify the
   //    r.h.s. vector b.
   ParGridFunction x(fespace);
   VectorFunctionCoefficient E(3, E_exact);
   x.ProjectCoefficient(E);

   // 8. Set up the parallel bilinear form corresponding to the EM diffusion
   //    operator curl muinv curl + sigma I, by adding the curl-curl and the
   //    mass domain integrators and finally imposing non-homogeneous Dirichlet
   //    boundary conditions. The boundary conditions are implemented by
   //    marking all the boundary attributes from the mesh as essential
   //    (Dirichlet). After serial and parallel assembly we extract the
   //    parallel matrix A.
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
   *X = 0.0;

   delete a;
   delete sigma;
   delete muinv;
   delete b;

   // 10. Define and apply a parallel PCG solver for AX=B with the AMS
   //     preconditioner from hypre.

   // HypreSolver *ams = new HypreAMS(*A, fespace);
   HyprePCG *pcg = new HyprePCG(*A);
   pcg->SetTol(1e-12);
   pcg->SetMaxIter(500);
   pcg->SetPrintLevel(2);
   // pcg->SetPreconditioner(*ams);
   HypreDiagScale *diag = new HypreDiagScale;
   pcg->SetPreconditioner(*diag);
   pcg->SetMaxIter(5000);
   pcg->Mult(*B, *X);
   delete diag;

   // 11. Extract the parallel grid function corresponding to the finite element
   //     approximation X. This is the local solution on each processor.
   x = *X;

   // 12. Compute and print the L^2 norm of the error.
   {
      double err = x.ComputeL2Error(E);
      if (myid == 0)
         cout << "\n|| E_h - E ||_{L^2} = " << err << '\n' << endl;
   }

   // 13. Save the refined mesh and the solution in parallel. This output can
   //     be viewed later using GLVis: "glvis -np <np> -m mesh -g sol".
   {
      ostringstream mesh_name, sol_name;
      mesh_name << "mesh." << setfill('0') << setw(6) << myid;
      sol_name << "sol." << setfill('0') << setw(6) << myid;

      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh->Print(mesh_ofs);

      ofstream sol_ofs(sol_name.str().c_str());
      sol_ofs.precision(8);
      x.Save(sol_ofs);
   }

   // 14. (Optional) Send the solution by socket to a GLVis server.
   char vishost[] = "localhost";
   int  visport   = 19916;
   osockstream sol_sock(visport, vishost);
   sol_sock << "parallel " << num_procs << " " << myid << "\n";
   sol_sock << "solution\n";
   sol_sock.precision(8);
   pmesh->Print(sol_sock);
   x.Save(sol_sock);
   sol_sock.send();

   // 15. Free the used memory.
   delete pcg;
   // delete ams;
   delete X;
   delete B;
   delete A;
   delete fespace;
   delete fec;
   delete pmesh;

   MPI_Finalize();

   return 0;
}

// A parameter for the exact solution.
const double kappa = M_PI/2;

const double sx = 1./3.;
const double sy = 5./7.;
const double sz = 7./23.;

void E_exact(const Vector &x, Vector &E)
{
   double Xx, Yy, Zz;
   Xx = kappa*(x(0) - sx);
   Yy = kappa*(x(1) - sy);
   Zz = kappa*(x(2) - sz);
   E(0) = sin(Xx)*sin(Yy)*cos(Zz);
   E(1) = cos(Xx)*cos(Yy)*cos(Zz);
   E(2) = cos(Xx)*cos(Yy)*sin(Zz);
}

void f_exact(const Vector &x, Vector &f)
{
   double Xx, Yy, Zz, c;
   Xx = kappa*(x(0) - sx);
   Yy = kappa*(x(1) - sy);
   Zz = kappa*(x(2) - sz);
   c = kappa*kappa;
   f(0) = cos(Zz)*sin(Xx)*(-c*cos(Yy) + (1 + 3*c)*sin(Yy));
   f(1) = cos(Xx)*cos(Zz)*((1 + 3*c)*cos(Yy) - c*sin(Yy));
   f(2) = (1 + 2*c)*cos(Xx)*cos(Yy)*sin(Zz);
}
