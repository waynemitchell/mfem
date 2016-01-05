//               Parallel Refinement/Derefinement/Rebalance Demo
//
// Compile with: make rdr-demo
//
// Sample runs:  mpirun -np 4 ./rdr-demo
//
// Description:  This is a demo of the parallel refinement, derefinement and
//               rebalance features including grid function transformations
//               for these operations.
//

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;


void Visualize(socketstream &sout, ParMesh &mesh, ParGridFunction &x, bool pause)
{
   int num_procs, myid;
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   sout.precision(8);
   sout << "parallel " << num_procs << " " << myid << "\n";
   sout << "solution\n" << mesh << x;
   sout << flush;

   if (pause)
   {
#if 0
      sout << "pause\n" << flush;
      if (myid == 0)
      {
         cout << "Visualization paused. Press space in the window to resume.\n";
      }
#else
      if (myid == 0)
      {
         cout << "Press Enter..." << endl;
         cin.ignore();
      }
#endif
   }
   MPI_Barrier(MPI_COMM_WORLD);
}


int main(int argc, char *argv[])
{
   /////// PART 1 -- based on Example 1 ////////////////////////////////////////

   // 1. Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // 2. Parse command-line options.
   const char *mesh_file = "../data/star.mesh";
   int order = 3;
   bool vis = 1;
   bool pause = 0;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&vis, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&pause, "-p", "--pause", "-np", "--no-pause",
                  "Wait for SPACE in the visualization window.");
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

   mesh->EnsureNCMesh();

   // empty processors at the beginning tend to confuse GLVis (?)
   while (mesh->GetNE() < num_procs)
   {
      mesh->UniformRefinement();
   }

   // 5. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted.
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;

   // 6. Define a parallel finite element space on the parallel mesh. Here we
   //    use continuous Lagrange finite elements of the specified order. If
   //    order < 1, we instead use an isoparametric/isogeometric space.
   FiniteElementCollection *fec;
   if (order < 0) { order = 1; }
   fec = new H1_FECollection(order, dim);

   ParFiniteElementSpace *fespace = new ParFiniteElementSpace(pmesh, fec);
   HYPRE_Int size = fespace->GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of unknowns: " << size << endl;
   }

   // 7. Set up the parallel linear form b(.) which corresponds to the
   //    right-hand side of the FEM linear system, which in this case is
   //    (1,phi_i) where phi_i are the basis functions in fespace.
   ParLinearForm *b = new ParLinearForm(fespace);
   ConstantCoefficient one(1.0);
   b->AddDomainIntegrator(new DomainLFIntegrator(one));
   b->Assemble();

   // 8. Define the solution vector x as a parallel finite element grid function
   //    corresponding to fespace. Initialize x with initial guess of zero,
   //    which satisfies the boundary conditions.
   ParGridFunction x(fespace);
   x = 0.0;

   // 9. Set up the parallel bilinear form a(.,.) on the finite element space
   //    corresponding to the Laplacian operator -Delta, by adding the Diffusion
   //    domain integrator and imposing homogeneous Dirichlet boundary
   //    conditions. The boundary conditions are implemented by marking all the
   //    boundary attributes from the mesh as essential. After serial and
   //    parallel assembly we extract the corresponding parallel matrix A.
   ParBilinearForm *a = new ParBilinearForm(fespace);
   a->AddDomainIntegrator(new DiffusionIntegrator(one));
   a->Assemble();
   a->Finalize();

   // 10. Define the parallel (hypre) matrix and vectors representing a(.,.),
   //     b(.) and the finite element approximation.
   HypreParMatrix *A = a->ParallelAssemble();
   HypreParVector *B = b->ParallelAssemble();
   HypreParVector *X = x.ParallelProject();

   // 11. Eliminate essential BC from the parallel system
   Array<int> ess_bdr(pmesh->bdr_attributes.Max());
   ess_bdr = 1;
   a->ParallelEliminateEssentialBC(ess_bdr, *A, *X, *B);

   delete a;
   delete b;

   // 12. Define and apply a parallel PCG solver for AX=B with the BoomerAMG
   //     preconditioner from hypre.
   HypreSolver *amg = new HypreBoomerAMG(*A);
   HyprePCG *pcg = new HyprePCG(*A);
   pcg->SetTol(1e-12);
   pcg->SetMaxIter(200);
   pcg->SetPrintLevel(2);
   pcg->SetPreconditioner(*amg);
   pcg->Mult(*B, *X);

   // 13. Extract the parallel grid function corresponding to the finite element
   //     approximation X. This is the local solution on each processor.
   x = *X;

   // 16. Free the used memory.
   delete pcg;
   delete amg;
   delete X;
   delete B;
   delete A;


   /////// PART 2 -- play with the solution 'x' ////////////////////////////////

   socketstream sout;
   if (vis)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      sout.open(vishost, visport);
      if (!sout)
      {
         cout << "Unable to connect to GLVis server at "
              << vishost << ':' << visport << endl;
         vis = false;
         cout << "GLVis visualization disabled.\n";
      }
   }

   if (vis) { Visualize(sout, *pmesh, x, pause); }

   srand(myid);

   for (int it = 0; it < 100; it++)
   {
      const int levels = 6;
      int frac = (rand() % 3) + 2;

      // refine/rebalance
      for (int i = 0; i < levels; i++)
      {
         pmesh->ncmesh->MarkCoarseLevel();
         pmesh->RandomRefinement(1, frac, false, 1, -1, rand());
         fespace->Update();
         x.Update();

         pmesh->Rebalance();
         fespace->Update();
         x.Update();

         if (vis) { Visualize(sout, *pmesh, x, pause); }
      }

      // derefine/rebalance
      for (int i = 0; i < levels; i++)
      {
         const Table &dtable = pmesh->GetDerefinementTable();
         Array<int> derefs;
         for (int i = 0; i < dtable.Size(); i++) { derefs.Append(i); }
         pmesh->NonconformingDerefinement(derefs);

         fespace->Update();
         x.Update();

         pmesh->Rebalance();
         fespace->Update();
         x.Update();

         if (vis) { Visualize(sout, *pmesh, x, pause); }
      }
   }

   delete fespace;
   delete fec;
   delete pmesh;

   MPI_Finalize();

   return 0;
}
