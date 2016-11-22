//                       MFEM Example 1 - Parallel Version
//
// Compile with: make checkpoint
//
// Saving a checkpoint file:
// mpirun -np 4 checkpoint
//
// Loading a checkpoint file:
// mpirun -np 4 checkpoint -c output/problem.root
//
// Description:
// The example highlights the use of dumping a checkpoint file
// using the sidre data collection, and of loading the grid
// functions back in from the checkpoint file.
//
// It is based on the ex1p problem.

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
   const char *mesh_file = "../../data/star-q2.mesh";
   int order = 1;
   bool static_cond = false;
   bool visualization = 1;
   const char * checkpoint_file = "";

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&checkpoint_file, "-c", "--checkpoint",
                  "Restore problem state from checkpoint file.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
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

   bool isRestart = (checkpoint_file != "");

   // 3. Read the (serial) mesh from the given mesh file on all processors.  We
   //    can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
   //    and volume meshes with the same code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   // 4. Refine the serial mesh on all processors to increase the resolution. In
   //    this example we do 'ref_levels' of uniform refinement. We choose
   //    'ref_levels' to be the largest number that gives a final mesh with no
   //    more than 10,000 elements.
   {
      int ref_levels =
         (int)floor(log(10000./mesh->GetNE())/log(2.)/dim);
      for (int l = 0; l < ref_levels; l++)
      {
         mesh->UniformRefinement();
      }
   }

   // 5. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted.
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   {
      int par_ref_levels = 2;
      for (int l = 0; l < par_ref_levels; l++)
      {
         pmesh->UniformRefinement();
      }
   }
   // Create sidre data collection.  Designate that sidre DC should own the
   // memory for the mesh by passing in 'true' for owns_mesh_data param.  This
   // is required to support restarting using the sidre DC.
   SidreDataCollection * dc = new SidreDataCollection("problem", pmesh, false);
   dc->SetPrefixPath("output");

   // 6. Define a parallel finite element space on the parallel mesh. Here we
   //    use continuous Lagrange finite elements of the specified order. If
   //    order < 1, we instead use an isoparametric/isogeometric space.
   FiniteElementCollection *fec;
   if (order > 0)
   {
      fec = new H1_FECollection(order, dim);
   }
   else if (pmesh->GetNodes())
   {
      fec = pmesh->GetNodes()->OwnFEC();
      if (myid == 0)
      {
         cout << "Using isoparametric FEs: " << fec->Name() << endl;
      }
   }
   else
   {
      fec = new H1_FECollection(order = 1, dim);
   }
   ParFiniteElementSpace *fespace = new ParFiniteElementSpace(pmesh, fec);
   HYPRE_Int size = fespace->GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of finite element unknowns: " << size << endl;
   }

   // 6.  If checkpoint file path provided, then load problem state and verify
   // solution PGF matches output from normal run.
   if (isRestart)
   {
      dc->Load(checkpoint_file, "sidre_hdf5");
   }

   // 7. Define the solution vector x as a parallel finite element grid function
   //    corresponding to fespace. Initialize x with initial guess of zero,
   //    which satisfies the boundary conditions.
   //
   ParGridFunction x;
   DataCollectionUtility::AllocateGridFunc(&x,
                                           fespace,
                                           dc,
                                           "solution",
                                           -1,
                                           !isRestart);

   //   ParGridFunction x(fespace);
   //   x = 0.0;

   // Register the solution PGF with the data collection so it will be written out/read in
   // during restart.
   //dc->RegisterField("solution", &x);

   // Only continue with the problem if not loading in the checkpoint file.
   if (!isRestart)
   {
      // 8. Determine the list of true (i.e. parallel conforming) essential
      //    boundary dofs. In this example, the boundary conditions are defined
      //    by marking all the boundary attributes from the mesh as essential
      //    (Dirichlet) and converting them to a list of true dofs.
      Array<int> ess_tdof_list;
      if (pmesh->bdr_attributes.Size())
      {
         Array<int> ess_bdr(pmesh->bdr_attributes.Max());
         ess_bdr = 1;
         fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
      }

      // 9. Set up the parallel linear form b(.) which corresponds to the
      //    right-hand side of the FEM linear system, which in this case is
      //    (1,phi_i) where phi_i are the basis functions in fespace.
      ParLinearForm *b = new ParLinearForm(fespace);
      ConstantCoefficient one(1.0);
      b->AddDomainIntegrator(new DomainLFIntegrator(one));
      b->Assemble();

      // 10. Set up the parallel bilinear form a(.,.) on the finite element space
      //     corresponding to the Laplacian operator -Delta, by adding the Diffusion
      //     domain integrator.
      ParBilinearForm *a = new ParBilinearForm(fespace);
      a->AddDomainIntegrator(new DiffusionIntegrator(one));

      // 11. Assemble the parallel bilinear form and the corresponding linear
      //     system, applying any necessary transformations such as: parallel
      //     assembly, eliminating boundary conditions, applying conforming
      //     constraints for non-conforming AMR, static condensation, etc.
      if (static_cond) { a->EnableStaticCondensation(); }
      a->Assemble();

      HypreParMatrix A;
      Vector B, X;
      a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);

      if (myid == 0)
      {
         cout << "Size of linear system: " << A.GetGlobalNumRows() << endl;
      }

      // 12. Define and apply a parallel PCG solver for AX=B with the BoomerAMG
      //     preconditioner from hypre.
      HypreSolver *amg = new HypreBoomerAMG(A);
      HyprePCG *pcg = new HyprePCG(A);
      pcg->SetTol(1e-12);
      pcg->SetMaxIter(200);
      pcg->SetPrintLevel(2);
      pcg->SetPreconditioner(*amg);
      pcg->Mult(B, X);

      // 13. Recover the parallel grid function corresponding to X. This is the
      //     local finite element solution on each processor.
      a->RecoverFEMSolution(X, *b, x);
      // Save data collection contents to checkpoint/viz file ( mesh, and registered GFs ).
      dc->Save();
   }

   // 14. Save the refined mesh and the solution in parallel. This output can
   //     be viewed later using GLVis: "glvis -np <np> -m mesh -g sol".
   {
      std::string suffix;
      if (isRestart)
      {
         suffix = ".post";
      }
      else
      {
         suffix = ".pre";
      }

      ostringstream mesh_name, sol_name;
      mesh_name << "output/mesh." << setfill('0') << setw(6) << myid << suffix;
      sol_name << "output/sol." << setfill('0') << setw(6) << myid << suffix;

      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh->Print(mesh_ofs);

      ofstream sol_ofs(sol_name.str().c_str());
      sol_ofs.precision(8);
      x.Save(sol_ofs);
   }


   // 15. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << *pmesh << x << flush;
   }

#if 0
   // 16. Free the used memory.
   delete pcg;
   delete amg;
   delete a;
   delete b;
   delete fespace;
   if (order > 0) { delete fec; }
   delete pmesh;
#endif
   MPI_Finalize();

   return 0;
}
