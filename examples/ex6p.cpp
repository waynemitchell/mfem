//                       MFEM Example 6 - Parallel Version
//
// Compile with: make ex6p
//
// Sample runs:  mpirun -np 4 ex6p -m ../data/square-disc.mesh -o 1
//               mpirun -np 4 ex6p -m ../data/square-disc.mesh -o 2
//               mpirun -np 4 ex6p -m ../data/square-disc-nurbs.mesh -o 2
//               mpirun -np 4 ex6p -m ../data/star.mesh -o 3
//               mpirun -np 4 ex6p -m ../data/escher.mesh -o 1
//               mpirun -np 4 ex6p -m ../data/fichera.mesh -o 2
//               mpirun -np 4 ex6p -m ../data/disc-nurbs.mesh -o 2
//               mpirun -np 4 ex6p -m ../data/ball-nurbs.mesh
//               mpirun -np 4 ex6p -m ../data/pipe-nurbs.mesh
//               mpirun -np 4 ex6p -m ../data/star-surf.mesh -o 2
//               mpirun -np 4 ex6p -m ../data/square-disc-surf.mesh -o 2
//
// Description:  This is a version of Example 1 with a simple adaptive mesh
//               refinement loop. The problem being solved is again the Laplace
//               equation -Delta u = 1 with homogeneous Dirichlet boundary
//               conditions. The problem is solved on a sequence of meshes which
//               are locally refined in a conforming (triangles, tetrahedrons)
//               or non-conforming (quadrilateral, hexahedrons) manner according
//               to a simple ZZ error estimator.
//
//               The example demonstrates MFEM's capability to work with both
//               conforming and nonconforming refinements, in 2D and 3D, on
//               linear, curved and surface meshes. Interpolation of functions
//               from coarse to fine meshes, as well as persistent GLVis
//               visualization are also illustrated.
//
//               We recommend viewing Example 1 before viewing this example.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

#include "HYPRE_sstruct_ls.h"

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
   const char *mesh_file = "../data/star.mesh";
   int order = 1;
   bool visualization = true;

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

   // 3. Refine the serial mesh on all processors to increase the resolution.
   //    Also project a NURBS mesh to a piecewise-quadratic curved mesh.
   for (int i = 0; i < 1; i++)
   {
      mesh->UniformRefinement();
   }

   if (mesh->NURBSext)
   {
      FiniteElementCollection* nfec = new H1_FECollection(2, dim);
      FiniteElementSpace* nfes = new FiniteElementSpace(mesh, nfec, dim);
      mesh->SetNodalFESpace(nfes);
      mesh->GetNodes()->MakeOwner(nfec);
   }

   mesh->GeneralRefinement(Array<Refinement>()); // FIXME

   // 4. Define a parallel mesh by partitioning the serial mesh.
   //    Once the parallel mesh is defined, the serial mesh can be deleted.
   ParMesh pmesh(MPI_COMM_WORLD, *mesh);
   delete mesh;

   // 5. Define a finite element space on the mesh. The polynomial order is
   //    one (linear) by default, but this can be changed on the command line.
   H1_FECollection fec(order, dim);
   ParFiniteElementSpace fespace(&pmesh, &fec);

   // 6. As in Example 1p, we set up bilinear and linear forms corresponding to
   //    the Laplace problem -\Delta u = 1. We don't assemble the discrete
   //    problem yet, this will be done in the main loop.
   ParBilinearForm a(&fespace);
   ParLinearForm b(&fespace);

   ConstantCoefficient one(1.0);
   ConstantCoefficient zero(0.0);

   a.AddDomainIntegrator(new DiffusionIntegrator(one));
   b.AddDomainIntegrator(new DomainLFIntegrator(one));

   // 7. The solution vector x and the associated finite element grid function
   //    will be maintained over the AMR iterations. We initialize it to zero.
   ParGridFunction x(&fespace);
   x = 0;

   Array<int> ess_bdr(pmesh.bdr_attributes.Max());
   ess_bdr = 1;

   // 9. Connect to GLVis.
   char vishost[] = "localhost";
   int  visport   = 19916;

   socketstream sout;
   if (visualization)
   {
      sout.open(vishost, visport);
      if (!sout)
      {
         if (myid == 0)
         {
            cout << "Unable to connect to GLVis server at "
                 << vishost << ':' << visport << endl;
            cout << "GLVis visualization disabled.\n";
         }
         visualization = false;
      }

      sout.precision(8);
      /*if (myid == 0)
      {
         sout << "keys Am\n";
      }*/
   }

   // 9. The main AMR loop. In each iteration we solve the problem on the
   //    current mesh, visualize the solution, estimate the error on all
   //    elements, refine the worst elements and update all objects to work
   //    with the new mesh.
   const int max_it = 25;
   for (int it = 0; it < max_it; it++)
   {
      if (myid == 0)
      {
         cout << "\nIteration " << it << endl;
         cout << "Number of unknowns: " << fespace.GetNConformingDofs() << endl;
      }

      if (myid == 0) { tic(); }

      fespace.Dof_TrueDof_Matrix();

      if (myid == 0)
      {
         cout << "P matrix time: " << tic_toc.RealTime() << endl;
      }

      // 10. Assemble the stiffness matrix and the right-hand side. Note that
      //     MFEM doesn't care at this point if the mesh is nonconforming (i.e.,
      //     contains hanging nodes). The FE space is considered 'cut' along
      //     hanging edges/faces.
      a.Assemble();
      a.Finalize();
      b.Assemble();

      //x.ProjectBdrCoefficient(zero, ess_bdr);
      x = 0;

      HypreParMatrix *A = a.ParallelAssemble();
      HypreParVector *B = b.ParallelAssemble();
      HypreParVector *X = x.ParallelAverage();

      // 12. As usual, we also need to eliminate the essential BC from the
      //     system. This needs to be done after ConformingAssemble.
      a.ParallelEliminateEssentialBC(ess_bdr, *A, *X, *B);

      // 11. Define and apply a parallel PCG solver for AX=B with the BoomerAMG
      //     preconditioner from hypre.
      HypreBoomerAMG amg(*A);
      HyprePCG pcg(*A);
      pcg.SetTol(1e-12);
      pcg.SetMaxIter(200);
      pcg.SetPrintLevel(2);
      pcg.SetPreconditioner(amg);
      pcg.Mult(*B, *X);

      // 12. Extract the parallel grid function corresponding to the finite element
      //     approximation X. This is the local solution on each processor.
      x = *X;

      // 14. Send the solution by socket to a GLVis server.
      if (visualization)
      {
         sout << "parallel " << num_procs << " " << myid << "\n";
         sout << "solution\n" << pmesh << x << flush;
      }

      // 16. Estimate element errors using the Zienkiewicz-Zhu error estimator.
      //     The bilinear form integrator must have the 'ComputeElementFlux'
      //     method defined.
      Vector errors(pmesh.GetNE());
      {
         ParFiniteElementSpace flux_fespace(&pmesh, &fec, dim);
         DiffusionIntegrator flux_integrator(one);
         ParGridFunction flux(&flux_fespace);
         ZZErrorEstimator(flux_integrator, x, flux, errors);
      }

      // 17. Make a list of elements whose error is larger than a fraction
      //     of the maximum element error. These elements will be refined.
      Array<int> ref_list;
      const double frac = 0.7;
      // the 'errors' are squared, so we need to square the fraction
      double threshold = (frac*frac) * errors.Max();
      for (int i = 0; i < errors.Size(); i++)
         if (errors[i] >= threshold)
         {
            ref_list.Append(i);
         }

      // 18. Refine the selected elements. Since we are going to transfer the
      //     grid function x from the coarse mesh to the new fine mesh in the
      //     next step, we need to request the "two-level state" of the mesh.
      //pmesh.UseTwoLevelState(1);
      pmesh.GeneralRefinement(ref_list);

      // 19. Update the space to reflect the new state of the mesh. Also,
      //     interpolate the solution x so that it lies in the new space but
      //     represents the same function. This saves solver iterations since
      //     we'll have a good initial guess of x in the next step.
      //     The interpolation algorithm needs the mesh to hold some information
      //     about the previous state, which is why the call UseTwoLevelState
      //     above is required.
      //fespace.UpdateAndInterpolate(&x);

      // Note: If interpolation was not needed, we could just use the following
      //     two calls to update the space and the grid function. (No need to
      //     call UseTwoLevelState in this case.)
      fespace.Update();
      x.Update();

      // 20. Inform also the bilinear and linear forms that the space has
      //     changed.
      a.Update();
      b.Update();

      delete A;
      delete B;
      delete X;

   }

   MPI_Finalize();
   return 0;
}
