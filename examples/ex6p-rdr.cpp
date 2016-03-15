//                    MFEM Example 6 (RDR) - Parallel Version
//
// Compile with: make ex6p-rdr
//
// Sample runs:  mpirun -np 4 ex6p-rdr -m ../data/square-disc.mesh -o 1
//               mpirun -np 4 ex6p-rdr -m ../data/square-disc.mesh -o 2
//               mpirun -np 4 ex6p-rdr -m ../data/square-disc-nurbs.mesh -o 2
//               mpirun -np 4 ex6p-rdr -m ../data/star.mesh -o 3
//               mpirun -np 4 ex6p-rdr -m ../data/escher.mesh -o 1
//               mpirun -np 4 ex6p-rdr -m ../data/fichera.mesh -o 2
//               mpirun -np 4 ex6p-rdr -m ../data/disc-nurbs.mesh -o 2
//               mpirun -np 4 ex6p-rdr -m ../data/ball-nurbs.mesh
//               mpirun -np 4 ex6p-rdr -m ../data/pipe-nurbs.mesh
//               mpirun -np 4 ex6p-rdr -m ../data/star-surf.mesh -o 2
//               mpirun -np 4 ex6p-rdr -m ../data/square-disc-surf.mesh -o 2
//               mpirun -np 4 ex6p-rdr -m ../data/amr-quad.mesh
//
// Description:  This is a version of Example 1 with a simple adaptive mesh
//               refinement/derefinement/rebalance loop. The problem being
//               solved is again the Laplace equation -Delta u = f with
//               homogeneous Dirichlet boundary conditions. At each outer
//               iteration the right hand side function is changed to mimic a
//               time dependent problem.  Within each outer iteration the
//               problem is solved on a sequence of meshes which are locally
//               refined in a conforming (triangles, tetrahedrons) or
//               non-conforming (quadrilateral, hexahedrons) manner according to
//               a simple ZZ error estimator.  At the end of the outer iteration
//               the ZZ error estimator is used to identify any elements which
//               may be over-refined and a single derefinement step is
//               performed.  After each refinement or derefinement a rebalance
//               operation is performed to keep the mesh evenly distributed
//               amongst the available processors.
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

using namespace std;
using namespace mfem;

double exp_func(const Vector & pt, double t)
{
   const double w = 0.02;
   double x = pt(0), y = pt(1);
   double t1 = x*x;
   double t2 = y*y;
   double t4 = sqrt(t1+t2);
   double t6 = pow(t4-t,2.0);
   double t7 = w*w;
   double t11 = exp(-1/t7*t6/2.0);
   double t18 = t*t;
   double t25 = t7*t7;
   double t29 = 1/t4/t25*(t1*t4-2.0*t1*t-2.0*t4*t7+t*t7+t18*t4-2.0*t2*t+t2*t4)*t11;
   return -t29;
}

void ComputeField(ParBilinearForm & a, ParLinearForm & b,
                  ParFiniteElementSpace & fespace, Array<int> & ess_bdr,
                  ParGridFunction & x);

double EstimateErrors(int order, int dim, int sdim, ParMesh & pmesh,
                      const ParGridFunction & x, Vector & errors);

void UpdateAndRebalance(ParMesh &pmesh, ParFiniteElementSpace &fespace,
                        ParGridFunction &x, ParBilinearForm &a, ParLinearForm &b);


int main(int argc, char *argv[])
{
   // 1. Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // 2. Parse command-line options.
   const char *mesh_file = "../data/star.mesh";
   int order = 2;
   double max_elem_error = 1.0e-4;
   double hysteresis = 0.25; // derefinement safety coefficient
   int nc_limit = 3;
   bool visualization = true;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&max_elem_error, "-e", "--max-err",
                  "Maximum element error");
   args.AddOption(&hysteresis, "-y", "--hysteresis",
                  "Derefinement safety coefficent.");
   args.AddOption(&nc_limit, "-l", "--nc-limit",
                  "Maximum level of hanging nodes.");
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
   int sdim = mesh->SpaceDimension();

   // 4. Refine the serial mesh on all processors to increase the resolution.
   //    Also project a NURBS mesh to a piecewise-quadratic curved mesh. Make
   //    sure that the mesh is non-conforming.
   if (mesh->NURBSext)
   {
      mesh->UniformRefinement();
      mesh->SetCurvature(2);
   }
   mesh->EnsureNCMesh();

   // 5. Define a parallel mesh by partitioning the serial mesh.  Once the
   //    parallel mesh is defined, the serial mesh can be deleted.
   ParMesh pmesh(MPI_COMM_WORLD, *mesh);
   delete mesh;

   MFEM_VERIFY(pmesh.bdr_attributes.Size() > 0,
               "Boundary attributes required in the mesh.");
   Array<int> ess_bdr(pmesh.bdr_attributes.Max());
   ess_bdr = 1;

   // 6. Define a finite element space on the mesh. The polynomial order is one
   //    (linear) by default, but this can be changed on the command line.
   H1_FECollection fec(order, dim);
   ParFiniteElementSpace fespace(&pmesh, &fec);

   // 7. As in Example 1p, we set up bilinear and linear forms corresponding to
   //    the Laplace problem -\Delta u = 1. We don't assemble the discrete
   //    problem yet, this will be done in the main loop.
   ParBilinearForm a(&fespace);
   ParLinearForm b(&fespace);

   ConstantCoefficient one(1.0);
   FunctionCoefficient rhs(exp_func);

   a.AddDomainIntegrator(new DiffusionIntegrator(one));
   b.AddDomainIntegrator(new DomainLFIntegrator(rhs));

   // 8. The solution vector x and the associated finite element grid function
   //    will be maintained over the AMR iterations. We initialize it to zero.
   ParGridFunction x(&fespace);
   x = 0;

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
   }

   // 10. The main time loop. In each iteration we update the right hand side,
   //     solve the problem on the current mesh, visualize the solution,
   //     estimate the error on all elements, refine bad elements and
   //     update all objects to work with the new mesh.  The recompute the
   //     errors and derefine any elements which have very small errors.
   for (double time = 0.0; time < 0.9; time += 0.01)
   {
      if (myid == 0)
      {
         cout << "\nTime " << time << "\n\nRefinement:" << endl;
      }

      // Set the current time in the source term
      rhs.SetTime(time);

      Vector errors;

      // 11. The main refinement loop
      for (int ref_it = 1; ; ref_it++)
      {
         if (myid == 0)
         {
            cout << "Iteration: " << ref_it << ", number of unknowns: "
                 << fespace.GlobalTrueVSize() << flush;
         }

         // 11a. Recompute the field on the current mesh
         ComputeField(a, b, fespace, ess_bdr, x);

         // 11b. Send the solution by socket to a GLVis server.
         if (visualization)
         {
            sout << "parallel " << num_procs << " " << myid << "\n";
            sout << "solution\n" << pmesh << x << flush;
         }
         // 11c. Estimate element errors using the Zienkiewicz-Zhu error
         //      estimator. The bilinear form integrator must have the
         //      'ComputeElementFlux' method defined.
         double tot_error = EstimateErrors(order, dim, sdim, pmesh, x, errors);

         if (myid == 0)
         {
            cout << ", total error: " << tot_error << endl;
         }

         // 11d. Refine elements
         if (!pmesh.RefineByError(errors, max_elem_error, -1, nc_limit))
         {
            break;
         }

         // 11e. Update the space and interpolate the solution,
         //      load balance the mesh.
         UpdateAndRebalance(pmesh, fespace, x, a, b);
      }

      // 12. Derefine any elements if desirable
      if (pmesh.Nonconforming())
      {
         // 12a. Derefine any elements with small enough errors if possible
         double threshold = hysteresis * max_elem_error;
         if (pmesh.DerefineByError(errors, threshold, nc_limit))
         {
            if (myid == 0)
            {
               cout << "\nDerefined elements." << endl;
            }

            // 12b. Update the space and interpolate the solution,
            //      load balance the mesh.
            UpdateAndRebalance(pmesh, fespace, x, a, b);
         }
      }
   }

   // 13. Exit
   MPI_Finalize();
   return 0;
}

void ComputeField(ParBilinearForm & a, ParLinearForm & b,
                  ParFiniteElementSpace & fespace, Array<int> & ess_bdr,
                  ParGridFunction & x)
{
   // Assemble the stiffness matrix and the right-hand side. Note that
   // MFEM doesn't care at this point that the mesh is nonconforming
   // and parallel. The FE space is considered 'cut' along hanging
   // edges/faces, and also across processor boundaries.
   a.Assemble();
   b.Assemble();

   // Set the initial estimate of the solution and the Dirichlet DOFs,
   // here we just use zero everywhere.
   x = 0.0;

   // Create the parallel linear system: eliminate boundary conditions,
   // constrain hanging nodes and nodes across processor boundaries.
   // The system will be solved for true (unconstrained/unique) DOFs only.
   Array<int> ess_tdof_list;
   fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   HypreParMatrix A;
   Vector B, X;
   a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

   // Define and apply a parallel PCG solver for AX=B with the BoomerAMG
   // preconditioner from hypre.
   HypreBoomerAMG amg(A);
   amg.SetPrintLevel(0);
   HyprePCG pcg(A);
   pcg.SetTol(1e-12);
   pcg.SetMaxIter(200);
   pcg.SetPrintLevel(0);
   pcg.SetPreconditioner(amg);
   pcg.Mult(B, X);

   // Extract the parallel grid function corresponding to the finite element
   // approximation X. This is the local solution on each processor.
   a.RecoverFEMSolution(X, b, x);
}

double EstimateErrors(int order, int dim, int sdim, ParMesh & pmesh,
                      const ParGridFunction & x, Vector & errors)
{
   // Space for the discontinuous (original) flux
   DiffusionIntegrator flux_integrator;
   L2_FECollection flux_fec(order, dim);
   ParFiniteElementSpace flux_fes(&pmesh, &flux_fec, sdim);

   // Space for the smoothed (conforming) flux
   double norm_p = 1;
   RT_FECollection smooth_flux_fec(order-1, dim);
   ParFiniteElementSpace smooth_flux_fes(&pmesh, &smooth_flux_fec);

   // Another possible set of options for the smoothed flux space:
   // norm_p = 1;
   // H1_FECollection smooth_flux_fec(order, dim);
   // ParFiniteElementSpace smooth_flux_fes(&pmesh, &smooth_flux_fec, dim);

   return L2ZZErrorEstimator(flux_integrator, x,
                             smooth_flux_fes, flux_fes, errors, norm_p);
}

void UpdateAndRebalance(ParMesh &pmesh, ParFiniteElementSpace &fespace,
                        ParGridFunction &x, ParBilinearForm &a, ParLinearForm &b)
{
   // Update the space and interpolate the solution.
   fespace.Update();
   x.Update();

   // Load balance the mesh.
   if (pmesh.Nonconforming())
   {
      pmesh.Rebalance();

      // Updat the space and solution again.
      fespace.Update();
      x.Update();
   }

   a.Update();
   b.Update();
}
