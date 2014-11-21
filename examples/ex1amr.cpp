//                     MFEM Example 1 - AMR Version
//
// Compile with: make ex1amr
//
// Sample runs:  ex1amr ../data/square-disc.mesh 1
//               ex1amr ../data/square-disc.mesh 2
//               ex1amr ../data/square-disc-nurbs.mesh 2
//               ex1amr ../data/star.mesh 3
//               ex1amr ../data/escher.mesh 1
//               ex1amr ../data/fichera.mesh 2
//               ex1amr ../data/disc-nurbs.mesh 2
//               ex1amr ../data/ball-nurbs.mesh
//               ex1amr ../data/pipe-nurbs.mesh
//
// Description:  This is a follow-up to "ex1.cpp" and a minimal adaptive mesh
//               refinement (AMR) code using MFEM. The problem being solved is
//               again the Laplace equation -\Delta u = 1 with homogeneous
//               Dirichlet boundary conditions. The problem is solved on a
//               sequence of meshes. In each step the mesh is refined locally
//               according to an error estimator.
//
//               The example demostrates MFEM's capability to work with both
//               conforming and nonconforming refinements, in 2D and 3D, linear
//               and curved meshes. Interpolation of functions from coarse to
//               fine meshes is also covered.

#include <fstream>
#include "mfem.hpp"


int main(int argc, char *argv[])
{
   if (argc == 1)
   {
      cout << "Usage: ex1 <mesh_file> [poly_order]" << endl;
      return 1;
   }

   const char* mesh_file = argv[1];
   int poly_order = (argc >= 3) ? atoi(argv[2]) : 1;

   // 1. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral or hexahedral elements with the same code.
   ifstream f(mesh_file);
   if (!f)
   {
      cerr << "Cannot open mesh file: " << mesh_file << endl;
      return 2;
   }
   Mesh mesh(f, 1, 1);
   f.close();

   int dim = mesh.Dimension();

   // 2. Since a NURBS mesh can currently only be refined uniformly, we need to
   //    convert it to a piecewise-polynomial curved mesh. First we refine the
   //    NURBS mesh a bit and then project the curvature to quadratic Nodes.
   if (mesh.NURBSext)
   {
      for (int i = 0; i < 2; i++)
         mesh.UniformRefinement();

      FiniteElementCollection* nfec = new H1_FECollection(2, dim);
      FiniteElementSpace* nfes = new FiniteElementSpace(&mesh, nfec, dim);
      mesh.SetNodalFESpace(nfes);
      mesh.GetNodes()->MakeOwner(nfec);
   }

   // 3. Define a finite element space on the mesh. The polynomial order is
   //    one (linear) by default, but this can be changed on the command line.
   H1_FECollection fec(poly_order, dim);
   FiniteElementSpace fespace(&mesh, &fec);

   // 4. As in ex1.cpp, we set up bilinear and linear forms corresponding to
   //    the Laplace problem -\Delta u = 1. We don't assemble the discrete
   //    problem yet, this will be done in the main loop.
   BilinearForm a(&fespace);
   LinearForm b(&fespace);

   ConstantCoefficient one(1.0);
   ConstantCoefficient zero(0.0);

   a.AddDomainIntegrator(new DiffusionIntegrator(one));
   b.AddDomainIntegrator(new DomainLFIntegrator(one));

   // 5. The solution vector x and the associated finite element grid function
   //    will be maintained over the AMR iterations. We initialize it to zero.
   GridFunction x(&fespace);
   x = 0;

   // 6. All boundary attributes will be used for essential (Dirichlet) BC.
   Array<int> ess_bdr(mesh.bdr_attributes.Max());
   ess_bdr = 1;

   // 7. Connect to GLVis
   char vishost[] = "localhost";
   int  visport   = 19916;
   osockstream sol_sock(visport, vishost);

   // 8. The main AMR loop. In each iteration we solve the problem on the
   //    current mesh, visualize the solution, estimate the error on all
   //    elements, refine the worst elements and update all objects to work
   //    with the new mesh.
   const int max_it = 15;
   for (int it = 0; it < max_it; it++)
   {
      cout << "\nIteration " << it << endl;
      cout << "Number of unknowns: " << fespace.GetNConformingDofs() << endl;

      // 9. Assemble the stiffness matrix and the right-hand side. Note that
      //    MFEM doesn't care at this point if the mesh is nonconforming
      //    (i.e., contains hanging nodes). The FE space is considered 'cut'
      //    along hanging edges/faces.
      a.Assemble();
      b.Assemble();

      x.ProjectBdrCoefficient(zero, ess_bdr);

      // 10. Take care of nonconforming meshes by applying the interpolation
      //     matrix P to a, b and x, so that slave degrees of freedom get
      //     eliminated from the linear system. The system becomes P'AP x = P'b.
      //     (If the mesh is conforming this step does nothing.)
      a.ConformingAssemble(x, b);

      // 11. As usual, we also need to eliminate the essential BC from the
      //     system. This needs to be done after ConformingAssemble.
      a.EliminateEssentialBC(ess_bdr, x, b);

      const SparseMatrix &A = a.SpMat();
#ifndef MFEM_USE_SUITESPARSE
      // 12. Define a simple symmetric Gauss-Seidel preconditioner and use it to
      //     solve the linear system with PCG.
      GSSmoother M(A);
      PCG(A, M, b, x, 1, 200, 1e-12, 0.0);
#else
      // 12. If MFEM was compiled with SuiteSparse, use UMFPACK to solve the system.
      UMFPackSolver umf_solver;
      umf_solver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
      umf_solver.SetOperator(A);
      umf_solver.Mult(*b, x);
#endif

      // 13. For nonconforming meshes, bring the solution vector back from
      //     the conforming space to the nonconforming (cut) space, i.e.,
      //     x = Px. Slave DOFs receive the correct values to make the solution
      //     continuous. (This step does nothing if the mesh is conforming.)
      x.ConformingProlongate();

      // 14. Send solution by socket to the GLVis server.
      sol_sock << "solution\n";
      sol_sock.precision(8);
      mesh.Print(sol_sock);
      x.Save(sol_sock);
      sol_sock.send();

      // 15. Estimate element errors using the Zienkiewicz-Zhu error estimator.
      //     The bilinear form integrator must have 'ComputeElementFlux' defined.
      Vector errors(mesh.GetNE());
      {
         FiniteElementSpace flux_fespace(&mesh, &fec, dim);
         DiffusionIntegrator flux_integrator(one);
         GridFunction flux(&flux_fespace);
         ComputeFlux(flux_integrator, x, flux);
         ZZErrorEstimator(flux_integrator, x, flux, errors, 1);
      }

      // 16. Make a list of elements whose error is larger than some fraction of
      //     the maximum element error. These elements will be refined.
      Array<int> ref_list;
      double emax = errors.Max();
      for (int i = 0; i < errors.Size(); i++)
         if (errors[i] >= 0.7*emax)
            ref_list.Append(i);

      // 17. Refine the selected elements. Since we are going to transfer the
      //     grid function x from the coarse mesh to the new fine mesh in the
      //     next step, we need to request the "two-level state" of the mesh.
      mesh.UseTwoLevelState(1);
      mesh.GeneralRefinement(ref_list);

      // 18. Update the space to reflect the new state of the mesh. Also,
      //     interpolate the solution x so that it lies in the new space but
      //     represents the same function. This saves solver iterations since
      //     we'll have a good initial guess of x in the next step.
      //     The interpolation algorithm needs the mesh to hold some information
      //     about the previous state, which is why the call UseTwoLevelState
      //     above is required.
      fespace.UpdateAndInterpolate(&x);

      // Note: If interpolation was not needed, we could just use the following
      //     two calls to update the space and the grid function. (No need for
      //     UseTwoLevelState in this case.)
      //fespace.Update();
      //x.Update();

      // 19. Inform also the bilinear and linear forms that the space has
      //     changed.
      a.Update();
      b.Update();
   }

   return 0;
}

