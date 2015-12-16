//                       MFEM Example ?? - Parallel Version
//
// Compile with: make ex??p
//
// Sample runs:  mpirun -np 4 ex??p -m ../data/square-torus-n1.mesh
//               mpirun -np 4 ex??p -m ../data/square-angled-pipe-n1.mesh
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
   int ref_levels = 1;
   bool visualization = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&ref_levels, "-r", "--ref-levels",
                  "Number of refinement levels.");
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

   if (mesh->MeshGenerator() == 2)
   {
      mesh->GeneralRefinement(Array<Refinement>(), 1);
   }

   // 4. Refine the serial mesh on all processors to increase the resolution. In
   //    this example we do 'ref_levels' of uniform refinement. We choose
   //    'ref_levels' to be the largest number that gives a final mesh with no
   //    more than 100 elements.
   {
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
      int par_ref_levels = 0;
      for (int l = 0; l < par_ref_levels; l++)
      {
         pmesh->UniformRefinement();
      }
   }
   if (order >= 2)
   {
      pmesh->ReorientTetMesh();
   }

   socketstream curl_sock, flux_sock, err_sock;
   char vishost[] = "localhost";
   int  visport   = 19916;
   if (visualization)
   {
      curl_sock.open(vishost, visport);
      curl_sock.precision(8);
      //curl_sock << "window_title 'Magnetic Field'" << flush;

      MPI_Barrier(MPI_COMM_WORLD);

      flux_sock.open(vishost, visport);
      flux_sock.precision(8);

      MPI_Barrier(MPI_COMM_WORLD);

      err_sock.open(vishost, visport);
      err_sock.precision(8);
   }

   for (int it = 1; it <= 100; it++)
   {
      // 6. Define a parallel finite element space on the parallel mesh. Here we
      //    use the lowest order Nedelec finite elements, but we can easily switch
      //    to higher-order spaces by changing the value of p.
      FiniteElementCollection *H1FEC        = new H1_FECollection(order, dim);
      FiniteElementCollection *HCurlFEC     = new ND_FECollection(order, dim);
      FiniteElementCollection *HDivFEC      = new RT_FECollection(order-1, dim);

      ParFiniteElementSpace   *H1FESpace    = new ParFiniteElementSpace(pmesh,
                                                                        H1FEC);
      ParFiniteElementSpace   *HCurlFESpace = new ParFiniteElementSpace(pmesh,
                                                                        HCurlFEC);
      ParFiniteElementSpace   *HDivFESpace  = new ParFiniteElementSpace(pmesh,
                                                                        HDivFEC);

      HYPRE_Int size = HCurlFESpace->GlobalTrueVSize();
      if (myid == 0)
      {
         cout << "Number of unknowns: " << size << endl;
      }

      // 7. - epsilon
      ParLinearForm *b1 = new ParLinearForm(H1FESpace);
      b1->Assemble();

      ParGridFunction x1(H1FESpace);
      ConstantCoefficient one(1.0);
      ConstantCoefficient one_half(0.5);
      ConstantCoefficient neg_one_half(-0.5);
      x1 = 0.0;
      Array<int> ess_bdr1_p(pmesh->bdr_attributes.Max());
      Array<int> ess_bdr1_n(pmesh->bdr_attributes.Max());
      ess_bdr1_p = 0; ess_bdr1_p[1] = 1;
      ess_bdr1_n = 0; ess_bdr1_n[0] = 1;
      x1.ProjectBdrCoefficient(one_half,ess_bdr1_p);
      x1.ProjectBdrCoefficient(neg_one_half,ess_bdr1_n);

      ParBilinearForm *a1 = new ParBilinearForm(H1FESpace);
      PWConstCoefficient *sigma =
         new PWConstCoefficient(pmesh->bdr_attributes.Max());
      (*sigma) = 0.0; (*sigma)(3) = 1.0;
      a1->AddBoundaryIntegrator(new DiffusionIntegrator(*sigma));
      a1->Assemble();
      a1->Finalize();

      HypreParMatrix *A1 = a1->ParallelAssemble();
      HypreParVector *B1 = b1->ParallelAssemble();
      HypreParVector *X1 = x1.ParallelProject();

      Array<int> ess_bdr1(pmesh->bdr_attributes.Max());
      ess_bdr1 = 0; ess_bdr1[0] = 1; ess_bdr1[1] = 1;
      // a1->EliminateEssentialBC(ess_bdr1, x1, *b1);
      a1->ParallelEliminateEssentialBC(ess_bdr1, *A1, *X1, *B1);

      delete a1;
      delete b1;
      delete sigma;

      HypreBoomerAMG *amg1 = new HypreBoomerAMG(*A1);
      amg1->SetPrintLevel(0);
      HyprePCG *pcg1 = new HyprePCG(*A1);
      pcg1->SetTol(1e-12);
      pcg1->SetMaxIter(200);
      pcg1->SetPrintLevel(2);
      pcg1->SetPreconditioner(*amg1);
      pcg1->Mult(*B1, *X1);
      delete pcg1;
      delete amg1;

      x1 = *X1;

      ParDiscreteLinearOperator *grad =
         new ParDiscreteLinearOperator(H1FESpace, HCurlFESpace);
      grad->AddDomainInterpolator(new GradientInterpolator);
      grad->Assemble();
      grad->Finalize();
      HypreParMatrix *Grad = grad->ParallelAssemble();
      HypreParVector *GradX1 = new HypreParVector(HCurlFESpace);
      Grad->Mult(*X1,*GradX1);
      ParGridFunction gradx1(HCurlFESpace,GradX1);

      delete GradX1;
      delete Grad;
      delete grad;
      delete X1;
      delete B1;
      delete A1;

      // 7. Set up the parallel linear form b(.) which corresponds to the
      //    right-hand side of the FEM linear system, which in this case is
      //    zero.
      ParLinearForm *b = new ParLinearForm(HCurlFESpace);
      b->Assemble();

      // 8. Define the solution vector x as a parallel finite element
      // grid function corresponding to HCurlFESpace. Initialize x by
      // projecting the boundary conditions onto the appropriate edges.
      ParGridFunction x(HCurlFESpace);
      Vector vZero(3); vZero = 0.0;
      VectorConstantCoefficient Zero(vZero);
      Array<int> ess_bdr_1(pmesh->bdr_attributes.Max());
      ess_bdr_1 = 1; ess_bdr_1[2] = 0;

      // Set x to be the unit vector in the z direction on the first surface
      x = gradx1;
      x.ProjectBdrCoefficientTangent(Zero,ess_bdr_1);

      // 9. Set up the parallel bilinear form corresponding to the
      // magnetostatic operator curl muinv curl, by adding the curl-curl
      // domain integrator and finally imposing non-homogeneous Dirichlet
      // boundary conditions. The boundary conditions are implemented by
      // marking all the boundary attributes from the mesh as essential
      // (Dirichlet). After serial and parallel assembly we extract the
      // parallel matrix A.
      ConstantCoefficient muinv(1.0);
      ParBilinearForm *a = new ParBilinearForm(HCurlFESpace);
      a->AddDomainIntegrator(new CurlCurlIntegrator(muinv));
      a->Assemble();
      a->Finalize();

      // 10. Define the parallel (hypre) matrix and vectors representing a(.,.),
      //     b(.) and the finite element approximation.
      HypreParMatrix *A = a->ParallelAssemble();
      HypreParVector *B = b->ParallelAssemble();
      HypreParVector *X = x.ParallelProject();

      // The entire outer surface of the mesh is held fixed at zero
      // except for the first surface which is set to he unit vector in
      // the z direction.
      Array<int> ess_bdr(pmesh->bdr_attributes.Max());
      ess_bdr = 1;
      a->ParallelEliminateEssentialBC(ess_bdr, *A, *X, *B);

      delete a;
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

      delete CurlX;
      delete Curl;
      delete curl;

      // 14. Save the refined mesh and the solution in parallel. This output can
      //     be viewed later using GLVis: "glvis -np <np> -m mesh -g sol".
      /*{
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
      }*/

      // 15. Send the solution by socket to a GLVis server.
      if (visualization)
      {
         /*socketstream sol_sock(vishost, visport);
         sol_sock << "parallel " << num_procs << " " << myid << "\n";
         sol_sock.precision(8);
         sol_sock << "solution\n" << *pmesh << x
             << "window_title 'Vector Potential'" << flush;*/

         /*socketstream sol_sock(vishost, visport);
         sol_sock << "parallel " << num_procs << " " << myid << "\n";
         sol_sock.precision(8);
         sol_sock << "solution\n" << *pmesh << error_gf
             << "window_title 'Error'" << flush;

         MPI_Barrier(pmesh->GetComm());*/

         curl_sock << "parallel " << num_procs << " " << myid << "\n";
         curl_sock << "solution\n" << *pmesh << curlx << flush;
         curl_sock << "window_title 'Flux in H(div)'\n" << flush;
      }

      // 16. Estimate element errors using the Zienkiewicz-Zhu error estimator.
      ND_FECollection flux_fec(order+1, dim);
      ParFiniteElementSpace flux_fes(pmesh, &flux_fec);
      CurlCurlIntegrator flux_integrator(muinv);
      // ParGridFunction flux(HCurlFESpace);
      ParGridFunction flux(&flux_fes);
      Vector est_errors;
      Array<int> aniso_flags;
      ZZErrorEstimator(flux_integrator, x, flux, est_errors, &aniso_flags);

      if (visualization)
      {
         L2_FECollection l2_fec(order+1, dim, 1);
         ParFiniteElementSpace flux_error_fes(pmesh, &l2_fec, dim);
         ParGridFunction hdiv_flux(&flux_error_fes);
         ParGridFunction flux_error(&flux_error_fes);
         curlx.ProjectVectorFieldOn(hdiv_flux);
         flux.ProjectVectorFieldOn(flux_error);
         subtract(flux_error, hdiv_flux, flux_error);

         flux_sock << "parallel " << num_procs << " " << myid << "\n";
         // flux_sock << "solution\n" << *pmesh << flux << flush;
         // flux_sock << "window_title 'Averaged Flux'\n" << flush;
         flux_sock << "solution\n" << *pmesh << flux_error << flush;
         flux_sock << "window_title 'Flux Error'\n" << flush;
      }

      if (visualization)
      {
         L2_FECollection l2_fec(0, dim);
         ParFiniteElementSpace err_fes(pmesh, &l2_fec);
         ParGridFunction errs(&err_fes);
         errs = est_errors;

         err_sock << "parallel " << num_procs << " " << myid << "\n";
         err_sock << "solution\n" << *pmesh << errs << flush;
         err_sock << "window_title 'Scalar Error Estimates'\n" << flush;
      }

      /*L2_FECollection l2_fec(0, dim);
      ParFiniteElementSpace error_space(pmesh, &l2_fec);
      ParGridFunction error_gf(&error_space);
      error_gf = est_errors;*/

      // 18. Free the used memory.
      delete X;
      delete B;
      delete A;
      delete HDivFESpace;
      delete HCurlFESpace;
      delete H1FESpace;
      delete HDivFEC;
      delete HCurlFEC;
      delete H1FEC;

      // 17. Make a list of elements whose error is larger than a fraction
      //     of the maximum element error. These elements will be refined.
      const double frac = 0.5;
      // the 'errors' are squared, so we need to square the fraction
      double threshold = (frac*frac) * est_errors.Max();
      if (pmesh->MeshGenerator() == 2)
      {
         Array<Refinement> ref_list;
         for (int i = 0; i < est_errors.Size(); i++)
         {
            if (est_errors[i] >= threshold)
            {
               MFEM_VERIFY(aniso_flags[i] != 0, "");
               ref_list.Append(Refinement(i, aniso_flags[i]));
               // ref_list.Append(Refinement(i, 7));
            }
         }
         cout << "Refining " << ref_list.Size() << " / " << pmesh->GetNE()
              << " elements ..." << endl;
         pmesh->GeneralRefinement(ref_list);
      }
      else if (order == 1)
      {
         Array<int> ref_list;
         for (int i = 0; i < est_errors.Size(); i++)
         {
            if (est_errors[i] >= threshold)
            {
               ref_list.Append(i);
            }
         }
         cout << "Refining " << ref_list.Size() << " / " << pmesh->GetNE()
              << " elements ..." << endl;
         pmesh->GeneralRefinement(ref_list);
      }
      else
      {
         break;
      }

      char c;
      if (myid == 0)
      {
         cout << "press (q)uit or (c)ontinue --> " << flush;
         cin >> c;
      }
      MPI_Bcast(&c, 1, MPI_CHAR, 0, MPI_COMM_WORLD);

      if (c != 'c')
      {
         break;
      }
   }

   delete pmesh;

   MPI_Finalize();

   return 0;
}
