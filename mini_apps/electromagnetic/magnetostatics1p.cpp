//               MFEM Electromagnetics Mini App 1 - Parallel Version
//
// Compile with: make magnetostatics1p
//
// Sample runs:
//   mpirun -np 4 magnetostatics1p -m ../../data/square-angled-pipe-n1.mesh
//   mpirun -np 4 magnetostatics1p -m ../../data/square-torus-n1.mesh
//
// Description:  This example code solves a simple 3D magnetostatic
//               problem corresponding to the second order
//               semi-definite Maxwell equation
//                  curl curl A = 0
//               with boundary condition
//                                / J on surface 3
//                  n x (A x n) = |
//                                \ (0,0,0) elsewhere

//               Where the surface current, J, is computed as a
//               divergence free current induced by constant
//               potentials on surfaces 1 and 2.
//
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
   const char *mesh_file = "../../data/square-torus-n1.mesh";
   int order = 1;
   int sr = 1, pr = 0;
   bool visualization = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&sr, "-sr", "--serial-ref-levels",
                  "Number of serial refinement levels.");
   args.AddOption(&pr, "-pr", "--parallel-ref-levels",
                  "Number of parallel refinement levels.");
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
      int ref_levels = sr;
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
      int par_ref_levels = pr;
      for (int l = 0; l < par_ref_levels; l++)
      {
         pmesh->UniformRefinement();
      }
   }
   if (order >= 2)
   {
      pmesh->ReorientTetMesh();
   }

   socketstream psi_sock, sol_sock, curl_sock;
   char vishost[] = "localhost";
   int  visport   = 19916;
   if (visualization)
   {
      psi_sock.open(vishost, visport);
      psi_sock.precision(8);

      MPI_Barrier(MPI_COMM_WORLD);

      sol_sock.open(vishost, visport);
      sol_sock.precision(8);

      MPI_Barrier(MPI_COMM_WORLD);

      curl_sock.open(vishost, visport);
      curl_sock.precision(8);
   }

   // 6. Define compatible parallel finite element spaces on the
   // parallel mesh. Here we use arbitrary order H1, Nedelec, and
   // Raviart-Thomas finite elements.
   FiniteElementCollection *H1FEC        = new H1_FECollection(order, dim);
   FiniteElementCollection *HCurlFEC     = new ND_FECollection(order, dim);
   FiniteElementCollection *HDivFEC      = new RT_FECollection(order-1, dim);

   ParFiniteElementSpace   *H1FESpace    = new ParFiniteElementSpace(pmesh,
								     H1FEC);
   ParFiniteElementSpace   *HCurlFESpace = new ParFiniteElementSpace(pmesh,
								     HCurlFEC);
   ParFiniteElementSpace   *HDivFESpace  = new ParFiniteElementSpace(pmesh,
								     HDivFEC);

   HYPRE_Int size_h1 = H1FESpace->GlobalTrueVSize();
   HYPRE_Int size_nd = HCurlFESpace->GlobalTrueVSize();
   HYPRE_Int size_rt = HDivFESpace->GlobalTrueVSize();
   if (myid == 0)
   {
     cout << "Number of H1      unknowns: " << size_h1 << endl;
     cout << "Number of H(Curl) unknowns: " << size_nd << endl;
     cout << "Number of H(Div)  unknowns: " << size_rt << endl;
   }

   // 7. Setup the equations for the surface current.  The current
   // will flow along surface 3 of the mesh from surface 1 to
   // surface 2.  We want the surface current to be divergence free
   // so we solve:
   //   Div(Grad Psi) = 0
   // with boundary conditions:
   //   Psi = -1/2 on surface 1 and Psi = 1/2 on surface 2
   // We then define the surface current as:
   //   J = Grad Psi

   // 8. Define the boundary conditions for Psi
   ParGridFunction psi(H1FESpace);
   ConstantCoefficient one_half(0.5);
   ConstantCoefficient neg_one_half(-0.5);
   psi = 0.0;
   Array<int> ess_bdr1_p(pmesh->bdr_attributes.Max());
   Array<int> ess_bdr1_n(pmesh->bdr_attributes.Max());
   Array<int> ess_bdr1(pmesh->bdr_attributes.Max());

   ess_bdr1_p = 0; ess_bdr1_p[1] = 1;
   ess_bdr1_n = 0; ess_bdr1_n[0] = 1;
   ess_bdr1 = 0; ess_bdr1[0] = 1; ess_bdr1[1] = 1;

   psi.ProjectBdrCoefficient(one_half,ess_bdr1_p);
   psi.ProjectBdrCoefficient(neg_one_half,ess_bdr1_n);

   // 9. Create the right hand side vector which is simply zero.
   ParLinearForm *b1 = new ParLinearForm(H1FESpace);
   b1->Assemble();


   // 10. Create the Div(Grad()) operator on the outer surface of the
   // mesh.  This operator will be zero everywhere in the interior of
   // the mesh so technically it will remain singular even after the
   // boundary conditions are applied.  However, the AMG solver we are
   // about to use ignores rows which are identically zero so it will
   // be able to successfully solve the surface equations and leave
   // all internal DoFs zero which is precisely what we need.
   ParBilinearForm *a1 = new ParBilinearForm(H1FESpace);
   ConstantCoefficient sigma(1.0);
   a1->AddBoundaryIntegrator(new DiffusionIntegrator(sigma));
   a1->Assemble();
   a1->EliminateEssentialBC(ess_bdr1, psi, *b1);
   a1->Finalize();

   HypreParMatrix *A1 = a1->ParallelAssemble();
   HypreParVector *B1 = b1->ParallelAssemble();
   HypreParVector *X1 = psi.ParallelAverage();

   delete a1;
   delete b1;

   // 11. Setup and use the AMG solver to compute Psi on the boundary.
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

   psi = *X1;

   // 12. Create the Gradient operator and compute J = Grad Psi.
   ParDiscreteLinearOperator *grad =
     new ParDiscreteLinearOperator(H1FESpace, HCurlFESpace);
   grad->AddDomainInterpolator(new GradientInterpolator);
   grad->Assemble();
   grad->Finalize();
   HypreParMatrix *Grad = grad->ParallelAssemble();
   HypreParVector *GradPsi = new HypreParVector(HCurlFESpace);
   Grad->Mult(*X1,*GradPsi);
   ParGridFunction gradPsi(HCurlFESpace,GradPsi);

   delete GradPsi;
   delete Grad;
   delete grad;
   delete X1;
   delete B1;
   delete A1;

   // 13. Now we are ready to setup and solve the magnetostatics problem.

   // 14. Set up the parallel linear form b(.) which corresponds to the
   //     right-hand side of the FEM linear system, which in this case is
   //     zero.
   ParLinearForm *b = new ParLinearForm(HCurlFESpace);
   b->Assemble();

   // 15. Define the solution vector x as a parallel finite element
   // grid function corresponding to HCurlFESpace. Initialize x by
   // projecting the boundary conditions onto the appropriate edges.
   ParGridFunction x(HCurlFESpace);
   Vector vZero(3); vZero = 0.0;
   VectorConstantCoefficient Zero(vZero);
   Array<int> ess_bdr_1(pmesh->bdr_attributes.Max());
   ess_bdr_1 = 1; ess_bdr_1[2] = 0;

   // Set x to be the surface current, Grad Psi, computed above.  Then
   // overwrite this with zero on the outer surface with the exception
   // of surface 3.
   x = gradPsi;
   x.ProjectBdrCoefficientTangent(Zero,ess_bdr_1);

   // 16. Set up the parallel bilinear form corresponding to the
   // magnetostatic operator curl muinv curl, by adding the curl-curl
   // domain integrator and finally imposing non-homogeneous Dirichlet
   // boundary conditions. The boundary conditions are implemented by
   // marking all the boundary attributes from the mesh as essential
   // (Dirichlet). After serial and parallel assembly we extract the
   // parallel matrix A.
   ConstantCoefficient muinv(1.0);

   // The entire outer surface of the mesh is held fixed at zero
   // except for the third surface which is set to J = Grad Psi.
   Array<int> ess_bdr(pmesh->bdr_attributes.Max());
   ess_bdr = 1;

   ParBilinearForm *a = new ParBilinearForm(HCurlFESpace);
   a->AddDomainIntegrator(new CurlCurlIntegrator(muinv));
   a->Assemble();
   a->EliminateEssentialBC(ess_bdr, x, *b);
   a->Finalize();

   // 17. Define the parallel (hypre) matrix and vectors representing a(.,.),
   //     b(.) and the finite element approximation.
   HypreParMatrix *A = a->ParallelAssemble();
   HypreParVector *B = b->ParallelAssemble();
   HypreParVector *X = x.ParallelAverage();

   delete a;
   delete b;

   // 18. Define and apply a parallel PCG solver for AX=B with the AMS
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

   // 19. Extract the parallel grid function corresponding to the finite
   //     element approximation X. This is the local solution on each
   //     processor.
   x = *X;

   // 20. Compute the Curl of the solution vector.  This is the
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

   // 21. Save the refined mesh and the solution in parallel. This output can
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

   // 22. Send the solution by socket to a GLVis server.
   if (visualization)
   {
     psi_sock << "parallel " << num_procs << " " << myid << "\n";
     psi_sock << "solution\n" << *pmesh << psi
	      << "window_title 'Driving Potential'" << flush;

     MPI_Barrier(pmesh->GetComm());

     sol_sock << "parallel " << num_procs << " " << myid << "\n";
     sol_sock << "solution\n" << *pmesh << x
	      << "window_title 'Vector Potential'" << flush;

     MPI_Barrier(pmesh->GetComm());

     curl_sock << "parallel " << num_procs << " " << myid << "\n";
     curl_sock << "solution\n" << *pmesh << curlx
	       << "window_title 'Magnetic Field'\n" << flush;
   }

   // 23. Free the used memory.
   delete X;
   delete B;
   delete A;
   delete HDivFESpace;
   delete HCurlFESpace;
   delete H1FESpace;
   delete HDivFEC;
   delete HCurlFEC;
   delete H1FEC;
   delete pmesh;

   MPI_Finalize();

   return 0;
}
