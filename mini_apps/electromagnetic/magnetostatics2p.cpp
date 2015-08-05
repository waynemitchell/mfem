//               MFEM Electromagnetics Mini App 2 - Parallel Version
//
// Compile with: make magnetostatics2p
//
// Sample runs:
//   mpirun -np 4 magnetostatics2p -m ../../data/inline_hex.mesh
//
// Description:  This example code solves a simple 3D magnetostatic
//               problem corresponding to the second order
//               semi-definite Maxwell equation
//                  curl curl A = J
//               Where the volume current, J, is computed as a
//               divergence free current.
//               The boundary condition is
//                  n x (A x n) = (0,0,0) on all exterior surfaces
//               This is a perfect electrical conductor (PEC) boundary
//               condition which results in a magnetic field sasifying:
//                  n . B = 0 on all surfaces
//               i.e. the magnetic field lines will be tangent to the
//               boundary.
//
//               We discretize the vector potential and current source
//               with Nedelec finite elements.  The magnetic field,
//               which is the curl of the vector potential, is
//               discretized with Raviart-Thomas finite elements.
//
//               The example demonstrates the use of H(curl) finite
//               element spaces with the curl-curl bilinear form.  It
//               also shows the divergence cleaning procedure needed
//               to ensure the right hand side is in the range of the
//               curl-curl operator.
//

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

// Current for the solenoid
void J4pi_exact(const Vector &, Vector &);

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

   socketstream psi_sock, j_raw_sock, j_sock, sol_sock, curl_sock;
   char vishost[] = "localhost";
   int  visport   = 19916;
   if (visualization)
   {
      psi_sock.open(vishost, visport);
      psi_sock.precision(8);

      MPI_Barrier(MPI_COMM_WORLD);

      j_raw_sock.open(vishost, visport);
      j_raw_sock.precision(8);

      MPI_Barrier(MPI_COMM_WORLD);

      j_sock.open(vishost, visport);
      j_sock.precision(8);

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

   // 7. Create the Gradient Operator and Mass matrix needed during
   // the divergence cleaning procedure.
   ParDiscreteLinearOperator *grad =
      new ParDiscreteLinearOperator(H1FESpace,
                                    HCurlFESpace);
   grad->AddDomainInterpolator(new GradientInterpolator);
   grad->Assemble();
   grad->Finalize();
   HypreParMatrix *Grad = grad->ParallelAssemble();

   ParBilinearForm *m1 = new ParBilinearForm(HCurlFESpace);
   m1->AddDomainIntegrator(new VectorFEMassIntegrator());
   m1->Assemble();
   m1->Finalize();
   HypreParMatrix *M1 = m1->ParallelAssemble();

   // 8. Divergence Cleaning:
   //    The Curl-Curl operator has a large null space which means
   //    that an arbitrary vector in H(Curl) is unlikely to be in the
   //    range of this operator.  In fact great care is required to
   //    ensure that a vector is in this range.  The procedure is as
   //    follows:
   //
   //    Start with J_raw which is the discretization of a volumetric
   //    current source.  Define a modified current source
   //       J = J_raw - Grad(Psi)
   //    Where Psi will be chosen such that the divergence of J is
   //    zero.  This means that Div(J) = Div(Grad(Psi)).  The discrete
   //    equations can be written:
   //       S*Psi = T_{01}^T*M*J_raw
   //    Where S is the stiffness matrix corresponding to the
   //    Laplacian operator, T_{01} is the discrete gradient operator,
   //    and M is the mass matrix for Nedelec finite elements.  When
   //    solving for Psi we can choose to set Psi equal to zero on any
   //    portion of the boundary where J_raw is zero.
   //

   // 9. Project the desired current source onto the grid function j_raw
   ParGridFunction j_raw(HCurlFESpace);
   ParGridFunction j(HCurlFESpace);
   VectorFunctionCoefficient j_func(3, J4pi_exact);
   j_raw.ProjectCoefficient(j_func);
   j = j_raw;

   // 10. Define the boundary conditions for Psi (Psi=0 on entire boundary)
   ParGridFunction psi(H1FESpace);
   psi = 0.0;
   Array<int> ess_bdr1(pmesh->bdr_attributes.Max());
   ess_bdr1 = 1;

   Array<int> ess_bdr1_v, dof_list1;
   H1FESpace->GetEssentialVDofs(ess_bdr1,ess_bdr1_v);

   for (int i = 0; i < ess_bdr1_v.Size(); i++)
   {
      if (ess_bdr1_v[i])
      {
         int loctdof = H1FESpace->GetLocalTDofNumber(i);
         if ( loctdof >= 0 )
         {
            dof_list1.Append(loctdof);
         }
      }
   }

   // 11. Create the Div(Grad()) operator for computing Psi
   ParBilinearForm *a1 = new ParBilinearForm(H1FESpace);
   ConstantCoefficient sigma(1.0);
   a1->AddDomainIntegrator(new DiffusionIntegrator(sigma));
   a1->Assemble();
   a1->Finalize();

   // 12. Setup the linear system for Psi
   HypreParMatrix *A1  = a1->ParallelAssemble();
   HypreParVector *X1  = psi.ParallelAverage();
   HypreParVector *B1  = new HypreParVector(*A1);
   HypreParVector *JR1 = j_raw.ParallelAverage();

   // JD1 is the vector dual to JR1
   HypreParVector *JD1 = new HypreParVector(*Grad,1);

   M1->Mult(*JR1,*JD1);
   Grad->MultTranspose(*JD1,*B1);

   A1->EliminateRowsCols(dof_list1, *X1, *B1);

   delete a1;
   delete m1;
   delete JD1;
   delete JR1;

   // 13. Setup and use the AMG solver to compute Psi.
   HypreBoomerAMG *amg1 = new HypreBoomerAMG(*A1);
   amg1->SetPrintLevel(0);
   HyprePCG *pcg1 = new HyprePCG(*A1);
   pcg1->SetTol(1e-14);
   pcg1->SetMaxIter(200);
   pcg1->SetPrintLevel(2);
   pcg1->SetPreconditioner(*amg1);
   pcg1->Mult(*B1, *X1);
   delete pcg1;
   delete amg1;

   psi = *X1;

   // 14. Use the Gradient operator to compute the correction to J_raw.
   HypreParVector *GradPsi = new HypreParVector(HCurlFESpace);
   Grad->Mult(*X1,*GradPsi);
   ParGridFunction gradPsi(HCurlFESpace,GradPsi);

   // j is now in the range of the Curl-Curl operator
   j -= gradPsi;

   delete GradPsi;
   delete Grad;
   delete grad;
   delete X1;
   delete B1;
   delete A1;

   // 15. Now we are ready to setup and solve the magnetostatics problem.

   // 16. Determine the boundary DoFs in the HCurl finite element space.
   Array<int> ess_bdr(pmesh->bdr_attributes.Max());
   ess_bdr = 1;

   Array<int> ess_bdr_v, dof_list;
   HCurlFESpace->GetEssentialVDofs(ess_bdr,ess_bdr_v);

   for (int i = 0; i < ess_bdr_v.Size(); i++)
   {
      if (ess_bdr_v[i])
      {
         int loctdof = HCurlFESpace->GetLocalTDofNumber(i);
         if ( loctdof >= 0 )
         {
            dof_list.Append(loctdof);
         }
      }
   }

   // 17. Set up the parallel bilinear form corresponding to the
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

   // 18. Define the parallel (hypre) matrix and vectors representing a(.,.),
   //     b(.) and the finite element approximation.
   ParGridFunction x(HCurlFESpace);
   HypreParMatrix *A = a->ParallelAssemble();
   HypreParVector *X = x.ParallelAverage();
   HypreParVector *J = j.ParallelAverage();
   HypreParVector *B = new HypreParVector(*A);

   // 19. The right hand side, B, must be the dual of J
   M1->Mult(*J,*B);

   delete J;

   // 20. Apply the boundary conditions to the assembled matrix and vectors
   A->EliminateRowsCols(dof_list, *X, *B);

   delete a;

   // 21. Define and apply a parallel PCG solver for AX=B with the AMS
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

   // 22. Extract the parallel grid function corresponding to the finite
   //     element approximation X. This is the local solution on each
   //     processor.
   x = *X;

   // 23. Compute the Curl of the solution vector.  This is the
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

   // 24. Save the refined mesh and the solution in parallel. This output can
   //     be viewed later using GLVis: "glvis -np <np> -m mesh -g sol".
   {
      ostringstream mesh_name, psi_name, sol_name, curl_name;
      mesh_name << "mesh." << setfill('0') << setw(6) << myid;
      psi_name  << "psi." << setfill('0') << setw(6) << myid;
      sol_name  << "sol." << setfill('0') << setw(6) << myid;
      curl_name << "sol_curl." << setfill('0') << setw(6) << myid;

      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh->Print(mesh_ofs);

      ofstream psi_ofs(psi_name.str().c_str());
      psi_ofs.precision(8);
      psi.Save(psi_ofs);

      ofstream sol_ofs(sol_name.str().c_str());
      sol_ofs.precision(8);
      x.Save(sol_ofs);

      ofstream curl_ofs(curl_name.str().c_str());
      curl_ofs.precision(8);
      curlx.Save(curl_ofs);
   }

   // 25. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      psi_sock << "parallel " << num_procs << " " << myid << "\n";
      psi_sock << "solution\n" << *pmesh << psi
               << "window_title 'Divergence Cleaning Potential'" << flush;

      MPI_Barrier(pmesh->GetComm());

      j_raw_sock << "parallel " << num_procs << " " << myid << "\n";
      j_raw_sock << "solution\n" << *pmesh << j_raw
                 << "window_title 'J Raw'" << flush;

      MPI_Barrier(pmesh->GetComm());

      j_sock << "parallel " << num_procs << " " << myid << "\n";
      j_sock << "solution\n" << *pmesh << j
             << "window_title 'J'" << flush;

      MPI_Barrier(pmesh->GetComm());

      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock << "solution\n" << *pmesh << x
               << "window_title 'Vector Potential'" << flush;

      MPI_Barrier(pmesh->GetComm());

      curl_sock << "parallel " << num_procs << " " << myid << "\n";
      curl_sock << "solution\n" << *pmesh << curlx
                << "window_title 'Magnetic Field'\n" << flush;
   }

   // 26. Free the used memory.
   delete X;
   delete B;
   delete A;
   delete M1;
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

//Current going around an idealized solenoid that has an inner radius of 0.2
//an outer radius of 0.22 and a height (in x) of 0.2 centered on (0.5, 0.5, 0.5)
void J4pi_exact(const Vector &x, Vector &J)
{
   const double r0 = 0.1;
   const double r1 = 0.2;
   const double h  = 0.1;
   double r = sqrt((x(1) - 0.5)*(x(1) - 0.5) + (x(2) - 0.5)*(x(2) - 0.5));
   J(0) = J(1) = J(2) = 0.0;

   if ( r >= r0 && r <= r1 && x(0) >= 0.5-0.5*h && x(0) <= 0.5+0.5*h )
   {
      J(1) = -(x(2) - 0.5);
      J(2) = (x(1) - 0.5);

      double scale = 4.*M_PI*sqrt(J(1)*J(1) + J(2)*J(2));
      J(1) *= scale;
      J(2) *= scale;
   }
}
