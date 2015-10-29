//               MFEM Magnetostatics Mini App 3
//               Vector Potential with a Volumetric Current
//
// Compile with: make magnetostatics3p
//
// Sample runs:
//   mpirun -np 4 magnetostatics3p
//
// Description:  This example code solves a simple 3D magnetostatic
//               problem corresponding to the second order
//               semi-definite Maxwell equation
//                  curl muInv curl A = J
//               Where the volume current, J, is computed as a
//               divergence free vector field in H(curl).
//               The boundary condition is
//                  n x (A x n) = (0,0,0) on all exterior surfaces
//               This is a perfect electrical conductor (PEC) boundary
//               condition which results in a magnetic flux sasifying:
//                  n . B = 0 on all surfaces
//               i.e. the magnetic flux lines will be tangent to the
//               boundary.
//
//               We discretize the vector potential and current source
//               with Nedelec finite elements.  The magnetic flux,
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
#include "pfem_extras.hpp"

using namespace std;
using namespace mfem;

// Current for the solenoid
void J4pi_exact(const Vector &, Vector &);

int main(int argc, char *argv[])
{
   // Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // Parse command-line options.
   const char *mesh_file = "./butterfly_3d.mesh";
   int order = 1;
   int sr = 0, pr = 0;
   bool visualization = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
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

   // Read the (serial) mesh from the given mesh file on all processors.  We
   // can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
   // and volume meshes with the same code.
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

   // Refine the serial mesh on all processors to increase the resolution. In
   // this example we do 'ref_levels' of uniform refinement. We choose
   // 'ref_levels' to be the largest number that gives a final mesh with no
   // more than 100 elements.
   {
      int ref_levels = sr;
      for (int l = 0; l < ref_levels; l++)
      {
         mesh->UniformRefinement();
      }
   }

   // Define a parallel mesh by a partitioning of the serial mesh. Refine
   // this mesh further in parallel to increase the resolution. Once the
   // parallel mesh is defined, the serial mesh can be deleted. Tetrahedral
   // meshes need to be reoriented before we can define high-order Nedelec
   // spaces on them.
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;

   // Refine this mesh in parallel to increase the resolution.
   int par_ref_levels = pr;
   for (int l = 0; l < par_ref_levels; l++)
   {
      pmesh->UniformRefinement();
   }

   socketstream psi_sock, j_raw_sock, j_sock, a_sock, b_sock;
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

      a_sock.open(vishost, visport);
      a_sock.precision(8);

      MPI_Barrier(MPI_COMM_WORLD);

      b_sock.open(vishost, visport);
      b_sock.precision(8);
   }

   // Define compatible parallel finite element spaces on the
   // parallel mesh. Here we use arbitrary order H1, Nedelec, and
   // Raviart-Thomas finite elements.
   H1_ParFESpace *H1FESpace    = new H1_ParFESpace(pmesh,order,dim);
   ND_ParFESpace *HCurlFESpace = new ND_ParFESpace(pmesh,order,dim);
   RT_ParFESpace *HDivFESpace  = new RT_ParFESpace(pmesh,order,dim);

   HYPRE_Int size_h1 = H1FESpace->GlobalTrueVSize();
   HYPRE_Int size_nd = HCurlFESpace->GlobalTrueVSize();
   HYPRE_Int size_rt = HDivFESpace->GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of H1      unknowns: " << size_h1 << endl;
      cout << "Number of H(Curl) unknowns: " << size_nd << endl;
      cout << "Number of H(Div)  unknowns: " << size_rt << endl;
   }

   // Create the Gradient Operator, Curl Operator, and a Mass
   // matrix needed during the divergence cleaning procedure.
   ParDiscreteGradOperator *Grad =
     new ParDiscreteGradOperator(H1FESpace, HCurlFESpace);

   ParDiscreteCurlOperator *Curl =
     new ParDiscreteCurlOperator(HCurlFESpace, HDivFESpace);

   ParBilinearForm *mass_nd = new ParBilinearForm(HCurlFESpace);
   mass_nd->AddDomainIntegrator(new VectorFEMassIntegrator);
   mass_nd->Assemble();
   mass_nd->Finalize();
   HypreParMatrix *Mass_nd = mass_nd->ParallelAssemble();

   // Divergence Cleaning:
   // The Curl-Curl operator has a large null space which means
   // that an arbitrary vector in H(Curl) is unlikely to be in the
   // range of this operator.  In fact great care is required to
   // ensure that a vector is in this range.  The procedure is as
   // follows:
   //
   // Start with J_raw which is the discretization of a volumetric
   // current source.  Define a modified current source
   //    J = J_raw - Grad(Psi)
   // Where Psi will be chosen such that the divergence of J is
   // zero.  This means that Div(J) = Div(Grad(Psi)).  The discrete
   // equations can be written:
   //    S*Psi = G^T*M*J_raw
   // Where S is the stiffness matrix corresponding to the
   // Laplacian operator, G is the discrete gradient operator,
   // and M is the mass matrix for Nedelec finite elements.  When
   // solving for Psi we can choose to set Psi equal to zero on any
   // portion of the boundary where J_raw is zero.
   //

   // Project the desired current source onto the grid function j_raw
   ParGridFunction j_raw(HCurlFESpace);
   ParGridFunction j(HCurlFESpace);
   VectorFunctionCoefficient j_func(3, J4pi_exact);
   j_raw.ProjectCoefficient(j_func);
   j = j_raw;

   // Define the boundary conditions for Psi (Psi=0 on entire boundary)
   ParGridFunction psi(H1FESpace); psi = 0.0;
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

   // Create the Div(Grad()) operator for computing Psi
   ParBilinearForm *laplacian = new ParBilinearForm(H1FESpace);
   laplacian->AddDomainIntegrator(new DiffusionIntegrator);
   laplacian->Assemble();
   laplacian->Finalize();
   HypreParMatrix *Laplacian = laplacian->ParallelAssemble();

   // Setup the linear system for Psi
   HypreParVector *Psi  = psi.ParallelAverage();
   HypreParVector *RHS1 = new HypreParVector(H1FESpace);
   HypreParVector *JR1  = j_raw.ParallelAverage();

   // JD1 is the vector dual to JR1
   HypreParVector *JD1 = new HypreParVector(HCurlFESpace);

   Mass_nd->Mult(*JR1,*JD1);
   Grad->MultTranspose(*JD1,*RHS1);

   Laplacian->EliminateRowsCols(dof_list1, *Psi, *RHS1);

   delete laplacian;
   delete mass_nd;
   delete JD1;
   delete JR1;

   // Setup and use the AMG solver to compute Psi.
   HypreBoomerAMG *amg1 = new HypreBoomerAMG(*Laplacian);
   amg1->SetPrintLevel(0);
   HyprePCG *pcg1 = new HyprePCG(*Laplacian);
   pcg1->SetTol(1e-14);
   pcg1->SetMaxIter(200);
   pcg1->SetPrintLevel(2);
   pcg1->SetPreconditioner(*amg1);
   pcg1->Mult(*RHS1, *Psi);
   delete pcg1;
   delete amg1;

   psi = *Psi;

   // Use the Gradient operator to compute the correction to J_raw.
   HypreParVector *GradPsi = new HypreParVector(HCurlFESpace);
   Grad->Mult(*Psi,*GradPsi);
   ParGridFunction gradPsi(HCurlFESpace,GradPsi);

   // j is now in the range of the Curl-Curl operator
   j -= gradPsi;

   delete GradPsi;
   delete Grad;
   delete Psi;
   delete RHS1;
   delete Laplacian;

   // Now we are ready to setup and solve the magnetostatics problem.

   // Determine the boundary DoFs in the HCurl finite element space.
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

   // Set up the parallel bilinear form corresponding to the
   // magnetostatic operator curl muinv curl, by adding the curl-curl
   // domain integrator and finally imposing non-homogeneous Dirichlet
   // boundary conditions. The boundary conditions are implemented by
   // marking all the boundary attributes from the mesh as essential
   // (Dirichlet). After serial and parallel assembly we extract the
   // parallel matrix CurlMuInvCurl.
   ConstantCoefficient muinv(1.0);
   ParBilinearForm *curlMuInvCurl = new ParBilinearForm(HCurlFESpace);
   curlMuInvCurl->AddDomainIntegrator(new CurlCurlIntegrator(muinv));
   curlMuInvCurl->Assemble();
   curlMuInvCurl->Finalize();
   HypreParMatrix *CurlMuInvCurl = curlMuInvCurl->ParallelAssemble();

   // Define the parallel (hypre) matrix and vectors representing a(.,.),
   // b(.) and the finite element approximation.
   ParGridFunction a(HCurlFESpace); a = 0.0;
   HypreParVector *A  = a.ParallelAverage();
   HypreParVector *J  = j.ParallelAverage();
   HypreParVector *JD = new HypreParVector(HCurlFESpace);

   // The right hand side, JD, must be the dual of J
   Mass_nd->Mult(*J,*JD);

   delete J;

   // Apply the boundary conditions to the assembled matrix and vectors
   CurlMuInvCurl->EliminateRowsCols(dof_list, *A, *JD);

   delete curlMuInvCurl;

   // Define and apply a parallel PCG solver for AX=B with the AMS
   // preconditioner from hypre.

   HypreSolver *ams = new HypreAMS(*CurlMuInvCurl, HCurlFESpace, 1);
   HyprePCG *pcg = new HyprePCG(*CurlMuInvCurl);
   pcg->SetTol(1e-12);
   pcg->SetMaxIter(500);
   pcg->SetPrintLevel(2);
   pcg->SetPreconditioner(*ams);
   pcg->Mult(*JD, *A);

   delete ams;
   delete pcg;

   // Extract the parallel grid function corresponding to the finite
   // element approximation X. This is the local solution on each
   // processor.
   a = *A;

   // Compute the Curl of the solution vector.  This is the
   // magnetic flux corresponding to the vector potential
   // represented by a.
   HypreParVector *B = new HypreParVector(HDivFESpace);
   Curl->Mult(*A,*B);
   ParGridFunction b(HDivFESpace,B);

   delete B;
   delete Curl;

   // Save the refined mesh and the solution in parallel. This output can
   // be viewed later using GLVis: "glvis -np <np> -m mesh -g sol".
   {
      ostringstream mesh_name, psi_name, a_name, b_name;
      mesh_name << "mesh." << setfill('0') << setw(6) << myid;
      psi_name  << "psi."  << setfill('0') << setw(6) << myid;
      a_name    << "a."    << setfill('0') << setw(6) << myid;
      b_name    << "b."    << setfill('0') << setw(6) << myid;

      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh->Print(mesh_ofs);

      ofstream psi_ofs(psi_name.str().c_str());
      psi_ofs.precision(8);
      psi.Save(psi_ofs);

      ofstream a_ofs(a_name.str().c_str());
      a_ofs.precision(8);
      a.Save(a_ofs);

      ofstream b_ofs(b_name.str().c_str());
      b_ofs.precision(8);
      b.Save(b_ofs);
   }

   // Send the solution by socket to a GLVis server.
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
             << "window_title 'Volumetric Current (J)'" << flush;

      MPI_Barrier(pmesh->GetComm());

      a_sock << "parallel " << num_procs << " " << myid << "\n";
      a_sock << "solution\n" << *pmesh << a
	     << "window_title 'Vector Potential (A)'" << flush;

      MPI_Barrier(pmesh->GetComm());

      b_sock << "parallel " << num_procs << " " << myid << "\n";
      b_sock << "solution\n" << *pmesh << b
	     << "window_title 'Magnetic Flux (B)'\n" << flush;
   }

   // Free the used memory.
   delete A;
   delete JD;
   delete CurlMuInvCurl;
   delete Mass_nd;
   delete HDivFESpace;
   delete HCurlFESpace;
   delete H1FESpace;
   delete pmesh;

   MPI_Finalize();

   return 0;
}

// Current going around an idealized solenoid that has an inner radius
// of 0.25 an outer radius of 0.3 and a height (in x) of 0.2 centered
// on (0.0, 0.0, 0.0)
void J4pi_exact(const Vector &x, Vector &J)
{
   const double r0 = 0.25;
   const double r1 = 0.35;
   const double h  = 0.2;
   Vector c(3);
   c(0) = 0.0; c(1) = 0.0; c(2) = 0.0;

   double r = sqrt((x(0) - c(0))*(x(0) - c(0)) + (x(1) - c(1))*(x(1) - c(1)));
   J(0) = J(1) = J(2) = 0.0;

   if ( r >= r0 && r <= r1 && x(2) >= c(2)-0.5*h && x(2) <= c(2)+0.5*h )
   {
      J(0) = -(x(1) - c(1));
      J(1) =  (x(0) - c(0));

      double scale = 4.*M_PI/sqrt(J(0)*J(0) + J(1)*J(1));
      J(0) *= scale;
      J(1) *= scale;
   }
}
