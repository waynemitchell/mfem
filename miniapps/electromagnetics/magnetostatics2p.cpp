//               MFEM Magnetostatics Mini App 2
//               Vector Potential with a Surface Current
//
// Compile with: make magnetostatics2p
//
// Sample runs:
//   mpirun -np 4 magnetostatics2p
//   mpirun -np 4 magnetostatics2p -m ./square-torus-n1.mesh
//
// Description:  This mini app solves a simple 3D magnetostatic
//               problem corresponding to the second order
//               semi-definite Maxwell equation
//                  curl muInv curl A = 0
//               with boundary condition
//                                / K on surface 3
//                  n x (A x n) = |
//                                \ (0,0,0) elsewhere
//
//               Where the surface current, K, is computed as a
//               divergence free current induced by constant
//               potentials on surfaces 1 and 2.
//
//               This is a perfect electrical conductor (PEC) boundary
//               condition which results in a magnetic flux satisfying:
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
#include "pfem_extras.hpp"

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   // Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // Parse command-line options.
   const char *mesh_file = "./square-angled-pipe.mesh";
   int order = 1;
   int sr = 2, pr = 1;
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
   // this example we do 'ref_levels' of uniform refinement.
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

   socketstream psi_sock, a_sock, b_sock;
   char vishost[] = "localhost";
   int  visport   = 19916;
   if (visualization)
   {
      psi_sock.open(vishost, visport);
      psi_sock.precision(8);

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

   // Setup the equations for the surface current.  The current
   // will flow along surface 3 of the mesh from surface 1 to
   // surface 2.  We want the surface current to be divergence free
   // so we solve:
   //   Div(Grad Psi) = 0
   // with boundary conditions:
   //   Psi = -1/2 on surface 1 and Psi = 1/2 on surface 2
   // We then define the surface current as:
   //   K = Grad Psi

   // Define the boundary conditions for Psi
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

   // Create the right hand side vector which is simply zero.
   ParLinearForm *rhs1 = new ParLinearForm(H1FESpace);
   rhs1->Assemble();


   // Create the Div(Grad()) operator on the outer surface of the
   // mesh.  This operator will be zero everywhere in the interior of
   // the mesh so technically it will remain singular even after the
   // boundary conditions are applied.  However, the AMG solver we are
   // about to use ignores rows which are identically zero so it will
   // be able to successfully solve the surface equations and leave
   // all internal DoFs zero which is precisely what we need.
   ParBilinearForm *laplacian = new ParBilinearForm(H1FESpace);
   ConstantCoefficient sigma(1.0);
   laplacian->AddBoundaryIntegrator(new DiffusionIntegrator(sigma));
   laplacian->Assemble();
   laplacian->EliminateEssentialBC(ess_bdr1, psi, *rhs1);
   laplacian->Finalize();

   HypreParMatrix *Laplacian = laplacian->ParallelAssemble();
   HypreParVector *RHS1      = rhs1->ParallelAssemble();
   HypreParVector *Psi       = psi.ParallelAverage();

   delete laplacian;
   delete rhs1;

   // Setup and use the AMG solver to compute Psi on the boundary.
   HypreBoomerAMG *amg1 = new HypreBoomerAMG(*Laplacian);
   amg1->SetPrintLevel(0);
   HyprePCG *pcg1 = new HyprePCG(*Laplacian);
   pcg1->SetTol(1e-12);
   pcg1->SetMaxIter(200);
   pcg1->SetPrintLevel(2);
   pcg1->SetPreconditioner(*amg1);
   pcg1->Mult(*RHS1, *Psi);
   delete pcg1;
   delete amg1;

   psi = *Psi;

   // Create the Gradient operator and compute K = Grad Psi.
   ParDiscreteGradOperator * Grad =
      new ParDiscreteGradOperator(H1FESpace, HCurlFESpace);

   HypreParVector *K = new HypreParVector(HCurlFESpace);
   Grad->Mult(*Psi,*K);
   ParGridFunction k(HCurlFESpace,K);

   delete K;
   delete Grad;
   delete Psi;
   delete RHS1;
   delete Laplacian;

   // Now we are ready to setup and solve the magnetostatics problem.
   //
   // Set up the parallel linear form rhs(.) which corresponds to the
   // right-hand side of the FEM linear system, which in this case is
   // zero.
   ParLinearForm *rhs = new ParLinearForm(HCurlFESpace);
   rhs->Assemble();

   // Define the solution vector a as a parallel finite element
   // grid function corresponding to HCurlFESpace. Initialize a by
   // projecting the boundary conditions onto the appropriate edges.
   ParGridFunction a(HCurlFESpace);
   Vector vZero(3); vZero = 0.0;
   VectorConstantCoefficient Zero(vZero);
   Array<int> ess_bdr_1(pmesh->bdr_attributes.Max());
   ess_bdr_1 = 1; ess_bdr_1[2] = 0;

   // Set a to be the surface current, Grad Psi, computed above.  Then
   // overwrite this with zero on the outer surface with the exception
   // of surface 3.
   a = k;
   a.ProjectBdrCoefficientTangent(Zero,ess_bdr_1);

   // Set up the parallel bilinear form corresponding to the
   // magnetostatic operator curl muinv curl, by adding the curl-curl
   // domain integrator and finally imposing non-homogeneous Dirichlet
   // boundary conditions. The boundary conditions are implemented by
   // marking all the boundary attributes from the mesh as essential
   // (Dirichlet). After serial and parallel assembly we extract the
   // parallel matrix CurlMuInvCurl.
   ConstantCoefficient muinv(1.0);

   // The entire outer surface of the mesh is held fixed at zero
   // except for the third surface which is set to K = Grad Psi.
   Array<int> ess_bdr(pmesh->bdr_attributes.Max());
   ess_bdr = 1;

   ParBilinearForm *curlMuInvCurl = new ParBilinearForm(HCurlFESpace);
   curlMuInvCurl->AddDomainIntegrator(new CurlCurlIntegrator(muinv));
   curlMuInvCurl->Assemble();
   curlMuInvCurl->EliminateEssentialBC(ess_bdr, a, *rhs);
   curlMuInvCurl->Finalize();
   HypreParMatrix *CurlMuInvCurl = curlMuInvCurl->ParallelAssemble();

   // Define the parallel (hypre) vectors representing the vector
   // potential, A, and the right-hand-side.
   HypreParVector *A   = a.ParallelAverage();
   HypreParVector *RHS = rhs->ParallelAssemble();

   // These objects are no longer needed
   delete curlMuInvCurl;
   delete rhs;

   // Define and apply a parallel PCG solver for the linear system
   // with the AMS preconditioner from hypre.

   HypreAMS *ams = new HypreAMS(*CurlMuInvCurl, HCurlFESpace);
   ams->SetSingularProblem();

   HyprePCG *pcg = new HyprePCG(*CurlMuInvCurl);
   pcg->SetTol(1e-12);
   pcg->SetMaxIter(500);
   pcg->SetPrintLevel(2);
   pcg->SetPreconditioner(*ams);
   pcg->Mult(*RHS, *A);

   delete ams;
   delete pcg;

   // Extract the parallel grid function corresponding to the finite
   // element approximation A. This is the local solution on each
   // processor.
   a = *A;

   // Compute the Curl of the solution vector.  This is the
   // magnetic flux corresponding to the vector potential
   // represented by A.
   ParDiscreteCurlOperator *Curl =
      new ParDiscreteCurlOperator(HCurlFESpace, HDivFESpace);
   HypreParVector *B = new HypreParVector(HDivFESpace);

   Curl->Mult(*A,*B);
   ParGridFunction b(HDivFESpace,B);

   delete B;
   delete Curl;

   // Save the refined mesh and the solution in parallel. This output can
   // be viewed later using GLVis: "glvis -np <np> -m mesh -g sol".
   {
      ostringstream mesh_name, a_name, b_name;
      mesh_name << "mesh." << setfill('0') << setw(6) << myid;
      a_name << "a." << setfill('0') << setw(6) << myid;
      b_name << "b." << setfill('0') << setw(6) << myid;

      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh->Print(mesh_ofs);

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
               << "window_title 'Driving Potential'" << flush;

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
   delete RHS;
   delete CurlMuInvCurl;
   delete HDivFESpace;
   delete HCurlFESpace;
   delete H1FESpace;
   delete pmesh;

   MPI_Finalize();

   return 0;
}
