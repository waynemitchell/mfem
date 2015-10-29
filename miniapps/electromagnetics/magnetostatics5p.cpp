//               MFEM Magnetostatics Mini App 5
//               Magnetic Scalar Potential and a Permanent Magnet
//
// Compile with: make magnetostatics5p
//
// Sample runs:
//   mpirun -np 4 magnetostatics5p
//
// Description:  This example code solves a simple 3D magnetostatic
//               problem using a magnetic scalar potential.  When the
//               volumetric current is zero Ampere's law become curl H
//               = 0.  This implies that H can be written as the
//               gradient of a scalar potential, H = -grad Phi_M.  We
//               then use the facts that B = mu ( H + M ) and div B = 0
//               to arrive at the equation
//                  div mu grad Phi_M = - div mu M
//               with boundary condition
//                  n x (A x n) = (0,0,0) on all exterior surfaces
//               This is a perfect electrical conductor (PEC) boundary
//               condition which results in a magnetic field sasifying:
//                  n . B = 0 on all surfaces
//               i.e. the magnetic field lines will be tangent to the
//               boundary.
//
//               We discretize the vector potential with Nedelec finite
//               elements.  The magnetization M and magnetic field B =
//               curl A are discretized with Raviart-Thomas finite
//               elements.
//
//               The example demonstrates the use of H(curl) finite element
//               spaces with the curl-curl bilinear form.
//

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "pfem_extras.hpp"

using namespace std;
using namespace mfem;

// Constant Magnetization
void M_exact(const Vector &, Vector &);

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

   socketstream phi_m_sock, m_sock, h_sock, b_sock;
   char vishost[] = "localhost";
   int  visport   = 19916;
   if (visualization)
   {
      phi_m_sock.open(vishost, visport);
      phi_m_sock.precision(8);

      MPI_Barrier(MPI_COMM_WORLD);

      m_sock.open(vishost, visport);
      m_sock.precision(8);

      MPI_Barrier(MPI_COMM_WORLD);

      h_sock.open(vishost, visport);
      h_sock.precision(8);

      MPI_Barrier(MPI_COMM_WORLD);

      b_sock.open(vishost, visport);
      b_sock.precision(8);
   }

   // Define compatible parallel finite element spaces on the
   // parallel mesh. Here we use arbitrary order Nedelec and
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

   // Choose a single DoF to be the essential boundary condition
   Array<int> dof_list;

   if ( myid == 0 )
   {
     dof_list.Append(0);
   }

   // Set up the parallel bilinear form corresponding to the
   // magnetostatic operator curl muinv curl, by adding the curl-curl
   // domain integrator and finally imposing non-homogeneous Dirichlet
   // boundary conditions. The boundary conditions are implemented by
   // marking all the boundary attributes from the mesh as essential
   // (Dirichlet). After serial and parallel assembly we extract the
   // parallel matrix A.
   ConstantCoefficient mu(1.0);

   ParBilinearForm *laplacian_mu = new ParBilinearForm(H1FESpace);
   laplacian_mu->AddDomainIntegrator(new DiffusionIntegrator(mu));
   laplacian_mu->Assemble();
   laplacian_mu->Finalize();

   // Define the parallel (hypre) matrix and vectors representing a(.,.),
   // b(.) and the finite element approximation.
   ParGridFunction phi_m(H1FESpace); phi_m = 0.0;

   ParDiscreteGradOperator *Grad =
     new ParDiscreteGradOperator(H1FESpace, HCurlFESpace);

   ParBilinearForm *m1_mu = new ParBilinearForm(HCurlFESpace);
   m1_mu->AddDomainIntegrator(new VectorFEMassIntegrator(mu));
   m1_mu->Assemble();
   m1_mu->Finalize();
   HypreParMatrix *M1_mu = m1_mu->ParallelAssemble();

   delete m1_mu;

   ParGridFunction m(HCurlFESpace);
   VectorFunctionCoefficient m_func(3, M_exact);
   m.ProjectCoefficient(m_func);

   HypreParVector *M    = m.ParallelAverage();
   HypreParVector *MD   = new HypreParVector(HCurlFESpace);
   HypreParVector *DivM = new HypreParVector(H1FESpace);

   M1_mu->Mult(*M,*MD);
   Grad->MultTranspose(*MD,*DivM);

   delete MD;
   delete M1_mu;

   HypreParMatrix *Laplacian_mu = laplacian_mu->ParallelAssemble();
   HypreParVector *Phi_M        = phi_m.ParallelAverage();

   // Apply the boundary conditions to the assembled matrix and vectors
   Laplacian_mu->EliminateRowsCols(dof_list, *Phi_M, *DivM);

   delete laplacian_mu;

   // Define and apply a parallel PCG solver for AX=B with the AMS
   // preconditioner from hypre.

   HypreSolver *amg = new HypreBoomerAMG(*Laplacian_mu);
   HyprePCG *pcg = new HyprePCG(*Laplacian_mu);
   pcg->SetTol(1e-12);
   pcg->SetMaxIter(500);
   pcg->SetPrintLevel(2);
   pcg->SetPreconditioner(*amg);
   pcg->Mult(*DivM, *Phi_M);

   delete amg;
   delete pcg;

   // Extract the parallel grid function corresponding to the finite
   // element approximation X. This is the local solution on each
   // processor.
   phi_m = *Phi_M;

   // Compute the negative Gradient of the solution vector.  This is
   // the magnetic field corresponding to the scalar potential
   // represented by phi_m.
   HypreParVector *H = new HypreParVector(HCurlFESpace);
   Grad->Mult(*Phi_M,*H,-1.0);
   ParGridFunction h(HCurlFESpace,H);

   delete Grad;

   ParMixedBilinearForm *hodge_mu =
     new ParMixedBilinearForm(HCurlFESpace,HDivFESpace);
   hodge_mu->AddDomainIntegrator(new VectorFEMassIntegrator(mu));
   hodge_mu->Assemble();
   hodge_mu->Finalize();

   ParBilinearForm *mass_rt =
     new ParBilinearForm(HDivFESpace);
   mass_rt->AddDomainIntegrator(new VectorFEMassIntegrator());
   mass_rt->Assemble();
   mass_rt->Finalize();

   HypreParMatrix *Hodge_mu = hodge_mu->ParallelAssemble();
   HypreParMatrix *Mass_rt  = mass_rt->ParallelAssemble();
   HypreParVector *BD       = new HypreParVector(HDivFESpace);
   HypreParVector *B        = new HypreParVector(HDivFESpace);

   delete hodge_mu;
   delete mass_rt;

   Hodge_mu->Mult(*H,*BD);
   Hodge_mu->Mult(*M,*BD,1.0,1.0);

   delete Hodge_mu;

   HypreSolver *ds_rt = new HypreDiagScale(*Mass_rt);
   HyprePCG *pcg_rt = new HyprePCG(*Mass_rt);
   pcg_rt->SetTol(1e-12);
   pcg_rt->SetMaxIter(500);
   pcg_rt->SetPrintLevel(2);
   pcg_rt->SetPreconditioner(*ds_rt);
   pcg_rt->Mult(*BD, *B);

   delete Mass_rt;
   delete ds_rt;
   delete pcg_rt;

   ParGridFunction b(HDivFESpace,B);

   delete BD;
   delete B;

   // Save the refined mesh and the solution in parallel. This output can
   // be viewed later using GLVis: "glvis -np <np> -m mesh -g sol".
   {
      ostringstream mesh_name, phi_m_name, h_name, b_name;
      mesh_name  << "mesh."  << setfill('0') << setw(6) << myid;
      phi_m_name << "phi_m." << setfill('0') << setw(6) << myid;
      h_name     << "h."     << setfill('0') << setw(6) << myid;
      b_name     << "b."     << setfill('0') << setw(6) << myid;

      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh->Print(mesh_ofs);

      ofstream phi_m_ofs(phi_m_name.str().c_str());
      phi_m_ofs.precision(8);
      phi_m.Save(phi_m_ofs);

      ofstream h_ofs(h_name.str().c_str());
      h_ofs.precision(8);
      h.Save(h_ofs);

      ofstream b_ofs(b_name.str().c_str());
      b_ofs.precision(8);
      b.Save(b_ofs);
   }

   // Send the solution by socket to a GLVis server.
   if (visualization)
   {
      phi_m_sock << "parallel " << num_procs << " " << myid << "\n";
      phi_m_sock << "solution\n" << *pmesh << phi_m
		 << "window_title 'Magnetic Scalar Potential (Phi_M)'\n"
		 << flush;

      MPI_Barrier(pmesh->GetComm());

      m_sock << "parallel " << num_procs << " " << myid << "\n";
      m_sock << "solution\n" << *pmesh << m
	     << "window_title 'Magnetisation (M)'\n" << flush;

      MPI_Barrier(pmesh->GetComm());

      h_sock << "parallel " << num_procs << " " << myid << "\n";
      h_sock << "solution\n" << *pmesh << h
	     << "window_title 'Magnetic Field (H)'\n" << flush;

      MPI_Barrier(pmesh->GetComm());

      b_sock << "parallel " << num_procs << " " << myid << "\n";
      b_sock << "solution\n" << *pmesh << b
	     << "window_title 'Magnetic Flux (B)'\n" << flush;
   }

   // Free the used memory.
   delete H;
   delete M;
   delete DivM;
   delete Phi_M;
   delete Laplacian_mu;
   delete HDivFESpace;
   delete HCurlFESpace;
   delete H1FESpace;
   delete pmesh;

   MPI_Finalize();

   return 0;
}

// A sphere of constant magnetization directed along the z axis.  The
// sphere has a radius of 0.25 and is centered at the origin.
void M_exact(const Vector &x, Vector &M)
{
   const double r0 = 0.25;
   double r = sqrt(x(0)*x(0) + x(1)*x(1) + x(2)*x(2));

   M(0) = M(1) = M(2) = 0.0;

   if ( r <= r0 )
   {
      M(2) = 1.0;
   }
}
