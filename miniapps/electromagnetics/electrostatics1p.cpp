//               MFEM Electrostatics Mini App 1
//               Dielectric sphere in a uniform electric field
//
// Compile with: make electrostatics1p
//
// Sample runs:
//   mpirun -np 4 electrostatics1p
//
// Description:

//               This mini app solves a simple 3D electrostatic
//               problem with non-uniform dielectric permittivity.
//                  Div eps Grad Phi = 0
//               The uniform field is imposed through the boundary
//               conditions.
//                  Phi = -z on all exterior surfaces
//               This will produce a uniform electric field in the z
//               direction.
//
//               We discretize the electric potential with H1 finite
//               elements.  The electric field E is discretized with
//               Nedelec finite elements.
//

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "pfem_extras.hpp"

using namespace std;
using namespace mfem;

// Constant Magnetization
double Eps_exact(const Vector &);

// Phi Boundary Condition
double phi_bc_exact(const Vector &);

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
   // parallel mesh is defined, the serial mesh can be deleted.
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;

   // 6. Refine this mesh in parallel to increase the resolution.
   int par_ref_levels = pr;
   for (int l = 0; l < par_ref_levels; l++)
   {
      pmesh->UniformRefinement();
   }

   socketstream phi_sock, e_sock;
   char vishost[] = "localhost";
   int  visport   = 19916;
   if (visualization)
   {
      phi_sock.open(vishost, visport);
      phi_sock.precision(8);

      MPI_Barrier(MPI_COMM_WORLD);

      e_sock.open(vishost, visport);
      e_sock.precision(8);
   }

   // Define compatible parallel finite element spaces on the parallel
   // mesh. Here we use arbitrary order H1 and Nedelec finite
   // elements.
   H1_ParFESpace *H1FESpace    = new H1_ParFESpace(pmesh,order,dim);
   ND_ParFESpace *HCurlFESpace = new ND_ParFESpace(pmesh,order,dim);

   HYPRE_Int size_h1 = H1FESpace->GlobalTrueVSize();
   HYPRE_Int size_nd = HCurlFESpace->GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of H1      unknowns: " << size_h1 << endl;
      cout << "Number of H(Curl) unknowns: " << size_nd << endl;
   }

   // Select DoFs on surface 1 as Dirichlet BCs
   Array<int> ess_bdr(pmesh->bdr_attributes.Max());
   ess_bdr = 1;

   Array<int> ess_bdr_v, dof_list;
   H1FESpace->GetEssentialVDofs(ess_bdr,ess_bdr_v);

   for (int i = 0; i < ess_bdr_v.Size(); i++)
   {
      if (ess_bdr_v[i])
      {
         int loctdof = H1FESpace->GetLocalTDofNumber(i);
         if ( loctdof >= 0 )
         {
            dof_list.Append(loctdof);
         }
      }
   }

   // Set up the parallel bilinear form corresponding to the
   // electrostatic operator div eps grad, by adding the diffusion
   // domain integrator and finally imposing Dirichlet boundary
   // conditions. The boundary conditions are implemented by marking
   // all the boundary attributes from the mesh as essential
   // (Dirichlet). After serial and parallel assembly we extract the
   // parallel matrix A.
   FunctionCoefficient eps(Eps_exact);

   ParBilinearForm *laplacian_eps = new ParBilinearForm(H1FESpace);
   laplacian_eps->AddDomainIntegrator(new DiffusionIntegrator(eps));
   laplacian_eps->Assemble();
   laplacian_eps->Finalize();

   // The solution vector to approximate the electric potential
   ParGridFunction phi(H1FESpace);
   FunctionCoefficient phi_bc(phi_bc_exact);
   phi.ProjectCoefficient(phi_bc);

   // The gradient operator needed to compute E from Phi
   ParDiscreteGradOperator *Grad =
      new ParDiscreteGradOperator(H1FESpace, HCurlFESpace);

   // Create the dual of the charge density which is simply zero in this case
   HypreParVector *RhoD = new HypreParVector(H1FESpace);
   (*RhoD) = 0.0;

   HypreParMatrix *Laplacian_eps = laplacian_eps->ParallelAssemble();
   HypreParVector *Phi           = phi.ParallelAverage();

   // Apply the boundary conditions to the assembled matrix and vectors
   Laplacian_eps->EliminateRowsCols(dof_list, *Phi, *RhoD);

   delete laplacian_eps;

   // Define and apply a parallel PCG solver for AX=B with the AMS
   // preconditioner from hypre.

   HypreSolver *amg = new HypreBoomerAMG(*Laplacian_eps);
   HyprePCG *pcg = new HyprePCG(*Laplacian_eps);
   pcg->SetTol(1e-12);
   pcg->SetMaxIter(500);
   pcg->SetPrintLevel(2);
   pcg->SetPreconditioner(*amg);
   pcg->Mult(*RhoD, *Phi);

   delete amg;
   delete pcg;

   // Extract the parallel grid function corresponding to the finite
   // element approximation Phi. This is the local solution on each
   // processor.
   phi = *Phi;

   // Compute the negative Gradient of the solution vector.  This is
   // the magnetic field corresponding to the scalar potential
   // represented by phi.
   HypreParVector *E = new HypreParVector(HCurlFESpace);
   Grad->Mult(*Phi,*E,-1.0);
   ParGridFunction e(HCurlFESpace,E);

   delete Grad;

   // Save the refined mesh and the solution in parallel. This output can
   // be viewed later using GLVis: "glvis -np <np> -m mesh -g sol".
   {
      ostringstream mesh_name, phi_name, e_name;
      mesh_name  << "mesh."  << setfill('0') << setw(6) << myid;
      phi_name   << "phi."   << setfill('0') << setw(6) << myid;
      e_name     << "e."     << setfill('0') << setw(6) << myid;

      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh->Print(mesh_ofs);

      ofstream phi_ofs(phi_name.str().c_str());
      phi_ofs.precision(8);
      phi.Save(phi_ofs);

      ofstream e_ofs(e_name.str().c_str());
      e_ofs.precision(8);
      e.Save(e_ofs);
   }

   // Send the solution by socket to a GLVis server.
   if (visualization)
   {
      phi_sock << "parallel " << num_procs << " " << myid << "\n";
      phi_sock << "solution\n" << *pmesh << phi
               << "window_title 'Scalar Potential (Phi)'\n"
               << flush;

      MPI_Barrier(pmesh->GetComm());

      e_sock << "parallel " << num_procs << " " << myid << "\n";
      e_sock << "solution\n" << *pmesh << e
             << "window_title 'Electric Field (E)'\n" << flush;
   }

   // Free the used memory.

   delete E;
   delete RhoD;
   delete Phi;
   delete Laplacian_eps;
   delete HCurlFESpace;
   delete H1FESpace;
   delete pmesh;

   MPI_Finalize();

   return 0;
}

// A sphere with constant permittivity of 10.  The sphere has a radius
// of 0.25 and is centered at the origin.
double Eps_exact(const Vector &x)
{
   const double r0 = 0.25;
   double r = sqrt(x(0)*x(0) + x(1)*x(1) + x(2)*x(2));

   if ( r <= r0 )
   {
      return 10.0;
   }
   return 1.0;
}

// To produce a uniform electric field in the z-direction the
// potential can be set to -z.
double phi_bc_exact(const Vector &x)
{
   return -x(2);
}
