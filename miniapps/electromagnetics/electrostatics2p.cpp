//               MFEM Electrostatics Mini App 2
//               Sphere of charge in a PEC box
//
// Compile with: make electrostatics2p
//
// Sample runs:
//   mpirun -np 4 electrostatics2p
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
double Rho_exact(const Vector &);

int main(int argc, char *argv[])
{
   // 1. Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // 2. Parse command-line options.
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

   // 6. Refine this mesh in parallel to increase the resolution.
   int par_ref_levels = pr;
   for (int l = 0; l < par_ref_levels; l++)
   {
      pmesh->UniformRefinement();
   }

   socketstream phi_sock, rho_sock, e_sock;
   char vishost[] = "localhost";
   int  visport   = 19916;
   if (visualization)
   {
      phi_sock.open(vishost, visport);
      phi_sock.precision(8);

      MPI_Barrier(MPI_COMM_WORLD);

      rho_sock.open(vishost, visport);
      rho_sock.precision(8);

      MPI_Barrier(MPI_COMM_WORLD);

      e_sock.open(vishost, visport);
      e_sock.precision(8);
   }

   // 7. Define compatible parallel finite element spaces on the
   // parallel mesh. Here we use arbitrary order Nedelec and
   // Raviart-Thomas finite elements.
   H1_ParFESpace *H1FESpace    = new H1_ParFESpace(pmesh,order,dim);
   ND_ParFESpace *HCurlFESpace = new ND_ParFESpace(pmesh,order,dim);

   HYPRE_Int size_h1 = H1FESpace->GlobalTrueVSize();
   HYPRE_Int size_nd = HCurlFESpace->GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of H1      unknowns: " << size_h1 << endl;
      cout << "Number of H(Curl) unknowns: " << size_nd << endl;
   }

   // 8. Choose a single DoF to be the essential boundary condition
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

   // 9. Set up the parallel bilinear form corresponding to the
   // magnetostatic operator curl muinv curl, by adding the curl-curl
   // domain integrator and finally imposing non-homogeneous Dirichlet
   // boundary conditions. The boundary conditions are implemented by
   // marking all the boundary attributes from the mesh as essential
   // (Dirichlet). After serial and parallel assembly we extract the
   // parallel matrix A.
   ConstantCoefficient eps(1.0);

   ParBilinearForm *laplacian_eps = new ParBilinearForm(H1FESpace);
   laplacian_eps->AddDomainIntegrator(new DiffusionIntegrator(eps));
   laplacian_eps->Assemble();
   laplacian_eps->Finalize();

   // 10. Define the parallel (hypre) matrix and vectors representing a(.,.),
   //     b(.) and the finite element approximation.
   ParGridFunction phi(H1FESpace); phi = 0.0;

   ParDiscreteGradOperator *Grad =
     new ParDiscreteGradOperator(H1FESpace, HCurlFESpace);

   ParBilinearForm *mass = new ParBilinearForm(H1FESpace);
   mass->AddDomainIntegrator(new MassIntegrator);
   mass->Assemble();
   mass->Finalize();
   HypreParMatrix *Mass = mass->ParallelAssemble();

   delete mass;

   ParGridFunction rho(H1FESpace);
   FunctionCoefficient rho_func(Rho_exact);
   rho.ProjectCoefficient(rho_func);

   HypreParVector *Rho  = rho.ParallelAverage();
   HypreParVector *RhoD = new HypreParVector(H1FESpace);

   Mass->Mult(*Rho,*RhoD);

   delete Rho;
   delete Mass;

   HypreParMatrix *Laplacian_eps = laplacian_eps->ParallelAssemble();
   HypreParVector *Phi           = phi.ParallelAverage();

   // 11. Apply the boundary conditions to the assembled matrix and vectors
   Laplacian_eps->EliminateRowsCols(dof_list, *Phi, *RhoD);

   delete laplacian_eps;

   // 12. Define and apply a parallel PCG solver for AX=B with the AMS
   //     preconditioner from hypre.

   HypreSolver *amg = new HypreBoomerAMG(*Laplacian_eps);
   HyprePCG *pcg = new HyprePCG(*Laplacian_eps);
   pcg->SetTol(1e-12);
   pcg->SetMaxIter(500);
   pcg->SetPrintLevel(2);
   pcg->SetPreconditioner(*amg);
   pcg->Mult(*RhoD, *Phi);

   delete amg;
   delete pcg;

   // 13. Extract the parallel grid function corresponding to the finite
   //     element approximation X. This is the local solution on each
   //     processor.
   phi = *Phi;

   // 14. Compute the negative Gradient of the solution vector.  This is
   //     the magnetic field corresponding to the scalar potential
   //     represented by phi.
   HypreParVector *E = new HypreParVector(HCurlFESpace);
   Grad->Mult(*Phi,*E,-1.0);
   ParGridFunction e(HCurlFESpace,E);

   delete Grad;

   // 15. Save the refined mesh and the solution in parallel. This output can
   //     be viewed later using GLVis: "glvis -np <np> -m mesh -g sol".
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

   // 16. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      phi_sock << "parallel " << num_procs << " " << myid << "\n";
      phi_sock << "solution\n" << *pmesh << phi
	       << "window_title 'Scalar Potential (Phi)'\n"
	       << flush;

      MPI_Barrier(pmesh->GetComm());

      rho_sock << "parallel " << num_procs << " " << myid << "\n";
      rho_sock << "solution\n" << *pmesh << rho
	       << "window_title 'Charge Density (Rho)'\n" << flush;

      MPI_Barrier(pmesh->GetComm());

      e_sock << "parallel " << num_procs << " " << myid << "\n";
      e_sock << "solution\n" << *pmesh << e
	     << "window_title 'Electric Field (E)'\n" << flush;
   }

   // 17. Free the used memory.
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

// A sphere of constant magnetization directed along the z axis.  The
// sphere has a radius of 0.25 and is centered at the origin.
double Rho_exact(const Vector &x)
{
   const double r0 = 0.25;
   double r = sqrt(x(0)*x(0) + x(1)*x(1) + x(2)*x(2));

   if ( r <= 1.0001 * r0 )
   {
      return 1.0;
   }
   return 0.0;
}
