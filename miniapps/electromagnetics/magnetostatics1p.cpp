//               MFEM Magnetostatics Mini App 1
//               Vector Potential with a Permanent Magnet
//
// Compile with: make magnetostatics1p
//
// Sample runs:
//   mpirun -np 4 magnetostatics1p
//
// Description:  This mini app solves a simple 3D magnetostatic
//               problem corresponding to the second order
//               semi-definite Maxwell equation
//                  curl muInv curl A = curl M
//               with boundary condition
//                  n x (A x n) = (0,0,0) on all exterior surfaces
//               This is a perfect electrical conductor (PEC) boundary
//               condition which results in a magnetic flux satisfying:
//                  n . B = 0 on all surfaces
//               i.e. the magnetic flux lines will be tangent to the
//               boundary.
//
//               We discretize the vector potential with Nedelec finite
//               elements.  The magnetization M and magnetic flux B =
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

   socketstream m_sock, a_sock, b_sock;
   char vishost[] = "localhost";
   int  visport   = 19916;
   if (visualization)
   {
      m_sock.open(vishost, visport);
      m_sock.precision(8);

      MPI_Barrier(MPI_COMM_WORLD);

      a_sock.open(vishost, visport);
      a_sock.precision(8);

      MPI_Barrier(MPI_COMM_WORLD);

      b_sock.open(vishost, visport);
      b_sock.precision(8);
   }

   // Define compatible parallel finite element spaces on the
   // parallel mesh. Here we use arbitrary order Nedelec and
   // Raviart-Thomas finite elements.
   ND_ParFESpace *HCurlFESpace = new ND_ParFESpace(pmesh,order,dim);
   RT_ParFESpace *HDivFESpace  = new RT_ParFESpace(pmesh,order,dim);

   HYPRE_Int size_nd = HCurlFESpace->GlobalTrueVSize();
   HYPRE_Int size_rt = HDivFESpace->GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of H(Curl) unknowns: " << size_nd << endl;
      cout << "Number of H(Div)  unknowns: " << size_rt << endl;
   }

   ParDiscreteCurlOperator *Curl =
      new ParDiscreteCurlOperator(HCurlFESpace, HDivFESpace);

   ParBilinearForm *mass_rt = new ParBilinearForm(HDivFESpace);
   mass_rt->AddDomainIntegrator(new VectorFEMassIntegrator());
   mass_rt->Assemble();
   mass_rt->Finalize();
   HypreParMatrix *Mass_rt = mass_rt->ParallelAssemble();

   delete mass_rt;

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

   // Define the parallel (hypre) vectors representing the vector
   // potential and the curl of the magnetization

   // The vector potential, A, must be in the space H(Curl)
   ParGridFunction a(HCurlFESpace); a = 0.0;
   HypreParVector *A = a.ParallelAverage();

   // The magnetization, M, must be in the space H(Div) so that its
   // curl, a weak curl in this case, will be in H(Curl).
   //
   // First project the vector function onto an H(Div) vector
   ParGridFunction m(HDivFESpace);
   VectorFunctionCoefficient m_func(3, M_exact);
   m.ProjectCoefficient(m_func);

   // Grab the assembled parallel vector representing M
   HypreParVector *M     = m.ParallelAverage();

   // Compute the dual of M
   HypreParVector *MD    = new HypreParVector(HDivFESpace);
   Mass_rt->Mult(*M,*MD);

   // Compute the dual of the weak curl of M (this will be the
   // right-hand-side vector in our linear solve).
   HypreParVector *CurlM = new HypreParVector(HCurlFESpace);
   Curl->MultTranspose(*MD,*CurlM);

   // These objects are no longer needed
   delete curlMuInvCurl;
   delete Mass_rt;

   // Apply the boundary conditions to the assembled matrix and vectors
   CurlMuInvCurl->EliminateRowsCols(dof_list, *A, *CurlM);

   // Define and apply a parallel PCG solver for the linear system
   // with the AMS preconditioner from hypre.

   HypreSolver *ams = new HypreAMS(*CurlMuInvCurl, HCurlFESpace, 1);
   HyprePCG *pcg = new HyprePCG(*CurlMuInvCurl);
   pcg->SetTol(1e-12);
   pcg->SetMaxIter(500);
   pcg->SetPrintLevel(2);
   pcg->SetPreconditioner(*ams);
   pcg->Mult(*CurlM, *A);

   delete ams;
   delete pcg;

   // Extract the parallel grid function corresponding to the finite
   // element approximation A. This is the local solution on each
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
      ostringstream mesh_name, a_name, b_name;
      mesh_name << "mesh." << setfill('0') << setw(6) << myid;
      a_name  << "a." << setfill('0') << setw(6) << myid;
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
      m_sock << "parallel " << num_procs << " " << myid << "\n";
      m_sock << "solution\n" << *pmesh << m
             << "window_title 'Magnetization (M)'" << flush;

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
   delete CurlM;
   delete CurlMuInvCurl;
   delete HDivFESpace;
   delete HCurlFESpace;
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
