//                       MFEM Example 11 - Parallel Version
//
// Compile with: make ex3p
//
// Sample runs:  mpirun -np 4 ex3p -m ../data/beam-tet.mesh
//               mpirun -np 4 ex3p -m ../data/beam-hex.mesh
//               mpirun -np 4 ex3p -m ../data/escher.mesh
//               mpirun -np 4 ex3p -m ../data/fichera.mesh
//               mpirun -np 4 ex3p -m ../data/fichera-q2.vtk
//               mpirun -np 4 ex3p -m ../data/fichera-q3.mesh
//               mpirun -np 4 ex3p -m ../data/beam-hex-nurbs.mesh
//
// Description:  This example code solves a simple 3D electromagnetic diffusion
//               problem corresponding to the second order definite Maxwell
//               equation curl (1/mu) curl A = J with boundary condition
//               A x n = 0.  A is the magnetic vector potential
//               so that magnetic flux density B = curl A, and J is the 
//               electric current density.  We discretize with Nedelec finite elements.
//
//               The example demonstrates the use of H(curl) finite element
//               spaces with the curl-curl and the (vector finite element) mass
//               bilinear form, as well as the computation of discretization
//               error when the exact solution is known.
//
//               We recommend viewing examples 1-2 before viewing this example.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

// Current for the solenoid
void J_exact(const Vector &, Vector &);

int main(int argc, char *argv[])
{
   // 1. Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // 2. Parse command-line options.
   const char *mesh_file = "../data/beam-hex.mesh";
   int order = 1;
   bool visualization = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
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
   //    more than 1,000 elements.
   {
      int ref_levels =
         (int)floor(log(100./mesh->GetNE())/log(2.)/dim);
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
      int par_ref_levels = 2;
      for (int l = 0; l < par_ref_levels; l++)
      {
         pmesh->UniformRefinement();
      }
   }
   pmesh->ReorientTetMesh();

   // 6. Define a parallel finite element space on the parallel mesh. Here we
   //    use the lowest order Nedelec finite elements, but we can easily switch
   //    to higher-order spaces by changing the value of p.
   FiniteElementCollection *HcurlFec = new ND_FECollection(order, dim);
   FiniteElementCollection *HdivFec  = new RT_FECollection(order, dim);
   ParFiniteElementSpace *HcurlFespace = new ParFiniteElementSpace(pmesh, HcurlFec);
   ParFiniteElementSpace *HdivFespace = new ParFiniteElementSpace(pmesh, HdivFec);
   HYPRE_Int size = HcurlFespace->GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of unknowns: " << size << endl;
   }


   // 7. Define the solution vector x and bfield as a parallel finite element grid functions
   //    corresponding to fespace. Initialize x with zeros. 
   //    Note that only values from the boundary edges will be used
   //    when eliminating the non-homogeneous boundary condition to modify the
   //    r.h.s. vector b.
   Vector zerovector(3);
   zerovector = 0.0;
   VectorConstantCoefficient zerovectorcoeff(zerovector);
   ParGridFunction x(HcurlFespace);
   ParGridFunction bfield(HdivFespace);
   x.ProjectCoefficient(zerovectorcoeff);
   bfield.ProjectCoefficient(zerovectorcoeff);
   HypreParVector *X = x.ParallelAverage();
   HypreParVector *BFIELD = bfield.ParallelAverage();

   /// 8. Set up the discrete curl operator for the B-field computation.
   ///    Alsoe set up C^t, and C^tC for the divergence cleaning operation on J.
   ParDiscreteLinearOperator *curl = new ParDiscreteLinearOperator(HcurlFespace, HdivFespace);
   curl->AddDomainInterpolator(new CurlInterpolator);
   curl->Assemble();
   curl->Finalize();
   HypreParMatrix *C = curl->ParallelAssemble();

   ParDiscreteLinearOperator *curlT = new ParDiscreteLinearOperator(HcurlFespace, HdivFespace);
   curlT->AddDomainInterpolator(new CurlInterpolator);
   curlT->Assemble();
   curlT->Finalize();   
   HypreParMatrix *Ct = curlT->ParallelAssemble()->Transpose();
   HypreParMatrix *CtC = ParMult(Ct, C);

   // 9. Project J for the solenoid on to the jdirty in H(Curl).  This may have
   //    left over divergence so we will clean that up by solving for u in 
   //    a least squares sense s.t. b = curl u = jdirty, and then taking curl u to
   //    get the divergence free version.  This least squares problem is C^tC U = C^t JDIRTY.
   VectorFunctionCoefficient f(3, J_exact);

   //Set up the RHS
   ParLinearForm *jdirty = new ParLinearForm(HcurlFespace);
   jdirty->AddDomainIntegrator(new VectorFEDomainLFIntegrator(f));
   jdirty->Assemble();
   HypreParVector *JDIRTY = jdirty->ParallelAssemble();
   HypreParVector *CtJDIRTY = new HypreParVector(*Ct);
   Ct->Mult(*JDIRTY, *CtJDIRTY);

   //Set up U
   ParGridFunction u(HcurlFespace);
   u.ProjectCoefficient(zerovectorcoeff);
   HypreParVector *U = u.ParallelAverage();
   *U = 0.0;

   //Solve for U
   std::cout << "Solving the curl u = J system." << std::endl;
   HypreSolver *diag_scale = new HypreDiagScale();
   HyprePCG *pcg_u = new HyprePCG(*CtC);
   pcg_u->SetTol(1e-12);
   pcg_u->SetMaxIter(1000);
   pcg_u->SetPrintLevel(2);
   pcg_u->SetPreconditioner(*diag_scale);
   pcg_u->Mult(*CtJDIRTY, *U);

   //Now B = C*U
   HypreParVector *B = new HypreParVector(*C, 1);
   C->Mult(*U, *B);

   // 10. Set up the parallel bilinear form corresponding to the magnetic diffusion
   //     operator curl muinv curl, by adding the curl-curl and the
   //     mass domain integrators and finally imposing non-homogeneous Dirichlet
   //     boundary conditions.
   Coefficient *muinv = new ConstantCoefficient(1.0 / (4.0e-7*M_PI));
   ParBilinearForm *a = new ParBilinearForm(HcurlFespace);
   a->AddDomainIntegrator(new CurlCurlIntegrator(*muinv));
   a->Assemble();
   a->Finalize();
   HypreParMatrix *A = a->ParallelAssemble();

   // 12. Define and apply a parallel PCG solver for AX=B with the AMS
   //     preconditioner from hypre.
   std::cout << "\nSolving the curl curl A = J system." << std::endl;
   HypreSolver *ams = new HypreAMS(*A, HcurlFespace);
   HyprePCG *pcg = new HyprePCG(*A);
   pcg->SetTol(1e-12);
   pcg->SetMaxIter(200);
   pcg->SetPrintLevel(2);
   pcg->SetPreconditioner(*ams);
   *X = 0.0;
   pcg->Mult(*B, *X);

   // 12. Interpolate to values of the BFIELD = curl X.     
   //     We could probably reformulate the problem to avoid this interpolation, but
   //     this will do for now.
   C->Mult(*X, *BFIELD);

   // 13. Extract the parallel grid functions corresponding to the finite element
   //     approximations of X and BFIELD. This is the local solution on each processor.
   x = *X;
   bfield = *BFIELD;

   // 14. Save the refined mesh and the solution in parallel. This output can
   //     be viewed later using GLVis: "glvis -np <np> -m mesh -g sol".
   {
      ostringstream mesh_name, sol_name;
      mesh_name << "mesh." << setfill('0') << setw(6) << myid;
      sol_name << "sol." << setfill('0') << setw(6) << myid;

      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh->Print(mesh_ofs);

      ofstream sol_ofs(sol_name.str().c_str());
      sol_ofs.precision(8);
      x.Save(sol_ofs);
   }

   // 15. Save data in the VisIt format
   VisItDataCollection visit_dc("Example11p", pmesh);
   visit_dc.RegisterField("Afield", &x);
   visit_dc.RegisterField("BField", &bfield);
   visit_dc.Save();

   // 16. Free the used memory.
   delete pcg;
   delete pcg_u;
   delete ams;
   delete diag_scale;
   delete HcurlFespace;
   delete HdivFespace;
   delete HcurlFec;
   delete HdivFec;
   delete pmesh;
   delete a;
   delete muinv;
   delete curl;
   delete curlT;
   delete jdirty;
   delete X;
   delete U;
   delete B;
   delete A;
   delete C;
   delete Ct;
   delete CtC;
   delete JDIRTY;
   delete CtJDIRTY;
   delete BFIELD;

   MPI_Finalize();

   return 0;
}

//Current going around an idealized solenoid that has an inner radius of 0.1
//an outer radius of 0.11 and a height (in x) of 2.0 centered on (4, 0.5, 0.5)
void J_exact(const Vector &x, Vector &J)
{
   double rsqr = (x(1) - 0.5)*(x(1) - 0.5) + (x(2) - 0.5)*(x(2) - 0.5);
   J(0) = J(1) = J(2) = 0.0;

   if (rsqr >= 0.1 && rsqr <= 0.11 && x(0) >= 3.0 && x(0) <= 5.0)
   {
      double r = sqrt(rsqr);
      J(1) = -x(2) / r;
      J(2) = x(1) / r;
   }
}
