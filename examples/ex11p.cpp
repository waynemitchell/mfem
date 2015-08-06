//                       MFEM Example 11 - Parallel Version
//
// Compile with: make ex11p
//
// Sample runs:  mpirun -np 4 ex11p -m ../data/inline_hex.mesh

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

ConstantCoefficient muinv(1.0);

// Current for the solenoid
void J4pi_exact(const Vector &, Vector &);

void Solve(int myid, int order, ParMesh* pmesh, DataCollection &result);


int main(int argc, char *argv[])
{
   // 1. Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // 2. Parse command-line options.
   int order = 1;
   bool visualization = 1;

   OptionsParser args(argc, argv);
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

   // 3. Generate a reasonable mesh for this problem
   Mesh *mesh = new Mesh(4, 4, 4, Element::HEXAHEDRON, 1, 1.0, 1.0, 1.0);
   mesh->GeneralRefinement(Array<int>(), 1); // ensure NC mesh

   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;

   socketstream sol_sock;
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      sol_sock.open(vishost, visport);
   }

   DataCollection result("", pmesh);

   for (int it = 0; it < 50; it++)
   {
      Solve(myid, order, pmesh, result);

      GridFunction* A = result.GetField("AField");
      GridFunction* B = result.GetField("BField");

      // 16. Send the solution by socket to a GLVis server.
      if (visualization)
      {
         sol_sock << "parallel " << num_procs << " " << myid << "\n";
         sol_sock.precision(8);
         sol_sock << "solution\n" << *pmesh << *B << flush;
      }

      Vector est_errors;
      Array<int> aniso_flags;
      {
         ND_FECollection flux_fec(order+1, pmesh->Dimension());
         ParFiniteElementSpace flux_fes(pmesh, &flux_fec);
         CurlCurlIntegrator flux_integrator(muinv);
         ParGridFunction flux(&flux_fes);
         ZZErrorEstimator(flux_integrator, *A, flux, est_errors, &aniso_flags);
      }

      double l_emax = est_errors.Max(), g_emax;
      MPI_Allreduce(&l_emax, &g_emax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

      const double frac = 0.5;
      double threshold = (frac*frac) * g_emax;

      Array<Refinement> ref_list;
      for (int i = 0; i < est_errors.Size(); i++)
      {
         if (est_errors[i] >= threshold)
         {
            MFEM_VERIFY(aniso_flags[i] != 0, "");
            ref_list.Append(Refinement(i, aniso_flags[i]));
         }
      }
      cout << "Refining " << ref_list.Size() << " / " << pmesh->GetNE()
           << " elements ..." << endl;
      pmesh->GeneralRefinement(ref_list);
   }


   MPI_Finalize();
   return 0;
}


void Solve(int myid, int order, ParMesh* pmesh, DataCollection &result)
{
   int dim = pmesh->Dimension();

   // 6. Define parallel finite element spaces on the parallel mesh.
   FiniteElementCollection *H1Fec    = new H1_FECollection(order, dim);
   FiniteElementCollection *HcurlFec = new ND_FECollection(order, dim);
   FiniteElementCollection *HdivFec  = new RT_FECollection(order, dim);
   ParFiniteElementSpace *H1Fespace = new ParFiniteElementSpace(pmesh, H1Fec);
   ParFiniteElementSpace *HcurlFespace = new ParFiniteElementSpace(pmesh, HcurlFec);
   ParFiniteElementSpace *HdivFespace = new ParFiniteElementSpace(pmesh, HdivFec);
   HYPRE_Int size = HcurlFespace->GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of unknowns: " << size << endl;
   }

   /// 7. Set up discrete gradient
   ParDiscreteLinearOperator *grad =
         new ParDiscreteLinearOperator(H1Fespace, HcurlFespace);

   grad->AddDomainInterpolator(new GradientInterpolator);
   grad->Assemble();
   grad->Finalize();
   HypreParMatrix *G = grad->ParallelAssemble();

   // 8. Define the solution vector x, bfield, and j as a parallel
   //    finite element grid functions corresponding to fespace.
   ParGridFunction* x      = new ParGridFunction(HcurlFespace);
   ParGridFunction* bfield = new ParGridFunction(HdivFespace);
   ParGridFunction* j      = new ParGridFunction(HcurlFespace);
   ParGridFunction* jdirty = new ParGridFunction(HcurlFespace);

   x->MakeOwner(HcurlFec);      // x and bfield will own the spaces too
   bfield->MakeOwner(HdivFec);

   // 9. Define J the current of the solenoid.  In order to converge
   //    we must ensure that the version of J on our RHS is divergence
   //    free.
   VectorFunctionCoefficient f(3, J4pi_exact);
   ParGridFunction psi(H1Fespace);
   ParGridFunction divj(H1Fespace);
   jdirty->ProjectCoefficient(f);

   HypreParVector *JDIRTY = jdirty->ParallelAssemble();
   HypreParVector *JTEMP = new HypreParVector(HcurlFespace);
   HypreParVector *JCLEAN = new HypreParVector(HcurlFespace);
   HypreParVector *DIVJ = new HypreParVector(H1Fespace);
   HypreParVector *PSI  = new HypreParVector(H1Fespace);
   *PSI = 0.0;
   psi = *PSI;

   // The problem is that JDIRTY may have a non-zero divergence so we
   // need to approximate its divergence.
   //
   // We are free to modify J by adding the gradient of a scalar
   // function so we want to find a scalar function whose gradient
   // will cancel the portion of J that has a non-zero divergence.
   // The scalar function will be called Psi.
   //
   // We need to solve:
   // Div(Grad(Psi)) = Div(JDIRTY)
   //
   // We can then compute JCLEAN as:
   // JCLEAN = JDIRTY - Grad(Psi)
   //

   // Compute the weak divergence of JDIRTY
   G->MultTranspose(*JDIRTY,*DIVJ);
   divj = *DIVJ;

   // Compute the Div(Grad()) operator
   ParBilinearForm *a_psi = new ParBilinearForm(H1Fespace);
   ConstantCoefficient one(1.0);
   a_psi->AddDomainIntegrator(new DiffusionIntegrator(one));
   a_psi->Assemble();
   //a_psi->EliminateEssentialBC(ess_bdr, psi, divj);
   a_psi->Finalize();

   Array<int> ess_bdr(pmesh->bdr_attributes.Max());
   ess_bdr = 1;

   HypreParMatrix *A_psi = a_psi->ParallelAssemble();
   a_psi->ParallelEliminateEssentialBC(ess_bdr, *A_psi, *PSI, *DIVJ);

   // Solve for Psi
   HypreSolver *amg_psi = new HypreBoomerAMG(*A_psi);
   HyprePCG *pcg_psi = new HyprePCG(*A_psi);
   pcg_psi->SetTol(1e-18);
   pcg_psi->SetMaxIter(200);
   pcg_psi->SetPrintLevel(2);
   pcg_psi->SetPreconditioner(*amg_psi);
   pcg_psi->Mult(*DIVJ, *PSI);
   delete a_psi;
   delete A_psi;
   delete pcg_psi;
   delete amg_psi;

   // Modify J
   ParBilinearForm *m1 = new ParBilinearForm(HcurlFespace);
   m1->AddDomainIntegrator(new VectorFEMassIntegrator(one));
   m1->Assemble();
   m1->Finalize();
   HypreParMatrix *M1 = m1->ParallelAssemble();
   *JCLEAN = *JDIRTY;
   G->Mult(*PSI,*JTEMP);
   M1->Mult(*JTEMP,*JCLEAN,-1.0,1.0);
   *j = *JCLEAN;
   delete grad;
   delete G;
   delete m1;
   delete M1;

   // 10. Set up the parallel bilinear form corresponding to the magnetic diffusion
   //     operator curl muinv curl, by adding the curl-curl and the
   //     mass domain integrators and finally imposing non-homogeneous Dirichlet
   //     boundary conditions.  We are using scaled CGS units so muinv = 1. in this case.
   ParBilinearForm *a = new ParBilinearForm(HcurlFespace);
   a->AddDomainIntegrator(new CurlCurlIntegrator(muinv));
   a->Assemble();
   a->Finalize();
   HypreParMatrix *A = a->ParallelAssemble();
   HypreParVector *B = j->ParallelAssemble();
   HypreParVector *X = x->ParallelAssemble();
   a->ParallelEliminateEssentialBC(ess_bdr, *A, *X, *B);


   // 12. Define and apply a parallel PCG solver for AX=B with the AMS
   //     preconditioner from hypre.
   HypreSolver *ams = new HypreAMS(*A, HcurlFespace, 1);
   HyprePCG *pcg = new HyprePCG(*A);
   pcg->SetTol(1e-10);
   pcg->SetMaxIter(200);
   pcg->SetPrintLevel(2);
   pcg->SetPreconditioner(*ams);
   pcg->Mult(*JCLEAN, *X);
   delete A;
   delete B;
   delete a;
   delete pcg;
   delete ams;

   // 12. Compute the values of the BFIELD = curl X.
   ParDiscreteLinearOperator *curl = new ParDiscreteLinearOperator(HcurlFespace,
                                                                   HdivFespace);
   curl->AddDomainInterpolator(new CurlInterpolator);
   curl->Assemble();
   curl->Finalize();
   HypreParMatrix *C = curl->ParallelAssemble();
   HypreParVector *BFIELD = new HypreParVector(HdivFespace);
   C->Mult(*X, *BFIELD);
   delete curl;
   delete C;

   // 13. Extract the parallel grid functions corresponding to the finite element
   //     approximations of X and BFIELD. This is the local solution on each processor.
   *x = *X;
   *bfield = *BFIELD;

   // 14. Save the refined mesh and the solution in parallel. This output can
   //     be viewed later using GLVis: "glvis -np <np> -m mesh -g sol".
   /*{
      ostringstream mesh_name, sol_name;
      mesh_name << "mesh." << setfill('0') << setw(6) << myid;
      sol_name << "sol." << setfill('0') << setw(6) << myid;

      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh->Print(mesh_ofs);

      ofstream sol_ofs(sol_name.str().c_str());
      sol_ofs.precision(8);
      bfield.Save(sol_ofs);
   }*/

   // 15. Save data in the VisIt format
   /*VisItDataCollection visit_dc("Example11p", pmesh);
   visit_dc.RegisterField("Afield", &x);
   visit_dc.RegisterField("BField", &bfield);*/

   //Remove the factor of 4pi from j for plotting
   *j *= 1.0 / (4.0*M_PI);
   *jdirty *= 1.0 / (4.0*M_PI);
   /*visit_dc.RegisterField("JField", &j);
   visit_dc.RegisterField("JDirtyField", &jdirty);
   visit_dc.Save();*/

   result.RegisterField("AField", x);
   result.RegisterField("BField", bfield);
   result.RegisterField("JField", j);
   result.RegisterField("JDirtyField", jdirty);
   result.SetOwnData(true);

   // 17. Free the used memory.
   delete H1Fespace;
   delete H1Fec;
   delete DIVJ;
   delete PSI;
   delete X;
   delete JDIRTY;
   delete JTEMP;
   delete JCLEAN;
   delete BFIELD;

   // these must now outlive this function:
   //delete HcurlFespace;
   //delete HdivFespace;
   //delete HcurlFec;
   //delete HdivFec;
   //delete pmesh;
}


//Current going around an idealized solenoid that has an inner radius of 0.2
//an outer radius of 0.22 and a height (in x) of 0.2 centered on (0.5, 0.5, 0.5).
//Since we are in scaled CGS units there is a factor of 4pi on the RHS.
void J4pi_exact(const Vector &x, Vector &J)
{
   const double sol_inner_r = 0.2;
   double r = sqrt((x(1) - 0.5)*(x(1) - 0.5) + (x(2) - 0.5)*(x(2) - 0.5));
   J(0) = J(1) = J(2) = 0.0;

   if (r >= sol_inner_r && r <= 1.1*sol_inner_r && x(0) >= 0.4 && x(0) <= 0.6)
   {
      J(1) = -(x(2) - 0.5);
      J(2) = (x(1) - 0.5);

      double scale = 4.*M_PI/sqrt(J(1)*J(1) + J(2)*J(2));
      J(1) *= scale;
      J(2) *= scale;
   }
}
