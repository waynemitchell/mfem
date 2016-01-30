//               MFEM Volta Mini App
//               Simple Electrostatics Simulation Code
//
// Compile with: make volta
//
// Sample runs:
//
//   By default the sources and fields are all zero
//     mpirun -np 4 volta
//
//   A cylinder at constant voltage in a square, grounded metal pipe.
//     mpirun -np 4 volta -m ../../data/square-disc.mesh
//                        -dbcs '1 2 3 4 5 6 7 8' -dbcv '0 0 0 0 1 1 1 1'
//
//   A cylinder with a constant surface charge density in a square,
//   grounded metal pipe.
//     mpirun -np 4 volta -m ../../data/square-disc.mesh
//                        -nbcs '5 6 7 8' -nbcv '5e-11 5e-11 5e-11 5e-11'
//                        -dbcs '1 2 3 4' 
//
//   A charged sphere, off-center, within a grounded metal sphere.
//     mpirun -np 4 volta -dbcs 1 -cs '0.0 0.5 0.0 0.2 2.0e-11'
//
//   A dielectric sphere suspended in a uniform electric field.
//     mpirun -np 4 volta -dbcs 1 -dbcg -ds '0.0 0.0 0.0 0.2 8.0'
//
// Description:
//               This mini app solves a simple 2D or 3D electrostatic
//               problem with non-uniform dielectric permittivity.
//                  Div eps Grad Phi = rho
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

// Permittivity Function
static Vector ds_params_(0);  // Center, Radius, and Permittivity
//                               of dielectric sphere
double dielectric_sphere(const Vector &);

// Charge Density Function
static Vector cs_params_(0);  // Center, Radius, and Total Charge
//                               of charged sphere
double charged_sphere(const Vector &);

// Phi Boundary Condition
double phi_bc_uniform(const Vector &);

// Physical Constants
// Permittivity of Free Space (units F/m)
static double epsilon0_ = 8.8541878176e-12;

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
   bool visualization = true;
   bool visit = true;

   Array<int> dbcs;
   Array<int> nbcs;

   Vector dbcv;
   Vector nbcv;

   bool dbcg = false;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&sr, "-rs", "--serial-ref-levels",
                  "Number of serial refinement levels.");
   args.AddOption(&pr, "-rp", "--parallel-ref-levels",
                  "Number of parallel refinement levels.");
   args.AddOption(&ds_params_, "-ds", "--dielectric-sphere-params",
                  "Center, Radius, and Permittivity of Dielectric Sphere");
   args.AddOption(&cs_params_, "-cs", "--charged-sphere-params",
                  "Center, Radius, and Total Charge of Charged Sphere");
   args.AddOption(&dbcs, "-dbcs", "--dirichlet-bc-surf",
                  "Dirichlet Boundary Condition Surfaces");
   args.AddOption(&dbcv, "-dbcv", "--dirichlet-bc-vals",
                  "Dirichlet Boundary Condition Values");
   args.AddOption(&dbcg, "-dbcg", "--dirichlet-bc-gradient",
                  "-no-dbcg", "--no-dirichlet-bc-gradient",
                  "Dirichlet Boundary Condition Gradient (phi = -z)");
   args.AddOption(&nbcs, "-nbcs", "--neumann-bc-surf",
                  "Neumann Boundary Condition Surfaces");
   args.AddOption(&nbcv, "-nbcv", "--neumann-bc-vals",
                  "Neumann Boundary Condition Values");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&visit, "-visit", "--visit", "-no-visit",
                  "--no-visualization",
                  "Enable or disable VisIt visualization.");
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

   int sdim = mesh->SpaceDimension();
   int dim = mesh->Dimension();

   if ( ds_params_.Size() != sdim + 2 )
   {
      // The dielectric sphere parameters have not been set.
      // We will set them to default values.
      ds_params_.SetSize(sdim+2);
      ds_params_ = 0.0;
      ds_params_(sdim+1) = 1.0;
   }
   if ( cs_params_.Size() != sdim + 2 )
   {
      // The charged sphere parameters have not been set.
      // We will set them to default values.
      cs_params_.SetSize(sdim+2);
      cs_params_ = 0.0;
      cs_params_(sdim+1) = 0.0;
   }

   // If values for Dirichlet BCs were not set assume they are zero
   if (dbcv.Size() < dbcs.Size() && !dbcg )
   {
      dbcv.SetSize(dbcs.Size());
      dbcv = 0.0;
   }

   // If values for Neumann BCs were not set assume they are zero
   if (nbcv.Size() < nbcs.Size() )
   {
      nbcv.SetSize(nbcs.Size());
      nbcv = 0.0;
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
   mesh->EnsureNCMesh();

   // Define a parallel mesh by a partitioning of the serial mesh. Refine
   // this mesh further in parallel to increase the resolution. Once the
   // parallel mesh is defined, the serial mesh can be deleted.
   ParMesh pmesh(MPI_COMM_WORLD, *mesh);
   delete mesh;

   // Refine this mesh in parallel to increase the resolution.
   int par_ref_levels = pr;
   for (int l = 0; l < par_ref_levels; l++)
   {
      pmesh.UniformRefinement();
   }

   socketstream phi_sock, e_sock, err_sock;
   char vishost[] = "localhost";
   int  visport   = 19916;
   if (visualization)
   {
      phi_sock.open(vishost, visport);
      phi_sock.precision(8);

      MPI_Barrier(MPI_COMM_WORLD);

      e_sock.open(vishost, visport);
      e_sock.precision(8);

      MPI_Barrier(MPI_COMM_WORLD);

      err_sock.open(vishost, visport);
      err_sock.precision(8);
   }

   // Define compatible parallel finite element spaces on the parallel
   // mesh. Here we use arbitrary order H1 and Nedelec finite
   // elements.
   H1_ParFESpace H1FESpace(&pmesh,order,dim);
   ND_ParFESpace HCurlFESpace(&pmesh,order,dim);

   // Select DoFs on the requested surfaces as Dirichlet BCs
   Array<int> ess_bdr(pmesh.bdr_attributes.Max());
   ess_bdr = 0;
   for (int i=0; i<dbcs.Size(); i++)
   {
      ess_bdr[dbcs[i]-1] = 1;
   }

   // Set up the parallel bilinear form corresponding to the
   // electrostatic operator div eps grad, by adding the diffusion
   // domain integrator and finally imposing Dirichlet boundary
   // conditions. The boundary conditions are implemented by marking
   // all the boundary attributes from the mesh as essential
   // (Dirichlet). After serial and parallel assembly we extract the
   // parallel matrix A.
   FunctionCoefficient eps(dielectric_sphere);
   FunctionCoefficient rho_func(charged_sphere);

   ParBilinearForm laplacian_eps(&H1FESpace);
   laplacian_eps.AddDomainIntegrator(new DiffusionIntegrator(eps));

   ParBilinearForm mass(&H1FESpace);
   mass.AddDomainIntegrator(new MassIntegrator);

   ParBilinearForm mass_s(&H1FESpace);
   mass_s.AddBoundaryIntegrator(new MassIntegrator);

   // The gradient operator needed to compute E from Phi
   ParDiscreteGradOperator Grad(&H1FESpace, &HCurlFESpace);

   // Create various grid functions
   ParGridFunction phi(&H1FESpace);    // Electric Potential
   ParGridFunction rho(&H1FESpace);    // Volumetric Charge Density
   ParGridFunction sigma(&H1FESpace);  // Surface Charge Density
   ParGridFunction e(&HCurlFESpace);   // Electric Field

   // Create coefficient for optional applied field
   FunctionCoefficient phi_bc(phi_bc_uniform);

   // Setup VisIt visualization class
   VisItDataCollection visit_dc("Volta-AMR-Parallel", &pmesh);

   if ( visit )
   {
      visit_dc.RegisterField("Phi", &phi);
      visit_dc.RegisterField("Rho", &rho);
      visit_dc.RegisterField("E", &e);
   }

   // The main AMR loop. In each iteration we solve the problem on the
   // current mesh, visualize the solution, estimate the error on all
   // elements, refine the worst elements and update all objects to work
   // with the new mesh.
   const int max_dofs = 100000;
   for (int it = 1; it <= 100; it++)
   {
      HYPRE_Int size_h1 = H1FESpace.GlobalTrueVSize();
      HYPRE_Int size_nd = HCurlFESpace.GlobalTrueVSize();
      if (myid == 0)
      {
         cout << "\nIteration " << it << endl;
         cout << "Number of H1      unknowns: " << size_h1 << endl;
         cout << "Number of H(Curl) unknowns: " << size_nd << endl;
      }

      // Assemble Matrices
      laplacian_eps.Assemble();
      laplacian_eps.Finalize();

      mass.Assemble();
      mass.Finalize();

      if ( nbcs.Size() > 0 )
      {
         mass_s.Assemble();
         mass_s.Finalize();
      }

      // Initialize the electric potential with its boundary conditions
      phi = 0.0;

      if ( dbcs.Size() > 0 )
      {
         if ( dbcg )
         {
            // Apply gradient boundary condition
            phi.ProjectBdrCoefficient(phi_bc, ess_bdr);
         }
         else
         {
            // Apply piecewise constant boundary condition
            Array<int> dbc_bdr_attr(pmesh.bdr_attributes.Max());
            for (int i=0; i<dbcs.Size(); i++)
            {
               ConstantCoefficient voltage(dbcv[i]);
               dbc_bdr_attr = 0;
               dbc_bdr_attr[dbcs[i]-1] = 1;
               phi.ProjectBdrCoefficient(voltage, dbc_bdr_attr);
            }
         }
      }

      // Initialize the volumetric charge density
      rho.ProjectCoefficient(rho_func);

      HypreParMatrix *Mass = mass.ParallelAssemble();
      HypreParVector *Rho   = rho.ParallelProject();
      HypreParVector *RhoD  = new HypreParVector(&H1FESpace);

      Mass->Mult(*Rho,*RhoD);

      // Initialize the suface charge density
      if ( nbcs.Size() > 0 )
      {
         Array<int> nbc_bdr_attr(pmesh.bdr_attributes.Max());
         for (int i=0; i<nbcs.Size(); i++)
         {
            ConstantCoefficient sigma_coef(nbcv[i]);
            nbc_bdr_attr = 0;
            nbc_bdr_attr[nbcs[i]-1] = 1;
            sigma.ProjectBdrCoefficient(sigma_coef, nbc_bdr_attr);
         }

         HypreParMatrix *Mass_s = mass_s.ParallelAssemble();
         HypreParVector *Sigma = sigma.ParallelProject();

         Mass_s->Mult(*Sigma,*RhoD,1.0,1.0);

         delete Mass_s;
         delete Sigma;
      }

      // Apply Dirichlet BCs to matrix and right hand side
      HypreParMatrix *Laplacian_eps = laplacian_eps.ParallelAssemble();
      HypreParVector *Phi           = phi.ParallelProject();

      // Apply the boundary conditions to the assembled matrix and vectors
      if ( dbcs.Size() > 0 )
      {
         // According to the selected surfaces
         laplacian_eps.ParallelEliminateEssentialBC(ess_bdr,
                                                    *Laplacian_eps,
                                                    *Phi, *RhoD);
      }
      else
      {
         // No surfaces were labeled as Dirichlet so eliminate one DoF
         Array<int> dof_list(0);
         if ( myid == 0 )
         {
            dof_list.SetSize(1);
            dof_list[0] = 0;
         }
         Laplacian_eps->EliminateRowsCols(dof_list, *Phi, *RhoD);
      }

      // Define and apply a parallel PCG solver for AX=B with the AMG
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
      HypreParVector *E = new HypreParVector(&HCurlFESpace);
      Grad.Mult(*Phi,*E,-1.0);
      e = *E;

      // Send the solution by socket to a GLVis server.
      if (visualization)
      {
         phi_sock << "parallel " << num_procs << " " << myid << "\n";
         phi_sock << "solution\n" << pmesh << phi
                  << "window_title 'Scalar Potential (Phi)'\n"
                  << flush;

         MPI_Barrier(pmesh.GetComm());

         e_sock << "parallel " << num_procs << " " << myid << "\n";
         e_sock << "solution\n" << pmesh << e
                << "window_title 'Electric Field (E)'\n" << flush;
      }

      // Estimate element errors using the Zienkiewicz-Zhu error estimator.
      // The bilinear form integrator must have the 'ComputeElementFlux'
      // method defined.
      Vector errors(pmesh.GetNE());
      {
         // Space for the discontinuous (original) flux
         DiffusionIntegrator flux_integrator(eps);
         L2_FECollection flux_fec(order, dim);
         ParFiniteElementSpace flux_fes(&pmesh, &flux_fec, sdim);

         // Space for the smoothed (conforming) flux
         double norm_p = 1;
         RT_FECollection smooth_flux_fec(order-1, dim);
         ParFiniteElementSpace smooth_flux_fes(&pmesh, &smooth_flux_fec);

         // Another possible set of options for the smoothed flux space:
         // norm_p = 1;
         // H1_FECollection smooth_flux_fec(order, dim);
         // ParFiniteElementSpace smooth_flux_fes(&pmesh, &smooth_flux_fec, dim);

         L2ZZErrorEstimator(flux_integrator, phi,
                            smooth_flux_fes, flux_fes, errors, norm_p);
      }
      double local_max_err = errors.Max();
      double global_max_err;
      MPI_Allreduce(&local_max_err, &global_max_err, 1,
                    MPI_DOUBLE, MPI_MAX, pmesh.GetComm());

      if ( visit )
      {
         visit_dc.SetCycle(it);
         visit_dc.SetTime(global_max_err);
         visit_dc.Save();
      }

      if (size_h1 > max_dofs)
      {
         break;
      }

      // Make a list of elements whose error is larger than a fraction
      // of the maximum element error. These elements will be refined.
      Array<int> ref_list;
      const double frac = 0.7;
      double threshold = frac * global_max_err;
      for (int i = 0; i < errors.Size(); i++)
      {
         if (errors[i] >= threshold) { ref_list.Append(i); }
      }

      // Refine the selected elements. Since we are going to transfer the
      // grid function x from the coarse mesh to the new fine mesh in the
      // next step, we need to request the "two-level state" of the mesh.
      pmesh.GeneralRefinement(ref_list);

      // Update the space to reflect the new state of the mesh. Also,
      // interpolate the solution x so that it lies in the new space but
      // represents the same function. This saves solver iterations since
      // we'll have a good initial guess of x in the next step.
      // The interpolation algorithm needs the mesh to hold some information
      // about the previous state, which is why the call UseTwoLevelState
      // above is required.
      //fespace.UpdateAndInterpolate(&x);

      // Note: If interpolation was not needed, we could just use the following
      //     six calls to update the space and the grid function. (No need to
      //     call UseTwoLevelState in this case.)
      H1FESpace.Update();
      HCurlFESpace.Update();
      phi.Update();
      rho.Update();
      sigma.Update();
      e.Update();

      // Inform the bilinear forms that the space has changed.
      Grad.Update();
      mass.Update();
      mass_s.Update();
      laplacian_eps.Update();

      // Free the used memory.
      delete E;
      delete Rho;
      delete RhoD;
      delete Mass;
      delete Phi;
      delete Laplacian_eps;

      char c;
      if (myid == 0)
      {
         cout << "press (q)uit or (c)ontinue --> " << flush;
         cin >> c;
      }
      MPI_Bcast(&c, 1, MPI_CHAR, 0, MPI_COMM_WORLD);

      if (c != 'c')
      {
         break;
      }
   }

   MPI_Finalize();

   return 0;
}

// A sphere with constant permittivity.  The sphere has a radius,
// center, and permittivity specified on the command line and stored
// in ds_params_.
double dielectric_sphere(const Vector &x)
{
   double r2 = 0.0;

   for (int i=0; i<x.Size(); i++)
   {
      r2 += (x(i)-ds_params_(i))*(x(i)-ds_params_(i));
   }

   if ( sqrt(r2) <= ds_params_(x.Size()) )
   {
      return ds_params_(x.Size()+1) * epsilon0_;
   }
   return epsilon0_;
}

// A sphere with constant charge density.  The sphere has a radius,
// center, and total charge specified on the command line and stored
// in cs_params_.
double charged_sphere(const Vector &x)
{
   double r2 = 0.0;
   double rho = 0.0;

   if ( cs_params_(x.Size()) > 0.0 )
   {
      switch ( x.Size() )
      {
         case 2:
            rho = cs_params_(x.Size()+1)/(M_PI*pow(cs_params_(x.Size()),2));
            break;
         case 3:
            rho = 0.75*cs_params_(x.Size()+1)/(M_PI*pow(cs_params_(x.Size()),3));
            break;
         default:
            rho = 0.0;
      }
   }

   for (int i=0; i<x.Size(); i++)
   {
      r2 += (x(i)-cs_params_(i))*(x(i)-cs_params_(i));
   }

   if ( sqrt(r2) <= cs_params_(x.Size()) )
   {
      return rho;
   }
   return 0.0;
}

// To produce a uniform electric field the potential can be set
// to -z (or -y in 2D).
double phi_bc_uniform(const Vector &x)
{
   return -x(x.Size()-1);
}
