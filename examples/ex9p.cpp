//                                MFEM Example 9
//
// Compile with: make ex9p
//
// Sample runs:  mpirun -np 4 ex9p -m ../data/periodic-segment.mesh -p 0 -rs 2 -rp 0 -dt 0.005
//               mpirun -np 4 ex9p -m ../data/periodic-square.mesh -p 0 -rs 2 -rp 0 -dt 0.01 -tf 10
//               mpirun -np 4 ex9p -m ../data/periodic-hexagon.mesh -p 0 -rs 2 -rp 0 -dt 0.01 -tf 10
//               mpirun -np 4 ex9p -m ../data/periodic-square.mesh -p 1 -rs 2 -rp 0 -dt 0.005 -tf 9
//               mpirun -np 4 ex9p -m ../data/periodic-hexagon.mesh -p 1 -rs 2 -rp 0 -dt 0.005 -tf 9
//               mpirun -np 4 ex9p -m ../data/disc-nurbs.mesh -p 2 -rs 2 -rp 1 -dt 0.005 -tf 9
//               mpirun -np 4 ex9p -m ../data/periodic-square.mesh -p 3 -rs 2 -rp 2 -dt 0.0025 -tf 9 -vs 20
//
// Description:  This example code solves the simple time-dependent advection
//               equation du/dt = v.grad(u), where v is the fluid velocity, and
//               u0(x)=u(0,x) is a given initial condition.
//
//               The example demonstrates the use of Discontinuous Galerkin (DG)
//               bilinear forms in MFEM (face integrators), the use of explicit
//               ODE time integrators, the definition of periodic boundary
//               conditions through periodic meshes, as well as the use of GLVis
//               for the persistent visualization of time-evolving solution.

#include "mfem.hpp"
#include <iostream>
#include <fstream>

using namespace std;
using namespace mfem;

// Choice for the problem setup. The fluid velocity, initial condition and
// inflow boundary condition are chosen based on this parameter.
int problem;

// Velocity coefficient
void velocity_function(const Vector &x, Vector &v);

// Initial condition
double u0_function(Vector &x);

// Inflow boundary condition
double inflow_function(Vector &x);


/** A time-dependent operator for the right-hand side of the ODE. The DG weak
    form of du/dt = v.grad(u) is M du/dt = K u + b, where M and K are the mass
    and advection matrices, and b describes the flow on the boundary. This can
    be written as a general ODE, du/dt = M^{-1} (K u + b), and this class is used
    to evaluate the right-hand side. */
class FE_Evolution : public TimeDependentOperator
{
private:
   HypreParMatrix &M, &K;
   const Vector &b;
   HypreSmoother M_prec;
   CGSolver M_solver;

   mutable Vector z;

public:
   FE_Evolution(HypreParMatrix &_M, HypreParMatrix &_K, const Vector &_b);

   virtual void Mult(const Vector &x, Vector &y) const;

   virtual ~FE_Evolution() { }
};


int main(int argc, char *argv[])
{
   // 1. Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // 2. Parse command-line options.
   problem = 0;
   const char *mesh_file = "../data/periodic-hexagon.mesh";
   int ser_ref_levels = 2;
   int par_ref_levels = 2;
   int order = 3;
   int ode_solver_type = 4;
   double t_final = 10.0;
   double dt = 0.01;
   bool visualization = 1;
   int vis_steps = 5;

   int precision = 8;
   cout.precision(precision);

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&problem, "-p", "--problem",
                  "Problem setup to use. See options in velocity_function().");
   args.AddOption(&ser_ref_levels, "-rs", "--refine_serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&par_ref_levels, "-rp", "--refine_parallel",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&order, "-o", "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                  "ODE solver: 1 - Forw. Euler, 2 - RK2 SSP, 3 - RK3 SSP,"
                  " 4 - RK4, 6 - RK6.");
   args.AddOption(&t_final, "-tf", "--t-final",
                  "Final time; start time is 0.");
   args.AddOption(&dt, "-dt", "--time-step",
                  "Time step.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&vis_steps, "-vs", "--visualization-steps",
                  "Visualize every n-th timestep.");

   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      MPI_Finalize();
      return 1;
   }
   if(myid == 0)
      args.PrintOptions(cout);

   // 3. Read the mesh from the given mesh file. We can handle geometrically
   //    periodic meshes in this code.
   Mesh *mesh;
   {
      ifstream imesh(mesh_file);
      if (!imesh)
      {
         cout << "Can not open mesh: " << mesh_file << endl;
         MPI_Finalize();
         return 2;
      }
      mesh = new Mesh(imesh, 1, 1);
   }
   int dim = mesh->Dimension();

   // 4. Define the ODE solver used for time integration. Several explicit
   //    Runge-Kutta methods are available.
   ODESolver *ode_solver = NULL;
   switch (ode_solver_type)
   {
   case 1: ode_solver = new ForwardEulerSolver; break;
   case 2: ode_solver = new RK2Solver(1.0); break;
   case 3: ode_solver = new RK3SSPSolver; break;
   case 4: ode_solver = new RK4Solver; break;
   case 6: ode_solver = new RK6Solver; break;
   default:
	  if(myid == 0)
         cout << "Unknown ODE solver type: " << ode_solver_type << '\n';
	  MPI_Finalize();
      return 3;
   }

   // 5. Refine the mesh in serial to increase the resolution. In this example we do
   //    'ser_ref_levels' of uniform refinement, where 'ser_ref_levels' is a
   //    command-line parameter. If the mesh is of NURBS type, we convert it to
   //    a (piecewise-polynomial) high-order mesh.
   for (int lev = 0; lev < ser_ref_levels; lev++)
      mesh->UniformRefinement();

   if (mesh->NURBSext)
   {
      int mesh_order = std::max(order, 1);
      FiniteElementCollection *mfec = new H1_FECollection(mesh_order, dim);
      FiniteElementSpace *mfes = new FiniteElementSpace(mesh, mfec, dim);
      mesh->SetNodalFESpace(mfes);
      mesh->GetNodes()->MakeOwner(mfec);
   }

   // 6. Define the parallel mesh
   ParMesh * pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;

   // 7. Refine the mesh in parallel to increase the resolution. In this example we do
   //    'par_ref_levels' of uniform refinement, where 'par_ref_levels' is a
   //    command-line parameter.
   for (int lev = 0; lev < par_ref_levels; lev++)
      pmesh->UniformRefinement();

   // 8. Define the discontinuous DG finite element space on the mesh of the
   //    given polynomial order.
   DG_FECollection fec(order, dim);
   ParFiniteElementSpace * fes = new ParFiniteElementSpace(pmesh, &fec);

   int global_vSize = fes->GlobalTrueVSize();
   if(myid == 0)
      cout << "Total number of dofs = " << global_vSize << endl;

   // 9. Set up and assemble the bilinear and linear forms corresponding to the
   //    DG discretization. The DGTraceIntegrator object involves integrals over
   //    the interior faces in the mesh.
   VectorFunctionCoefficient velocity(dim, velocity_function);
   FunctionCoefficient inflow(inflow_function);
   FunctionCoefficient u0(u0_function);

   ParBilinearForm * mVarf, *kVarf;
   mVarf = new ParBilinearForm(fes);
   mVarf->AddDomainIntegrator(new MassIntegrator);
   kVarf = new ParBilinearForm(fes);
   kVarf->AddDomainIntegrator(new ConvectionIntegrator(velocity, -1.0));
   kVarf->AddInteriorFaceIntegrator(
      new TransposeIntegrator(new DGTraceIntegrator(velocity, 1.0, -0.5)));
   kVarf->AddBdrFaceIntegrator(
      new TransposeIntegrator(new DGTraceIntegrator(velocity, 1.0, -0.5)));

   ParLinearForm * bForm = new ParLinearForm(fes);
   bForm->AddBdrFaceIntegrator(
      new BoundaryFlowIntegrator(inflow, velocity, -1.0, -0.5));

   mVarf->Assemble();
   mVarf->Finalize();
   int skip_zeros = 0;
   kVarf->Assemble(skip_zeros);
   kVarf->Finalize(skip_zeros);
   bForm->Assemble();

   HypreParMatrix * M = mVarf->ParallelAssemble();
   HypreParMatrix * K = kVarf->ParallelAssemble();
   HypreParVector * b = bForm->ParallelAssemble();

   // 10. Define the initial conditions, save the corresponding function to a
   //    file and (optionally) initialize GLVis visualization.
   ParGridFunction *u = new ParGridFunction(fes);
   u->ProjectCoefficient(u0);
   HypreParVector * x = u->GetTrueDofs();

   {
      ostringstream mesh_name, sol_name;
	  mesh_name << "ex9p." << setfill('0') << setw(6) << myid;
	  sol_name << "ex9-init." << setfill('0') << setw(6) << myid;

      ofstream omesh(mesh_name.str().c_str());
      omesh.precision(precision);
      pmesh->Print(omesh);
      ofstream osol(sol_name.str().c_str());
      osol.precision(precision);
      u->Save(osol);
   }

   socketstream sout;
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      sout.open(vishost, visport);
      if (!sout)
      {
         if(myid == 0)
            cout << "Unable to connect to GLVis server at "
                 << vishost << ':' << visport << endl;
         visualization = false;
         if(myid == 0)
            cout << "GLVis visualization disabled.\n";
      }
      else
      {
         sout << "parallel " << num_procs << " " << myid << "\n";
         sout.precision(precision);
         sout << "solution\n" << *pmesh << *u;
         sout << "pause\n";
         sout << flush;
         if(myid==0)
            cout << "GLVis visualization paused."
                 << " Press space (in the GLVis window) to resume it.\n";
      }
   }

   // 11. Define the time-dependent evolution operator describing the ODE
   //    right-hand side, and perform time-integration (looping over the time
   //    iterations, ti, with a time-step dt).
   FE_Evolution adv(*M, *K, *b);
   ode_solver->Init(adv);

   double t = 0.0;
   for (int ti = 0; true; )
   {
      if (t >= t_final - dt/2)
         break;

      ode_solver->Step(*x, t, dt);
      ti++;

      if (visualization && (ti % vis_steps == 0))
      {
         *u = *x;
         sout << "parallel " << num_procs << " " << myid << "\n";
         sout << "solution\n" << *pmesh << *u << flush;
      }
   }

   // 12. Save the final solution.
   {
	  *u = *x;
	  ostringstream sol_name;
      sol_name << "ex9-final." << setfill('0') << setw(6) << myid;
      ofstream osol(sol_name.str().c_str());
      osol.precision(precision);
      u->Save(osol);
   }

   // 13. Free the used memory.
   delete x;
   delete u;
   delete b;
   delete K;
   delete M;
   delete bForm;
   delete kVarf;
   delete mVarf;
   delete fes;
   delete pmesh;
   delete ode_solver;

   MPI_Finalize();
   return 0;
}


// Implementation of class FE_Evolution
FE_Evolution::FE_Evolution(HypreParMatrix &_M, HypreParMatrix &_K, const Vector &_b)
   : TimeDependentOperator(_M.Height()),
   M(_M),
   K(_K),
   b(_b),
   M_solver(M.GetComm()),
   z(_M.Height())
{
   M_prec.SetType(HypreSmoother::Jacobi);
   M_solver.SetPreconditioner(M_prec);
   M_solver.SetOperator(M);

   M_solver.iterative_mode = false;
   M_solver.SetRelTol(1e-9);
   M_solver.SetAbsTol(0.0);
   M_solver.SetMaxIter(100);
   M_solver.SetPrintLevel(0);
   // M_solver.SetPrintLevel(2);
}

void FE_Evolution::Mult(const Vector &x, Vector &y) const
{
   // y = M^{-1} (K x + b)
   K.Mult(x, z);
   z += b;
   M_solver.Mult(z, y);
}


// Velocity coefficient
void velocity_function(const Vector &x, Vector &v)
{
   int dim = x.Size();

   switch (problem)
   {
   case 0:
   {
      // Translations in 1D, 2D, and 3D
      switch (dim)
      {
      case 1: v(0) = 1.0; break;
      case 2: v(0) = sqrt(2./3.); v(1) = sqrt(1./3.); break;
      case 3: v(0) = sqrt(3./6.); v(1) = sqrt(2./6.); v(2) = sqrt(1./6.); break;
      }
      break;
   }
   case 1:
   case 2:
   {
      // Clockwise rotation in 2D around the origin
      const double w = M_PI/2;
      switch (dim)
      {
      case 1: v(0) = 1.0; break;
      case 2: v(0) = w*x(1); v(1) = -w*x(0); break;
      case 3: v(0) = w*x(1); v(1) = -w*x(0); v(2) = 0.0; break;
      }
      break;
   }
   case 3:
   {
      // Clockwise twisting rotation in 2D around the origin
      const double w = M_PI/2;
      double d = max((x(0)+1.)*(1.-x(0)),0.) * max((x(1)+1.)*(1.-x(1)),0.);
      d = d*d;
      switch (dim)
      {
      case 1: v(0) = 1.0; break;
      case 2: v(0) = d*w*x(1); v(1) = -d*w*x(0); break;
      case 3: v(0) = d*w*x(1); v(1) = -d*w*x(0); v(2) = 0.0; break;
      }
      break;
   }
   }
}

// Initial condition
double u0_function(Vector &x)
{
   int dim = x.Size();

   switch (problem)
   {
   case 0:
   case 1:
   {
      switch (dim)
      {
      case 1:
         return exp(-40.*pow(x(0)-0.5,2));
      case 2:
      {
         const double rx = 0.45, ry = 0.25, cx = 0., cy = -0.2, w = 10.;
         return ( erfc(w*(x(0)-cx-rx))*erfc(-w*(x(0)-cx+rx)) *
                  erfc(w*(x(1)-cy-ry))*erfc(-w*(x(1)-cy+ry)) )/16;
      }
      case 3:
         return 0.0;
      }
   }
   case 2:
   {
      const double r = sqrt(8.);
      double x_ = x(0), y_ = x(1), rho, phi;
      rho = hypot(x_, y_) / r;
      phi = atan2(y_, x_);
      return pow(sin(M_PI*rho),2)*sin(3*phi);
   }
   case 3:
   {
      const double f = M_PI;
      return sin(f*x(0))*sin(f*x(1));
   }
   }
   return 0.0;
}

// Inflow boundary condition
double inflow_function(Vector &x)
{
   switch (problem)
   {
   case 0:
   case 1:
   case 2:
   case 3: return 0.0;
   }
   return 0.0;
}
