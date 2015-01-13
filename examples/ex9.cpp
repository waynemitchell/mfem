//                             MFEM Example 9
//
// Compile with: make ex9
//
// Sample runs:
//      ./ex9 -m ../data/periodic-segment.mesh -p 0 -r 2 -dt 0.005
//      ./ex9 -m ../data/periodic-square.mesh -p 0 -r 2 -dt 0.01 -tf 10
//      ./ex9 -m ../data/periodic-hexagon.mesh -p 0 -r 2 -dt 0.01 -tf 10
//      ./ex9 -m ../data/periodic-square.mesh -p 1 -r 2 -dt 0.005 -tf 9
//      ./ex9 -m ../data/periodic-hexagon.mesh -p 1 -r 2 -dt 0.005 -tf 9
//      ./ex9 -m ../data/disc-nurbs.mesh -p 2 -r 3 -dt 0.005 -tf 9
//      ./ex9 -m ../data/periodic-square.mesh -p 3 -r 4 -dt 0.0025 -tf 9 -vs 20
//
// Description:  TODO

#include <iostream>
#include <fstream>
#include "mfem.hpp"

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


/// TODO: short description
class FE_Evolution : public TimeDependentOperator
{
private:
   SparseMatrix &M, &K;
   const Vector &b;
   DSmoother M_prec;
   CGSolver M_solver;

   mutable Vector z;

public:
   FE_Evolution(SparseMatrix &_M, SparseMatrix &_K, const Vector &_b);

   virtual void Mult(const Vector &x, Vector &y) const;

   virtual ~FE_Evolution() { }
};


int main(int argc, char *argv[])
{
   OptionsParser args(argc, argv);

   problem = 0;
   const char *mesh_file = "../data/periodic-hexagon.mesh";
   int ref_levels = 2;
   int order = 3;
   int ode_solver_type = 4;
   double t_final = 10.0;
   double dt = 0.01;

   bool visualization = true;
   int vis_steps = 5;

   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&problem, "-p", "--problem",
                  "Problem setup to use.");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly.");
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

   int precision = 8;

   cout.precision(precision);

   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   Mesh *mesh;
   {
      ifstream imesh(mesh_file);
      if (!imesh)
      {
         cout << "Can not open mesh: " << mesh_file << endl;
         return 2;
      }
      mesh = new Mesh(imesh, 1, 1);
   }

   ODESolver *ode_solver = NULL;
   switch (ode_solver_type)
   {
   case 1: ode_solver = new ForwardEulerSolver; break;
   case 2: ode_solver = new RK2Solver(1.0); break;
   case 3: ode_solver = new RK3SSPSolver; break;
   case 4: ode_solver = new RK4Solver; break;
   case 6: ode_solver = new RK6Solver; break;
   default:
      cout << "Unknown ODE solver type: " << ode_solver_type << '\n';
      return 3;
   }

   for (int lev = 0; lev < ref_levels; lev++)
      mesh->UniformRefinement();

   int dim = mesh->Dimension();
   if (mesh->NURBSext)
   {
      int mesh_order = std::max(order, 1);
      FiniteElementCollection *mfec = new H1_FECollection(mesh_order, dim);
      FiniteElementSpace *mfes = new FiniteElementSpace(mesh, mfec, dim);
      mesh->SetNodalFESpace(mfes);
      mesh->GetNodes()->MakeOwner(mfec);
   }

   {
      const char out_mesh_file[] = "ex9.mesh";
      ofstream omesh(out_mesh_file);
      omesh.precision(precision);
      mesh->Print(omesh);
   }

   DG_FECollection fec(order, dim);
   FiniteElementSpace fes(mesh, &fec);

   cout << "Total number of dofs = " << fes.GetVSize() << endl;

   VectorFunctionCoefficient velocity(dim, velocity_function);
   FunctionCoefficient inflow(inflow_function);
   FunctionCoefficient u0(u0_function);

   BilinearForm M(&fes), K(&fes);
   M.AddDomainIntegrator(new MassIntegrator);
   K.AddDomainIntegrator(new ConvectionIntegrator(velocity, -1.0));
   K.AddInteriorFaceIntegrator(
      new TransposeIntegrator(new DGTraceIntegrator(velocity, 1.0, -0.5)));
   K.AddBdrFaceIntegrator(
      new TransposeIntegrator(new DGTraceIntegrator(velocity, 1.0, -0.5)));

   LinearForm b(&fes);
   b.AddBdrFaceIntegrator(
      new BoundaryFlowIntegrator(inflow, velocity, -1.0, -0.5));

   M.Assemble();
   M.Finalize();
   int skip_zeros = 0;
   K.Assemble(skip_zeros);
   K.Finalize(skip_zeros);
   b.Assemble();

   GridFunction u(&fes);
   u.ProjectCoefficient(u0);

   {
      const char init_solution_file[] = "ex9-init.sol";
      ofstream osol(init_solution_file);
      osol.precision(precision);
      u.Save(osol);
   }

   FE_Evolution adv(M.SpMat(), K.SpMat(), b);
   ode_solver->Init(adv);

   socketstream sout;
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      sout.open(vishost, visport);
      if (!sout)
      {
         cout << "Unable to connect to GLVis server at "
              << vishost << ':' << visport << endl;
         visualization = false;
         cout << "GLVis visualization disabled.\n";
      }
      else
      {
         sout.precision(precision);
         sout << "solution\n" << *mesh << u;
         sout << "pause\n";
         sout << flush;
         cout << "GLVis visualization paused."
              << " Press space (in the GLVis window) to resume it.\n";
      }
   }

   // time stepping loop
   double t = 0.0;
   for (int ti = 0; true; )
   {
      if (t >= t_final - dt/2)
         break;

      ode_solver->Step(u, t, dt);
      ti++;

      if (visualization && (ti % vis_steps == 0))
      {
         sout << "solution\n" << *mesh << u << flush;
      }
   }

   {
      const char final_solution_file[] = "ex9-final.sol";
      ofstream osol(final_solution_file);
      osol.precision(precision);
      u.Save(osol);
   }

   delete ode_solver;
   delete mesh;

   return 0;
}


// Implementation of class FE_Evolution
FE_Evolution::FE_Evolution(SparseMatrix &_M, SparseMatrix &_K, const Vector &_b)
   : TimeDependentOperator(_M.Size()), M(_M), K(_K), b(_b), z(_M.Size())
{
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
