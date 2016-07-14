//                                MFEM Example 16
//
// Compile with: make ex16
//
// Sample runs:
//    ex10 -m ../data/star.mesh -s 3 -r 2 -o 2 -dt 3
//
// Description:  This examples solves a time dependent nonlinear heat equation
//               problem of the form du/dt = C(u), where C is a
//               non-linear Laplacian C(u) = \nabla \cdot (\alpha u) \nabla u.
//
//               The example demonstrates the use of nonlinear operators (the
//               class ConductionOperator defining C(x)), as well as their
//               implicit time integration. Note that implementing the
//               method ConductionOperator::ImplicitSolve is the only
//               requirement for high-order implicit (SDIRK) time integration.
//
//               We recommend viewing examples 2 and 9 before viewing this
//               example.


#include "mfem.hpp"
#include <memory>
#include <iostream>
#include <fstream>

using namespace std;
using namespace mfem;

/** After spatial discretization, the conduction model can be written as:
 *     du/dt = -M^{-1}((\alpha u) Ku)
 *  where u is the vector representing the temperature,
 *  M is the mass matrix, and K is the diffusion matrix.
 *
 *  Class ConductionOperator represents the right-hand side of the above ODE
 */
class ConductionOperator : public TimeDependentOperator
{
protected:
   FiniteElementSpace &fespace;

   BilinearForm *M;
   BilinearForm *K;

   CGSolver M_solver; // Krylov solver for inverting the mass matrix M
   DSmoother M_prec;  // Preconditioner for the mass matrix M

   Solver *K_solver; // Implicit solver for M + dt K
   Solver *K_prec; // Preconditioner for the implicit solver

   Vector u_alpha; // u * alpha at the previous time

   Array<int> &ess_bdr;

   double alpha;

   mutable Vector z; // auxiliary vector

public:
   ConductionOperator(FiniteElementSpace &f, Array<int> &ess_bdr,
                      double alpha, const Vector &u_);

   virtual void Mult(const Vector &u, Vector &du_dt) const;
   /** Solve the Backward-Euler equation: k = f(u + dt*k, t), for the unknown k.
       This is the only requirement for high-order SDIRK implicit integration.*/
   virtual void ImplicitSolve(const double dt, const Vector &u, Vector &k);

   void SetParameters(const Vector &u_);

   virtual ~ConductionOperator();
};

double InitialTemperature(const Vector &x);

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../data/star.mesh";
   int ref_levels = 2;
   int order = 2;
   int ode_solver_type = 3;
   double t_final = 300.0;
   double dt = 3.0;
   double alpha = 10;
   bool visualization = true;
   int vis_steps = 1;

   int precision = 8;
   cout.precision(precision);

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly.");
   args.AddOption(&order, "-o", "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                  "ODE solver: 1 - Backward Euler, 2 - SDIRK2, 3 - SDIRK3,\n\t"
                  "\t   11 - Forward Euler, 12 - RK2, 13 - RK3 SSP, 14 - RK4.");
   args.AddOption(&t_final, "-tf", "--t-final",
                  "Final time; start time is 0.");
   args.AddOption(&dt, "-dt", "--time-step",
                  "Time step.");
   args.AddOption(&alpha, "-a", "--alpha",
                  "Alpha coefficient.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&vis_steps, "-vs", "--visualization-steps",
                  "Visualize every n-th timestep.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   // 2. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral and hexahedral meshes with the same code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   // 3. Define the ODE solver used for time integration. Several implicit
   //    singly diagonal implicit Runge-Kutta (SDIRK) methods, as well as
   //    explicit Runge-Kutta methods are available.
   ODESolver *ode_solver;
   switch (ode_solver_type)
   {
      // Implicit L-stable methods
      case 1:  ode_solver = new BackwardEulerSolver; break;
      case 2:  ode_solver = new SDIRK23Solver(2); break;
      case 3:  ode_solver = new SDIRK33Solver; break;
      // Explicit methods
      case 11: ode_solver = new ForwardEulerSolver; break;
      case 12: ode_solver = new RK2Solver(0.5); break; // midpoint method
      case 13: ode_solver = new RK3SSPSolver; break;
      case 14: ode_solver = new RK4Solver; break;
      // Implicit A-stable methods (not L-stable)
      case 22: ode_solver = new ImplicitMidpointSolver; break;
      case 23: ode_solver = new SDIRK23Solver; break;
      case 24: ode_solver = new SDIRK34Solver; break;
      default:
         cout << "Unknown ODE solver type: " << ode_solver_type << '\n';
         return 3;
   }

   // 4. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement, where 'ref_levels' is a
   //    command-line parameter.
   for (int lev = 0; lev < ref_levels; lev++)
   {
      mesh->UniformRefinement();
   }

   // 5. Define the vector finite element space representing the current and the 
   //    initial temperature, u_ref.
   H1_FECollection fe_coll(order, dim);
   FiniteElementSpace fespace(mesh, &fe_coll);

   int fe_size = fespace.GetVSize();
   cout << "Number of temperature unknowns: " << fe_size << endl;

   Vector u(fe_size);;
   GridFunction u_gf;
   u_gf.MakeRef(&fespace, u, 0);

   // 6. Set the initial conditions for u, and the boundary conditions. All 
   // boundaries are considered essential.
   FunctionCoefficient u_0(InitialTemperature);
   u_gf.ProjectCoefficient(u_0);

   Array<int> ess_bdr(fespace.GetMesh()->bdr_attributes.Max());
   ess_bdr = 1;

   // 7. Initialize the conduction operator and the VisIt visualization 
   ConductionOperator oper(fespace, ess_bdr, alpha, u);

   VisItDataCollection visit_dc("Example16", mesh);
   visit_dc.RegisterField("temperature", &u_gf);
   if (visualization)
   {
      visit_dc.SetCycle(0);
      visit_dc.SetTime(0.0);
      visit_dc.Save();
   }

   // 8. Perform time-integration (looping over the time iterations, ti, with a
   //    time-step dt).
   ode_solver->Init(oper);
   double t = 0.0;

   bool last_step = false;
   for (int ti = 1; !last_step; ti++)
   {
      if (t + dt >= t_final - dt/2)
      {
         last_step = true;
      }

      ode_solver->Step(u, t, dt);

      if (last_step || (ti % vis_steps) == 0)
      {
         cout << "step " << ti << ", t = " << t << endl;

         if (visualization)
         {
            visit_dc.SetCycle(ti);
            visit_dc.SetTime(t);
            visit_dc.Save();
         }
      }
      oper.SetParameters(u);
   }


   // 9. Free the used memory.
   delete ode_solver;
   delete mesh;

   return 0;
}

ConductionOperator::ConductionOperator(FiniteElementSpace &f,
                                       Array<int> &ess_bdr_, double al, const Vector &u_)
   : TimeDependentOperator(f.GetVSize(), 0.0), fespace(f), ess_bdr(ess_bdr_), z(height)
{
   const double rel_tol = 1e-8;
   const int skip_zero_entries = 0;

   M = new BilinearForm(&fespace);
   M->AddDomainIntegrator(new MassIntegrator());
   M->Assemble(skip_zero_entries);
   M->EliminateEssentialBC(ess_bdr);
   M->Finalize(skip_zero_entries);

   M_solver.iterative_mode = false;
   M_solver.SetRelTol(rel_tol);
   M_solver.SetAbsTol(0.0);
   M_solver.SetMaxIter(30);
   M_solver.SetPrintLevel(0);
   M_solver.SetPreconditioner(M_prec);
   M_solver.SetOperator(M->SpMat());

   alpha = al;

   K_prec = new DSmoother(1);
   MINRESSolver *K_minres = new MINRESSolver;
   K_minres->SetRelTol(rel_tol);
   K_minres->SetAbsTol(0.0);
   K_minres->SetMaxIter(300);
   K_minres->SetPrintLevel(-1);
   K_minres->SetPreconditioner(*K_prec);
   K_solver = K_minres;

   SetParameters(u_);

}

void ConductionOperator::Mult(const Vector &u, Vector &du_dt) const
{
   K->Mult(u, z);
   z.Neg(); // z = -z
   M_solver.Mult(z, du_dt);
}

void ConductionOperator::ImplicitSolve(const double dt,
                                       const Vector &u, Vector &du_dt)
{
   // Solve the equation:
   //    du_dt = -M^{-1}*[K(u + dt*du_dt)]

   SparseMatrix *T = Add(1.0, M->SpMat(), dt, K->SpMat());
   K_solver->SetOperator(*T);
   K->Mult(u, z);
   K_solver->Mult(z, du_dt);
}

void ConductionOperator::SetParameters(const Vector &u_)
{
   const int skip_zero_entries = 0;

   u_alpha = u_;
   u_alpha *= alpha;

   if (K != NULL) {
      delete K;
   } 

   K = new BilinearForm(&fespace);
   GridFunction u_alpha_gf;
   u_alpha_gf.MakeRef(&fespace, u_alpha, 0);

   GridFunctionCoefficient u_coeff(&u_alpha_gf);

   K->AddDomainIntegrator(new DiffusionIntegrator(u_coeff));
   K->Assemble(skip_zero_entries);
   K->EliminateEssentialBC(ess_bdr);
   K->Finalize(skip_zero_entries);

}

ConductionOperator::~ConductionOperator()
{
   delete K_solver;
   delete K_prec;
}

double InitialTemperature(const Vector &x)
{
   int dim = x.Size();
   switch (dim)
      {
      case 1:
         if (abs(x(0)) < 0.5) {
            return 1.0;
         }
         else {
            return 0.0;
         }
      case 2:
      case 3:         
         if (sqrt(x(0)*x(0) + x(1) * x(1)) < 0.5) {
            return 1.0;
         }
         else {
            return 0.0;
         }

      }
   return 0.0;
}

