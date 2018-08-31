//                           Example using MFEM / HYPRE / SUNDIALS
//
// Compile with: make ex16
//
// Sample runs:
//
// Description:  This example solves a time dependent nonlinear heat equation
//               problem of the form du/dt = C(u), with a reaction-diffusion
//               operator of the form C(u) = -Delta u + alpha * u^2.
//

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "backends/occa/vector.hpp"

#if !defined(MFEM_USE_OCCA) || !defined(MFEM_USE_BACKENDS)
#error Requires OCCA backend support
#endif

using namespace std;
using namespace mfem;


class TimeDerivativeOperator : public Operator
{
   Operator *Moper;
   Operator *Koper;
   mutable Vector Kdu;
   const double dt0;
   double dt;

public:
   TimeDerivativeOperator(Operator *Moper_, const double dt_, Operator *Koper_)
      : Operator(*Koper_->InLayout(), *Moper_->OutLayout()),
        Moper(Moper_),
        Koper(Koper_),
        Kdu(Moper_->OutLayout()),
        dt0(dt_), dt(dt_) { }

   void SetTimestep(const double dt_) {
      // update internal dt used in Mult()
      dt = dt_;
   }

   virtual void Mult(const Vector &x, Vector &y) const
   {
      Moper->Mult(x, y);
      Koper->Mult(x, Kdu);

      y.Axpby(1.0, y, dt, Kdu);
   }
};

/** After spatial discretization, the model can be written as:
 *
 *     du/dt = M^{-1}(-Ku + b)
 *
 *  where u is the vector representing the temperature, M is the mass matrix,
 *  and K is the diffusion operator with a constant coefficient.
 *
 *  Class ModelOperator represents the right-hand side of the above ODE.
 */
class ModelOperator : public TimeDependentOperator
{
protected:
   ParFiniteElementSpace &fespace;
   Array<int> ess_tdof_list; // this list remains empty for pure Neumann b.c.

   ParBilinearForm M;
   ParBilinearForm K;

   OperatorHandle Mh, Kh;
   Operator *Moper, *Koper;
   TimeDerivativeOperator *T;

   CGSolver M_solver; // Krylov solver for inverting the mass matrix M
   CGSolver T_solver; // Implicit solver for T = M + dt K

   double alpha;

   ParGridFunction u_gf;
   GridFunctionCoefficient uc;
   ParLinearForm b;
   Vector b_vec;

   mutable Vector z; // auxiliary vector

   ::occa::device device;

public:
   ModelOperator(ParFiniteElementSpace &f,
                 const char *Mspec,
                 const char *Kspec,
                 double alpha_,
                 const Vector &u,
                 ::occa::device device_);

   virtual void Mult(const Vector &u, Vector &du_dt) const;
   /** Solve the Backward-Euler equation: k = f(u + dt*k, t), for the unknown k.
       This is the only requirement for high-order SDIRK implicit integration.*/
   virtual void ImplicitSolve(const double dt, const Vector &u, Vector &k);

   /// Update the diffusion BilinearForm K using the given true-dof vector `u`.
   void SetParameters(const Vector &u);

   virtual ~ModelOperator() { delete T; }
};

double InitialCondition(const Vector &x);

int main(int argc, char *argv[])
{
   MPI_Session mpi_session(argc, argv);
   const int num_procs = mpi_session.WorldSize();
   const int myid = mpi_session.WorldRank();

   // 1. Parse command-line options.
   const char *mesh_file = "../data/star.mesh"; // star.mesh or fichera.mesh are good options
   int ser_ref_levels = 1;
   int par_ref_levels = 1;
   int order = 2;
   int ode_solver_type = 1;
   double t_final = 0.5;
   double dt = 1.0e-2;
   double alpha = 1.0e-2;
   bool visualization = false;
   int vis_steps = 5;
   const char *Mspec = "representation: 'partial'";
   const char *Kspec = "representation: 'partial'";
   const char *occa_spec = "mode: 'Serial', integrator: 'acrotensor'";

   int precision = 8;
   cout.precision(precision);

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly before parallelizing.");
   args.AddOption(&par_ref_levels, "-rp", "--refine-parallel",
                  "Number of times to refine the mesh uniformly after parallelizing.");
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
   args.AddOption(&Mspec, "-ms", "--mass-spec", "Mass operator specification");
   args.AddOption(&Kspec, "-ks", "--laplace-spec", "Laplacian operator specification");
   args.AddOption(&occa_spec, "-os", "--occa-spec", "OCCA engine specification");
   args.Parse();
   if (!args.Good())
   {
      if (mpi_session.Root()) args.PrintUsage(cout);
      return 1;
   }
   if (mpi_session.Root()) args.PrintOptions(cout);

   // Examples for OCCA specifications:
   //   - CPU (serial): "mode: 'Serial'"
   //   - CUDA GPU: "mode: 'CUDA', device_id: 0"
   //   - OpenMP on CPUs: "mode: 'OpenMP', threads: 4"
   //   - OpenCL on device 0: "mode: 'OpenCL', device_id: 0, platform_id: 0"

   SharedPtr<Engine> engine(new mfem::occa::Engine(MPI_COMM_WORLD, occa_spec));

   // 2. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral and hexahedral meshes with the same code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   mesh->SetEngine(*engine);
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
      case 15: ode_solver = new GeneralizedAlphaSolver(0.5); break;
      // Implicit A-stable methods (not L-stable)
      case 22: ode_solver = new ImplicitMidpointSolver; break;
      case 23: ode_solver = new SDIRK23Solver; break;
      case 24: ode_solver = new SDIRK34Solver; break;
      default:
         cout << "Unknown ODE solver type: " << ode_solver_type << '\n';
         delete mesh;
         return 3;
   }

   // 4. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement, where 'ref_levels' is a
   //    command-line parameter.
   for (int lev = 0; lev < ser_ref_levels; lev++)
   {
      mesh->UniformRefinement();
   }

   // 6. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted.
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   for (int lev = 0; lev < par_ref_levels; lev++)
   {
      pmesh->UniformRefinement();
   }

   // 5. Define the vector finite element space representing the current and the
   //    initial temperature, u_ref.
   H1_FECollection fe_coll(order, dim);
   ParFiniteElementSpace fespace(pmesh, &fe_coll);

   if (mpi_session.Root())
   {
      cout << "Number of temperature unknowns: " << fespace.GlobalTrueVSize() << endl;
   }

   ParGridFunction u_gf(&fespace);

   // 6. Set the initial conditions for u. All boundaries are considered
   //    natural. This computes this on the host, so pull/push is needed.
   u_gf.Pull();
   FunctionCoefficient u_0(InitialCondition);
   u_gf.ProjectCoefficient(u_0);
   u_gf.Push();

   Vector u;
   u_gf.GetTrueDofs(u);

   // 7. Initialize the conduction operator and the visualization.
   ModelOperator oper(fespace, Mspec, Kspec, alpha, u,
                      engine.As<mfem::occa::Engine>()->GetDevice());

   u_gf.SetFromTrueDofs(u);
   {
      ostringstream mesh_name, sol_name;
      mesh_name << "ex16-mesh." << setfill('0') << setw(6) << myid;
      sol_name << "ex16-init." << setfill('0') << setw(6) << myid;
      ofstream omesh(mesh_name.str().c_str());
      omesh.precision(precision);
      pmesh->Print(omesh);
      ofstream osol(sol_name.str().c_str());
      osol.precision(precision);
      u_gf.Pull(); // Pull from backend before saving
      u_gf.Save(osol);
   }

   socketstream sout;
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      sout.open(vishost, visport);
      sout << "parallel " << num_procs << " " << myid << endl;
      int good = sout.good(), all_good;
      MPI_Allreduce(&good, &all_good, 1, MPI_INT, MPI_MIN, pmesh->GetComm());
      if (!all_good)
      {
         sout.close();
         visualization = false;
         if (mpi_session.Root())
         {
            cout << "Unable to connect to GLVis server at "
                 << vishost << ':' << visport << endl;
            cout << "GLVis visualization disabled.\n";
         }
      }
      else
      {
         sout.precision(precision);
         sout << "solution\n" << *pmesh << u_gf;
         sout << "pause\n";
         sout << flush;
         if (mpi_session.Root())
         {
            cout << "GLVis visualization paused."
                 << " Press space (in the GLVis window) to resume it.\n";
         }
      }
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
         if (mpi_session.Root())
         {
            cout << "step " << ti << ", t = " << t << endl;
         }

         u_gf.SetFromTrueDofs(u);
         if (visualization)
         {
            u_gf.SetFromTrueDofs(u);
            sout << "parallel " << num_procs << " " << myid << "\n";
            sout << "solution\n" << *pmesh << u_gf << flush;
         }
      }
      oper.SetParameters(u);
   }

   // 11. Save the final solution in parallel. This output can be viewed later
   //     using GLVis: "glvis -np <np> -m ex16-mesh -g ex16-final".
   {
      ostringstream sol_name;
      sol_name << "ex16-final." << setfill('0') << setw(6) << myid;
      ofstream osol(sol_name.str().c_str());
      osol.precision(precision);
      u_gf.Pull(); // Pull from backend before saving
      u_gf.Save(osol);
   }

   // 10. Free the used memory.
   delete pmesh;
   delete ode_solver;

   return 0;
}

// #include "backends/hypre/parmatrix.hpp"

ModelOperator::ModelOperator(ParFiniteElementSpace &f,
                             const char *Mspec, const char *Kspec, double alpha_,
                             const Vector &u, ::occa::device device_)
   : TimeDependentOperator(*f.GetTrueVLayout()), fespace(f),
     M(&fespace), K(&fespace), Mh(Mspec), Kh(Kspec), T(NULL),
     M_solver(fespace.GetComm()), T_solver(fespace.GetComm()),
     alpha(alpha_), u_gf(&fespace), uc(&u_gf),
     b(&fespace), b_vec(f.GetTrueVLayout()), z(f.GetTrueVLayout()), device(device_)
{
   const double rel_tol = 1e-8;

   M.AddDomainIntegrator(new MassIntegrator());
   M.Assemble();
   M.FormSystemMatrix(ess_tdof_list, Mh); Moper = Mh.Ptr();
   // mfem::hypre::ParMatrix *mat = static_cast<mfem::hypre::ParMatrix*>(Moper);
   // mat->Print("matrix_M.txt");

   K.AddDomainIntegrator(new DiffusionIntegrator());
   K.Assemble();
   K.FormSystemMatrix(ess_tdof_list, Kh); Koper = Kh.Ptr();
   // mfem::hypre::ParMatrix *matK = static_cast<mfem::hypre::ParMatrix*>(Koper);
   // matK->Print("matrix_K.txt");

   b.AddDomainIntegrator(new DomainLFIntegrator(uc));

   M_solver.iterative_mode = false;
   M_solver.SetRelTol(rel_tol);
   M_solver.SetAbsTol(0.0);
   M_solver.SetMaxIter(200);
   M_solver.SetPrintLevel(1);
   M_solver.SetOperator(*Moper);

   T_solver.iterative_mode = false;
   T_solver.SetRelTol(rel_tol);
   T_solver.SetAbsTol(0.0);
   T_solver.SetMaxIter(200);
   T_solver.SetPrintLevel(0);

   SetParameters(u);
}

void ModelOperator::Mult(const Vector &u, Vector &du_dt) const
{
   // Compute:
   //    du_dt = -M^{-1}*(K(u) + b(u)) b(u) = u^2
   // for du_dt
   Koper->Mult(u, z);
   z.Axpby(-1.0, z, -1.0, b_vec);
   M_solver.Mult(z, du_dt);
}

void ModelOperator::ImplicitSolve(const double dt,
                                  const Vector &u, Vector &du_dt)
{
   // Solve the equation:
   //    du_dt = M^{-1}*[-K(u + dt*du_dt) + b]
   // for du_dt
   // T = M + dt*K
   // T_solver is solving T du_dt = -(Ku + b)
   if (!T)
   {
      T = new TimeDerivativeOperator(Moper, dt, Koper);
      T_solver.SetOperator(*T);
   }
   else
   {
      T->SetTimestep(dt);
   }

   Koper->Mult(u, z);  // z = Ku
   z.Axpby(-1.0, z, -1.0, b_vec); // z = -Ku - b
   T_solver.Mult(z, du_dt);
}

void ModelOperator::SetParameters(const Vector &u)
{
   // Call an occa kernel that computes b = alpha * u^2
   static ::occa::kernelBuilder rhs_builder =
      ::occa::linalg::customLinearMethod(
         "modeloperator_rhs",
         "v0[i] = c0 * v0[i] * v0[i];",
         "defines: {"
         "  CTYPE0: 'double',"
         "  VTYPE0: 'double',"
         "  TILESIZE: '128',"
         "}");

   u_gf.SetFromTrueDofs(u);

   ::occa::kernel kernel = rhs_builder.build(device);
   ::occa::memory u_data = u_gf.Get_PVector()->As<mfem::occa::Vector>().OccaMem();
   kernel((int)u_gf.Size(), alpha, u_data);

   // u changed, so reassemble b
   u_gf.Pull();
   b.Assemble();
   b.ParallelAssemble(b_vec);
   b_vec.Push();
}

double InitialCondition(const Vector &x)
{
   if (x.Norml2() < 0.5)
   {
      return 2.0;
   }
   else
   {
      return 1.0;
   }
}
