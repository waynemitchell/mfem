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
#include "backends/hypre/solvers.hpp"
#include "cuda_profiler_api.h"

#if !defined(MFEM_USE_OCCA) || !defined(MFEM_USE_BACKENDS)
#error Requires OCCA backend support
#endif

using namespace std;
using namespace mfem;


// The (HO) operator used in the implicit solver inside ModelOperator
class TimeDerivativeOperator : public Operator
{
   Operator *Moper;
   Operator *Koper;
   mutable Vector Kdu;
   double dt;

public:
   TimeDerivativeOperator(Operator *Moper_,
                          const double dt_,
                          Operator *Koper_)
      : Operator(*Koper_->InLayout(), *Moper_->OutLayout()),
        Moper(Moper_),
        Koper(Koper_),
        Kdu(Moper_->OutLayout()),
        dt(dt_) { }

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

   double TimeStep() const { return dt; }

   virtual ~TimeDerivativeOperator() {}
};

/* The (LOR) AMG operator used in the SetPreconditioner call for the
 * T_solver inside ModelOperator */
class LORAMGSolver : public Solver
{
   const int basis;
   ParMesh mesh;
   FiniteElementCollection *fec;
   ParFiniteElementSpace fespace;
   ParBilinearForm Mform, Kform;

   OperatorHandle Mh, Kh;

   double dt;

   mfem::hypre::ParMatrix *M, *K, *A;
   mfem::hypre::AMGSolver *amg_solver;

public:
   LORAMGSolver(ParFiniteElementSpace &f, const int order, const int fbasis, const double dt_)
      : Solver(f.GetTrueVLayout(), f.GetTrueVLayout()),
        basis(fbasis == BasisType::Positive ? BasisType::ClosedUniform : fbasis),
        mesh(f.GetParMesh(), order, basis),
        fec(new H1_FECollection(1, mesh.Dimension())),
        fespace(&mesh, fec),
        Mform(&fespace), Kform(&fespace),
        Mh("representation: 'full'"),
        Kh("representation: 'full'"),
        dt(dt_),
        M(NULL), K(NULL), A(NULL),
        amg_solver(NULL)
   {
      mfem::Array<int> ess_tdof_list; // empty for now

      // Build LOR M
      Mform.AddDomainIntegrator(new MassIntegrator());
      Mform.Assemble();
      Mform.FormSystemMatrix(ess_tdof_list, Mh);
      Mh.Get<mfem::hypre::ParMatrix>(M);

      // Build LOR K
      Kform.AddDomainIntegrator(new DiffusionIntegrator());
      Kform.Assemble();
      Kform.FormSystemMatrix(ess_tdof_list, Kh);
      Kh.Get<mfem::hypre::ParMatrix>(K);

      // Clone M to a new matrix
      A = new mfem::hypre::ParMatrix(*K);

      // Do an axpy operation that assumes the same sparsity
      A->HypreAxpy(1.0, *M, dt, *K);

      // Build an AMG based on A -- calls AMG setup as part of constructor
      amg_solver = new mfem::hypre::AMGSolver(A);
   }

   virtual ~LORAMGSolver() {
      delete amg_solver;
      delete A;
      // M and K are destroyed as part of the handles
   }

   virtual void SetOperator(const Operator &op)
   {
      mfem_error("Not supported");
   }

   void Reassemble(const int dt_)
   {
      dt = dt_;

      int rank;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      if (!rank) { std::cout << "Reassembling preconditioner matrix" << std::endl; }

      A->HypreAxpy(1.0, *M, dt, *K);

      // The AMGSolver does not need to be re-setup
   }

   virtual void Mult(const Vector &x, Vector &y) const
   {
      amg_solver->Mult(x, y);
   }

   double SetupTime() const { return amg_solver->SetupTime(); }
   double SolveTime() const { return amg_solver->SolveTime(); }
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
   int basis;
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

   LORAMGSolver *lor_amg_solver;

   int precond;

   double solve_time;

public:
   ModelOperator(ParFiniteElementSpace &f,
                 const int basis_,
                 const char *Mspec,
                 const char *Kspec,
                 const bool precond_,
                 const double alpha_,
                 const Vector &u,
                 ::occa::device device_);

   virtual void Mult(const Vector &u, Vector &du_dt) const;
   /** Solve the Backward-Euler equation: k = f(u + dt*k, t), for the unknown k.
       This is the only requirement for high-order SDIRK implicit integration.*/
   virtual void ImplicitSolve(const double dt, const Vector &u, Vector &k);

   /// Update the diffusion BilinearForm K using the given true-dof vector `u`.
   void SetParameters(const Vector &u);

   /** Solve the system (M + dt K) y = M b. The result y replaces the input b.
       This method is used by the implicit SUNDIALS solvers. */
   void SundialsSolve(const double dt, const Vector &b, Vector &x, double tol);

   virtual ~ModelOperator() { delete T; delete lor_amg_solver; }

   void PrintStats() const {
      std::cout << "==== Stats ====" << std::endl;
      std::cout << "Solver time = " << solve_time << std::endl;
      if (lor_amg_solver) {
         std::cout << "LOR AMG setup time = " << lor_amg_solver->SetupTime() << std::endl;
         std::cout << "LOR AMG solve time = " << lor_amg_solver->SolveTime() << std::endl;
      }
   }
};

/// Custom Jacobian system solver for the SUNDIALS time integrators.
/** For the ODE system represented by ModelOperator

    M du/dt = -K - b(u),

    this class facilitates the solution of linear systems of the form

    (M + γK) y = M b(u),

    for given b, u (not used), and γ = GetTimeStep(). */
class SundialsJacSolver : public SundialsLinearSolver
{
private:
   ModelOperator *oper;

public:
   SundialsJacSolver(ModelOperator *oper_) : oper(oper_) { }

   int InitializeSystem(void *sundials_mem);
   int SetupSystem(void *sundials_mem);
   int SolveSystem(void *sundials_mem, Vector &x, Vector &b, double tol);
   int FreeSystem(void *sundials_mem);
};

enum MethodType {
   METHOD_CPU_PA=1,
   METHOD_CPU_FA=2,
   METHOD_GPU_PA=11,
   METHOD_GPU_FA=12
};

enum PreconditionerType {
   PRECOND_NONE,
   PRECOND_AMG
};

double InitialCondition(const Vector &x);

int main(int argc, char *argv[])
{
   MPI_Session mpi_session(argc, argv);
   const int num_procs = mpi_session.WorldSize();
   const int myid = mpi_session.WorldRank();

   // 1. Parse command-line options.
   const char *mesh_file = "../data/star.mesh"; // star.mesh or fichera.mesh are good options
   const char *basis_type = "G"; // Gauss-Lobatto
   int ser_ref_levels = 1;
   int par_ref_levels = 1;
   int order = 2;
   int ode_solver_type = 1;
   double t_final = 0.5;
   double dt = 1.0e-2;
   double alpha = 1.0e-2;
   bool visualization = false;
   int vis_steps = 5;
   int method = -1;
   int precond = PRECOND_NONE;
   const char *Mspec = "representation: 'partial'";
   const char *Kspec = "representation: 'partial'";
   const char *occa_spec = "mode: 'Serial', integrator: 'acrotensor'";

   int precision = 8;
   cout.precision(precision);

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&basis_type, "-b", "--basis-type",
                  "Basis: G - Gauss-Lobatto, P - Positive, U - Uniform");
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
   args.AddOption(&method, "-sm", "--solver-method", "Solver method (see example for enum)");
   args.AddOption(&precond, "-p", "--preconditioner", "Preconditioner (see example for enum)");

   args.Parse();
   if (method >= 0) {
      switch (method)
      {
      case METHOD_CPU_PA:
         occa_spec = "mode: 'Serial', integrator: 'acrotensor'";
         Mspec = Kspec = "representation: 'partial'";
         break;
      case METHOD_CPU_FA:
         occa_spec = "mode: 'Serial', integrator: 'acrotensor'";
         Mspec = Kspec = "representation: 'full'";
         break;
      case METHOD_GPU_PA:
         occa_spec = "mode: 'CUDA', device_id: 0, integrator: 'acrotensor', memory:{unified:true}";
         Mspec = Kspec = "representation: 'partial'";
         break;
      case METHOD_GPU_FA:
         occa_spec = "mode: 'CUDA', device_id: 0, integrator: 'acrotensor', memory:{unified:true}";
         Mspec = Kspec = "representation: 'full'";
         break;
      default:
         mfem_error("Not supported");
         break;
      };
   }
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

   // See class BasisType in fem/fe_coll.hpp for available basis types
   int basis = BasisType::GetType(basis_type[0]);
   if (mpi_session.Root())
   {
      cout << "Using " << BasisType::Name(basis) << " basis ..." << endl;
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
   H1_FECollection fe_coll(order, dim, basis);
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
   ModelOperator oper(fespace, basis, Mspec, Kspec, precond, alpha, u,
                      engine.As<mfem::occa::Engine>()->GetDevice());

   // 3. Define the ODE solver used for time integration. Several implicit
   //    singly diagonal implicit Runge-Kutta (SDIRK) methods, as well as
   //    explicit Runge-Kutta methods are available.
   ODESolver *ode_solver;
   CVODESolver *cvode = NULL;
   ARKODESolver *arkode = NULL;
   SundialsJacSolver sun_solver(&oper); // Used by the implicit SUNDIALS ode solvers.
   // Relative and absolute tolerances for CVODE and ARKODE.
   const double reltol = 1e-4, abstol = 1e-4;

   switch (ode_solver_type)
   {
   case 1:  ode_solver = new BackwardEulerSolver; break;
   case 2:
      cvode = new CVODESolver(MPI_COMM_WORLD, CV_ADAMS, CV_FUNCTIONAL,
                              fespace.GetTrueDofOffsets(), fespace.GlobalTrueVSize());
      cvode->SetSStolerances(reltol, abstol);
      cvode->SetMaxStep(dt);
      ode_solver = cvode; break;
   case 3:
      cvode = new CVODESolver(MPI_COMM_WORLD, CV_BDF, CV_NEWTON,
                              fespace.GetTrueDofOffsets(), fespace.GlobalTrueVSize());
      cvode->SetNewLinearSolver(sun_solver, oper.InLayout());
      cvode->SetSStolerances(reltol, abstol);
      cvode->SetMaxStep(dt);
      ode_solver = cvode; break;
      // TODO: Add the same lazy initialization and constructor arguments to ARKODESolver to enable these.
      // case 4:
      //       arkode = new ARKODESolver(MPI_COMM_WORLD, ARKODESolver::EXPLICIT);
      //       arkode->SetSStolerances(reltol, abstol);
      //       arkode->SetMaxStep(dt);
      //       if (ode_solver_type == 3) { arkode->SetERKTableNum(FEHLBERG_13_7_8); }
      //       ode_solver = arkode; break;
      // case 5:
      //       arkode = new ARKODESolver(MPI_COMM_WORLD, ARKODESolver::IMPLICIT);
      //       arkode->SetLinearSolver(sun_solver);
      //       arkode->SetSStolerances(reltol, abstol);
      //       arkode->SetMaxStep(dt);
      //       ode_solver = arkode; break;
   default:
      cout << "Unknown ODE solver type: " << ode_solver_type << '\n';
      delete mesh;
      return 3;
   }

   // Since we want to update the diffusion coefficient after every time step,
   // we need to use the "one-step" mode of the SUNDIALS solvers.
   if (cvode) { cvode->SetStepMode(CV_ONE_STEP); }
   if (arkode) { arkode->SetStepMode(ARK_ONE_STEP); }

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

         if (mpi_session.Root())
         {
            if (cvode) { cvode->PrintInfo(); }
            if (arkode) { arkode->PrintInfo(); }
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

   if (mpi_session.Root()) { oper.PrintStats(); }
   if (cvode && mpi_session.Root()) { cvode->PrintTiming(); }

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

ModelOperator::ModelOperator(ParFiniteElementSpace &f, const int basis_,
                             const char *Mspec, const char *Kspec, const bool precond_,
                             const double alpha_, const Vector &u, ::occa::device device_)
   : TimeDependentOperator(*f.GetTrueVLayout()), fespace(f), basis(basis_),
     M(&fespace), K(&fespace), Mh(Mspec), Kh(Kspec), T(NULL),
     M_solver(fespace.GetComm()), T_solver(fespace.GetComm()),
     alpha(alpha_), u_gf(&fespace), uc(&u_gf),
     b(&fespace), b_vec(f.GetTrueVLayout()), z(f.GetTrueVLayout()),
     device(device_), lor_amg_solver(NULL), precond(precond_)
{
   const double rel_tol = 1e-4;

   // TODO: Switch to FormLinearSystem as this will not work if there are Dirichlet BCs
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
   M_solver.SetPrintLevel(2);
   M_solver.SetOperator(*Moper);

   T_solver.iterative_mode = false;
   T_solver.SetRelTol(rel_tol);
   T_solver.SetAbsTol(0.0);
   T_solver.SetMaxIter(200);
   T_solver.SetPrintLevel(2);

   SetParameters(u);
}

void ModelOperator::Mult(const Vector &u, Vector &du_dt) const
{
   // Compute:
   //    du_dt = -M^{-1}*(K(u) + b(u))
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

      if (precond == PRECOND_AMG)
      {
         lor_amg_solver = new LORAMGSolver(fespace, fespace.GetFE(0)->GetOrder(), basis, dt);
         T_solver.SetPreconditioner(*lor_amg_solver);
      }
   }
   else
   {
      T->SetTimestep(dt);
      if (precond == PRECOND_AMG)
      {
         const double last_dt = T->TimeStep();
         const double factor = 5;
         if ((dt < (1.0/factor) * last_dt) || (dt > factor * last_dt))
         {
            lor_amg_solver->Reassemble(dt);
         }
         // preconditioner object set by SetPreconditioner() call above is still valid
      }
   }

   Koper->Mult(u, z);  // z = Ku
   z.Axpby(-1.0, z, -1.0, b_vec); // z = -Ku - b
   // z.Axpby(-1.0, z, 0.0, z); // z = -Ku
   tic();
   T_solver.Mult(z, du_dt);
   solve_time += toc();
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
   u_gf.Pull(); // after pulling u_gf

   ::occa::kernel kernel = rhs_builder.build(device);
   ::occa::memory u_data = u_gf.Get_PVector()->As<mfem::occa::Vector>().OccaMem();
   kernel((int)u_gf.Size(), alpha, u_data);

   // u changed, so reassemble b
   b.Assemble();
   b.Push();
   b.ParallelAssemble(b_vec);
}

void ModelOperator::SundialsSolve(const double dt, const Vector &b, Vector &x, double tol)
{
   if (!T)
   {
      T = new TimeDerivativeOperator(Moper, dt, Koper);
      T_solver.SetOperator(*T);

      if (precond == PRECOND_AMG)
      {
         lor_amg_solver = new LORAMGSolver(fespace, fespace.GetFE(0)->GetOrder(), basis, dt);
         T_solver.SetPreconditioner(*lor_amg_solver);
      }
   }
   else
   {
      T->SetTimestep(dt);
      if (precond == PRECOND_AMG)
      {
         const double last_dt = T->TimeStep();
         const double factor = 5;
         if ((dt < (1.0/factor) * last_dt) || (dt > factor * last_dt))
         {
            lor_amg_solver->Reassemble(dt);
         }
         // preconditioner object set by SetPreconditioner() call above is still valid
      }
   }

   Moper->Mult(b, z);  // z = Mb
   tic();
   // T_solver.SetRelTol(sqrt(tol));
   T_solver.Mult(z, x);
   solve_time += toc();
}

int SundialsJacSolver::InitializeSystem(void *sundials_mem)
{
   return 0;
}

int SundialsJacSolver::SetupSystem(void *sundials_mem)
{
   return 0;
}

int SundialsJacSolver::SolveSystem(void *sundials_mem, Vector &x, Vector &b, double tol)
{
   oper->SundialsSolve(GetTimeStep(sundials_mem), b, x, tol);
   return 0;
}

int SundialsJacSolver::FreeSystem(void *sundials_mem)
{
   return 0;
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
