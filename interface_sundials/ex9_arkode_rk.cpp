//                                MFEM Example 9
//
// Compile with: make ex9_arkode_rk
//
// Sample runs:
//    ex9_arkode_rk -m ../data/periodic-segment.mesh -p 0 -r 2 -dt 0.005
//    ex9_arkode_rk -m ../data/periodic-square.mesh -p 0 -r 2 -dt 0.01 -tf 10
//    ex9_arkode_rk -m ../data/periodic-hexagon.mesh -p 0 -r 2 -dt 0.01 -tf 10
//    ex9_arkode_rk -m ../data/periodic-square.mesh -p 1 -r 2 -dt 0.005 -tf 9
//    ex9_arkode_rk -m ../data/periodic-hexagon.mesh -p 1 -r 2 -dt 0.005 -tf 9
//    ex9_arkode_rk -m ../data/disc-nurbs.mesh -p 2 -r 3 -dt 0.005 -tf 9
//    ex9_arkode_rk -m ../data/periodic-square.mesh -p 3 -r 4 -dt 0.0025 -tf 9 -vs 20
//    ex9_arkode_rk -m ../data/periodic-cube.mesh -p 0 -r 2 -o 2 -dt 0.02 -tf 8
//
// Description:  This example code solves the time-dependent advection equation
//               du/dt = v.grad(u), where v is a given fluid velocity, and
//               u0(x)=u(0,x) is a given initial condition.
//
//               The example demonstrates the use of Discontinuous Galerkin (DG)
//               bilinear forms in MFEM (face integrators), the use of explicit
//               ODE time integrators, the definition of periodic boundary
//               conditions through periodic meshes, as well as the use of GLVis
//               for persistent visualization of a time-evolving solution. The
//               saving of time-dependent data files for external visualization
//               with VisIt (visit.llnl.gov) is also illustrated.

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* Header files with a description of contents used in ARKbanx.c */

#include <arkode/arkode.h>             /* prototypes for ARKODE fcts., consts. */
#include <nvector/nvector_serial.h>  /* serial N_Vector types, fcts., macros */
#include <sundials/sundials_band.h>  /* definitions of type DlsMat and macros */
#include <sundials/sundials_types.h> /* definition of type realtype */
#include <sundials/sundials_math.h>  /* definition of ABS and EXP */

/* Problem Constants */

#define RTOL  RCONST(1.0e-9) /* scalar absolute tolerance */
#define ATOL  RCONST(1.0e-12)    /* scalar absolute tolerance */
#define T0    RCONST(0.0)    /* initial time              */

#define ZERO RCONST(0.0)

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

/* Type : UserData (contains grid constants) */

typedef struct {
  TimeDependentOperator* f_op;
} *UserData;

/* Private function to check function return values */

static int check_flag(void *flagvalue, char *funcname, int opt);

/* Functions Called by the Solver */
static int f(realtype t, N_Vector y, N_Vector ydot, void *user_data);

/** A time-dependent operator for the right-hand side of the ODE. The DG weak
    form of du/dt = v.grad(u) is M du/dt = K u + b, where M and K are the mass
    and advection matrices, and b describes the flow on the boundary. This can
    be written as a general ODE, du/dt = M^{-1} (K u + b), and this class is
    used to evaluate the right-hand side. */
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

   // 1. Parse command-line options.
   problem = 0;
   const char *mesh_file = "../data/periodic-hexagon.mesh";
   int ref_levels = 2;
   int order = 3;
   int ode_solver_type = 4;
   double t_final = 10.0;
   double dt = 0.01;
   bool visualization = true;
   bool visit = false;
   int vis_steps = 5;

   int precision = 8;
   cout.precision(precision);

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&problem, "-p", "--problem",
                  "Problem setup to use. See options in velocity_function().");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly.");
   args.AddOption(&order, "-o", "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                  "ODE solver: 1 - Forward Euler, 2 - RK2 SSP, 3 - RK3 SSP,"
                  " 4 - RK4, 6 - RK6.");
   args.AddOption(&t_final, "-tf", "--t-final",
                  "Final time; start time is 0.");
   args.AddOption(&dt, "-dt", "--time-step",
                  "Time step.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&visit, "-visit", "--visit-datafiles", "-no-visit",
                  "--no-visit-datafiles",
                  "Save data files for VisIt (visit.llnl.gov) visualization.");
   args.AddOption(&vis_steps, "-vs", "--visualization-steps",
                  "Visualize every n-th timestep.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   // 2. Read the mesh from the given mesh file. We can handle geometrically
   //    periodic meshes in this code.
   Mesh *mesh;
   ifstream imesh(mesh_file);
   if (!imesh)
   {
      cerr << "\nCan not open mesh file: " << mesh_file << '\n' << endl;
      return 2;
   }
   mesh = new Mesh(imesh, 1, 1);
   imesh.close();
   int dim = mesh->Dimension();
  
   // 4. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement, where 'ref_levels' is a
   //    command-line parameter. If the mesh is of NURBS type, we convert it to
   //    a (piecewise-polynomial) high-order mesh.
   for (int lev = 0; lev < ref_levels; lev++)
      mesh->UniformRefinement();

   if (mesh->NURBSext)
   {
      int mesh_order = std::max(order, 1);
      FiniteElementCollection *mfec = new H1_FECollection(mesh_order, dim);
      FiniteElementSpace *mfes = new FiniteElementSpace(mesh, mfec, dim);
      mesh->SetNodalFESpace(mfes);
      mesh->GetNodes()->MakeOwner(mfec);
   }

   // 5. Define the discontinuous DG finite element space of the given
   //    polynomial order on the refined mesh.
   DG_FECollection fec(order, dim);
   FiniteElementSpace fes(mesh, &fec);

   cout << "Number of unknowns: " << fes.GetVSize() << endl;

   // 6. Set up and assemble the bilinear and linear forms corresponding to the
   //    DG discretization. The DGTraceIntegrator involves integrals over mesh
   //    interior faces.
   VectorFunctionCoefficient velocity(dim, velocity_function);
   FunctionCoefficient inflow(inflow_function);
   FunctionCoefficient u0(u0_function);

   BilinearForm m(&fes);
   m.AddDomainIntegrator(new MassIntegrator);
   BilinearForm k(&fes);
   k.AddDomainIntegrator(new ConvectionIntegrator(velocity, -1.0));
   k.AddInteriorFaceIntegrator(
      new TransposeIntegrator(new DGTraceIntegrator(velocity, 1.0, -0.5)));
   k.AddBdrFaceIntegrator(
      new TransposeIntegrator(new DGTraceIntegrator(velocity, 1.0, -0.5)));

   LinearForm b(&fes);
   b.AddBdrFaceIntegrator(
      new BoundaryFlowIntegrator(inflow, velocity, -1.0, -0.5));

   m.Assemble();
   m.Finalize();
   int skip_zeros = 0;
   k.Assemble(skip_zeros);
   k.Finalize(skip_zeros);
   b.Assemble();

   // 7. Define the initial conditions, save the corresponding grid function to
   //    a file and (optionally) save data in the VisIt format and initialize
   //    GLVis visualization.
   GridFunction u(&fes);
   u.ProjectCoefficient(u0);

   {
      ofstream omesh("ex9.mesh");
      omesh.precision(precision);
      mesh->Print(omesh);
      ofstream osol("ex9-init.gf");
      osol.precision(precision);
      u.Save(osol);
   }

   VisItDataCollection visit_dc("Example9", mesh);
   visit_dc.RegisterField("solution", &u);
   if (visit)
   {
      visit_dc.SetCycle(0);
      visit_dc.SetTime(0.0);
      visit_dc.Save();
   }

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

   // 8. Define the time-dependent evolution operator describing the ODE
   //    right-hand side, and perform time-integration (looping over the time
   //    iterations, ti, with a time-step dt).
   FE_Evolution adv(m.SpMat(), k.SpMat(), b);
   
   // 9. Initialize the ODE solver
   //goes into init
   realtype reltol, abstol, tin, tout;
   long int yin_length;
   yin_length=u.Size();
   long int n = adv.Width();
   cout<<yin_length<<n<<endl;
   //intial time
   realtype t = 0.0;
   realtype *yin;
   N_Vector y;
   yin= u.GetData();
   UserData data;
   void *arkode_mem;
   int iout, flag;
   realtype tF=t_final;

   y = NULL;
   data = NULL;
   arkode_mem = NULL;
   /* Create a serial vector */

   y = N_VMake_Serial(n,yin);   /* Allocate y vector */
   if(check_flag((void*)y, "N_VNew_Serial", 0)) return(1);

   reltol = RTOL;   /* Set the tolerances */
   abstol = ATOL;

   data = (UserData) malloc(sizeof *data);   /* Allocate data memory */
   if(check_flag((void *)data, "malloc", 2)) return(1);
   
   TimeDependentOperator *tmp=&adv;
   data->f_op=tmp;

   /* Call ARKodeCreate to create the solver memory */
   arkode_mem=ARKodeCreate();
   if(check_flag((void *)arkode_mem, "ARKodeCreate", 0)) return(1);

   /* Call ARKodeInit to initialize the integrator memory and specify the
    * user's right hand side function in u'=f(t,u), the inital time t, and
    * the initial dependent variable vector y. */
   flag = ARKodeInit(arkode_mem, f, NULL, t, y);
   if(check_flag(&flag, "ARKodeInit", 1)) return(1);


   /* Call ARKodeSetERKTableNum to compare directly with method integration
    * in MFEM example 9. */
   flag = ARKodeSetERKTableNum(arkode_mem, 3);
   if(check_flag(&flag, "ARKodeSetERKTableNum", 1)) return(1);

   /* Call ARKodeSStolerances to specify the scalar relative tolerance
    * and scalar absolute tolerance */
   flag = ARKodeSStolerances(arkode_mem, reltol, abstol);
   if (check_flag(&flag, "ARKodeSStolerances", 1)) return(1);

   /* Set the pointer to user-defined data */
   flag = ARKodeSetUserData(arkode_mem, data);
   if(check_flag(&flag, "ARKodeSetUserData", 1)) return(1);

   /* Set the initial step size */
   flag = ARKodeSetFixedStep(arkode_mem, dt);
   if(check_flag(&flag, "ARKodeSetInitStep", 1)) return(1);

   /* Set the minimum step size*/
//   flag = ARKodeSetMinStep(arkode_mem, dt);
//   if(check_flag(&flag, "ARKodeSetMinStep", 1)) return(1);

   /* Set the maximum step size */
   flag = ARKodeSetMaxStep(arkode_mem, dt);
   if(check_flag(&flag, "ARKodeSetMaxStep", 1)) return(1);

/*
   flag = ARKodeSetStopTime(arkode_mem, tF+1e-9);
   if(check_flag(&flag, "ARKodeSetStopTime", 1)) return(1);
*/
   // 10. Loop over integration
   cout<<"tF before integration: "<<tF<<"\ntout, t before integration"<<tout<<"\n"<<t<<endl;
   for(iout=1, tout=dt; tout-1e-9<=tF; iout++, tout += dt) {
      flag = ARKode(arkode_mem, tout, y, &t, ARK_ONE_STEP);
//      flag = ARKode(arkode_mem, tout, y, &t, ARK_NORMAL);
      if(check_flag(&flag, "ARKode", 1)) break;

      u.SetData(NV_DATA_S(y));

      int ti=iout;
      if (ti % vis_steps == 0)
      {
         cout << "time step: " << ti << ", time: " << t << endl;

         if (visualization)
            sout << "solution\n" << *mesh << u << flush;

         if (visit)
         {
            visit_dc.SetCycle(ti);
            visit_dc.SetTime(t);
            visit_dc.Save();
         } //end if visit
      } //end if visualization step
  } //end for over integrating
  
  cout<<"tF after integration: "<<tF<<"\ntout, t after integration: "<<tout<<"\n"<<t<<endl;

/* Consider changing structure of for loop back to this:
   for (int ti = 0; true; )
   {
      if (t >= t_final - dt/2)
         break;

      ode_solver->Step(u, t, dt);
      ti++;


      if (ti % vis_steps == 0)
      {
         cout << "time step: " << ti << ", time: " << t << endl;

         if (visualization)
            sout << "solution\n" << *mesh << u << flush;

         if (visit)
         {
            visit_dc.SetCycle(ti);
            visit_dc.SetTime(t);
            visit_dc.Save();
         }
      }
      
   }
*/
   // 11. Save the final solution. This output can be viewed later using GLVis:
   //    "glvis -m ex9.mesh -g ex9_arkode_rk-final.gf".
   {
      ofstream osol("ex9_arkode_rk-final.gf");
      osol.precision(precision);
      u.Save(osol);
   }
   
   // 12. Print solver statistics.
   long int nst, nst_a, nfe, nfi, nsetups, nje, nfeLS, nni, ncfn, netf;
   /* Get/print some final statistics on how the solve progressed */
   flag = ARKodeGetNumSteps(arkode_mem, &nst);
   check_flag(&flag, "ARKodeGetNumSteps", 1);
   flag = ARKodeGetNumStepAttempts(arkode_mem, &nst_a);
   check_flag(&flag, "ARKodeGetNumStepAttempts", 1);
   flag = ARKodeGetNumRhsEvals(arkode_mem, &nfe, &nfi);
   check_flag(&flag, "ARKodeGetNumRhsEvals", 1);
   /*
   flag = ARKodeGetNumLinSolvSetups(arkode_mem, &nsetups);
   check_flag(&flag, "ARKodeGetNumLinSolvSetups", 1);
   */
   flag = ARKodeGetNumErrTestFails(arkode_mem, &netf);
   check_flag(&flag, "ARKodeGetNumErrTestFails", 1);
   /*
   flag = ARKodeGetNumNonlinSolvIters(arkode_mem, &nni);
   check_flag(&flag, "ARKodeGetNumNonlinSolvIters", 1);
   flag = ARKodeGetNumNonlinSolvConvFails(arkode_mem, &ncfn);
   check_flag(&flag, "ARKodeGetNumNonlinSolvConvFails", 1);
   flag = ARKDlsGetNumJacEvals(arkode_mem, &nje);
   check_flag(&flag, "ARKDlsGetNumJacEvals", 1);
   flag = ARKDlsGetNumRhsEvals(arkode_mem, &nfeLS);
   check_flag(&flag, "ARKDlsGetNumRhsEvals", 1);
   */
   
   printf("\nFinal Solver Statistics:\n");
   printf("    Internal solver steps = %li (attempted = %li)\n", nst, nst_a);
   printf("    Total RHS evals:   Fe = %li,   Fi = %li\n", nfe, nfi);
   printf("    Total number of error test failures = %li\n\n", netf);

   // 13. Free the used memory.
   /* Clean up and return with successful completion */
   N_VDestroy_Serial(y);       /* Free y vector */
   ARKodeFree(&arkode_mem);   /* Free integrator memory */   
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
}

void FE_Evolution::Mult(const Vector &x, Vector &y) const
{
   // y = M^{-1} (K x + b)
   K.Mult(x, z);
   z += b;
   M_solver.Mult(z, y);
}

static int f(realtype t, N_Vector y, N_Vector ydot,void *user_data)
{
  UserData udata;
  realtype *ydata, *ydotdata;
  long int ylen, ydotlen;
 
  udata = (UserData) user_data;
  //ydata is now a pointer to the realtype data array in y
  ydata = NV_DATA_S(y);
  ylen = NV_LENGTH_S(y);
  
  // probably unnecessary, since overwriting ydot as output
  //ydotdata is now a pointer to the realtype data array in ydot
  ydotdata = NV_DATA_S(ydot);
  ydotlen = NV_LENGTH_S(ydot);
  
  //f_op is now a pointer of abstract base class type TimeDependentOperator. It points to the TimeDependentOperator in the user_data struct
  TimeDependentOperator* f_op = udata->f_op;
  
  // Eventually add these MFEM Vectors to UserData struct
  // Creates mfem vectors with pointers to the data array in y and in ydot respectively
  // Have not explicitly set as owndata, so allocated size is -size
  Vector mfem_vector_y((double*) ydata, ylen);
  Vector mfem_vector_ydot((double*) ydotdata, ydotlen);
  f_op->SetTime(t);
  f_op->Mult(mfem_vector_y,mfem_vector_ydot);

  return(0);
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
      case 3:
      {
         double rx = 0.45, ry = 0.25, cx = 0., cy = -0.2, w = 10.;
         if (dim == 3)
         {
            const double s = (1. + 0.25*cos(2*M_PI*x(2)));
            rx *= s;
            ry *= s;
         }
         return ( erfc(w*(x(0)-cx-rx))*erfc(-w*(x(0)-cx+rx)) *
                  erfc(w*(x(1)-cy-ry))*erfc(-w*(x(1)-cy+ry)) )/16;
      }
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

// Inflow boundary condition (zero for the problems considered in this example)
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

/* Check function return value...
     opt == 0 means SUNDIALS function allocates memory so check if
              returned NULL pointer
     opt == 1 means SUNDIALS function returns a flag so check if
              flag >= 0
     opt == 2 means function allocates memory so check if returned
              NULL pointer */

static int check_flag(void *flagvalue, char *funcname, int opt)
{
  int *errflag;

  /* Check if SUNDIALS function returned NULL pointer - no memory allocated */

  if (opt == 0 && flagvalue == NULL) {
    fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed - returned NULL pointer\n\n",
            funcname);
    return(1); }

  /* Check if flag < 0 */

  else if (opt == 1) {
    errflag = (int *) flagvalue;
    if (*errflag < 0) {
      fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed with flag = %d\n\n",
              funcname, *errflag);
      return(1); }}

  /* Check if function returned NULL pointer - no memory allocated */

  else if (opt == 2 && flagvalue == NULL) {
    fprintf(stderr, "\nMEMORY_ERROR: %s() failed - returned NULL pointer\n\n",
            funcname);
    return(1); }

  return(0);
}
