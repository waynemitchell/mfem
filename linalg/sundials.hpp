#ifndef MFEM_SUNDIALS
#define MFEM_SUNDIALS

#include "../config/config.hpp"

#ifdef MFEM_USE_SUNDIALS

#ifdef MFEM_USE_MPI
#include <mpi.h>
#endif

#include "ode.hpp"
#include "operator.hpp"

#include <cvode/cvode.h>
#include <arkode/arkode.h>

namespace mfem
{
class SundialsLinearSolveOperator;

/// Wraps the CVODE library.
/// http://computation.llnl.gov/sites/default/files/public/cv_guide.pdf
class CVODESolver: public ODESolver
{
#ifdef MFEM_USE_MPI
private:
   MPI_Comm comm;
#endif

protected:
   N_Vector y;
   void *ode_mem;
   int solver_iteration_type;

   void (*connectNV)(Vector &, N_Vector &);

public:

   /// CVODE needs the initial condition (first argument).
   /// parallel specifies whether the calling code is parallel or not.
   /// lmm specifies the linear multistep method, the options are
   /// CV_ADAMS (explicit problems) or CV_BDF (implicit problems).
   /// iter specifies type of solver iteration, the options are
   /// CV_FUNCTIONAL (linear problems) or CV_NEWTON (nonlinear problems).
   CVODESolver(Vector &y_, bool parallel,
               int lmm = CV_ADAMS, int iter = CV_FUNCTIONAL);

   void Init(TimeDependentOperator &f_);

   /// Allows changing the operator, starting solution, current time.
   void ReInit(TimeDependentOperator &f_, Vector &y_, double &t_);

   /// Note that this MUST be called before the first call to Step().
   void SetSStolerances(realtype reltol, realtype abstol);

   /// Uses CVODE to integrate over (t, t + dt).
   /// Calls CVODE(), which is the main driver of the CVODE package.
   void Step(Vector &x, double &t, double &dt);

   /// Defines a custom Jacobian inversion for non-linear problems.
   void SetLinearSolve(SundialsLinearSolveOperator *op);

   /// Destroys the associated CVODE memory.
   ~CVODESolver();
};

/// Wraps the ARKODE library.
/// http://computation.llnl.gov/sites/default/files/public/ark_guide.pdf
class ARKODESolver: public ODESolver
{
#ifdef MFEM_USE_MPI
private:
   MPI_Comm comm;
#endif

protected:
   N_Vector y;
   void* ode_mem;
   bool use_explicit;

   void (*connectNV)(Vector &, N_Vector &);

public:

   /// ARKODE needs the initial condition (first argument).
   /// parallel specifies whether the calling code is parallel or not.
   /// exlicit_ specifies whether the time integration is explicit.
   ARKODESolver(Vector &y_, bool parallel, bool explicit_ = true);

   void Init(TimeDependentOperator &_f);

   /// Allows changing the operator, starting solution, current time.
   void ReInit(TimeDependentOperator &_f, Vector &y_, double &_t);

   /// Note that this MUST be called before the first call to Step().
   void SetSStolerances(realtype reltol, realtype abstol);

   /** \brief
    * ARKode supports two modes as specified by itask: ARK_NORMAL and
    * ARK_ONE_STEP. In the ARK_NORMAL mode, the solver steps until
    * it reaches or passes tout and then interpolates to obtain
    * y(tout). In the ARK_ONE_STEP mode, it takes one internal step
    * and returns. The behavior of both modes can be over-rided
    * through user-specification of ark_tstop (through the
    * ARKodeSetStopTime function), in which case if a solver step
    * would pass tstop, the step is shortened so that it stops at
    * exactly the specified stop time, and hence interpolation of
    * y(tout) is not required.
    */
   /// Uses ARKODE to integrate over (t, t + dt).
   /// Calls ARKODE(), which is the main driver of the CVODE package.
   /// TODO: keep the interpolation comment.
   void Step(Vector &x, double &t, double &dt);

   /// Defines a custom Jacobian inversion for non-linear problems.
   void SetLinearSolve(SundialsLinearSolveOperator*);

   /// Chooses a specific Butcher table for a RK method.
   void WrapSetERKTableNum(int table_num);

   /** Specifies to use a fixed time step size instead of performing
       any form of temporal adaptivity.  ARKode will use this step size
       for all steps (unless tstop is set, in which case it may need to
       modify that last step approaching tstop.  If any (non)linear
       solver failure occurs, ARKode will immediately return with an
       error message since the time step size cannot be modified.
       Any nonzero argument will result in the use of that fixed step
       size; an argument of 0 will re-enable temporal adaptivity. */
   // TODO: interface for tstop.
   void WrapSetFixedStep(double dt);

   /// Destroys the associated ARKODE memory.
   ~ARKODESolver();
};

/// Interface for custom Jacobian inversion in Sundials.
/// The Jacobian problem has the form I - dt inv(M) J(y) = b.
class SundialsLinearSolveOperator : public Operator
{
public:
   SundialsLinearSolveOperator(int s) : Operator(s)
   { }
   virtual void SolveJacobian(Vector* b, Vector* y, double dt) = 0;
};

/// Wraps the KINSOL library.
/// http://computation.llnl.gov/sites/default/files/public/kin_guide.pdf
class KinSolWrapper
{
private:
   void *kin_mem;
   N_Vector u;
   N_Vector u_scale;
   N_Vector f_scale;

   void (*connectNV)(Vector &, N_Vector &);

public:
   KinSolWrapper(Operator &oper, Vector &mfem_u,
                 bool parallel, bool use_oper_grad = false);
   ~KinSolWrapper();

   void SetPrintLevel(int level);
   void SetFuncNormTol(double tol);
   void SetScaledStepTol(double tol);

   void Solve(Vector &mfem_u,
              Vector &mfem_u_scale, Vector &mfem_f_scale);
};

}  // namespace mfem

#endif // MFEM_USE_SUNDIALS

#endif // MFEM_SUNDIALS
