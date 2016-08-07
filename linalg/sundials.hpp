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

   void Init(TimeDependentOperator &f_);

   /// Allows changing the operator, starting solution, current time.
   void ReInit(TimeDependentOperator &f_, Vector &y_, double &t_);

   /// Note that this MUST be called before the first call to Step().
   void SetSStolerances(realtype reltol, realtype abstol);

   /// Uses ARKODE to integrate over (t, t + dt).
   /// Calls ARKODE(), which is the main driver of the CVODE package.
   void Step(Vector &x, double &t, double &dt);

   /// Defines a custom Jacobian inversion for non-linear problems.
   void SetLinearSolve(SundialsLinearSolveOperator*);

   /// Chooses a specific Butcher table for a RK method.
   void WrapSetERKTableNum(int table_num);

   /// Specifies to use a fixed time step size instead of performing any form
   /// of temporal adaptivity.
   /// Use  of  this  function  is  not  recommended, since may there is no
   /// assurance  of  the  validity  of  the  computed solutions.
   /// It is primarily provided for code-to-code verification testing purposes.
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
   N_Vector u, u_scale, f_scale;

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
