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

/// Wraps the CVode library of linear multistep methods
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

   /** This constructor wraps the CVodeCreate function and sets the
    *  initial condition. By default, this uses Adams methods with
    *  functional iterations (no Newton solves).
    *
    *  CVodeCreate creates an internal memory block for a problem to
    *  be solved by CVODE.
    */
   CVODESolver(Vector &y_, bool parallel,
               int lmm = CV_ADAMS, int iter = CV_FUNCTIONAL);

   /** \brief
    * The Init function is used in the initial construction and initialization
    * of the CVODESolver object. It wraps CVodeInit to pass the initial
    * condition to the ode_mem struct.
    *
    * CVodeInit
    *
    * CVodeInit allocates and initializes memory for a problem. All
    * problem inputs are checked for errors. If any error occurs during
    * initialization, it is reported to the file whose file pointer is
    * errfp and an error flag is returned. Otherwise, it returns CV_SUCCESS.
    *
    */
   void Init(TimeDependentOperator &_f);

   /** \brief
    * The ReInit function is used to re-initialize initial condition and the
    * CVODESolver object. It wraps CVodeReInit to pass the initial condtion
    * to the ode_mem struct.
    *
    * CVodeReInit
    *
    * CVodeReInit re-initializes CVODE's memory for a problem, assuming
    * it has already been allocated in a prior CVodeInit call.
    * All problem specification inputs are checked for errors.
    * If any error occurs during initialization, it is reported to the
    * file whose file pointer is errfp.
    * The return value is CV_SUCCESS = 0 if no errors occurred, or
    * a negative value otherwise.
    */
   void ReInit(TimeDependentOperator &_f, Vector &y_, double &_t);

   /** \brief SetSStolerances wraps the CVode function CVodeSStolerances which
    * specifies scalar relative and absolute tolerances. These tolerances must
    * be set before the first integration step is called.
    *
    * CVodeSStolerances
    *
    * These functions specify the integration tolerances. One of them
    * MUST be called before the first call to CVode.
    *
    * CVodeSStolerances specifies scalar relative and absolute tolerances.
    */
   void SetSStolerances(realtype reltol, realtype abstol);

   /** \brief Step transfers vector pointers using TransferNVector and calls
    * CVode, which integrates over a user-defined time interval.
    *
    * CVode
    *
    * This routine is the main driver of the CVODE package.
    *
    * It integrates over a time interval defined by the user, by calling
    * cvStep to do internal time steps.
    *
    * The first time that CVode is called for a successfully initialized
    * problem, it computes a tentative initial step size h.
    *
    * CVode supports two modes, specified by itask: CV_NORMAL, CV_ONE_STEP.
    * In the CV_NORMAL mode, the solver steps until it reaches or passes tout
    * and then interpolates to obtain y(tout).
    * In the CV_ONE_STEP mode, it takes one internal step and returns.
    */
   void Step(Vector &x, double &t, double &dt);

   /** Defines a custom Jacobian inversion.
       First calls the existing cv_lfree routine if it is not NULL.
       It then sets the cv_linit, cv_lsetup, cv_lsolve, cv_lfree fields in
       the CVODE's memory structure. */
   void SetLinearSolve(SundialsLinearSolveOperator*);

   /** \brief Destroys associated memory. Calls CVodeFree and N_VDestroy.
    *
    * CVodeFree
    *
    * This routine frees the problem memory allocated by CVodeInit.
    * Such memory includes all the vectors allocated by cvAllocVectors,
    * and the memory lmem for the linear solver (deallocated by a call
    * to lfree).
    */
   ~CVODESolver();
};

/// Wraps the ARKode library of explicit, implicit and additive RK methods.
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

   /** \brief This constructor wraps the ARKodeCreate function,
    * calls the ReInit function to handle the inital condition,
    * and initializes pointers to null and flags to false.
    *
    * ARKodeCreate
    *
    * ARKodeCreate creates an internal memory block for a problem to
    * be solved by ARKODE.  If successful, ARKodeCreate returns a
    * pointer to the problem memory. This pointer should be passed to
    * ARKodeInit. If an initialization error occurs, ARKodeCreate
    * prints an error message to standard err and returns NULL.
    */
   ARKODESolver(Vector &mfem_y, bool parallel, bool _use_explicit = true);

   void Init(TimeDependentOperator &_f);
   /** \brief
    * The ReInit function is used in the initial construction and
    * initialization of the ARKODESolver object. It wraps either ARKodeInit or
    * ARKodeReInit to pass the initial condtion to the ode_mem struct.
    *
    * ARKodeInit
    *
    * ARKodeInit allocates and initializes memory for a problem. All
    * inputs are checked for errors. If any error occurs during
    * initialization, it is reported to the file whose file pointer
    * is errfp and an error flag is returned. Otherwise, it returns
    * ARK_SUCCESS.  This routine must be called prior to calling
    * ARKode to evolve the problem.
    *
    * ARKodeReInit
    *
    * ARKodeReInit re-initializes ARKODE's memory for a problem,
    * assuming it has already been allocated in a prior ARKodeInit
    * call.  All problem specification inputs are checked for errors.
    * If any error occurs during initialization, it is reported to
    * the file whose file pointer is errfp.  This routine should only
    * be called after ARKodeInit, and only when the problem dynamics
    * or desired solvers have changed dramatically, so that the
    * problem integration should resume as if started from scratch.
    *
    * The return value is ARK_SUCCESS = 0 if no errors occurred, or
    * a negative value otherwise.
    */
   void ReInit(TimeDependentOperator &_f, Vector &y_, double &_t);

   /** \brief
    * SetSStolerances wraps the ARKode function ARKodeSStolerances
    * which specifies scalar relative and absolute tolerances.
    *
    * These functions specify the integration tolerances. One of them
    * SHOULD be called before the first call to ARKode; otherwise
    * default values of reltol=1e-4 and abstol=1e-9 will be used,
    * which may be entirely incorrect for a specific problem.
    *
    * ARKodeSStolerances
    *
    * ARKodeSStolerances specifies scalar relative and absolute tolerances.
    */
   void SetSStolerances(realtype reltol, realtype abstol);

   /** \brief
    * Step transfers vector pointers using TransferNVector and calls
    * ARKode, which integrates over a user-defined time interval.
    *
    * ARKode
    *
    * This routine is the main driver of the ARKODE package.
    *
    * It integrates over a time interval defined by the user, by
    * calling arkStep to do internal time steps.
    *
    * The first time that ARKode is called for a successfully
    * initialized problem, it computes a tentative initial step size.
    *
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
   void Step(Vector &x, double &t, double &dt);

   /** Defines a custom Jacobian inversion.
       First calls the existing ark_lfree routine if it is not NULL.
       It then sets the ark_linit, ark_lsetup, ark_lsolve, ark_lfree fields in
       the ARKODE's memory structure. */
   void SetLinearSolve(SundialsLinearSolveOperator*);

   /** \brief
    * Wraps SetERKTable to choose a specific Butcher table for a RK method.
    *
    * ARKodeSetERKTable
    *
    * Specifies to use a customized Butcher table for the explicit
    * portion of the system.
    */
   void WrapSetERKTableNum(int table_num);

   /** \brief
    * Wraps SetFixedStep to force ARKode to take one internal step of size dt.
    *
    * ARKodeSetFixedStep
    *
    * Specifies to use a fixed time step size instead of performing
    * any form of temporal adaptivity.  ARKode will use this step size
    * for all steps (unless tstop is set, in which case it may need to
    * modify that last step approaching tstop.  If any (non)linear
    * solver failure occurs, ARKode will immediately return with an
    * error message since the time step size cannot be modified.
    *
    * Any nonzero argument will result in the use of that fixed step
    * size; an argument of 0 will re-enable temporal adaptivity.
    */
   void WrapSetFixedStep(double dt);

   /** \brief Destroys associated memory. Calls CVodeFree and N_VDestroy.
    *
    * ARKodeFree
    *
    * This routine frees the problem memory allocated by ARKodeInit.
    * Such memory includes all the vectors allocated by
    * arkAllocVectors, and arkAllocRKVectors, and the memory lmem for
    * the linear solver (deallocated by a call to lfree).
    */
   ~ARKODESolver();
};

/// Interface for custom Jacobian inversion in Sundials.
/// The Jacobian problem has the form I - dt inv(M) J(y) = b.
class SundialsLinearSolveOperator : public Operator
{
public:
   SundialsLinearSolveOperator(int s) : Operator(s)
   { }
   virtual void SolveJacobian(Vector* b, Vector* y, Vector* tmp, double dt) = 0;
};

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
