#include "sundials.hpp"

#ifdef MFEM_USE_SUNDIALS

#include "solvers.hpp"
#ifdef MFEM_USE_MPI
#include "hypre.hpp"
#endif

#include <nvector/nvector_serial.h>
#ifdef MFEM_USE_MPI
#include <nvector/nvector_parhyp.h>
#endif

#include <cvode/cvode_impl.h>
#include <cvode/cvode_spgmr.h>

// This just hides a warning (to be removed after it's fixed in Sundials).
#ifdef MSG_TIME_INT
  #undef MSG_TIME_INT
#endif

#include <arkode/arkode_impl.h>
#include <arkode/arkode_spgmr.h>

#include <kinsol/kinsol.h>
#include <kinsol/kinsol_spgmr.h>


/* Choose default tolerances to match ARKode defaults*/
#define RELTOL RCONST(1.0e-4)
#define ABSTOL RCONST(1.0e-9)

using namespace std;

// Creates N_Vector nv linked to the data in mv.
static void ConnectNVector(mfem::Vector &mv, N_Vector &nv)
{
   nv = N_VMake_Serial(mv.Size(),
                      static_cast<realtype *>(mv.GetData()));
   MFEM_ASSERT(static_cast<void *>(nv) != NULL, "N_VMake_Serial() failed!");
}

// Creates a parallel N_Vector nv linked to the data in mv.
static void ConnectParNVector(mfem::Vector &mv, N_Vector &nv)
{
#ifdef MFEM_USE_MPI
   mfem::HypreParVector *hpv = dynamic_cast<mfem::HypreParVector *>(&mv);
   MFEM_VERIFY(hpv != NULL, "Could not cast to HypreParVector!");
   nv = N_VMake_ParHyp(hpv->StealParVector());
#else
   MFEM_ABORT("This function should be called only with a parallel build!");
#endif
}

// Creates MFEM Vector mv linked to the data in nv.
static inline void ConnectMFEMVector(N_Vector &nv, mfem::Vector &mv)
{
   if (N_VGetVectorID(nv) == SUNDIALS_NVEC_SERIAL)
   {
      mv.NewDataAndSize(NV_DATA_S(nv), NV_LENGTH_S(nv));
   }
   else if (N_VGetVectorID(nv) == SUNDIALS_NVEC_PARHYP)
   {
#ifdef MFEM_USE_MPI
      mfem::HypreParVector hpv(N_VGetVector_ParHyp(nv));
      mv.NewDataAndSize(hpv.GetData(), hpv.Size());
#else
      MFEM_ABORT("The serial MFEM build somehow produced a parallel N_Vector!");
#endif
   }
   else
   {
      MFEM_ABORT("Unknown N_Vector type.");
   }
}

static int SundialsMult(realtype t, N_Vector y, N_Vector ydot, void *user_data)
{
   mfem::Vector mfem_y, mfem_ydot;
   ConnectMFEMVector(y, mfem_y);
   ConnectMFEMVector(ydot, mfem_ydot);

   // Compute y' = f(t, y).
   mfem::TimeDependentOperator *f =
      static_cast<mfem::TimeDependentOperator *>(user_data);
   f->SetTime(t);
   f->Mult(mfem_y, mfem_ydot);
   return 0;
}

// Computes the non-linear operator action F(u).
static int KinSolMult(N_Vector u, N_Vector fu, void *user_data)
{
   mfem::Vector mfem_u, mfem_fu;
   ConnectMFEMVector(u, mfem_u);
   ConnectMFEMVector(fu, mfem_fu);

   // Computes the non-linear action F(u).
   static_cast<mfem::Operator *>(user_data)->Mult(mfem_u, mfem_fu);
   return 0;
}

// Computes J(u)v.
// Here new_u tells you whether u has been updated since the
// last call to KinSolJacAction.
static int KinSolJacAction(N_Vector v, N_Vector Jv, N_Vector u,
                           booleantype *new_u, void *user_data)
{
   mfem::Vector mfem_u, mfem_v, mfem_Jv;
   ConnectMFEMVector(u, mfem_u);
   ConnectMFEMVector(v, mfem_v);
   ConnectMFEMVector(Jv, mfem_Jv);

   mfem::Operator &J =
         static_cast<mfem::Operator *>(user_data)->GetGradient(mfem_u);
   J.Mult(mfem_v, mfem_Jv);
   return 0;
}

static int WrapLinearCVSolveInit(CVodeMem cv_mem)
{
   return 0;
}

// Jean: setup may not be needed, since Jacobian is recomputed each iteration
// ypred is the predicted y at the current time, fpred is f(t,ypred).
static int WrapLinearCVSolveSetup(CVodeMem cv_mem, int convfail,
                                  N_Vector ypred, N_Vector fpred,
                                  booleantype *jcurPtr, N_Vector vtemp1,
                                  N_Vector vtemp2, N_Vector vtemp3)
{
   return 0;
}

static int WrapLinearCVSolve(CVodeMem cv_mem, N_Vector b,
                             N_Vector weight, N_Vector ycur,
                             N_Vector fcur)
{
   mfem::Vector solve_y, solve_b;
   ConnectMFEMVector(ycur, solve_y);
   ConnectMFEMVector(b, solve_b);

   mfem::SundialsLinearSolveOperator *op =
         static_cast<mfem::SundialsLinearSolveOperator *>(cv_mem->cv_lmem);
   op->SolveJacobian(&solve_b, &solve_y, cv_mem->cv_gamma);
   return 0;
}

static void WrapLinearCVSolveFree(CVodeMem cv_mem)
{
   return;
}

/*
 The purpose of ark_linit is to complete initializations for a
 specific linear solver, such as counters and statistics.
 An LInitFn should return 0 if it has successfully initialized
 the ARKODE linear solver and a negative value otherwise.
 If an error does occur, an appropriate message should be sent
 to the error handler function.
 */
static int WrapLinearARKSolveInit(ARKodeMem ark_mem)
{
   return 0;
}

/*
The job of ark_lsetup is to prepare the linear solver for
 subsequent calls to ark_lsolve. It may recompute Jacobian-
 related data as it deems necessary. Its parameters are as
 follows:

 ark_mem - problem memory pointer of type ARKodeMem. See the
          typedef earlier in this file.

 convfail - a flag to indicate any problem that occurred during
            the solution of the nonlinear equation on the
            current time step for which the linear solver is
            being used. This flag can be used to help decide
            whether the Jacobian data kept by a ARKODE linear
            solver needs to be updated or not.
            Its possible values have been documented above.

 ypred - the predicted y vector for the current ARKODE internal
         step.

 fpred - f(tn, ypred).

 jcurPtr - a pointer to a boolean to be filled in by ark_lsetup.
           The function should set *jcurPtr=TRUE if its Jacobian
           data is current after the call and should set
           *jcurPtr=FALSE if its Jacobian data is not current.
           Note: If ark_lsetup calls for re-evaluation of
           Jacobian data (based on convfail and ARKODE state
           data), it should return *jcurPtr=TRUE always;
           otherwise an infinite loop can result.

 vtemp1 - temporary N_Vector provided for use by ark_lsetup.

 vtemp3 - temporary N_Vector provided for use by ark_lsetup.

 vtemp3 - temporary N_Vector provided for use by ark_lsetup.

 The ark_lsetup routine should return 0 if successful, a positive
 value for a recoverable error, and a negative value for an
 unrecoverable error.
 */
//ypred is the predicted y at the current time, fpred is f(t,ypred)
static int WrapLinearARKSolveSetup(ARKodeMem ark_mem, int convfail,
                                   N_Vector ypred, N_Vector fpred,
                                   booleantype *jcurPtr, N_Vector vtemp1,
                                   N_Vector vtemp2, N_Vector vtemp3)
{
   return 0;
}

/*
 ark_lsolve must solve the linear equation P x = b, where
 P is some approximation to (M - gamma J), M is the system mass
 matrix, J = (df/dy)(tn,ycur), and the RHS vector b is input. The
 N-vector ycur contains the solver's current approximation to
 y(tn) and the vector fcur contains the N_Vector f(tn,ycur). The
 solution is to be returned in the vector b. ark_lsolve returns
 a positive value for a recoverable error and a negative value
 for an unrecoverable error. Success is indicated by a 0 return
 value.
*/
static int WrapLinearARKSolve(ARKodeMem ark_mem, N_Vector b,
                              N_Vector weight, N_Vector ycur,
                              N_Vector fcur)
{
   mfem::Vector solve_y, solve_b;
   ConnectMFEMVector(ycur, solve_y);
   ConnectMFEMVector(b, solve_b);

   mfem::SundialsLinearSolveOperator *op =
         static_cast<mfem::SundialsLinearSolveOperator *>(ark_mem->ark_lmem);
   op->SolveJacobian(&solve_b, &solve_y, ark_mem->ark_gamma);
   return 0;
}

// This should free up any memory allocated by the linear solver.
static void WrapLinearARKSolveFree(ARKodeMem ark_mem)
{
   return;
}

namespace mfem
{

CVODESolver::CVODESolver(Vector &y_, bool parallel, int lmm, int iter)
   : lin_method_type(lmm), solver_iteration_type(iter)
{
   connectNV = (parallel) ? ConnectParNVector : ConnectNVector;

   // Create the NVector y.
   (*connectNV)(y_, y);

   // Create the solver memory.
   ode_mem = CVodeCreate(lmm, iter);

   // Initialize integrator memory, specify the user's
   // RHS function in x' = f(t, x), initial time, initial condition.
   int flag = CVodeInit(ode_mem, SundialsMult, 0.0, y);
   MFEM_ASSERT(flag >= 0, "CVodeInit() failed!");

   // For some reason CVODE insists those to be set by hand (no defaults).
   SetSStolerances(RELTOL, ABSTOL);

   // When implicit method is chosen, one should specify the linear solver.
   if (lin_method_type == CV_BDF)
   {
      CVSpgmr(ode_mem, PREC_NONE, 0);
   }
}

void CVODESolver::Init(TimeDependentOperator &f_)
{
   f = &f_;

   // Set the pointer to user-defined data.
   int flag = CVodeSetUserData(ode_mem, f);
   MFEM_ASSERT(flag >= 0, "CVodeSetUserData() failed!");
}

void CVODESolver::ReInit(TimeDependentOperator &f_, Vector &y_, double &t_)
{
   f = &f_;
   (*connectNV)(y_, y);

   // Re-init memory, time and solution. The RHS action is known from Init().
   int flag = CVodeReInit(ode_mem, t_, y);
   MFEM_ASSERT(flag >= 0, "CVodeReInit() failed!");

   // Set the pointer to user-defined data.
   flag = CVodeSetUserData(ode_mem, f);
   MFEM_ASSERT(flag >= 0, "CVodeSetUserData() failed!");

   // When implicit method is chosen, one should specify the linear solver.
   if (lin_method_type == CV_BDF)
   {
      CVSpgmr(ode_mem, PREC_NONE, 0);
   }
}

void CVODESolver::SetSStolerances(realtype reltol, realtype abstol)
{
   // Specify the scalar relative tolerance and scalar absolute tolerance.
   int flag = CVodeSStolerances(ode_mem, reltol, abstol);
   MFEM_ASSERT(flag >= 0, "CVodeSStolerances() failed!");
}

void CVODESolver::Step(Vector &x, double &t, double &dt)
{
   (*connectNV)(x, y);
   realtype tout = t + dt;

   // Don't allow stepping over tout (then it interpolates in time).
   CVodeSetStopTime(ode_mem, tout);
   // Step.
   // CV_NORMAL - take many steps until reaching tout.
   // CV_ONE_STEP - take one step and return (might be before tout).
   int flag = CVode(ode_mem, tout, y, &t, CV_NORMAL);
   MFEM_ASSERT(flag >= 0, "CVode() failed!");

   // Record last incremental step size.
   flag = CVodeGetLastStep(ode_mem, &dt);
}

void CVODESolver::SetLinearSolve(SundialsLinearSolveOperator *op)
{
   MFEM_VERIFY(solver_iteration_type == CV_NEWTON,
               "The function is applicable only to CV_NEWTON iteration type.");
   MFEM_VERIFY(ode_mem != NULL, "CVODE memory error!");

   CVodeMem cv_mem = static_cast<CVodeMem>(ode_mem);
   if (cv_mem->cv_lfree != NULL)
   {
      cv_mem->cv_lfree(cv_mem);
   }

   // Set four main function fields in cv_mem.
   cv_mem->cv_linit  = WrapLinearCVSolveInit;
   cv_mem->cv_lsetup = WrapLinearCVSolveSetup;
   cv_mem->cv_lsolve = WrapLinearCVSolve;
   cv_mem->cv_lfree  = WrapLinearCVSolveFree;

   // Maximum number of Newton iterations.
   CVodeSetMaxNumSteps(cv_mem, 50);
   SetSStolerances(1e-2, 1e-4);

   cv_mem->cv_lmem = op;
}

CVODESolver::~CVODESolver()
{
   N_VDestroy(y);
   if (ode_mem != NULL)
   {
      CVodeFree(&ode_mem);
   }
}


ARKODESolver::ARKODESolver(Vector &y_, bool parallel, bool explicit_)
   : ODESolver(),
     use_explicit(explicit_)
{
   connectNV = (parallel) ? ConnectParNVector : ConnectNVector;

   // Create the NVector y.
   (*connectNV)(y_, y);

   // Create the solver memory.
   ode_mem = ARKodeCreate();

   // Initialize the integrator memory, specify the user's
   // RHS function in x' = f(t, x), the initial time, initial condition.
   int flag = use_explicit ?
              ARKodeInit(ode_mem, SundialsMult, NULL, 0.0, y) :
              ARKodeInit(ode_mem, NULL, SundialsMult, 0.0, y);
   MFEM_ASSERT(flag >= 0, "ARKodeInit() failed!");

   SetSStolerances(RELTOL, ABSTOL);

   // When implicit method is chosen, one should specify the linear solver.
   if (use_explicit == false)
   {
      ARKSpgmr(ode_mem, PREC_NONE, 0);
   }
}

void ARKODESolver::Init(TimeDependentOperator &f_)
{
   f = &f_;
   // Set the pointer to user-defined data.
   int flag = ARKodeSetUserData(ode_mem, this->f);
   MFEM_ASSERT(flag >= 0, "ARKodeSetUserData() failed!");
}

void ARKODESolver::ReInit(TimeDependentOperator &f_, Vector &y_, double &t_)
{
   f = &f_;
   (*connectNV)(y_, y);

   // Re-init memory, time and solution. The RHS action is known from Init().
   int flag = use_explicit ?
              ARKodeReInit(ode_mem, SundialsMult, NULL, t_, y) :
              ARKodeReInit(ode_mem, NULL, SundialsMult, t_, y);
   MFEM_ASSERT(flag >= 0, "ARKodeReInit() failed!");

   // Set the pointer to user-defined data.
   flag = ARKodeSetUserData(ode_mem, this->f);
   MFEM_ASSERT(flag >= 0, "ARKodeSetUserData() failed!");

   // When implicit method is chosen, one should specify the linear solver.
   if (use_explicit == false)
   {
      ARKSpgmr(ode_mem, PREC_NONE, 0);
   }
}

void ARKODESolver::SetSStolerances(realtype reltol, realtype abstol)
{
   // Specify the scalar relative tolerance and scalar absolute tolerance.
   int flag = ARKodeSStolerances(ode_mem, reltol, abstol);
   MFEM_ASSERT(flag >= 0, "ARKodeSStolerances() failed!");
}

void ARKODESolver::Step(Vector &x, double &t, double &dt)
{
   (*connectNV)(x, y);
   realtype tout = t + dt;

   // Don't allow stepping over tout (then it interpolates in time).
   ARKodeSetStopTime(ode_mem, tout);
   // Step.
   // ARK_NORMAL - take many steps until reaching tout.
   // ARK_ONE_STEP - take one step and return (might be before tout).
   int flag = ARKode(ode_mem, tout, y, &t, ARK_NORMAL);
   MFEM_ASSERT(flag >= 0, "ARKode() failed!");

   // Record last incremental step size.
   flag = ARKodeGetLastStep(ode_mem, &dt);
}

void ARKODESolver::SetERKTableNum(int table_num)
{
   ARKodeSetERKTableNum(ode_mem, table_num);
}

void ARKODESolver::SetFixedStep(double dt)
{
   ARKodeSetFixedStep(ode_mem, static_cast<realtype>(dt));
}

void ARKODESolver::SetLinearSolve(SundialsLinearSolveOperator* op)
{
   MFEM_VERIFY(use_explicit == false,
               "The function is applicable only to implicit time integration.");
   MFEM_VERIFY(ode_mem != NULL, "ARKODE memory error!");

   ARKodeMem ark_mem = static_cast<ARKodeMem>(ode_mem);
   if (ark_mem->ark_lfree != NULL)
   {
      ark_mem->ark_lfree(ark_mem);
   }

   // Tell ARKODE that the Jacobian inversion is custom.
   ark_mem->ark_lsolve_type = 4;
   // Set four main function fields in ark_mem.
   ark_mem->ark_linit  = WrapLinearARKSolveInit;
   ark_mem->ark_lsetup = WrapLinearARKSolveSetup;
   ark_mem->ark_lsolve = WrapLinearARKSolve;
   ark_mem->ark_lfree  = WrapLinearARKSolveFree;

   // Maximum number of Newton iterations.
   ARKodeSetMaxNumSteps(ode_mem, 10000);
   SetSStolerances(1e-2,1e-4);

   ark_mem->ark_lmem = op;
}

ARKODESolver::~ARKODESolver()
{
   N_VDestroy(y);
   if (ode_mem != NULL)
   {
      ARKodeFree(&ode_mem);
   }
}

KinSolver::KinSolver(Operator &oper, Vector &mfem_u,
                             bool parallel, bool use_oper_grad)
   : kin_mem(NULL)
{
   connectNV = (parallel) ? ConnectParNVector : ConnectNVector;

   kin_mem = KINCreate();

   (*connectNV)(mfem_u, u);
   KINInit(kin_mem, KinSolMult, u);

   // Set void pointer to user data.
   KINSetUserData(kin_mem, static_cast<void *>(&oper));

   // Set scaled preconditioned GMRES linear solver.
   KINSpgmr(kin_mem, 0);

   // Define the Jacobian action.
   if (use_oper_grad)
   {
      KINSpilsSetJacTimesVecFn(kin_mem, KinSolJacAction);
   }
}

void KinSolver::SetPrintLevel(int level)
{
   KINSetPrintLevel(kin_mem, level);
}

void KinSolver::SetFuncNormTol(double tol)
{
   KINSetFuncNormTol(kin_mem, tol);
}

void KinSolver::SetScaledStepTol(double tol)
{
   KINSetScaledStepTol(kin_mem, tol);
}

void KinSolver::Solve(Vector &mfem_u,
                          Vector &mfem_u_scale, Vector &mfem_f_scale)
{
   (*connectNV)(mfem_u, u);
   (*connectNV)(mfem_u_scale, u_scale);
   (*connectNV)(mfem_f_scale, f_scale);

   // LINESEARCH might be fancier, but more fragile near convergence.
   int strategy = KIN_LINESEARCH;
//   int strategy = KIN_NONE;
   int flag = KINSol(kin_mem, u, strategy, u_scale, f_scale);
   MFEM_VERIFY(flag == KIN_SUCCESS || flag == KIN_INITIAL_GUESS_OK,
               "KINSol returned " << flag << " that indicated a problem!");
}

KinSolver::~KinSolver()
{
   KINFree(&kin_mem);
}

} // namespace mfem

#endif
