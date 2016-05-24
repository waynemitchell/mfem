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

#include <cvode/cvode_band.h>
#include <cvode/cvode_spgmr.h>

#include <cvode/cvode_impl.h>
// This just hides a warning (to be removed after it's fixed in Sundials).
#ifdef MSG_TIME_INT
  #undef MSG_TIME_INT
#endif
#include <arkode/arkode_impl.h>

#include <kinsol/kinsol.h>
#include <kinsol/kinsol_spgmr.h>


/* Choose default tolerances to match ARKode defaults*/
#define RELTOL RCONST(1.0e-4)
#define ABSTOL RCONST(1.0e-9)

using namespace std;

// Connects v to mfem_v.
static void ConnectNVector(mfem::Vector &mfem_v, N_Vector &v)
{
#ifndef MFEM_USE_MPI
   v = N_VMake_Serial(mfem_v.Size(),
                      static_cast<realtype *>(mfem_v.GetData()));
   MFEM_ASSERT(static_cast<void *>(v) != NULL, "N_VMake_Serial() failed!");
#else
   mfem::HypreParVector *hv = dynamic_cast<mfem::HypreParVector *>(&mfem_v);
   MFEM_VERIFY(hv != NULL, "Could not cast to HypreParVector!");
   v = N_VMake_ParHyp(hv->StealParVector());
#endif
}

static int SundialsMult(realtype t, N_Vector y, N_Vector ydot, void *user_data)
{
   // Creates mfem Vectors linked to the data in y and in ydot.
#ifndef MFEM_USE_MPI
   mfem::Vector mfem_y(NV_DATA_S(y), NV_LENGTH_S(y));
   mfem::Vector mfem_ydot(NV_DATA_S(ydot), NV_LENGTH_S(ydot));
#else
   mfem::HypreParVector mfem_y =
      static_cast<mfem::HypreParVector>(N_VGetVector_ParHyp(y));
   mfem::HypreParVector mfem_ydot =
      static_cast<mfem::HypreParVector>(N_VGetVector_ParHyp(ydot));
#endif

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
   // Creates mfem Vectors linked to the data in y and in ydot.
#ifndef MFEM_USE_MPI
   mfem::Vector mfem_u(NV_DATA_S(u), NV_LENGTH_S(u));
   mfem::Vector mfem_fu(NV_DATA_S(fu), NV_LENGTH_S(fu));
#else
   mfem::HypreParVector mfem_u =
      static_cast<mfem::HypreParVector>(N_VGetVector_ParHyp(u));
   mfem::HypreParVector mfem_fu =
      static_cast<mfem::HypreParVector>(N_VGetVector_ParHyp(fu));
#endif

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
#ifndef MFEM_USE_MPI
   mfem::Vector mfem_u(NV_DATA_S(u), NV_LENGTH_S(u));
   mfem::Vector mfem_v(NV_DATA_S(v), NV_LENGTH_S(v));
   mfem::Vector mfem_Jv(NV_DATA_S(Jv), NV_LENGTH_S(Jv));
#else
   mfem::HypreParVector mfem_u =
      static_cast<mfem::HypreParVector>(N_VGetVector_ParHyp(u));
   mfem::HypreParVector mfem_v =
      static_cast<mfem::HypreParVector>(N_VGetVector_ParHyp(v));
   mfem::HypreParVector mfem_Jv =
      static_cast<mfem::HypreParVector>(N_VGetVector_ParHyp(Jv));
#endif

   mfem::Operator &J =
         static_cast<mfem::Operator *>(user_data)->GetGradient(mfem_u);
   J.Mult(mfem_v, mfem_Jv);
   return 0;
}

/// Linear solve associated with CVodeMem structs.
static int MFEMLinearCVSolve(void *cvode_mem, mfem::Solver* solve,
                             mfem::SundialsLinearSolveOperator* op);

/// Linear solve associated with ARKodeMem structs.
static int MFEMLinearARKSolve(void *arkode_mem, mfem::Solver*,
                              mfem::SundialsLinearSolveOperator*);

namespace mfem
{

CVODESolver::CVODESolver(Vector &y_, int lmm, int iter)
   : solver_iteration_type(iter)
{
   // Create the NVector y.
   ConnectNVector(y_, y);

   // Create the solver memory.
   ode_mem = CVodeCreate(lmm, iter);

   // Initialize integrator memory, specify the user's
   // RHS function in x' = f(t, x), initial time, initial condition.
   int flag = CVodeInit(ode_mem, SundialsMult, 0.0, y);
   MFEM_ASSERT(flag >= 0, "CVodeInit() failed!");

   // For some reason CVODE insists those to be set by hand (no defaults).
   SetSStolerances(RELTOL, ABSTOL);
}

#ifdef MFEM_USE_MPI
CVODESolver::CVODESolver(MPI_Comm comm_, Vector &y_, int lmm, int iter)
   : comm(comm_),
     solver_iteration_type(iter)
{
   // Create the NVector y.
   ConnectNVector(y_, y);

   // Create the solver memory.
   ode_mem = CVodeCreate(lmm, iter);

   // Initialize integrator memory, specify the user's
   // RHS function in x' = f(t, x), initial time, initial condition.
   int flag = CVodeInit(ode_mem, SundialsMult, 0.0, y);
   MFEM_ASSERT(flag >= 0, "CVodeInit() failed!");

   // For some reason CVODE insists those to be set by hand (no defaults).
   SetSStolerances(RELTOL, ABSTOL);
}
#endif

void CVODESolver::Init(TimeDependentOperator &_f)
{
   f = &_f;

   // Set the pointer to user-defined data.
   int flag = CVodeSetUserData(ode_mem, f);
   MFEM_ASSERT(flag >= 0, "CVodeSetUserData() failed!");

   // When newton iterations are chosen, one should specify the linear solver.
   if (solver_iteration_type == CV_NEWTON)
   {
      CVSpgmr(ode_mem, PREC_NONE, 0);
   }
}

void CVODESolver::ReInit(TimeDependentOperator &_f, Vector &_y, double & _t)
{
   f = &_f;
   ConnectNVector(_y, y);

   // Re-init memory, time and solution. The RHS action is known from Init().
   int flag = CVodeReInit(ode_mem, static_cast<realtype>(_t), y);
   MFEM_ASSERT(flag >= 0, "CVodeReInit() failed!");

   // Set the pointer to user-defined data.
   flag = CVodeSetUserData(ode_mem, this->f);
   MFEM_ASSERT(flag >= 0, "CVodeSetUserData() failed!");

   // When newton iterations are chosen, one should specify the linear solver.
   if (solver_iteration_type == CV_NEWTON)
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
#ifndef MFEM_USE_MPI
   NV_DATA_S(y) = x.GetData();
#else
   HypreParVector *hx = dynamic_cast<HypreParVector *>(&x);
   MFEM_ASSERT(hx != NULL, "Could not cast to HypreParVector!");
   y = N_VMake_ParHyp(hx->StealParVector());
#endif

   // Perform the step.
   realtype tout = t + dt;
   int flag = CVode(ode_mem, tout, y, &t, CV_NORMAL);
   MFEM_ASSERT(flag >= 0, "CVode() failed!");

   // Record last incremental step size.
   flag = CVodeGetLastStep(ode_mem, &dt);
}

void CVODESolver::SetLinearSolve(Solver* J_solve,
                                 SundialsLinearSolveOperator* op)
{
   // Jean comment: If linear solve should be Newton, recreate ode_mem object.
   //    Consider checking for CV_ADAMS vs CV_BDF as well.
   // TODO: Is there an error here ??
   if (solver_iteration_type == CV_FUNCTIONAL)
   {
      realtype t0 = ((CVodeMem) ode_mem)->cv_tn;
      CVodeFree(&ode_mem);
      ode_mem=CVodeCreate(CV_BDF, CV_NEWTON); // ??

      int flag = CVodeInit(ode_mem, SundialsMult, t0, y);
      MFEM_ASSERT(flag >= 0, "CVodeInit() failed!");

      CVodeSetUserData(ode_mem, this->f);
   }

   // Call CVodeSetMaxNumSteps to increase default.
   CVodeSetMaxNumSteps(ode_mem, 10000);
   SetSStolerances(1e-2,1e-4);

   MFEMLinearCVSolve(ode_mem, J_solve, op);
}

CVODESolver::~CVODESolver()
{
   N_VDestroy(y);
   if (ode_mem != NULL)
   {
      CVodeFree(&ode_mem);
   }
}

ARKODESolver::ARKODESolver(Vector &_y, int _use_explicit)
   : ODESolver(),
     use_explicit(_use_explicit)
{
   y = N_VMake_Serial(_y.Size(),
                      (realtype*) _y.GetData());   /* Allocate y vector */
   MFEM_ASSERT(static_cast<void *>(y) != NULL, "N_VMake_Serial() failed!");

   // Create the solver memory.
   ode_mem = ARKodeCreate();

   // Initialize the integrator memory, specify the user's
   // RHS function in x' = f(t, x), the inital time, initial condition.
   int flag = use_explicit ?
              ARKodeInit(ode_mem, SundialsMult, NULL, 0.0, y) :
              ARKodeInit(ode_mem, NULL, SundialsMult, 0.0, y);
   MFEM_ASSERT(flag >= 0, "ARKodeInit() failed!");

   SetSStolerances(RELTOL, ABSTOL);
}

#ifdef MFEM_USE_MPI
ARKODESolver::ARKODESolver(MPI_Comm _comm, Vector &_y, int _use_explicit)
   : ODESolver(),
     comm(_comm),
     use_explicit(_use_explicit)
{
   // Create the NVector y.
   ConnectNVector(_y, y);

   // Create the solver memory.
   ode_mem = ARKodeCreate();

   // Initialize the integrator memory, specify the user's
   // RHS function in x' = f(t, x), the inital time, initial condition.
   int flag = use_explicit ?
              ARKodeInit(ode_mem, SundialsMult, NULL, 0.0, y) :
              ARKodeInit(ode_mem, NULL, SundialsMult, 0.0, y);
   MFEM_ASSERT(flag >= 0, "ARKodeInit() failed!");

   SetSStolerances(RELTOL, ABSTOL);
}
#endif

void ARKODESolver::Init(TimeDependentOperator &_f)
{
   f = &_f;
   // Set the pointer to user-defined data.
   int flag = ARKodeSetUserData(ode_mem, this->f);
   MFEM_ASSERT(flag >= 0, "ARKodeSetUserData() failed!");
}

void ARKODESolver::ReInit(TimeDependentOperator &_f, Vector &_y, double &_t)
{
   f = &_f;
   ConnectNVector(_y, y);

   // Re-init memory, time and solution. The RHS action is known from Init().
   int flag = use_explicit ?
              ARKodeReInit(ode_mem, SundialsMult, NULL, (realtype) _t, y) :
              ARKodeReInit(ode_mem, NULL, SundialsMult, (realtype) _t, y);
   MFEM_ASSERT(flag >= 0, "ARKodeReInit() failed!");

   // Set the pointer to user-defined data.
   flag = ARKodeSetUserData(ode_mem, this->f);
   MFEM_ASSERT(flag >= 0, "ARKodeSetUserData() failed!");
}

void ARKODESolver::SetSStolerances(realtype reltol, realtype abstol)
{
   // Specify the scalar relative tolerance and scalar absolute tolerance.
   int flag = ARKodeSStolerances(ode_mem, reltol, abstol);
   MFEM_ASSERT(flag >= 0, "ARKodeSStolerances() failed!");
}

void ARKODESolver::Step(Vector &x, double &t, double &dt)
{
#ifndef MFEM_USE_MPI
      NV_DATA_S(y) = x.GetData();
#else
   HypreParVector *hx = dynamic_cast<HypreParVector *>(&x);
   MFEM_ASSERT(hx != NULL, "Could not cast to HypreParVector!");
   y = N_VMake_ParHyp(hx->StealParVector());
#endif

   // Step.
   realtype tout = t + dt;
   int flag = ARKode(ode_mem, tout, y, &t, ARK_NORMAL);
   MFEM_ASSERT(flag >= 0, "ARKode() failed!");

   // Record last incremental step size.
   flag = ARKodeGetLastStep(ode_mem, &dt);
}

void ARKODESolver::WrapSetERKTableNum(int table_num)
{
   ARKodeSetERKTableNum(ode_mem, table_num);
}

void ARKODESolver::WrapSetFixedStep(double dt)
{
   ARKodeSetFixedStep(ode_mem, static_cast<realtype>(dt));
}

void ARKODESolver::SetLinearSolve(Solver* solve,
                                  SundialsLinearSolveOperator* op)
{
   if (use_explicit)
   {
      realtype t0= ((ARKodeMem) ode_mem)->ark_tn;
      ARKodeFree(&ode_mem);
      ode_mem=ARKodeCreate();

      // TODO: why is this?
      use_explicit=false;
      //change init structure in order to switch to implicit method
      int flag = use_explicit ?
                    ARKodeInit(ode_mem, SundialsMult, NULL, t0, y) :
                    ARKodeInit(ode_mem, NULL, SundialsMult, t0, y);
      MFEM_ASSERT(flag >= 0, "ARKodeInit() failed!");

      ARKodeSetUserData(ode_mem, this->f);
   }

   // Call ARKodeSetMaxNumSteps to increase default.
   ARKodeSetMaxNumSteps(ode_mem, 10000);
   SetSStolerances(1e-2,1e-4);

   MFEMLinearARKSolve(ode_mem, solve, op);
}

ARKODESolver::~ARKODESolver()
{
   N_VDestroy(y);
   if (ode_mem != NULL)
   {
      ARKodeFree(&ode_mem);
   }
}

KinSolWrapper::KinSolWrapper(Operator &oper, Vector &mfem_u)
   : kin_mem(NULL)
{
   kin_mem = KINCreate();

   ConnectNVector(mfem_u, u);
   ConnectNVector(mfem_u, u_scale);
   ConnectNVector(mfem_u, f_scale);
   KINInit(kin_mem, KinSolMult, u);

   // Set void pointer to user data.
   KINSetUserData(kin_mem, static_cast<void *>(&oper));

   // Set scaled preconditioned GMRES linear solver.
   KINSpgmr(kin_mem, 0);

   // Define the Jacobian action.
   KINSpilsSetJacTimesVecFn(kin_mem, KinSolJacAction);
}

void KinSolWrapper::setPrintLevel(int level)
{
   KINSetPrintLevel(kin_mem, level);
}

void KinSolWrapper::setFuncNormTol(double tol)
{
   KINSetFuncNormTol(kin_mem, tol);
}

void KinSolWrapper::setScaledStepTol(double tol)
{
   KINSetScaledStepTol(kin_mem, tol);
}

void KinSolWrapper::solve(Vector &mfem_u,
                          Vector &mfem_u_scale, Vector &mfem_f_scale)
{
   ConnectNVector(mfem_u, u);
   ConnectNVector(mfem_u_scale, u_scale);
   ConnectNVector(mfem_f_scale, f_scale);

   setFuncNormTol(1e-6);
   setScaledStepTol(1e-8);

   // LINESEARCH might be fancier, but more fragile near convergence.
   int strategy = KIN_LINESEARCH;
//   int strategy = KIN_NONE;
   int flag = KINSol(kin_mem, u, strategy, u_scale, f_scale);
   MFEM_VERIFY(flag == KIN_SUCCESS || flag == KIN_INITIAL_GUESS_OK,
               "KINSol returned " << flag << " that indicated a problem!");
}

KinSolWrapper::~KinSolWrapper()
{
   KINFree(&kin_mem);
}


} // namespace mfem

static int WrapLinearSolveSetup(void* lmem, double tn,
                                mfem::Vector* ypred, mfem::Vector* fpred)
{
   return 0;
}

static int WrapLinearSolve(void* lmem, double tn, mfem::Vector* b,
                           mfem::Vector* ycur, mfem::Vector* yn,
                           mfem::Vector* fcur)
{
   mfem::MFEMLinearSolverMemory *tmp_lmem =
         static_cast<mfem::MFEMLinearSolverMemory *>(lmem);

   tmp_lmem->op_for_gradient->SolveJacobian(b, ycur, yn, tmp_lmem->J_solve,
                                            tmp_lmem->weight);
   return 0;
}

static int WrapLinearCVSolveInit(CVodeMem cv_mem)
{
   return 0;
}

//Setup may not be needed, since Jacobian is recomputed each iteration
//ypred is the predicted y at the current time, fpred is f(t,ypred)
// TODO: what's the point of this function? WrapLinearSolveSetup does nothing!
static int WrapLinearCVSolveSetup(CVodeMem cv_mem, int convfail,
                                  N_Vector ypred, N_Vector fpred,
                                  booleantype *jcurPtr, N_Vector vtemp1,
                                  N_Vector vtemp2, N_Vector vtemp3)
{
   mfem::MFEMLinearSolverMemory* lmem= (mfem::MFEMLinearSolverMemory*)
                                       cv_mem->cv_lmem;
#ifndef MFEM_USE_MPI
   lmem->setup_y->SetDataAndSize(NV_DATA_S(ypred),NV_LENGTH_S(ypred));
   lmem->setup_f->SetDataAndSize(NV_DATA_S(fpred),NV_LENGTH_S(fpred));
#else
   lmem->solve_y = new mfem::HypreParVector(N_VGetVector_ParHyp(ypred));
   lmem->solve_f = new mfem::HypreParVector(N_VGetVector_ParHyp(fpred));
#endif
   *jcurPtr=TRUE;
   cv_mem->cv_lmem = lmem;
   WrapLinearSolveSetup(cv_mem->cv_lmem, cv_mem->cv_tn, lmem->setup_y,
                        lmem->setup_f);
   return 0;
}

static int WrapLinearCVSolve(CVodeMem cv_mem, N_Vector b,
                             N_Vector weight, N_Vector ycur,
                             N_Vector fcur)
{
   mfem::MFEMLinearSolverMemory* lmem= (mfem::MFEMLinearSolverMemory*)
                                       cv_mem->cv_lmem;
#ifndef MFEM_USE_MPI
   lmem->solve_y->SetDataAndSize(NV_DATA_S(ycur),NV_LENGTH_S(ycur));
   lmem->solve_yn->SetDataAndSize(NV_DATA_S(cv_mem->cv_zn[0]),NV_LENGTH_S(ycur));
   lmem->solve_f->SetDataAndSize(NV_DATA_S(fcur),NV_LENGTH_S(fcur));
   lmem->solve_b->SetDataAndSize(NV_DATA_S(b),NV_LENGTH_S(b));
#else
   lmem->solve_y = new mfem::HypreParVector(N_VGetVector_ParHyp(ycur));
   lmem->solve_f = new mfem::HypreParVector(N_VGetVector_ParHyp(fcur));
   lmem->solve_b = new mfem::HypreParVector(N_VGetVector_ParHyp(b));
#endif

   lmem->weight = cv_mem->cv_gamma;

   cv_mem->cv_lmem = lmem;
   // TODO: remove this function.
   // TODO: we dont need so many arguments.
   // TODO: can't we only have the operator in the lmem?
   WrapLinearSolve(cv_mem->cv_lmem, cv_mem->cv_tn, lmem->solve_b, lmem->solve_y,
                   lmem->setup_y, lmem->solve_f);
   return 0;
}

static void WrapLinearCVSolveFree(CVodeMem cv_mem)
{
   return;
}

/*---------------------------------------------------------------
 MFEMLinearCVSolve:

 This routine initializes the memory record and sets various
 function fields specific to the linear solver module.
 MFEMLinearCVSolve first calls the existing lfree routine if this is not
 NULL. It then sets the cv_linit, cv_lsetup, cv_lsolve,
 cv_lfree fields in (*cvode_mem) to be WrapLinearCVSolveInit,
 WrapLinearCVSolveSetup, WrapLinearCVSolve, and WrapLinearCVSolveFree, respectively.
---------------------------------------------------------------*/
static int MFEMLinearCVSolve(void *ode_mem, mfem::Solver* solve,
                             mfem::SundialsLinearSolveOperator* op)
{
   CVodeMem cv_mem;

   MFEM_VERIFY(ode_mem != NULL, "CVODE memory error!");
   cv_mem = (CVodeMem) ode_mem;

   if (cv_mem->cv_lfree != NULL) { cv_mem->cv_lfree(cv_mem); }

   // Set four main function fields in ark_mem
   cv_mem->cv_linit  = WrapLinearCVSolveInit;
   cv_mem->cv_lsetup = WrapLinearCVSolveSetup;
   cv_mem->cv_lsolve = WrapLinearCVSolve;
   cv_mem->cv_lfree  = WrapLinearCVSolveFree;
   cv_mem->cv_setupNonNull = 1;
   // forces cvode to call lsetup prior to every time it calls lsolve
   cv_mem->cv_maxcor = 1;

   //void* for containing linear solver memory
   mfem::MFEMLinearSolverMemory* lmem = new mfem::MFEMLinearSolverMemory();

#ifndef MFEM_USE_MPI
   lmem->setup_y = new mfem::Vector();
   lmem->setup_f = new mfem::Vector();
   lmem->solve_y = new mfem::Vector();
   lmem->solve_yn = new mfem::Vector();
   lmem->solve_f = new mfem::Vector();
   lmem->solve_b = new mfem::Vector();
   lmem->vec_tmp = new mfem::Vector(NV_LENGTH_S(cv_mem->cv_zn[0]));
#else
   lmem->setup_y = new mfem::HypreParVector(N_VGetVector_ParHyp(
                                                cv_mem->cv_zn[0]));
   lmem->setup_f = new mfem::HypreParVector((N_VGetVector_ParHyp(
                                                cv_mem->cv_zn[0])));
   lmem->solve_y = new mfem::HypreParVector((N_VGetVector_ParHyp(
                                                cv_mem->cv_zn[0])));
   lmem->solve_f = new mfem::HypreParVector((N_VGetVector_ParHyp(
                                                cv_mem->cv_zn[0])));
   lmem->solve_b = new mfem::HypreParVector((N_VGetVector_ParHyp(
                                                cv_mem->cv_zn[0])));
   lmem->vec_tmp = new mfem::HypreParVector((N_VGetVector_ParHyp(
                                                cv_mem->cv_zn[0])));
#endif

   lmem->J_solve = solve;
   lmem->op_for_gradient = op;

   cv_mem->cv_lmem = lmem;
   return (CVSPILS_SUCCESS);
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
   mfem::MFEMLinearSolverMemory* lmem= (mfem::MFEMLinearSolverMemory*)
                                       ark_mem->ark_lmem;

#ifndef MFEM_USE_MPI
   lmem->setup_y->SetDataAndSize(NV_DATA_S(ypred),NV_LENGTH_S(ypred));
   lmem->setup_f->SetDataAndSize(NV_DATA_S(fpred),NV_LENGTH_S(fpred));
#else
   // TODO: delete the pointers.
   lmem->solve_y = new mfem::HypreParVector(N_VGetVector_ParHyp(ypred));
   lmem->solve_f = new mfem::HypreParVector(N_VGetVector_ParHyp(fpred));
#endif
   *jcurPtr=TRUE;
   ark_mem->ark_lmem = lmem;
   WrapLinearSolveSetup(ark_mem->ark_lmem, ark_mem->ark_tn, lmem->setup_y,
                        lmem->setup_f);
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
   if (ark_mem->ark_tn>0)
   {
      mfem::MFEMLinearSolverMemory* lmem= (mfem::MFEMLinearSolverMemory*)
                                          ark_mem->ark_lmem;

#ifndef MFEM_USE_MPI
      lmem->solve_y->SetDataAndSize(NV_DATA_S(ycur),NV_LENGTH_S(ycur));
      lmem->solve_yn->SetDataAndSize(NV_DATA_S(ark_mem->ark_y),NV_LENGTH_S(ycur));
      lmem->solve_f->SetDataAndSize(NV_DATA_S(fcur),NV_LENGTH_S(fcur));
      lmem->solve_b->SetDataAndSize(NV_DATA_S(b),NV_LENGTH_S(b));
#else
      lmem->solve_y = new mfem::HypreParVector(N_VGetVector_ParHyp(ycur));
      lmem->solve_f = new mfem::HypreParVector(N_VGetVector_ParHyp(fcur));
      lmem->solve_b = new mfem::HypreParVector(N_VGetVector_ParHyp(b));
#endif

      lmem->weight = ark_mem->ark_gamma;

      ark_mem->ark_lmem = lmem;
      WrapLinearSolve(ark_mem->ark_lmem, ark_mem->ark_tn, lmem->solve_b,
                      lmem->solve_y,
                      lmem->setup_y, lmem->solve_f);
   }
   return 0;
}

/*
 ark_lfree should free up any memory allocated by the linear
 solver. This routine is called once a problem has been
 completed and the linear solver is no longer needed.
 */
static void WrapLinearARKSolveFree(ARKodeMem ark_mem)
{
   return;
}

/*---------------------------------------------------------------
 MFEMLinearARKSolve:

 This routine initializes the memory record and sets various
 function fields specific to the linear solver module.
 MFEMLinearARKSolve first calls the existing lfree routine if this is not
 NULL. It then sets the ark_linit, ark_lsetup, ark_lsolve,
 ark_lfree fields in (*arkode_mem) to be WrapLinearARKSolveInit,
 WrapLinearARKSolveSetup, WrapARKLinearSolve, and WrapLinearARKSolveFree,
 respectively.
---------------------------------------------------------------*/
static int MFEMLinearARKSolve(void *arkode_mem, mfem::Solver* solve,
                              mfem::SundialsLinearSolveOperator* op)
{
   ARKodeMem ark_mem;

   MFEM_VERIFY(arkode_mem != NULL, "ARKODE memory error!");
   ark_mem = (ARKodeMem) arkode_mem;

   if (ark_mem->ark_lfree != NULL) { ark_mem->ark_lfree(ark_mem); }

   // Set four main function fields in ark_mem
   ark_mem->ark_linit  = WrapLinearARKSolveInit;
   ark_mem->ark_lsetup = WrapLinearARKSolveSetup;
   ark_mem->ark_lsolve = WrapLinearARKSolve;
   ark_mem->ark_lfree  = WrapLinearARKSolveFree;
   ark_mem->ark_lsolve_type = 0;
   ark_mem->ark_linear = TRUE;
   ark_mem->ark_setupNonNull = 1;
   // forces arkode to call lsetup prior to every time it calls lsolve
   ark_mem->ark_msbp = 0;

   //void* for containing linear solver memory
   mfem::MFEMLinearSolverMemory* lmem = new mfem::MFEMLinearSolverMemory();
#ifndef MFEM_USE_MPI
   lmem->setup_y = new mfem::Vector();
   lmem->setup_f = new mfem::Vector();
   lmem->solve_y = new mfem::Vector();
   lmem->solve_yn = new mfem::Vector();
   lmem->solve_f = new mfem::Vector();
   lmem->solve_b = new mfem::Vector();
   lmem->vec_tmp = new mfem::Vector();
#else
   lmem->setup_y = new mfem::HypreParVector((N_VGetVector_ParHyp(
                                                ark_mem->ark_ycur)));
   lmem->setup_f = new mfem::HypreParVector((N_VGetVector_ParHyp(
                                                ark_mem->ark_ycur)));
   lmem->solve_y = new mfem::HypreParVector((N_VGetVector_ParHyp(
                                                ark_mem->ark_ycur)));
   lmem->solve_f = new mfem::HypreParVector((N_VGetVector_ParHyp(
                                                ark_mem->ark_ycur)));
   lmem->solve_b = new mfem::HypreParVector((N_VGetVector_ParHyp(
                                                ark_mem->ark_ycur)));
   lmem->vec_tmp = new mfem::HypreParVector((N_VGetVector_ParHyp(
                                                ark_mem->ark_ycur)));
#endif
   lmem->J_solve=solve;
   lmem->op_for_gradient= op;

   ark_mem->ark_lmem = lmem;
   return (ARKSPILS_SUCCESS);
}

#endif
