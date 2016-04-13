#include "../config/config.hpp"
#ifdef MFEM_USE_SUNDIALS
#include "../general/tic_toc.hpp"
#include "../linalg/operator.hpp"
#include "../linalg/solvers.hpp"
#include "../linalg/linalg.hpp"
#include "sundials.hpp"
#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <typeinfo>
#include <exception>

#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <cvode/cvode.h>             /* prototypes for CVODE fcts., consts. */
#include <cvode/cvode_band.h>             /* prototypes for CVODE fcts., consts. */
#include <cvode/cvode_spgmr.h>
#include <arkode/arkode.h>             /* prototypes for ARKODE fcts., consts. */
#include <nvector/nvector_serial.h>  /* serial N_Vector types, fcts., macros */
#ifdef MFEM_USE_MPI
#include <nvector/nvector_parhyp.h>  /* parallel hypre N_Vector types, fcts., macros */
#endif
#include <sundials/sundials_band.h>  /* definitions of type DlsMat and macros */
#include <sundials/sundials_types.h> /* definition of type realtype */
#include <arkode/arkode_spils.h>
#include <arkode/arkode_impl.h>
#include <sundials/sundials_math.h>  /* definition of ABS and EXP */

/* Choose default tolerances to match ARKode defaults*/
#define RELTOL RCONST(1.0e-4)
#define ABSTOL RCONST(1.0e-9)

using namespace std;

static int WrapLinearCVSolveInit(CVodeMem cv_mem);

static int WrapLinearCVSolveSetup(CVodeMem cv_mem, int convfail,
                                  N_Vector ypred, N_Vector fpred,
                                  booleantype *jcurPtr, N_Vector vtemp1,
                                  N_Vector vtemp2, N_Vector vtemp3);

static int WrapLinearCVSolve(CVodeMem cv_mem, N_Vector b,
                             N_Vector weight, N_Vector ynow,
                             N_Vector fnow);

static void WrapLinearCVSolveFree(CVodeMem cv_mem);

/*
 The purpose of ark_linit is to complete initializations for a
 specific linear solver, such as counters and statistics.
 An LInitFn should return 0 if it has successfully initialized
 the ARKODE linear solver and a negative value otherwise.
 If an error does occur, an appropriate message should be sent
 to the error handler function.
 */
static int WrapLinearARKSolveInit(ARKodeMem ark_mem);

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
static int WrapLinearARKSolveSetup(ARKodeMem ark_mem, int convfail,
                                   N_Vector ypred, N_Vector fpred,
                                   booleantype *jcurPtr, N_Vector vtemp1,
                                   N_Vector vtemp2, N_Vector vtemp3);
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
                              N_Vector fcur);
/*
 ark_lfree should free up any memory allocated by the linear
 solver. This routine is called once a problem has been
 completed and the linear solver is no longer needed.
 */
static void WrapLinearARKSolveFree(ARKodeMem ark_mem);

static int WrapLinearSolveSetup(void* lmem, double tn,
                                mfem::Vector* ypred, mfem::Vector* fpred);

static int WrapLinearSolve(void* lmem, double tn, mfem::Vector* b,
                           mfem::Vector* ycur, mfem::Vector* yn,
                           mfem::Vector* fcur);

namespace mfem
{

CVODESolver::CVODESolver(Vector &_y, int lmm, int iter)
   : ODESolver(),
     initialized_sundials(false),
     tolerances_set_sundials(false),
     solver_iteration_type(iter)
{
   // Create the NVector y.
   long int yin_length = _y.Size();
   CreateNVector(yin_length, &_y);

   // Create the solver memory.
   ode_mem = CVodeCreate(lmm, iter);
}

#ifdef MFEM_USE_MPI
CVODESolver::CVODESolver(MPI_Comm _comm, Vector &_y, int lmm, int iter)
   : ODESolver(),
     comm(_comm),
     initialized_sundials(false),
     tolerances_set_sundials(false),
     solver_iteration_type(iter)
{
   // Create the NVector y.
   long int yin_length = _y.Size();
   CreateNVector(yin_length, &_y);

   // Create the solver memory.
   ode_mem = CVodeCreate(lmm, iter);
}
#endif

void CVODESolver::CreateNVector(long int& yin_length, Vector* _x)
{
#ifndef MFEM_USE_MPI
   // Create a serial NVector.
   y = N_VMake_Serial(yin_length,
                      (realtype*) _x->GetData());   /* Allocate y vector */
   MFEM_ASSERT(static_cast<void *>(y) != NULL, "N_VMake_Serial() failed!");
#else
   // Create a parallel NVector.
   HypreParVector *x = dynamic_cast<HypreParVector *>(_x);
   MFEM_ASSERT(x != NULL, "Could not cast to HypreParVector!");

   y = N_VMake_ParHyp(x->StealParVector());
#endif
}

void CVODESolver::TransferNVectorShallow(Vector* _x, N_Vector &_y)
{
#ifndef MFEM_USE_MPI
   NV_DATA_S(_y) = _x->GetData();
#else
   HypreParVector *x = dynamic_cast<HypreParVector *>(_x);
   MFEM_ASSERT(x != NULL, "Could not cast to HypreParVector!");

   y = N_VMake_ParHyp(x->StealParVector());
#endif
}

void CVODESolver::TransferNVectorShallow(N_Vector &_y, Vector* _x)
{
   _x->SetData(NV_DATA_S(_y));
}

void CVODESolver::DestroyNVector(N_Vector& _y)
{
   N_VDestroy(y);
}

void CVODESolver::Init(TimeDependentOperator &_f)
{
   f = &_f;

   // Initialize integrator memory, specify the user's
   // RHS function in x' = f(t,x), initial time, initial condition.
#ifndef MFEM_USE_MPI
   int flag = CVodeInit(ode_mem, sun_f_fun, 0.0, y);
#else
   int flag = CVodeInit(ode_mem, sun_f_fun_par, 0.0, y);
#endif
   MFEM_ASSERT(flag >= 0, "CVodeInit() failed!");

   initialized_sundials = true;
   SetSStolerances(RELTOL, ABSTOL);

   // Set the pointer to user-defined data.
   flag = CVodeSetUserData(ode_mem, f);
   MFEM_ASSERT(flag >= 0, "CVodeSetUserData() failed!");

   // TODO: what's going on with the tolerances. Should this be in Init()?
#ifndef MFEM_USE_MPI
   if (solver_iteration_type==CV_NEWTON)
   {
      SetSStolerances(1e-3,1e-6);
      CVBand(ode_mem, yin_length, yin_length*.5, yin_length*.5);
   }
#else
   if (solver_iteration_type==CV_NEWTON)
   {
      printf("Entering spgmr\n");
      SetSStolerances(1e-3,1e-6);
      CVSpgmr(ode_mem, PREC_NONE, 0);
   }
#endif
}

void CVODESolver::ReInit(TimeDependentOperator &_f, Vector &_y, double& _t)
{
   f = &_f;
   long int yin_length=_f.Width();
   // Create an NVector
   CreateNVector(yin_length, &_y);

   // Re-init memory, time and solution. The RHS action is known from Init().
   int flag = CVodeReInit(ode_mem, static_cast<realtype>(_t), y);
   MFEM_ASSERT(flag >= 0, "CVodeReInit() failed!");

   // Set the pointer to user-defined data.
   flag = CVodeSetUserData(ode_mem, this->f);
   MFEM_ASSERT(flag >= 0, "CVodeSetUserData() failed!");

   // TODO: what's going on with the tolerances. Should this be in Init()?
#ifndef MFEM_USE_MPI
   if (solver_iteration_type==CV_NEWTON)
   {
      SetSStolerances(1e-3,1e-6);
      CVBand(ode_mem, yin_length, yin_length*.5, yin_length*.5);
   }
#else
   if (solver_iteration_type==CV_NEWTON)
   {
      printf("Entering spgmr\n");
      SetSStolerances(1e-3,1e-6);
      CVSpgmr(ode_mem, PREC_NONE, 0);
   }
#endif
}

void CVODESolver::SetSStolerances(realtype reltol, realtype abstol)
{
   // Specify the scalar relative tolerance and scalar absolute tolerance.
   int flag = CVodeSStolerances(ode_mem, reltol, abstol);
   MFEM_ASSERT(flag >= 0, "CVodeSStolerances() failed!");

   tolerances_set_sundials = true;
}

void CVODESolver::SetLinearSolve(Solver* J_solve,
                                 SundialsLinearSolveOperator* op)
{
   //  N_Vector y0 = N_VClone_Serial( ((CVodeMem) ode_mem)->cv_zn[0]);
   //if linear solve should be newton, recreate ode_mem object
   //consider checking for CV_ADAMS vs CV_BDF as well
   if (solver_iteration_type==CV_FUNCTIONAL)
   {
      realtype t0= ((CVodeMem) ode_mem)->cv_tn;
      CVodeFree(&ode_mem);
      ode_mem=CVodeCreate(CV_BDF,CV_NEWTON);
      initialized_sundials=false;
      tolerances_set_sundials=false;

#ifndef MFEM_USE_MPI
   int flag = CVodeInit(ode_mem, sun_f_fun, t0, y);
#else
   int flag = CVodeInit(ode_mem, sun_f_fun_par, t0, y);
#endif
      MFEM_ASSERT(flag >= 0, "CVodeInit() failed!");

      CVodeSetUserData(ode_mem, this->f);
      if (!tolerances_set_sundials)
      {
         SetSStolerances(RELTOL,ABSTOL);
      }
   }
   /* Call CVodeSetMaxNumSteps to increase default */
   CVodeSetMaxNumSteps(ode_mem, 10000);
   SetSStolerances(1e-2,1e-4);

   MFEMLinearCVSolve(ode_mem, J_solve, op);

}
void CVODESolver::SetStopTime(double tf)
{
   CVodeSetStopTime(ode_mem, tf);
}

void CVODESolver::Step(Vector &x, double &t, double &dt)
{
   int flag = 0;
   realtype tout = t + dt;
   TransferNVectorShallow(&x, y);

   // Perform the step.
   flag = CVode(ode_mem, tout, y, &t, CV_NORMAL);
   MFEM_ASSERT(flag >= 0, "CVode() failed!");

   // Record last incremental step size.
   flag = CVodeGetLastStep(ode_mem, &dt);
}

CVODESolver::~CVODESolver()
{
   // Free the used memory.
   // Clean up and return with successful completion
   DestroyNVector(y);
   if (ode_mem!=NULL)
   {
      CVodeFree(&ode_mem);   // Free integrator memory
   }
}

ARKODESolver::ARKODESolver()
{
   y = NULL;
   f = NULL;

   /* Call ARKodeCreate to create the solver memory */
   ode_mem=ARKodeCreate();
   initialized_sundials=false;
   tolerances_set_sundials=false;
}

ARKODESolver::ARKODESolver(TimeDependentOperator &_f, Vector &_x, double&_t,
                           int _use_explicit)
{
   y = NULL;
   f = NULL;

   /* Call ARKodeCreate to create the solver memory */
   ode_mem=ARKodeCreate();
   initialized_sundials=false;
   tolerances_set_sundials=false;
   use_explicit=_use_explicit;
   ReInit(_f,_x,_t);
}

void ARKODESolver::CreateNVector(long int& yin_length, realtype* ydata)
{
   // Create a serial vector.
   y = N_VMake_Serial(yin_length,ydata);   /* Allocate y vector */
   MFEM_ASSERT(static_cast<void *>(y) != NULL, "N_VMake_Serial() failed!");
}

void ARKODESolver::CreateNVector(long int& yin_length, Vector* _x)
{

   // Create a serial vector
   y = N_VMake_Serial(yin_length,
                      (realtype*) _x->GetData());   /* Allocate y vector */
   MFEM_ASSERT(static_cast<void *>(y) != NULL, "N_VMake_Serial() failed!");
}

void ARKODESolver::TransferNVectorShallow(Vector* _x, N_Vector &_y)
{
   NV_DATA_S(_y)=_x->GetData();
}

void ARKODESolver::DestroyNVector(N_Vector& _y)
{

   if (NV_OWN_DATA_S(y)==true)
   {
      N_VDestroy_Serial(y);   // Free y vector
   }
}

int ARKODESolver::WrapARKodeInit(void* _ode_mem, double &_t, N_Vector &_y)
{
   return use_explicit ?
          ARKodeInit(_ode_mem, sun_f_fun, NULL, (realtype) _t, _y) :
          ARKodeInit(_ode_mem, NULL, sun_f_fun, (realtype) _t, _y);
}

int ARKODESolver::WrapARKodeReInit(void* _ode_mem, double &_t, N_Vector &_y)
{
   return use_explicit ?
          ARKodeReInit(_ode_mem, sun_f_fun, NULL, (realtype) _t, _y) :
          ARKodeReInit(_ode_mem, NULL, sun_f_fun, (realtype) _t, _y);
}

void ARKODESolver::Init(TimeDependentOperator &_f)
{
   //not checking that initial pointers set to NULL:
   f = &_f;
   long int yin_length=_f.Width(); //assume don't have initial condition in Init
   //intial time
   realtype t = 0.0;
   realtype *yin;
   yin= new realtype[yin_length];
   int flag;

   // Create an NVector
   CreateNVector(yin_length, yin);

   if (initialized_sundials)
   {
      /* Call ARKodeReInit to initialize the integrator memory and specify the inital time t,
       * and the initial dependent variable vector y. */
      flag = WrapARKodeReInit(ode_mem, t, y);
      MFEM_ASSERT(flag >= 0, "WrapARKodeReInit() failed!");
   }
   else
   {
      /* Call ARKodeInit to initialize the integrator memory and specify the
       * user's right hand side function in x'=f(t,x), the inital time t, and
       * the initial dependent variable vector y. */
      flag = WrapARKodeInit(ode_mem, t, y);
      MFEM_ASSERT(flag >= 0, "WrapARKodeInit() failed!");
      initialized_sundials = true;
      SetSStolerances(RELTOL, ABSTOL);
   }

   /* Set the pointer to user-defined data */
   flag = ARKodeSetUserData(ode_mem, this->f);
   MFEM_ASSERT(flag >= 0, "ARKodeSetUserData() failed!");
}

void ARKODESolver::ReInit(TimeDependentOperator &_f, Vector &_x, double& _t)
{
   //not checking that initial pointers set to NULL:
   f = &_f;
   long int yin_length=_f.Width(); //assume don't have initial condition in Init
   int flag;
   // Create an NVector
   CreateNVector(yin_length, &_x);

   if (initialized_sundials)
   {
      /* Call ARKodeReInit to initialize the integrator memory and specify the inital
       * time t, and the initial dependent variable vector y. */
      flag = WrapARKodeReInit(ode_mem, _t, y);
      MFEM_ASSERT(flag >= 0, "WrapARKodeReInit() failed!");
   }
   else
   {
      // Initialize the integrator memory and specify the
      // user's right hand side function in x'=f(t,x), the inital time t, and
      // the initial dependent variable vector y.
      flag = WrapARKodeInit(ode_mem, _t, y);
      MFEM_ASSERT(flag >= 0, "WrapARKodeInit() failed!");
      initialized_sundials = true;
      SetSStolerances(RELTOL,ABSTOL);
   }

   /* Set the pointer to user-defined data */
   flag = ARKodeSetUserData(ode_mem, this->f);
   MFEM_ASSERT(flag >= 0, "ARKodeSetUserData() failed!");
}

void ARKODESolver::SetSStolerances(realtype reltol, realtype abstol)
{
   int flag = 0;

   // Specify the scalar relative tolerance and scalar absolute tolerance.
   flag = ARKodeSStolerances(ode_mem, reltol, abstol);
   MFEM_ASSERT(flag >= 0, "ARKodeSStolerances() failed!");

   tolerances_set_sundials = true;
}

void ARKODESolver::WrapSetERKTableNum(int& table_num)
{
   ARKodeSetERKTableNum(ode_mem, table_num);
}

void ARKODESolver::WrapSetFixedStep(realtype dt)
{
   ARKodeSetFixedStep(ode_mem, dt);
}

void ARKODESolver::SetStopTime(double tf)
{
   ARKodeSetStopTime(ode_mem, tf);
}
void ARKODESolver::SetLinearSolve(Solver* solve,
                                  SundialsLinearSolveOperator* op)
{

   if (use_explicit)
   {
      realtype t0= ((ARKodeMem) ode_mem)->ark_tn;
      ARKodeFree(&ode_mem);
      ode_mem=ARKodeCreate();
      initialized_sundials=false;
      tolerances_set_sundials=false;
      use_explicit=false;
      //change init structure in order to switch to implicit method
      WrapARKodeInit(ode_mem,t0,y);
      ARKodeSetUserData(ode_mem, this->f);
      if (!tolerances_set_sundials)
      {
         SetSStolerances(RELTOL,ABSTOL);
      }
   }
   //   ARKodeSetIRKTableNum(ode_mem, 21);
   /* Call ARKodeSetMaxNumSteps to increase default */
   ARKodeSetMaxNumSteps(ode_mem, 10000);
   SetSStolerances(1e-2,1e-4);
   MFEMLinearARKSolve(ode_mem, solve, op);

}
void ARKODESolver::Step(Vector &x, double &t, double &dt)
{
   int flag=0;
   realtype tout=t+dt;
   TransferNVectorShallow(&x,y);

   // Step.
   flag = ARKode(ode_mem, tout, y, &t, ARK_NORMAL);
   MFEM_ASSERT(flag >= 0, "ARKode() failed!");

   // Record last incremental step size.
   flag = ARKodeGetLastStep(ode_mem, &dt);
}

ARKODESolver::~ARKODESolver()
{
   // Free the used memory.
   // Clean up and return with successful completion
   DestroyNVector(y);
   if (ode_mem != NULL)
   {
      ARKodeFree(&ode_mem);
   }
}

#ifdef MFEM_USE_MPI
ARKODEParSolver::ARKODEParSolver(MPI_Comm _comm, TimeDependentOperator &_f,
                                 Vector &_x, double &_t, int _use_explicit)
{
   y = NULL;
   f = &_f;

   /* Call ARKodeCreate to create the solver memory */
   ode_mem = ARKodeCreate();
   initialized_sundials = false;
   tolerances_set_sundials = false;
   use_explicit = _use_explicit;
   comm = _comm;
   ReInit(_f,_x,_t);
}

void ARKODEParSolver::CreateNVector(long int& yin_length, Vector* _x)
{
   HypreParVector *x = dynamic_cast<HypreParVector *>(_x);
   MFEM_ASSERT(x != NULL, "Could not cast to HypreParVector!");

   y = N_VMake_ParHyp(x->StealParVector());
}

void ARKODEParSolver::TransferNVectorShallow(Vector* _x, N_Vector &_y)
{
   HypreParVector *x = dynamic_cast<HypreParVector *>(_x);
   MFEM_ASSERT(x != NULL, "Could not cast to HypreParVector!");

   y = N_VMake_ParHyp(x->StealParVector());
}

void ARKODEParSolver::DestroyNVector(N_Vector& _y)
{
   if (NV_OWN_DATA_PH(y)==true)
   {
      N_VDestroy_ParHyp(y);   // Free y vector
   }
}

int ARKODEParSolver::WrapARKodeInit(void* _ode_mem, double &_t, N_Vector &_y)
{
   // Assumes integrating TimeDependentOperator f explicitly.
   // Consider adding a flag to switch between explicit and implicit.
   return (use_explicit) ? ARKodeInit(_ode_mem, sun_f_fun_par, NULL,
                                      (realtype) _t, _y)
          : ARKodeInit(_ode_mem, NULL, sun_f_fun_par,
                       (realtype) _t, _y);
}

int ARKODEParSolver::WrapARKodeReInit(void* _ode_mem, double &_t, N_Vector &_y)
{
   //Assumes integrating TimeDependentOperator f explicitly
   //Consider adding a flag to switch between explicit and implicit
   return (use_explicit) ? ARKodeReInit(_ode_mem, sun_f_fun_par, NULL,
                                        (realtype) _t,_y)
          : ARKodeInit(_ode_mem, NULL, sun_f_fun_par,
                       (realtype) _t, _y);
}

#endif

}


int sun_f_fun(realtype t, N_Vector y, N_Vector ydot, void *user_data)
{

   realtype *ydata, *ydotdata;
   long int ylen, ydotlen;

   // ydata is now a pointer to the realtype data array in y
   ydata = NV_DATA_S(y);
   ylen = NV_LENGTH_S(y);

   // probably unnecessary, since overwriting ydot as output
   // ydotdata is now a pointer to the realtype data array in ydot
   ydotdata = NV_DATA_S(ydot);
   ydotlen = NV_LENGTH_S(ydot);

   mfem::TimeDependentOperator* f =
      static_cast<mfem::TimeDependentOperator*>(user_data);

   // Creates mfem Vectors linked to the data in y and in ydot.
   // Have not explicitly set as owndata, so allocated size is -size.
   mfem::Vector mfem_vector_y((double*) ydata, ylen);
   mfem::Vector mfem_vector_ydot((double*) ydotdata, ydotlen);

   // Apply y' = f(t, y).
   f->SetTime(t);
   f->Mult(mfem_vector_y, mfem_vector_ydot);

   return 0;
}

#ifdef MFEM_USE_MPI
int sun_f_fun_par(realtype t, N_Vector y, N_Vector ydot, void *user_data)
{
   mfem::TimeDependentOperator* f =
      static_cast<mfem::TimeDependentOperator*>(user_data);

   // Creates mfem HypreParVectors mfem_vector_y and mfem_vector_ydot by
   // using the casting operators in HypreParVector to cast the hypre_ParVector
   // in y and in ydot respectively.
   // Have not explicitly set as owndata, so allocated size is -size
   mfem::HypreParVector mfem_vector_y =
      static_cast<mfem::HypreParVector>(NV_HYPRE_PARVEC_PH(y));
   mfem::HypreParVector mfem_vector_ydot =
      static_cast<mfem::HypreParVector>(NV_HYPRE_PARVEC_PH(ydot));

   // Apply y' = f(t, y).
   f->SetTime(t);
   f->Mult(mfem_vector_y, mfem_vector_ydot);

   return 0;
}
#endif

/*---------------------------------------------------------------
 MFEMLinearCVSolve:

 This routine initializes the memory record and sets various
 function fields specific to the linear solver module.
 MFEMLinearCVSolve first calls the existing lfree routine if this is not
 NULL. It then sets the cv_linit, cv_lsetup, cv_lsolve,
 cv_lfree fields in (*cvode_mem) to be WrapLinearCVSolveInit,
 WrapLinearCVSolveSetup, WrapLinearCVSolve, and WrapLinearCVSolveFree, respectively.
---------------------------------------------------------------*/
int MFEMLinearCVSolve(void *ode_mem, mfem::Solver* solve,
                      mfem::SundialsLinearSolveOperator* op)
{
   //   cout<<"entered linearcvsolve"<<endl;
   CVodeMem cv_mem;

   // Return immediately if arkode_mem is NULL
   if (ode_mem == NULL) {MFEM_ABORT("arkode_mem is NULL") }
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
   //   cout<<"created setup_y"<<endl;
   lmem->setup_f = new mfem::Vector();
   lmem->solve_y = new mfem::Vector();
   lmem->solve_yn = new mfem::Vector();
   lmem->solve_f = new mfem::Vector();
   lmem->solve_b = new mfem::Vector();
   lmem->vec_tmp = new mfem::Vector(NV_LENGTH_S(cv_mem->cv_zn[0]));
#else
   lmem->setup_y = new mfem::HypreParVector((NV_HYPRE_PARVEC_PH(
                                                cv_mem->cv_zn[0])));
   lmem->setup_f = new mfem::HypreParVector((NV_HYPRE_PARVEC_PH(
                                                cv_mem->cv_zn[0])));
   lmem->solve_y = new mfem::HypreParVector((NV_HYPRE_PARVEC_PH(
                                                cv_mem->cv_zn[0])));
   lmem->solve_f = new mfem::HypreParVector((NV_HYPRE_PARVEC_PH(
                                                cv_mem->cv_zn[0])));
   lmem->solve_b = new mfem::HypreParVector((NV_HYPRE_PARVEC_PH(
                                                cv_mem->cv_zn[0])));
   lmem->vec_tmp = new mfem::HypreParVector((NV_HYPRE_PARVEC_PH(
                                                cv_mem->cv_zn[0])));
#endif

   lmem->J_solve=solve;
   lmem->op_for_gradient= op;

   cv_mem->cv_lmem = lmem;
   return (CVSPILS_SUCCESS);
}

static int WrapLinearCVSolveInit(CVodeMem cv_mem)
{
   return 0;
}

//Setup may not be needed, since Jacobian is recomputed each iteration
//ypred is the predicted y at the current time, fpred is f(t,ypred)
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
   lmem->setup_y->SetData(NV_DATA_PH(ypred));
   lmem->setup_f->SetData(NV_DATA_PH(fpred));
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
   ((mfem::HypreParVector*) lmem->solve_y)->SetDataAndSize(NV_DATA_PH(ycur),
                                                           NV_LOCLENGTH_PH(ycur));
   ((mfem::HypreParVector*) lmem->solve_f)->SetDataAndSize(NV_DATA_PH(fcur),
                                                           NV_LOCLENGTH_PH(fcur));
   ((mfem::HypreParVector*) lmem->solve_b)->SetDataAndSize(NV_DATA_PH(b),
                                                           NV_LOCLENGTH_PH(b));
#endif

   lmem->weight = cv_mem->cv_gamma;

   cv_mem->cv_lmem = lmem;
   WrapLinearSolve(cv_mem->cv_lmem, cv_mem->cv_tn, lmem->solve_b, lmem->solve_y,
                   lmem->setup_y, lmem->solve_f);
   return 0;
}


static void WrapLinearCVSolveFree(CVodeMem cv_mem)
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
 WrapLinearARKSolveSetup, WrapARKLinearSolve, and WrapLinearARKSolveFree, respectively.
---------------------------------------------------------------*/
int MFEMLinearARKSolve(void *arkode_mem, mfem::Solver* solve,
                       mfem::SundialsLinearSolveOperator* op)
{
   ARKodeMem ark_mem;

   // Return immediately if arkode_mem is NULL
   if (arkode_mem == NULL) {MFEM_ABORT("arkode_mem is NULL") }
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
   lmem->setup_y = new mfem::HypreParVector((NV_HYPRE_PARVEC_PH(
                                                ark_mem->ark_ycur)));
   lmem->setup_f = new mfem::HypreParVector((NV_HYPRE_PARVEC_PH(
                                                ark_mem->ark_ycur)));
   lmem->solve_y = new mfem::HypreParVector((NV_HYPRE_PARVEC_PH(
                                                ark_mem->ark_ycur)));
   lmem->solve_f = new mfem::HypreParVector((NV_HYPRE_PARVEC_PH(
                                                ark_mem->ark_ycur)));
   lmem->solve_b = new mfem::HypreParVector((NV_HYPRE_PARVEC_PH(
                                                ark_mem->ark_ycur)));
   lmem->vec_tmp = new mfem::HypreParVector((NV_HYPRE_PARVEC_PH(
                                                ark_mem->ark_ycur)));
#endif
   lmem->J_solve=solve;
   lmem->op_for_gradient= op;

   ark_mem->ark_lmem = lmem;
   return (ARKSPILS_SUCCESS);
}

static int WrapLinearARKSolveInit(ARKodeMem ark_mem)
{
   return 0;
}


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
   lmem->setup_y->SetData(NV_DATA_PH(ypred));
   lmem->setup_f->SetData(NV_DATA_PH(fpred));
#endif
   *jcurPtr=TRUE;
   ark_mem->ark_lmem = lmem;
   WrapLinearSolveSetup(ark_mem->ark_lmem, ark_mem->ark_tn, lmem->setup_y,
                        lmem->setup_f);
   return 0;
}

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
      ((mfem::HypreParVector*) lmem->solve_y)->SetDataAndSize(NV_DATA_PH(ycur),
                                                              NV_LOCLENGTH_PH(ycur));
      ((mfem::HypreParVector*) lmem->solve_f)->SetDataAndSize(NV_DATA_PH(fcur),
                                                              NV_LOCLENGTH_PH(fcur));
      ((mfem::HypreParVector*) lmem->solve_b)->SetDataAndSize(NV_DATA_PH(b),
                                                              NV_LOCLENGTH_PH(b));
#endif

      lmem->weight = ark_mem->ark_gamma;

      ark_mem->ark_lmem = lmem;
      WrapLinearSolve(ark_mem->ark_lmem, ark_mem->ark_tn, lmem->solve_b,
                      lmem->solve_y,
                      lmem->setup_y, lmem->solve_f);
   }
   return 0;
}


static void WrapLinearARKSolveFree(ARKodeMem ark_mem)
{
   return;
}

static int WrapLinearSolveSetup(void* lmem, double tn,
                                mfem::Vector* ypred, mfem::Vector* fpred)
{
   return 0;
}

static int WrapLinearSolve(void* lmem, double tn, mfem::Vector* b,
                           mfem::Vector* ycur, mfem::Vector* yn,
                           mfem::Vector* fcur)
{

   mfem::MFEMLinearSolverMemory* tmp_lmem= (mfem::MFEMLinearSolverMemory*) lmem;
   mfem::Solver* prec=tmp_lmem->J_solve;
   /*
   //Original plan:
   // Take J_solve and Operator which has the relevant GetGradient operation
   // and solve the system. In the simple case where GetGradient gives you
   // something similart to I-gamma*J
   prec->SetOperator((tmp_lmem->op_for_gradient)->GetGradient(*ycur));
   prec->Mult(*b, *yn);  // c = [DF(x_i)]^{-1} [F(x_i)-b]

   *b=*yn;*/

   (tmp_lmem->op_for_gradient)->SolveJacobian(b,ycur, yn, prec, tmp_lmem->weight);
   return 0;
}
#endif
