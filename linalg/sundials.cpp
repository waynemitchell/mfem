// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#include "sundials.hpp"

#ifdef MFEM_USE_SUNDIALS

#include "solvers.hpp"
#ifdef MFEM_USE_MPI
#include "hypre.hpp"
#endif

#include <nvector/nvector_serial.h>
#ifdef MFEM_USE_MPI
#include <nvector/nvector_parallel.h>
#include <nvector/nvector_parhyp.h>
#endif

#include <cvode/cvode_spgmr.h>

#include <arkode/arkode_spgmr.h>

#include <kinsol/kinsol.h>
#include <kinsol/kinsol_spgmr.h>


using namespace std;

namespace mfem
{

static inline SundialsLinearSolver *get_spec(void *ptr)
{
   return static_cast<SundialsLinearSolver *>(ptr);
}

static int LinSysInit(CVodeMem cv_mem)
{
   return get_spec(cv_mem->cv_lmem)->InitSystem(cv_mem);
}

static int LinSysSetup(CVodeMem cv_mem, int convfail,
                       N_Vector ypred, N_Vector fpred, booleantype *jcurPtr,
                       N_Vector vtemp1, N_Vector vtemp2, N_Vector vtemp3)
{
   Vector yp(ypred), fp(fpred), vt1(vtemp1), vt2(vtemp2), vt3(vtemp3);
   return get_spec(cv_mem->cv_lmem)->SetupSystem(cv_mem, convfail, yp, fp,
                                                 *jcurPtr, vt1, vt2, vt3);
}

static int LinSysSolve(CVodeMem cv_mem, N_Vector b, N_Vector weight,
                       N_Vector ycur, N_Vector fcur)
{
   Vector bb(b), w(weight), yc(ycur), fc(fcur);
   return get_spec(cv_mem->cv_lmem)->SolveSystem(cv_mem, bb, w, yc, fc);
}

static int LinSysFree(CVodeMem cv_mem)
{
   return get_spec(cv_mem->cv_lmem)->FreeSystem(cv_mem);
}

static int LinSysInit(ARKodeMem ark_mem)
{
   return get_spec(ark_mem->ark_lmem)->InitSystem(ark_mem);
}

static int LinSysSetup(ARKodeMem ark_mem, int convfail,
                       N_Vector ypred, N_Vector fpred, booleantype *jcurPtr,
                       N_Vector vtemp1, N_Vector vtemp2, N_Vector vtemp3)
{
   Vector yp(ypred), fp(fpred), vt1(vtemp1), vt2(vtemp2), vt3(vtemp3);
   return get_spec(ark_mem->ark_lmem)->SetupSystem(ark_mem, convfail, yp, fp,
                                                   *jcurPtr, vt1, vt2, vt3);
}

static int LinSysSolve(ARKodeMem ark_mem, N_Vector b, N_Vector weight,
                       N_Vector ycur, N_Vector fcur)
{
   Vector bb(b), w(weight), yc(ycur), fc(fcur);
   return get_spec(ark_mem->ark_lmem)->SolveSystem(ark_mem, bb, w, yc, fc);
}

static int LinSysFree(ARKodeMem ark_mem)
{
   return get_spec(ark_mem->ark_lmem)->FreeSystem(ark_mem);
}


const double SundialsSolver::default_rel_tol = 1e-4;
const double SundialsSolver::default_abs_tol = 1e-9;

// static method
int SundialsSolver::ODEMult(realtype t, N_Vector y,
                            N_Vector ydot, void *td_oper)
{
   Vector mfem_y(y), mfem_ydot(ydot);

   // Compute y' = f(t, y).
   TimeDependentOperator *f = static_cast<TimeDependentOperator *>(td_oper);
   f->SetTime(t);
   f->Mult(mfem_y, mfem_ydot);
   return 0;
}

// static method
int SundialsSolver::Mult(N_Vector u, N_Vector fu, void *oper)
{
   Vector mfem_u(u), mfem_fu(fu);

   // Computes the non-linear action F(u).
   static_cast<Operator *>(oper)->Mult(mfem_u, mfem_fu);
   return 0;
}

// static method
int SundialsSolver::GradientMult(N_Vector v, N_Vector Jv, N_Vector u,
                                 booleantype *new_u, void *oper)
{
   Vector mfem_u(u), mfem_v(v), mfem_Jv(Jv);

   Operator &J = static_cast<Operator *>(oper)->GetGradient(mfem_u);
   J.Mult(mfem_v, mfem_Jv);
   return 0;
}

static inline CVodeMem Mem(const CVODESolver *self)
{
   return CVodeMem(self->SundialsMem());
}

void CVODESolver::SetStepMode(int itask)
{
   Mem(this)->cv_taskc = itask;
}

CVODESolver::CVODESolver(int lmm, int iter)
{
   // Allocate an empty serial N_Vector wrapper in y.
   y = N_VNewEmpty_Serial(0);
   MFEM_ASSERT(y, "error in N_VNew_Serial()");

   // Create the solver memory.
   sundials_mem = CVodeCreate(lmm, iter);
   MFEM_ASSERT(sundials_mem, "error in CVodeCreate()");

   SetStepMode(CV_NORMAL);

   // Replace the zero defaults with some positive numbers.
   SetSStolerances(default_rel_tol, default_abs_tol);

   flag = CV_SUCCESS;
}

#ifdef MFEM_USE_MPI

CVODESolver::CVODESolver(MPI_Comm comm, int lmm, int iter)
{
   if (comm == MPI_COMM_NULL)
   {
      // Allocate an empty serial N_Vector wrapper in y.
      y = N_VNewEmpty_Serial(0);
      MFEM_ASSERT(y, "error in N_VNew_Serial()");
   }
   else
   {
      // Allocate an empty parallel N_Vector wrapper in y.
      y = N_VNewEmpty_Parallel(comm, 0, 0); // calls MPI_Allreduce()
      MFEM_ASSERT(y, "error in N_VNewEmpty_Parallel()");
   }

   // Create the solver memory.
   sundials_mem = CVodeCreate(lmm, iter);
   MFEM_ASSERT(sundials_mem, "error in CVodeCreate()");

   SetStepMode(CV_NORMAL);

   // Replace the zero defaults with some positive numbers.
   SetSStolerances(default_rel_tol, default_abs_tol);

   flag = CV_SUCCESS;
}

#endif // MFEM_USE_MPI

void CVODESolver::SetLinearSolve(SundialsLinearSolver &ls_spec)
{
   CVodeMem mem = Mem(this);
   MFEM_ASSERT(mem->cv_iter == CV_NEWTON,
               "The function is applicable only to CV_NEWTON iteration type.");

   if (mem->cv_lfree != NULL) { (mem->cv_lfree)(mem); }

   // Set the linear solver function fields in mem.
   // Note that {linit,lsetup,lfree} can be NULL.
   mem->cv_linit  = LinSysInit;
   mem->cv_lsetup = LinSysSetup;
   mem->cv_lsolve = LinSysSolve;
   mem->cv_lfree  = LinSysFree;
   mem->cv_lmem   = &ls_spec;
   mem->cv_setupNonNull = TRUE;
}

void CVODESolver::SetSStolerances(double reltol, double abstol)
{
   CVodeMem mem = Mem(this);
   // For now store the values in cv_mem:
   mem->cv_reltol = reltol;
   mem->cv_Sabstol = abstol;
   // The call to CVodeSStolerances() is done after CVodeInit() in Step()
}

void CVODESolver::SetMaxOrder(int max_order)
{
   flag = CVodeSetMaxOrd(sundials_mem, max_order);
   if (flag == CV_ILL_INPUT)
   {
      MFEM_WARNING("CVodeSetMaxOrd() did not change the maximum order!");
   }
}

static inline void cvCopyInit(CVodeMem src, CVodeMem dest)
{
   dest->cv_linit  = src->cv_linit;
   dest->cv_lsetup = src->cv_lsetup;
   dest->cv_lsolve = src->cv_lsolve;
   dest->cv_lfree  = src->cv_lfree;
   dest->cv_lmem   = src->cv_lmem;
   dest->cv_setupNonNull = src->cv_setupNonNull;

   dest->cv_reltol  = src->cv_reltol;
   dest->cv_Sabstol = src->cv_Sabstol;

   dest->cv_taskc = src->cv_taskc;
   dest->cv_qmax = src->cv_qmax;
}

void CVODESolver::Init(TimeDependentOperator &f_)
{
   CVodeMem mem = Mem(this);
   CVodeMemRec backup;

   if (mem->cv_MallocDone == TRUE)
   {
      // TODO: preserve more options.
      cvCopyInit(mem, &backup);
      CVodeFree(&sundials_mem);
      sundials_mem = CVodeCreate(backup.cv_lmm, backup.cv_iter);
      MFEM_ASSERT(sundials_mem, "error in CVodeCreate()");
      cvCopyInit(&backup, mem);
   }

   ODESolver::Init(f_);

   // Set actual size and data in the N_Vector y.
   int loc_size = f_.Height();
   if (!Parallel())
   {
      NV_LENGTH_S(y) = loc_size;
      NV_DATA_S(y) = new double[loc_size](); // value-initizalize
   }
   else
   {
#ifdef MFEM_USE_MPI
      long local_size = loc_size, global_size;
      MPI_Allreduce(&local_size, &global_size, 1, MPI_LONG, MPI_SUM,
                    NV_COMM_P(y));
      NV_LOCLENGTH_P(y) = local_size;
      NV_GLOBLENGTH_P(y) = global_size;
      NV_DATA_P(y) = new double[loc_size](); // value-initalize
#endif
   }

   // Call CVodeInit().
   cvCopyInit(mem, &backup);
   flag = CVodeInit(mem, ODEMult, f_.GetTime(), y);
   MFEM_ASSERT(flag >= 0, "CVodeInit() failed!");
   cvCopyInit(&backup, mem);

   // Delete the allocated data in y.
   if (!Parallel())
   {
      delete [] NV_DATA_S(y);
      NV_DATA_S(y) = NULL;
   }
   else
   {
#ifdef MFEM_USE_MPI
      delete [] NV_DATA_P(y);
      NV_DATA_P(y) = NULL;
#endif
   }

   // The TimeDependentOperator pointer, f, will be the user-defined data.
   flag = CVodeSetUserData(sundials_mem, f);
   MFEM_ASSERT(flag >= 0, "CVodeSetUserData() failed!");

   flag = CVodeSStolerances(mem, mem->cv_reltol, mem->cv_Sabstol);
   MFEM_ASSERT(flag >= 0, "CVodeSStolerances() failed!");
}

void CVODESolver::Step(Vector &x, double &t, double &dt)
{
   CVodeMem mem = Mem(this);

   if (!Parallel())
   {
      NV_DATA_S(y) = x.GetData();
      MFEM_VERIFY(NV_LENGTH_S(y) == x.Size(), "");
   }
   else
   {
#ifdef MFEM_USE_MPI
      NV_DATA_P(y) = x.GetData();
      MFEM_VERIFY(NV_LOCLENGTH_P(y) == x.Size(), "");
#endif
   }

   if (mem->cv_nst == 0)
   {
      // Set default linear solver, if not already set.
      if (mem->cv_iter == CV_NEWTON && mem->cv_lsolve == NULL)
      {
         flag = CVSpgmr(sundials_mem, PREC_NONE, 0);
      }
      // Set the actual t0 and y0.
      mem->cv_tn = t;
      N_VScale(ONE, y, mem->cv_zn[0]);
   }

   double tout = t + dt;
   // The actual time integration.
   flag = CVode(sundials_mem, tout, y, &t, mem->cv_taskc);
   MFEM_ASSERT(flag >= 0, "CVode() failed!");

   // Return the last incremental step size.
   dt = mem->cv_hu;
}

void CVODESolver::PrintInfo() const
{
   CVodeMem mem = Mem(this);

   cout <<
        "CVODE:\n  "
        "num steps: " << mem->cv_nst << ", "
        "num evals: " << mem->cv_nfe << ", "
        "num lin setups: " << mem->cv_nsetups << ", "
        "num nonlin sol iters: " << mem->cv_nni << "\n  "
        "last order: " << mem->cv_qu << ", "
        "next order: " << mem->cv_next_q << ", "
        "last dt: " << mem->cv_hu << ", "
        "next dt: " << mem->cv_next_h
        << endl;
}

CVODESolver::~CVODESolver()
{
   N_VDestroy(y);
   CVodeFree(&sundials_mem);
}

static inline ARKodeMem Mem(const ARKODESolver *self)
{
   return ARKodeMem(self->SundialsMem());
}

ARKODESolver::ARKODESolver(Vector &y_, bool parallel, bool explicit_)
   : use_explicit(explicit_)
{
   // Create the NVector y.
   y = y_.ToNVector();

   // Create the solver memory.
   sundials_mem = ARKodeCreate();

   // Initialize the integrator memory, specify the user's
   // RHS function in x' = f(t, x), the initial time, initial condition.
   int flag = use_explicit ?
              ARKodeInit(sundials_mem, ODEMult, NULL, 0.0, y) :
              ARKodeInit(sundials_mem, NULL, ODEMult, 0.0, y);
   MFEM_ASSERT(flag >= 0, "ARKodeInit() failed!");

   SetSStolerances(default_rel_tol, default_abs_tol);

   // When implicit method is chosen, one should specify the linear solver.
   if (use_explicit == false)
   {
      ARKSpgmr(sundials_mem, PREC_NONE, 0);
   }
}

void ARKODESolver::Init(TimeDependentOperator &f_)
{
   f = &f_;
   // Set the pointer to user-defined data.
   int flag = ARKodeSetUserData(sundials_mem, this->f);
   MFEM_ASSERT(flag >= 0, "ARKodeSetUserData() failed!");
}

void ARKODESolver::ReInit(TimeDependentOperator &f_, Vector &y_, double &t_)
{
   f = &f_;
   y_.ToNVector(y);

   // Re-init memory, time and solution.
   int flag = use_explicit ?
              ARKodeReInit(sundials_mem, ODEMult, NULL, t_, y) :
              ARKodeReInit(sundials_mem, NULL, ODEMult, t_, y);
   MFEM_ASSERT(flag >= 0, "ARKodeReInit() failed!");

   // Set the pointer to user-defined data.
   flag = ARKodeSetUserData(sundials_mem, this->f);
   MFEM_ASSERT(flag >= 0, "ARKodeSetUserData() failed!");

   // When implicit method is chosen, one should specify the linear solver.
   if (use_explicit == false)
   {
      ARKSpgmr(sundials_mem, PREC_NONE, 0);
   }
}

void ARKODESolver::SetSStolerances(realtype reltol, realtype abstol)
{
   // Specify the scalar relative tolerance and scalar absolute tolerance.
   int flag = ARKodeSStolerances(sundials_mem, reltol, abstol);
   MFEM_ASSERT(flag >= 0, "ARKodeSStolerances() failed!");
}

void ARKODESolver::Step(Vector &x, double &t, double &dt)
{
   x.ToNVector(y);

   realtype tout = t + dt;

   // Don't allow stepping over tout (then it interpolates in time).
   ARKodeSetStopTime(sundials_mem, tout);
   // Step.
   // ARK_NORMAL - take many steps until reaching tout.
   // ARK_ONE_STEP - take one step and return (might be before tout).
   int flag = ARKode(sundials_mem, tout, y, &t, ARK_NORMAL);
   MFEM_ASSERT(flag >= 0, "ARKode() failed!");

   // Record last incremental step size.
   flag = ARKodeGetLastStep(sundials_mem, &dt);
}

void ARKODESolver::SetOrder(int order)
{
   int flag = ARKodeSetOrder(sundials_mem, order);
   if (flag == ARK_ILL_INPUT)
   {
      MFEM_WARNING("ARKodeSetOrder() did not change the order!");
   }
}

void ARKODESolver::SetERKTableNum(int table_num)
{
   ARKodeSetERKTableNum(sundials_mem, table_num);
}

void ARKODESolver::SetIRKTableNum(int table_num)
{
   ARKodeSetIRKTableNum(sundials_mem, table_num);
}

void ARKODESolver::SetFixedStep(double dt)
{
   ARKodeSetFixedStep(sundials_mem, static_cast<realtype>(dt));
}

void ARKODESolver::SetLinearSolve(SundialsLinearSolver &ls_spec)
{
   ARKodeMem mem = Mem(this);
   MFEM_VERIFY(mem->ark_implicit,
               "The function is applicable only to implicit time integration.");

   if (mem->ark_lfree != NULL) { mem->ark_lfree(mem); }

   // Tell ARKODE that the Jacobian inversion is custom.
   mem->ark_lsolve_type = 4;
   // Set the linear solver function fields in mem.
   // Note that {linit,lsetup,lfree} can be NULL.
   mem->ark_linit  = LinSysInit;
   mem->ark_lsetup = LinSysSetup;
   mem->ark_lsolve = LinSysSolve;
   mem->ark_lfree  = LinSysFree;
   mem->ark_lmem   = &ls_spec;
   mem->ark_setupNonNull = TRUE;
}

ARKODESolver::~ARKODESolver()
{
   if (y) { N_VDestroy(y); }
   if (sundials_mem) { ARKodeFree(&sundials_mem); }
}

KinSolver::KinSolver(Operator &oper, Vector &mfem_u,
                     bool parallel, bool use_oper_grad)
{
   sundials_mem = KINCreate();

   u = mfem_u.ToNVector();
   KINInit(sundials_mem, SundialsSolver::Mult, u);

   // Set void pointer to user data.
   KINSetUserData(sundials_mem, static_cast<void *>(&oper));

   // Set scaled preconditioned GMRES linear solver.
   KINSpgmr(sundials_mem, 0);

   // Define the Jacobian action.
   if (use_oper_grad)
   {
      KINSpilsSetJacTimesVecFn(sundials_mem, GradientMult);
   }
}

void KinSolver::SetPrintLevel(int level)
{
   KINSetPrintLevel(sundials_mem, level);
}

void KinSolver::SetFuncNormTol(double tol)
{
   KINSetFuncNormTol(sundials_mem, tol);
}

void KinSolver::SetScaledStepTol(double tol)
{
   KINSetScaledStepTol(sundials_mem, tol);
}

void KinSolver::Solve(Vector &mfem_u,
                      Vector &mfem_u_scale, Vector &mfem_f_scale)
{
   mfem_u.ToNVector(u);
   if (!u_scale)
   {
      u_scale = mfem_u_scale.ToNVector();
      f_scale = mfem_f_scale.ToNVector();
   }
   else
   {
      mfem_u_scale.ToNVector(u_scale);
      mfem_f_scale.ToNVector(f_scale);
   }

   // [VAD] Make 'strategy' a settable parameter.
   // LINESEARCH might be fancier, but more fragile near convergence.
   int strategy = KIN_LINESEARCH;
   //   int strategy = KIN_NONE;
   int flag = KINSol(sundials_mem, u, strategy, u_scale, f_scale);
   MFEM_VERIFY(flag == KIN_SUCCESS || flag == KIN_INITIAL_GUESS_OK,
               "KINSol returned " << flag << " that indicated a problem!");
}

KinSolver::~KinSolver()
{
   if (u) { N_VDestroy(u); }
   if (u_scale) { N_VDestroy(u_scale); }
   if (f_scale) { N_VDestroy(f_scale); }

   if (sundials_mem) { KINFree(&sundials_mem); }
}

} // namespace mfem

#endif
