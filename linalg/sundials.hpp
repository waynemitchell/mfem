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


class SundialsSolver
{
protected:
   void *sundials_mem;

   static const double default_rel_tol;
   static const double default_abs_tol;

   // Computes the action of a time-dependent operator.
   static int ODEMult(realtype t, N_Vector y, N_Vector ydot, void *td_oper);

   // Computes the non-linear operator action F(u).
   static int Mult(N_Vector u, N_Vector fu, void *oper);

   // Computes J(u)v.
   // Here new_u tells you whether u has been updated since the
   // last call to KinSolJacAction.
   static int GradientMult(N_Vector v, N_Vector Jv, N_Vector u,
                           booleantype *new_u, void *oper);

   // Note: the contructors are protected.
   SundialsSolver() : sundials_mem(NULL) { }
   SundialsSolver(void *mem) : sundials_mem(mem) { }

public:
   /// Access the underlying SUNDIALS object.
   void *SundialsMem() const { return sundials_mem; }
};

/// Wrapper for the CVODE library.
/// http://computation.llnl.gov/sites/default/files/public/cv_guide.pdf
class CVODESolver : public ODESolver, public SundialsSolver
{
protected:
   N_Vector y;
   int flag;

#ifdef MFEM_USE_MPI
   bool Parallel() { return (y->ops->nvgetvectorid != N_VGetVectorID_Serial); }
#else
   bool Parallel() { return false; }
#endif

public:
   /// Construct a serial CVODESolver, a wrapper for SUNDIALS' CVODE solver.
   /** @param[in] lmm   Specifies the linear multistep method, the options are
                        CV_ADAMS (explicit problems) or CV_BDF (implicit
                        problems).
       @param[in] iter  Specifies type of solver iteration, the options are
                        CV_FUNCTIONAL (linear problems) or CV_NEWTON (nonlinear
                        problems).
       For parameter desciption, see the CVodeCreate documentation (cvode.h).
   */
   CVODESolver(int lmm, int iter);

#ifdef MFEM_USE_MPI
   /// Construct a parallel CVODESolver, a wrapper for SUNDIALS' CVODE solver.
   /** @param[in] comm  The MPI communicator used to partition the ODE system.
       @param[in] lmm   Specifies the linear multistep method, the options are
                        CV_ADAMS (explicit problems) or CV_BDF (implicit
                        problems).
       @param[in] iter  Specifies type of solver iteration, the options are
                        CV_FUNCTIONAL (linear problems) or CV_NEWTON (nonlinear
                        problems).
       For parameter desciption, see the CVodeCreate documentation (cvode.h).
   */
   CVODESolver(MPI_Comm comm, int lmm, int iter);
#endif

   /** @brief CVode supports two modes, specified by itask: CV_NORMAL (default)
       and CV_ONE_STEP.

       In the CV_NORMAL mode, the solver steps until it reaches or passes
       tout = t + dt, where t and dt are specified in Step(), and then
       interpolates to obtain y(tout). In the CV_ONE_STEP mode, it takes one
       internal step and returns.
   */
   void SetStepMode(int itask);

   /// Return the flag returned by the last call to a CVODE function.
   int GetFlag() const { return flag; }

   /// Specify the scalar relative tolerance and scalar absolute tolerance.
   /** @note This method can be called before Init(). */
   void SetSStolerances(double reltol, double abstol);

   /// Sets the maximum order of the linear multistep method.
   /** The default is 12 (CV_ADAMS) or 5 (CV_BDF).
       CVODE uses adaptive-order integration, based on the local truncation
       error. Use this if you know a-priori that your system is such that
       higher order integration formulas are unstable.
       @note @a max_order can't be higher than the current maximum order. */
   void SetMaxOrder(int max_order);

   /// Set the ODE right-hand-side operator.
   /** The start time of CVODE is initialized from the current time of @a f_.
       @note This method calls CVodeInit(). Some CVODE parameters can be set
       (using the handle returned by SundialsMem()) only after this call. */
   virtual void Init(TimeDependentOperator &f_);

   /// Defines a custom Jacobian inversion for non-linear problems.
   void SetLinearSolve(SundialsLinearSolveOperator *op);

   /// Uses CVODE to integrate over [t, t + dt], using the specified step mode.
   /** Calls CVode(), which is the main driver of the CVODE package.
       @param[in,out] x  Solution vector to advance. On input/output x=x(t)
                         for t corresponding to the input/output value of t,
                         respectively.
       @param[in,out] t  Input: the starting time value. Output: the time value
                         of the solution output, as returned by CVode().
       @param[in,out] dt Input: desired time step. Output: the last incremental
                         time step used. */
   virtual void Step(Vector &x, double &t, double &dt);

   /// Print CVODE statistics.
   void PrintInfo() const;

   /// Destroys the associated CVODE memory.
   ~CVODESolver();
};

/// Wraps the ARKODE library.
/// http://computation.llnl.gov/sites/default/files/public/ark_guide.pdf
class ARKODESolver: public ODESolver, public SundialsSolver
{
protected:
   N_Vector y;
   bool use_explicit;

public:
   /// ARKODE needs the initial condition (first argument).
   /// parallel specifies whether the calling code is parallel or not.
   /// exlicit_ specifies whether the time integration is explicit.
   ARKODESolver(Vector &y_, bool parallel, bool explicit_ = true);

   void Init(TimeDependentOperator &f_);

   /// Allows changing the operator, starting solution, current time.
   /// Note that the linear solver set previously remains in effect.
   void ReInit(TimeDependentOperator &f_, Vector &y_, double &t_);

   /// Note that this MUST be called before the first call to Step().
   void SetSStolerances(realtype reltol, realtype abstol);

   /// Uses ARKODE to integrate over (t, t + dt).
   /// Calls ARKODE(), which is the main driver of the CVODE package.
   void Step(Vector &x, double &t, double &dt);

   /// Defines a custom Jacobian inversion for non-linear problems.
   void SetLinearSolve(SundialsLinearSolveOperator *op);

   /// Chooses integration order for all explicit / implicit / IMEX methods.
   /// The default is 4, and the allowed ranges are:
   /// [2, 8] for explicit; [2, 5] for implicit; [3, 5] for IMEX.
   void SetOrder(int order);

   /// Chooses a specific Butcher table for an explicit or implicit RK method.
   /// See the documentation for all possible options, stability regions, etc.
   /// For example, table_num = ARK548L2SA_DIRK_8_4_5 is 8-stage 5th order.
   void SetERKTableNum(int table_num);
   void SetIRKTableNum(int table_num);

   /// Specifies to use a fixed time step size instead of performing any form
   /// of temporal adaptivity.
   /// Use  of  this  function  is  not  recommended, since may there is no
   /// assurance  of  the  validity  of  the  computed solutions.
   /// It is primarily provided for code-to-code verification testing purposes.
   void SetFixedStep(double dt);

   /// Destroys the associated ARKODE memory.
   ~ARKODESolver();
};

/// Wraps the KINSOL library.
/// http://computation.llnl.gov/sites/default/files/public/kin_guide.pdf
class KinSolver : public SundialsSolver
{
private:
   N_Vector u, u_scale, f_scale;

public:
   /// Specifies the initial condition and non-linear operator.
   /// parallel specifies whether the calling code is parallel or not.
   /// use_oper_grad = true will use oper.GetGradient() to determine
   /// the action of the Jacobian.
   KinSolver(Operator &oper, Vector &mfem_u,
             bool parallel, bool use_oper_grad = false);
   ~KinSolver();

   void SetPrintLevel(int level);
   void SetFuncNormTol(double tol);
   void SetScaledStepTol(double tol);

   void Solve(Vector &mfem_u,
              Vector &mfem_u_scale, Vector &mfem_f_scale);
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

}  // namespace mfem

#endif // MFEM_USE_SUNDIALS

#endif // MFEM_SUNDIALS
