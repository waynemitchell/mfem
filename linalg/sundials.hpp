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
#include "solvers.hpp"

#include <cvode/cvode.h>
#include <arkode/arkode.h>
#include <kinsol/kinsol.h>

namespace mfem
{

/// Abstract base class, wrapping the custom linear solvers interface in
/// SUNDIALS' CVODE and ARKODE solvers.
class SundialsLinearSolver
{
public:
   enum {CVODE, ARKODE} type; ///< Is CVODE or ARKODE using this object?

protected:
   SundialsLinearSolver() { }
   virtual ~SundialsLinearSolver() { }

   /// Get the current scaled time step, gamma, from @a sundials_mem.
   double GetTimeStep(void *sundials_mem);
   /// Get the TimeDependentOperator associated with @a sundials_mem.
   TimeDependentOperator *GetTimeDependentOperator(void *sundials_mem);

public:
   /** @name Linear solver interface methods.
       These four functions and their parameters are explained in Section 7 of
       http://computation.llnl.gov/sites/default/files/public/cv_guide.pdf
       or Section 7.4 of
       http://computation.llnl.gov/sites/default/files/public/ark_guide.pdf

       The first argument, @a sundials_mem, is one of the pointer types,
       CVodeMem or ARKodeMem, depending on the value of the data member @a type.
   */
   ///@{
   virtual int InitSystem(void *sundials_mem) = 0;
   virtual int SetupSystem(void *sundials_mem, int conv_fail,
                           Vector &y_pred, Vector &f_pred, int &jac_cur,
                           Vector &v_temp1, Vector &v_temp2,
                           Vector &v_temp3) = 0;
   virtual int SolveSystem(void *sundials_mem, Vector &b, Vector &weight,
                           Vector &y_cur, Vector &f_cur) = 0;
   virtual int FreeSystem(void *sundials_mem) = 0;
   ///@}
};

class SundialsSolver
{
protected:
   void *sundials_mem;
   mutable int flag;

   N_Vector y;
#ifdef MFEM_USE_MPI
   bool Parallel() const
   { return (y->ops->nvgetvectorid != N_VGetVectorID_Serial); }
#else
   bool Parallel() const { return false; }
#endif

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

   /// Return the flag returned by the last call to a SUNDIALS function.
   int GetFlag() const { return flag; }
};

/// Wrapper for the CVODE library.
/// http://computation.llnl.gov/sites/default/files/public/cv_guide.pdf
class CVODESolver : public ODESolver, public SundialsSolver
{
public:
   /** Construct a serial CVODESolver, a wrapper for SUNDIALS' CVODE solver.
       @param[in] lmm   Specifies the linear multistep method, the options are
                        CV_ADAMS (explicit problems) or CV_BDF (implicit
                        problems).
       @param[in] iter  Specifies type of solver iteration, the options are
                        CV_FUNCTIONAL (linear problems) or CV_NEWTON (nonlinear
                        problems).
       For parameter desciption, see the CVodeCreate documentation (cvode.h). */
   CVODESolver(int lmm, int iter);

#ifdef MFEM_USE_MPI
   /** Construct a parallel CVODESolver, a wrapper for SUNDIALS' CVODE solver.
       @param[in] comm  The MPI communicator used to partition the ODE system.
       @param[in] lmm   Specifies the linear multistep method, the options are
                        CV_ADAMS (explicit problems) or CV_BDF (implicit
                        problems).
       @param[in] iter  Specifies type of solver iteration, the options are
                        CV_FUNCTIONAL (linear problems) or CV_NEWTON (nonlinear
                        problems).
       For parameter desciption, see the CVodeCreate documentation (cvode.h). */
   CVODESolver(MPI_Comm comm, int lmm, int iter);
#endif

   /** Specify the scalar relative tolerance and scalar absolute tolerance.
       @note This method can be called before Init(). */
   void SetSStolerances(double reltol, double abstol);

   /// Defines a custom Jacobian inversion for non-linear problems.
   void SetLinearSolve(SundialsLinearSolver &ls_spec);

   /** @brief CVode supports two modes, specified by itask: CV_NORMAL (default)
       and CV_ONE_STEP.

       In the CV_NORMAL mode, the solver steps until it reaches or passes
       tout = t + dt, where t and dt are specified in Step(), and then
       interpolates to obtain y(tout). In the CV_ONE_STEP mode, it takes one
       internal step and returns. */
   void SetStepMode(int itask);

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
   virtual ~CVODESolver();
};

/// Wraps the ARKODE library.
/// http://computation.llnl.gov/sites/default/files/public/ark_guide.pdf
class ARKODESolver: public ODESolver, public SundialsSolver
{
protected:
   bool use_implicit;
   int irk_table, erk_table;

public:
   /** Construct a serial ARKODESolver, a wrapper for SUNDIALS' ARKODE solver.
       @param[in] implicit  Specifies if the time integrator is implicit. */
   ARKODESolver(bool implicit = false);

#ifdef MFEM_USE_MPI
   /** Construct a serial ARKODESolver, a wrapper for SUNDIALS' ARKODE solver.
       @param[in] implicit  Specifies if the time integrator is implicit. */
   ARKODESolver(MPI_Comm comm, bool implicit = false);
#endif

   /** Specify the scalar relative tolerance and scalar absolute tolerance.
       @note This method can be called before Init(). */
   void SetSStolerances(realtype reltol, realtype abstol);

   /// Defines a custom Jacobian inversion for non-linear problems.
   void SetLinearSolve(SundialsLinearSolver &ls_spec);

   /** @brief ARKode supports two modes, specified by itask:
       ARK_NORMAL(default) and ARK_ONE_STEP.

       In the ARK_NORMAL mode, the solver steps until it reaches or passes
       tout = t + dt, where t and dt are specified in Step(), and then
       interpolates to obtain y(tout). In the ARK_ONE_STEP mode, it takes one
       internal step and returns. */
   void SetStepMode(int itask);

   /** Chooses integration order for all explicit / implicit / IMEX methods.
       The default is 4, and the allowed ranges are:
       [2, 8] for explicit; [2, 5] for implicit; [3, 5] for IMEX. */
   void SetOrder(int order);

   /** Chooses a specific Butcher table for an explicit or implicit RK method.
       See the documentation for all possible options, stability regions, etc.
       For example, table_num = ARK548L2SA_DIRK_8_4_5 is 8-stage 5th order. */
   void SetIRKTableNum(int table_num);
   void SetERKTableNum(int table_num);

   /** Specifies to use a fixed time step size instead of performing any form
       of temporal adaptivity. Use  of  this  function  is  not  recommended,
       since may there is no assurance  of  the  validity  of  the  computed
       solutions. It is primarily provided for code-to-code verification
       testing purposes. */
   void SetFixedStep(double dt);

   /** Set the ODE right-hand-side operator.
       The start time of ARKODE is initialized from the current time of @a f_.
       @note This method calls ARKodeInit(). Some ARKODE parameters can be set
       (using the handle returned by SundialsMem()) only after this call. */
   virtual void Init(TimeDependentOperator &f_);

   /** Uses ARKODE to integrate over [t, t + dt], using the specified step mode.
       Calls ARKode(), which is the main driver of the ARKODE package.
       @param[in,out] x  Solution vector to advance. On input/output x=x(t)
                         for t corresponding to the input/output value of t,
                         respectively.
       @param[in,out] t  Input: the starting time value. Output: the time value
                         of the solution output, as returned by CVode().
       @param[in,out] dt Input: desired time step. Output: the last incremental
                         time step used. */
   virtual void Step(Vector &x, double &t, double &dt);

   /// Print ARKODE statistics.
   void PrintInfo() const;

   /// Destroys the associated ARKODE memory.
   virtual ~ARKODESolver();
};

/// Wraps the KINSOL library.
/// http://computation.llnl.gov/sites/default/files/public/kin_guide.pdf
class KinSolver : public NewtonSolver, public SundialsSolver
{
private:
   bool use_oper_grad;
   mutable N_Vector y_scale, f_scale;

public:
   /** Construct a serial KinSolver, a wrapper for SUNDIALS' KINSOL solver.

       @param[in] strategy   Specifies the nonlinear solver strategy:
                             KIN_NONE / KIN_LINESEARCH / KIN_PICARD / KIN_FP.
       @param[in] oper_grad  Specifies whether the solver should use its
                             Operator's GetGradient() method to compute action
                             of the system's Jacobian. */
   KinSolver(int strategy, bool oper_grad = true);

#ifdef MFEM_USE_MPI
   /** Construct a parallel KinSolver, a wrapper for SUNDIALS' KINSOL solver.

       @param[in] strategy   Specifies the nonlinear solver strategy:
                             KIN_NONE / KIN_LINESEARCH / KIN_PICARD / KIN_FP.
       @param[in] oper_grad  Specifies whether the solver should use its
                             Operator's GetGradient() method to compute action
                             of the system's Jacobian. */
   KinSolver(MPI_Comm comm, int strategy, bool oper_grad = true);
#endif

   virtual ~KinSolver();

   virtual void SetSolver(Solver &solver)
   { MFEM_ABORT("This option is not implemented in class KinSolver yet."); }
   virtual void SetPreconditioner(Solver &pr)
   { MFEM_ABORT("This option is not implemented in class KinSolver yet."); }

   /** Sets the problem size, the action of the nonlinear system and
       the action of its Jacobian. This method calls KINInit(). */
   virtual void SetOperator(const Operator &op);

   /** Corresponds to the NewtonSolver's interface; calls the other
       Mult() method with u_scale = f_scale = 1.
       @note The parameter @a b is not used, KINSol always assumes no RHS. */
   virtual void Mult(const Vector &b, Vector &x) const;

   /** @brief Calls KINSol() to solve the nonlinear system.

       Before calling KINSol(), this functions uses its fields, see class
       IterativeSolver, to set various KINSOL options.
       @note The functions SetRelTol() and SetAbsTol() influence KINSOL's
       "scaled step" and "function norm" tolerances, respectively. */
   void Mult(Vector &x, Vector &x_scale, Vector &fx_scale) const;
};

}  // namespace mfem

#endif // MFEM_USE_SUNDIALS

#endif // MFEM_SUNDIALS
