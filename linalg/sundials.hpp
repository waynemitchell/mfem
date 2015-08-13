#ifndef MFEM_SUNDIALS
#define MFEM_SUNDIALS
#include "mfem.hpp"
#include "../linalg/operator.hpp"
#include "../linalg/ode.hpp"
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../config/config.hpp"

// cvode header files
#include <cvode/cvode.h>             /* prototypes for CVODE fcts., consts. */
#include <arkode/arkode.h>           /* prototypes for ARKODE fcts., consts. */
#include <nvector/nvector_serial.h>  /* serial N_Vector types, fcts., macros */
#ifdef MFEM_USE_MPI
#include <nvector/nvector_parhyp.h>  /* parallel hypre N_Vector types, fcts., macros */
#endif
#include <sundials/sundials_types.h> /* definition of type realtype */
#include <sundials/sundials_math.h>  /* definition of ABS and EXP */
#ifdef MFEM_USE_MPI
#include <mpi.h>
#endif
namespace mfem
{
/// Wraps the CVode library of linear multistep methods
class CVODESolver: public ODESolver
{
protected:
   N_Vector y;
   void* ode_mem;
   int step_type;
   bool initialized_sundials;
   bool tolerances_set_sundials;

public:
   /** \brief This constructor wraps the CVodeCreate function, and initializes pointers to null and flags to false.
    *
    * CVodeCreate
    *
    * CVodeCreate creates an internal memory block for a problem to
    * be solved by CVODE.
    * If successful, CVodeCreate returns a pointer to the problem memory.
    * This pointer should be passed to CVodeInit.
    * If an initialization error occurs, CVodeCreate prints an error
    * message to standard err and returns NULL.
    */
   CVODESolver();

   /** \brief This constructor wraps the CVodeCreate function,
    * calls the ReInit function to handle the inital condition, and initializes pointers
    * to null and flags to false.
    *
    * CVodeCreate creates an internal memory block for a problem to
    * be solved by CVODE.
    */
   CVODESolver(TimeDependentOperator &, Vector &, double &);

   void Init(TimeDependentOperator &);
   /** \brief The ReInit function is used in the initial construction and initialization of the CVODESolver object. It wraps either CVodeInit or CVodeReInit to pass the initial condtion to the ode_mem struct.
    *
    * CVodeInit
    *
    * CVodeInit allocates and initializes memory for a problem. All
    * problem inputs are checked for errors. If any error occurs during
    * initialization, it is reported to the file whose file pointer is
    * errfp and an error flag is returned. Otherwise, it returns CV_SUCCESS
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
   void ReInit(TimeDependentOperator &, Vector &, double &);

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
    * CVode, which integrates over a user-defined time interval
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
   void Step(Vector &, double&, double&);

   TimeDependentOperator* GetFOperator()
   {
      return f;
   }

   void SetStepType(int _step_type)
   {
      step_type = _step_type;
   }

   void SetStopTime(double);

   /** \brief Destroys associated memory. Calls CVodeFree and DestroyNVector
    *
    * CVodeFree
    *
    * This routine frees the problem memory allocated by CVodeInit.
    * Such memory includes all the vectors allocated by cvAllocVectors,
    * and the memory lmem for the linear solver (deallocated by a call
    * to lfree).
    */
   ~CVODESolver();

   virtual void CreateNVector(long int&, realtype*);

   /** \brief Creates an NVector of the appropriate type where the data is owned by the Vector* */
   virtual void CreateNVector(long int&, Vector*);

   /** \brief Transfers the data owned by the Vector* by copying the double* pointer */
   virtual void TransferNVectorShallow(Vector*,N_Vector&);

   /** \brief Destroys an NVector of the appropriate type */
   virtual void DestroyNVector(N_Vector&);

   /** \brief Wraps the CVodeInit function */
   virtual int WrapCVodeInit(void*,double&,N_Vector&);

private:

   /* Private function to check function return values */
   int check_flag(void *flagvalue, char *funcname, int opt);
};

#ifdef MFEM_USE_MPI
class CVODEParSolver: public CVODESolver
{
protected:
   MPI_Comm comm;

public:
   CVODEParSolver(MPI_Comm _comm, TimeDependentOperator &_f, Vector &_x,
                  double &_t);

   void CreateNVector();

   void CreateNVector(long int&, realtype*);

   void CreateNVector(long int&, Vector*);

   /** \brief Transfers the data owned by the Vector* by copying the double* pointer and the hypre_ParVector* pointer */
   void TransferNVectorShallow(Vector*,N_Vector&);

   void DestroyNVector(N_Vector&);

private:

   int WrapCVodeInit(void*,double&,N_Vector&);

};
#endif

/// Wraps the ARKode library of explicit, implicit and additive Runge Kutta methods
class ARKODESolver: public ODESolver
{
protected:
   N_Vector y;
   void* ode_mem;
   int step_type;
   bool initialized_sundials;
   bool tolerances_set_sundials;

public:
   /** \brief This constructor wraps the ARKodeCreate function, and initializes pointers to null and flags to false.
    *
    * ARKodeCreate
    *
    * ARKodeCreate creates an internal memory block for a problem to
    * be solved by ARKODE.  If successful, ARKodeCreate returns a
    * pointer to the problem memory. This pointer should be passed to
    * ARKodeInit. If an initialization error occurs, ARKodeCreate
    * prints an error message to standard err and returns NULL.
    */
   ARKODESolver();

   /** \brief This constructor wraps the ARKodeCreate function,
    * calls the ReInit function to handle the inital condition, and initializes pointers
    * to null and flags to false.
    *
    * ARKodeCreate creates an internal memory block for a problem to
    * be solved by ARKODE.
    */
   ARKODESolver(TimeDependentOperator &, Vector &, double &);

   void Init(TimeDependentOperator &);
   /** \brief The ReInit function is used in the initial construction and initialization of the ARKODESolver object. It wraps either ARKodeInit or ARKodeReInit to pass the initial condtion to the ode_mem struct.
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
   void ReInit(TimeDependentOperator &, Vector &, double &);

   /** \brief SetSStolerances wraps the ARKode function ARKodeSStolerances which
    * specifies scalar relative and absolute tolerances.
    *
    *  These functions specify the integration tolerances. One of them
    * SHOULD be called before the first call to ARKode; otherwise
    * default values of reltol=1e-4 and abstol=1e-9 will be used,
    * which may be entirely incorrect for a specific problem.
    *
    * ARKodeSStolerances
    *
    * ARKodeSStolerances specifies scalar relative and absolute tolerances.
    */
   void SetSStolerances(realtype reltol, realtype abstol);

   /* \brief Wraps SetERKTable to choose a specific Butcher table for a specific RK method
    * ARKodeSetERKTable
    *
    * Specifies to use a customized Butcher table for the explicit
    * portion of the system
    */
   void WrapSetERKTableNum(int&);

   /* \brief Wraps SetFixedStep to force ARKode to take one internal step of size dt
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
   void WrapSetFixedStep(realtype dt);

   /** \brief Step transfers vector pointers using TransferNVector and calls
    * ARKode, which integrates over a user-defined time interval
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
    * ARK_ONE_STEP.  In the ARK_NORMAL mode, the solver steps until
    * it reaches or passes tout and then interpolates to obtain
    * y(tout).  In the ARK_ONE_STEP mode, it takes one internal step
    * and returns.  The behavior of both modes can be over-rided
    * through user-specification of ark_tstop (through the
    * ARKodeSetStopTime function), in which case if a solver step
    * would pass tstop, the step is shortened so that it stops at
    * exactly the specified stop time, and hence interpolation of
    * y(tout) is not required.
    */
   void Step(Vector &, double&, double&);

   TimeDependentOperator* GetFOperator()
   {
      return f;
   }

   void SetStepType(int _step_type)
   {
      step_type = _step_type;
   }

   void SetStopTime(double);

   /** \brief Destroys associated memory. Calls CVodeFree and DestroyNVector
    *
    * ARKodeFree
    *
    * This routine frees the problem memory allocated by ARKodeInit.
    * Such memory includes all the vectors allocated by
    * arkAllocVectors, and arkAllocRKVectors, and the memory lmem for
    * the linear solver (deallocated by a call to lfree).
    */
   ~ARKODESolver();

   virtual void CreateNVector(long int&, realtype*);

   /** \brief Creates an NVector of the appropriate type where the data is owned by the Vector* */
   virtual void CreateNVector(long int&, Vector*);

   /** \brief Transfers the data owned by the Vector* by copying the double* pointer */
   virtual void TransferNVectorShallow(Vector*,N_Vector&);

   /** \brief Destroys an NVector of the appropriate type */
   virtual void DestroyNVector(N_Vector&);

private:
   virtual int WrapARKodeInit(void*,double&,N_Vector&);

   virtual int WrapARKodeReInit(void*,double&,N_Vector&);
   /* Private function to check function return values */
   int check_flag(void *flagvalue, char *funcname, int opt);
};

#ifdef MFEM_USE_MPI
class ARKODEParSolver: public ARKODESolver
{
protected:
   MPI_Comm comm;

public:
   ARKODEParSolver(MPI_Comm _comm, TimeDependentOperator &_f, Vector &_x,
                   double &_t);

   void CreateNVector();

   void CreateNVector(long int&, realtype*);

   void CreateNVector(long int&, Vector*);

   void TransferNVectorShallow(Vector*,N_Vector&);

   void DestroyNVector(N_Vector&);

private:

   int WrapARKodeReInit(void*,double&,N_Vector&);

   int WrapARKodeInit(void*,double&,N_Vector&);

};
#endif


}

int sun_f_fun(realtype t, N_Vector y, N_Vector ydot, void *user_data);

int sun_f_fun_par(realtype t, N_Vector y, N_Vector ydot, void *user_data);

#endif
