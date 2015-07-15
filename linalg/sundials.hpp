#ifndef MFEM_SUNDIALS
#define MFEM_SUNDIALS
#include "mfem.hpp"
#include "../linalg/operator.hpp"
#include "../linalg/ode.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../config/config.hpp"

// cvode header files
#include <cvode/cvode.h>             /* prototypes for CVODE fcts., consts. */
#include <arkode/arkode.h>           /* prototypes for ARKODE fcts., consts. */
#include <nvector/nvector_serial.h>  /* serial N_Vector types, fcts., macros */
#include <sundials/sundials_band.h>  /* definitions of type DlsMat and macros */
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
   bool initial_condition_set;
   bool tolerances_set;
   bool parallel;
#ifdef MFEM_USE_MPI
   MPI_Comm comm;
#endif
public:
   CVODESolver();

#ifdef MFEM_USE_MPI
   CVODESolver(MPI_Comm);
#endif

   void Init(TimeDependentOperator &);

   void Init(TimeDependentOperator &, Vector &, double&, double&);

   void ReInit(TimeDependentOperator &);

   void ReInit(TimeDependentOperator &, Vector &, double&, double&);

   void SetSStolerances(realtype reltol, realtype abstol);

   void Step(Vector &, double&, double&);

   void SetIC(Vector &, double&, double&);

   void GetY(Vector &, double&, double&);

   TimeDependentOperator* GetFOperator()
   {
      return f;
   }

   void SetStepType(int _step_type)
   {
      step_type = _step_type;
   }

   void SetStopTime(double);

//   void (CVODESolver::*PtrToStep)(Vector &, double&, double&);

   ~CVODESolver();

private:
   void CreateNVector();

   void CreateNVector(Vector&);

   /* Private function to check function return values */
   int check_flag(void *flagvalue, char *funcname, int opt);
};

/// Wraps the ARKode library of additive runge-kutta methods
class ARKODESolver: public ODESolver
{
protected:
   N_Vector y;
   void* ode_mem;
   int step_type;
   bool initial_condition_set;
   bool tolerances_set;
   bool parallel;
#ifdef MFEM_USE_MPI
   MPI_Comm comm;
#endif

public:
   ARKODESolver();
#ifdef MFEM_USE_MPI
   ARKODESolver(MPI_Comm);
#endif
   void Init(TimeDependentOperator &);

   void Init(TimeDependentOperator &, Vector &, double&, double&);

   void ReInit(TimeDependentOperator &);

   void ReInit(TimeDependentOperator &, Vector &, double&, double&);

   void SetSStolerances(realtype reltol, realtype abstol);

   void Step(Vector &, double&, double&);

   void SetIC(Vector &, double&, double&);

   void GetY(Vector &, double&, double&);

   TimeDependentOperator* GetFOperator()
   {
      return f;
   }

   void SetStepType(int _step_type)
   {
      step_type = _step_type;
   }

   void SetStopTime(double);

   void (ARKODESolver::*PtrToStep)(Vector &, double&, double&);

   ~ARKODESolver();

private:
   /* Private function to check function return values */
   int check_flag(void *flagvalue, char *funcname, int opt);
};


}

int sun_f_fun(realtype t, N_Vector y, N_Vector ydot, void *user_data);


#endif
