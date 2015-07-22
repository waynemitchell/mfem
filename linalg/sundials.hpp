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
   bool initialized_sundials;
   bool tolerances_set_sundials;

public:
   CVODESolver();

   CVODESolver(TimeDependentOperator &, Vector &, double &);

   void Init(TimeDependentOperator &);

   void ReInit(TimeDependentOperator &, Vector &, double &);

   void SetSStolerances(realtype reltol, realtype abstol);

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

   ~CVODESolver();
   
   virtual void CreateNVector(long int&, realtype*);
   
   virtual void CreateNVector(long int&, Vector*);
   
   virtual void TransferNVectorShallow(Vector*,N_Vector&);
   
   virtual void DestroyNVector(N_Vector&);
   
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
   CVODEParSolver(MPI_Comm _comm, TimeDependentOperator &_f, Vector &_x, double &_t);
   
   void CreateNVector();

   void CreateNVector(long int&, realtype*);
   
   void CreateNVector(long int&, Vector*);
   
   void TransferNVectorShallow(Vector*,N_Vector&);
   
   void DestroyNVector(N_Vector&);

private:
   
   int WrapCVodeInit(void*,double&,N_Vector&);

};
#endif

/// Wraps the ARKode library of additive runge-kutta methods
class ARKODESolver: public ODESolver
{
protected:
   N_Vector y;
   void* ode_mem;
   int step_type;
   bool initialized_sundials;
   bool tolerances_set_sundials;

public:
   ARKODESolver();

   ARKODESolver(TimeDependentOperator &, Vector &, double &);

   void Init(TimeDependentOperator &);

   void ReInit(TimeDependentOperator &, Vector &, double &);

   void SetSStolerances(realtype reltol, realtype abstol);

   void Step(Vector &, double&, double&);

   void GetY(Vector &);

   TimeDependentOperator* GetFOperator()
   {
      return f;
   }

   void SetStepType(int _step_type)
   {
      step_type = _step_type;
   }

   void SetStopTime(double);

   ~ARKODESolver();

private:
   /* Private function to check function return values */
   int check_flag(void *flagvalue, char *funcname, int opt);
};


}

int sun_f_fun(realtype t, N_Vector y, N_Vector ydot, void *user_data);

int sun_f_fun_par(realtype t, N_Vector y, N_Vector ydot, void *user_data);

#endif
