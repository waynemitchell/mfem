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
#include <arkode/arkode.h>             /* prototypes for ARKODE fcts., consts. */
#include <nvector/nvector_serial.h>  /* serial N_Vector types, fcts., macros */
#include <sundials/sundials_band.h>  /* definitions of type DlsMat and macros */
#include <sundials/sundials_types.h> /* definition of type realtype */
#include <sundials/sundials_math.h>  /* definition of ABS and EXP */

/* Type : UserData (contains grid constants) */

namespace mfem
{

typedef struct {
  TimeDependentOperator* f_op;
} *UserData;

class SunODESolver : public ODESolver
{

protected:
//   TimeDependentOperator *f; // f(.,t) : R^n --> R^n
   UserData data;
   N_Vector y;   
   void* ode_mem;

public:
   SunODESolver();
   
   void Init(TimeDependentOperator &);

   void Step(Vector &, double&, double&);
   
   void SetIC(Vector &, double&, double&);
   
   void GetY(Vector &, double&, double&);

   TimeDependentOperator* GetFOperator();
   
   void (SunODESolver::*PtrToStep)(Vector &, double&, double&);
 
//   ~SunODESolver();

/* Private function to check function return values */

 int check_flag(void *flagvalue, char *funcname, int opt);


};

class CVODESolver : public mfem::SunODESolver
{
/*
protected:
   TimeDependentOperator *f; // f(.,t) : R^n --> R^n
   UserData data;
   N_Vector y;   
   void* ode_mem;*/

public:
   CVODESolver();
   
   void Init(TimeDependentOperator &);

   void Step(Vector &, double&, double&);
   
   void SetIC(Vector &, double&, double&);
   
   void GetY(Vector &, double&, double&);
   
   void (CVODESolver::*PtrToStep)(Vector &, double&, double&);

   ~CVODESolver();

/* Private function to check function return values */

// int check_flag(void *flagvalue, char *funcname, int opt);

};

class ARKODESolver : public mfem::SunODESolver
{
/*
protected:
   TimeDependentOperator *f; // f(.,t) : R^n --> R^n
   UserData data;
   N_Vector y;   
   void* ode_mem;
*/
protected:
   int step_type;
   
public:
   ARKODESolver();
   
   void Init(TimeDependentOperator &);

   void Step(Vector &, double&, double&);
   
   void SetIC(Vector &, double&, double&);
   
   void GetY(Vector &, double&, double&);
   
   void (ARKODESolver::*PtrToStep)(Vector &, double&, double&);
   
   void SetStepType(int);

   ~ARKODESolver();

/* Private function to check function return values */

// int check_flag(void *flagvalue, char *funcname, int opt);

};


}
//leave N_Vector wrapping question for after SunODE_Solver inheritance finished
 int sun_f_fun(realtype t, N_Vector y, N_Vector ydot, void *user_data);


#endif
