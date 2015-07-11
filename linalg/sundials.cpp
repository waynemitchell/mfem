#include "../config/config.hpp"

#include "../linalg/operator.hpp"
#include "../linalg/ode.hpp"
#include "sundials.hpp"

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <cvode/cvode.h>             /* prototypes for CVODE fcts., consts. */
#include <arkode/arkode.h>             /* prototypes for ARKODE fcts., consts. */
#include <nvector/nvector_serial.h>  /* serial N_Vector types, fcts., macros */
#include <sundials/sundials_band.h>  /* definitions of type DlsMat and macros */
#include <sundials/sundials_types.h> /* definition of type realtype */
#include <sundials/sundials_math.h>  /* definition of ABS and EXP */


/* Problem Constants */

#define RTOL  RCONST(1.0e-3) /* scalar absolute tolerance */
#define ATOL  RCONST(1.0e-6)    /* scalar absolute tolerance */
#define T0    RCONST(0.0)    /* initial time              */

#define ZERO RCONST(0.0)

//#define ASSUME_LIMITS 1

using namespace std;

namespace mfem
{


CVODESolver::CVODESolver()
{
   y = NULL;
   f = NULL;
   PtrToStep=&CVODESolver::SetIC;

   /* Call CVodeCreate to create the solver memory */
   ode_mem=CVodeCreate(CV_ADAMS,CV_FUNCTIONAL);
}

void CVODESolver::Init(TimeDependentOperator &_f)
{
   //not checking that initial pointers set to NULL:
   f = &_f;
   realtype reltol, abstol;
   long int yin_length=_f.Width(); //assume don't have initial condition in Init
   //intial time
   realtype t = 0.0;
   realtype *yin;
   yin= new realtype[yin_length];
   int iout, flag;
   reltol = RTOL;   /* Set the tolerances */
   abstol = ATOL;

   // Create a serial vector
   y = N_VMake_Serial(yin_length,yin);   /* Allocate y vector */
   if (check_flag((void*)y, "N_VNew_Serial", 0)) { return; }

   /* Call CVodeInit to initialize the integrator memory and specify the
    * user's right hand side function in u'=f(t,u), the inital time t, and
    * the initial dependent variable vector y. */
   flag = CVodeInit(ode_mem, sun_f_fun, t, y);
   if (check_flag(&flag, "CVodeInit", 1)) { return; }

   /* Call CVodeSStolerances to specify the scalar relative tolerance
    * and scalar absolute tolerance */
   flag = CVodeSStolerances(ode_mem, reltol, abstol);
   if (check_flag(&flag, "CVodeSStolerances", 1)) { return; }

   /* Set the pointer to user-defined data */
   //currently pointed to something defined within this step
   flag = CVodeSetUserData(ode_mem, this->f);
   if (check_flag(&flag, "CVodeSetUserData", 1)) { return; }

}

void CVODESolver::SetStopTime(double tf)
{
   CVodeSetStopTime(ode_mem, tf);
}

void CVODESolver::SetIC(Vector &x, double&t, double&dt)
{
   int flag=0;
   realtype* yin=NV_DATA_S(y);
   delete yin;
   yin=NULL;
   NV_DATA_S(y)= x.GetData();
   flag = CVodeReInit(ode_mem, t, y);


   /* Set the minimum step size */
   //   flag = CVodeSetMinStep(ode_mem, dt);
   //   if(check_flag(&flag, "CVodeSetMinStep", 1)) return;

   /* Set the maximum step size */
#ifdef ASSUME_LIMITS
   flag = CVodeSetMaxStep(ode_mem, dt);
   if (check_flag(&flag, "CVodeSetMaxStep", 1)) { return; }
#endif
   PtrToStep=&CVODESolver::GetY;
}

void CVODESolver::GetY(Vector &x, double&t, double&dt)
{
   NV_DATA_S(y)=x.GetData();
}

void CVODESolver::Step(Vector &x, double &t, double &dt)
{
   int flag=0;
   /*   cout<<"dt="<<dt<<endl;*/
   realtype tout=t+dt;

   (this->*PtrToStep)(x,t,dt);

   //Step
   flag = CVode(ode_mem, tout, y, &t, CV_NORMAL);
   if (check_flag(&flag, "CVode", 1)) { return; }
   return;
   flag = CVodeGetLastStep(ode_mem, &dt);
   if (check_flag(&flag, "CVodeGetLastStep", 1)) { return; }
}

CVODESolver::~CVODESolver()
{
   // Free the used memory.
   // Clean up and return with successful completion
   if (NV_OWN_DATA_S(y)==true)
   {
      N_VDestroy_Serial(y);   // Free y vector
   }
   if (ode_mem!=NULL)
   {
      CVodeFree(&ode_mem);   // Free integrator memory
   }
}

int CVODESolver::check_flag(void *flagvalue, char *funcname, int opt)
{
  int *errflag;

  /* Check if SUNDIALS function returned NULL pointer - no memory allocated */

  if (opt == 0 && flagvalue == NULL) {
    fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed - returned NULL pointer\n\n",
            funcname);
    return(1); }

  /* Check if flag < 0 */

  else if (opt == 1) {
    errflag = (int *) flagvalue;
    if (*errflag < 0) {
      fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed with flag = %d\n\n",
              funcname, *errflag);
      return(1); }}

  /* Check if function returned NULL pointer - no memory allocated */

  else if (opt == 2 && flagvalue == NULL) {
    fprintf(stderr, "\nMEMORY_ERROR: %s() failed - returned NULL pointer\n\n",
            funcname);
    return(1); }

  return(0);
}


ARKODESolver::ARKODESolver()
{
   y = NULL;
   f = NULL;
   PtrToStep=&ARKODESolver::SetIC;

   /* Call CVodeCreate to create the solver memory */
   ode_mem=ARKodeCreate();
   step_type=ARK_NORMAL;
}

void ARKODESolver::Init(TimeDependentOperator &_f)
{
   f = &_f;
   realtype reltol, abstol;
   long int yin_length=_f.Width(); //assume don't have initial condition in Init
   //intial time
   realtype t = 0.0;
   realtype *yin;
   yin= new realtype[yin_length];
   int iout, flag;

   // Create a serial vector

   y = N_VMake_Serial(yin_length,yin);   /* Allocate y vector */
   if (check_flag((void*)y, "N_VNew_Serial", 0)) { return; }

   reltol = RTOL;   /* Set the tolerances */
   abstol = ATOL;

   /* Call ARKodeInit to initialize the integrator memory and specify the
    * user's right hand side function in u'=f(t,u), the inital time t, and
    * the initial dependent variable vector y. */
   flag = ARKodeInit(ode_mem, sun_f_fun, NULL, t, y);
   if (check_flag(&flag, "ARKodeInit", 1)) { return; }

#ifdef ASSUME_LIMITS
   /* Call ARKodeSetERKTableNum to compare directly with method integration
    * in MFEM example 9. */
   flag = ARKodeSetERKTableNum(ode_mem, 3);
   if (check_flag(&flag, "ARKodeSetERKTableNum", 1)) { return; }
#endif

   /* Call ARKodeSStolerances to specify the scalar relative tolerance
    * and scalar absolute tolerance */
   flag = ARKodeSStolerances(ode_mem, reltol, abstol);
   if (check_flag(&flag, "ARKodeSStolerances", 1)) { return; }

   /* Set the pointer to user-defined data */
   flag = ARKodeSetUserData(ode_mem, this->f);
   if (check_flag(&flag, "ARKodeSetUserData", 1)) { return; }

}

void ARKODESolver::SetStopTime(double tf)
{
   ARKodeSetStopTime(ode_mem, tf);
}

void ARKODESolver::SetIC(Vector &x, double&t, double&dt)
{
   int flag=0;
   realtype* yin=NV_DATA_S(y);
   delete yin;
   yin=NULL;
   NV_DATA_S(y)= x.GetData();
   flag = ARKodeReInit(ode_mem, sun_f_fun, NULL, t, y);

#ifdef ASSUME_LIMITS
   if (step_type==ARK_ONE_STEP)
   {
      flag = ARKodeSetFixedStep(ode_mem, dt);
      if (check_flag(&flag, "ARKodeSetInitStep", 1)) { return; }
   }
#endif

   /* Set the minimum step size */
   //   flag = ARKodeSetMinStep(ode_mem, dt);
   //   if(check_flag(&flag, "ARKodeSetMinStep", 1)) return;
#ifdef ASSUME_LIMITS
   /* Set the maximum step size */
   flag = ARKodeSetMaxStep(ode_mem, dt);
   if (check_flag(&flag, "ARKodeSetMaxStep", 1)) { return; }
#endif
   PtrToStep=&ARKODESolver::GetY;
}

void ARKODESolver::GetY(Vector &x, double&t, double&dt)
{
   NV_DATA_S(y)=x.GetData();
}

void ARKODESolver::Step(Vector &x, double &t, double &dt)
{
   int flag=0;
   /*   cout<<"dt="<<dt<<endl;*/
   realtype tout=t+dt;

   (this->*PtrToStep)(x,t,dt);

   //Step
   flag = ARKode(ode_mem, tout, y, &t, step_type);
   if (check_flag(&flag, "ARKode", 1)) { return; }
   //   x.SetData(NV_DATA_S(y));
   flag = ARKodeGetLastStep(ode_mem, &dt);
   if (check_flag(&flag, "ARKodeGetLastStep", 1)) { return; }
   return;
}

ARKODESolver::~ARKODESolver()
{
   // Free the used memory.
   // Clean up and return with successful completion
   if (NV_OWN_DATA_S(y)==true)
   {
      N_VDestroy_Serial(y);   // Free y vector
   }
   if (ode_mem!=NULL)
   {
      ARKodeFree(&ode_mem);   // Free integrator memory
   }
}

int ARKODESolver::check_flag(void *flagvalue, char *funcname, int opt)
{
  int *errflag;

  /* Check if SUNDIALS function returned NULL pointer - no memory allocated */

  if (opt == 0 && flagvalue == NULL) {
    fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed - returned NULL pointer\n\n",
            funcname);
    return(1); }

  /* Check if flag < 0 */

  else if (opt == 1) {
    errflag = (int *) flagvalue;
    if (*errflag < 0) {
      fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed with flag = %d\n\n",
              funcname, *errflag);
      return(1); }}

  /* Check if function returned NULL pointer - no memory allocated */

  else if (opt == 2 && flagvalue == NULL) {
    fprintf(stderr, "\nMEMORY_ERROR: %s() failed - returned NULL pointer\n\n",
            funcname);
    return(1); }

  return(0);
}

}


int sun_f_fun(realtype t, N_Vector y, N_Vector ydot,void *user_data)
{

   //using namespace mfem;
   realtype *ydata, *ydotdata;
   long int ylen, ydotlen;

   //ydata is now a pointer to the realtype data array in y
   ydata = NV_DATA_S(y);
   ylen = NV_LENGTH_S(y);
   
   // probably unnecessary, since overwriting ydot as output
   //ydotdata is now a pointer to the realtype data array in ydot
   ydotdata = NV_DATA_S(ydot);
   ydotlen = NV_LENGTH_S(ydot);
   
  //f is now a pointer of abstract base class type TimeDependentOperator. It points to the TimeDependentOperator in the user_data struct
  mfem::TimeDependentOperator* f = (mfem::TimeDependentOperator*) user_data;

   //if gridfunction information necessary for Mult, keep Vector in userdata
   //  Vector* u = udata->u;
   //  u->SetData((double*) ydata);
   // Creates mfem vectors with pointers to the data array in y and in ydot respectively
   // Have not explicitly set as owndata, so allocated size is -size
   mfem::Vector mfem_vector_y((double*) ydata, ylen);
   mfem::Vector mfem_vector_ydot((double*) ydotdata, ydotlen);

   f->SetTime(t);
   f->Mult(mfem_vector_y,mfem_vector_ydot);

   return (0);
}

