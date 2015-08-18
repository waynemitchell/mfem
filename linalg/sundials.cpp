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

namespace mfem
{

CVODESolver::CVODESolver()
{
   y = NULL;
   f = NULL;

   /* Call CVodeCreate to create the solver memory */
   /* Assumes Adams methods and funcitonal iterations, rather than BDF or Newton solves */
   ode_mem=CVodeCreate(CV_ADAMS,CV_FUNCTIONAL);
   initialized_sundials=false;
   tolerances_set_sundials=false;
}

CVODESolver::CVODESolver(TimeDependentOperator &_f, Vector &_x, double&_t,
                         int lmm, int iter)
{
   y = NULL;
   f = NULL;

   /* Call CVodeCreate to create the solver memory */
   /* Assumes Adams methods and funcitonal iterations, rather than BDF or Newton solves */
   ode_mem=CVodeCreate(lmm,iter);
   initialized_sundials=false;
   tolerances_set_sundials=false;
   linear_multistep_method_type=lmm;
   solver_iteration_type=iter;
   ReInit(_f,_x,_t);
}

void CVODESolver::CreateNVector(long int& yin_length, realtype* ydata)
{

   // Create a serial vector
   y = N_VMake_Serial(yin_length,ydata);   /* Allocate y vector */
   if (check_flag((void*)y, "N_VNew_Serial", 0)) { MFEM_ABORT("N_VNew_Serial"); }

}

void CVODESolver::CreateNVector(long int& yin_length, Vector* _x)
{

   // Create a serial vector
   y = N_VMake_Serial(yin_length,
                      (realtype*) _x->GetData());   /* Allocate y vector */
   if (check_flag((void*)y, "N_VNew_Serial", 0)) { MFEM_ABORT("N_VNew_Serial"); }

}

void CVODESolver::TransferNVectorShallow(Vector* _x, N_Vector &_y)
{
   NV_DATA_S(_y)=_x->GetData();
}

void CVODESolver::DestroyNVector(N_Vector& _y)
{
   if (NV_OWN_DATA_S(y)==true)
   {
      N_VDestroy_Serial(y);   // Free y vector
   }
}

int CVODESolver::WrapCVodeInit(void* _ode_mem, double &_t, N_Vector &_y)
{
   return CVodeInit(_ode_mem, sun_f_fun, (realtype) _t, _y);
}

void CVODESolver::Init(TimeDependentOperator &_f)
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
      /* Call CVodeReInit to initialize the integrator memory and specify the inital time t,
       * and the initial dependent variable vector y. */
      flag = CVodeReInit(ode_mem, (realtype) t, y);
      if (check_flag(&flag, "CVodeInit", 1)) { MFEM_ABORT("CVodeInit"); }
   }
   else
   {
      /* Call CVodeInit to initialize the integrator memory and specify the
       * user's right hand side function in x'=f(t,x), the inital time t, and
       * the initial dependent variable vector y. */
      flag = WrapCVodeInit(ode_mem, t, y);
      if (check_flag(&flag, "CVodeInit", 1)) { MFEM_ABORT("CVodeInit"); }
      initialized_sundials=true;
      SetSStolerances(RELTOL,ABSTOL);
   }

   /* Set the pointer to user-defined data */
   flag = CVodeSetUserData(ode_mem, this->f);
   if (check_flag(&flag, "CVodeSetUserData", 1)) { MFEM_ABORT("CVodeSetUserData"); }
}

void CVODESolver::ReInit(TimeDependentOperator &_f, Vector &_x, double& _t)
{
   //not checking that initial pointers set to NULL:
   f = &_f;
   long int yin_length=_f.Width(); //assume don't have initial condition in Init
   int flag;
   // Create an NVector
   CreateNVector(yin_length, &_x);

   if (initialized_sundials)
   {
      /* Call CVodeReInit to initialize the integrator memory and specify the inital time t,
       * and the initial dependent variable vector y. */
      flag = CVodeReInit(ode_mem, (realtype) _t, y);
      if (check_flag(&flag, "CVodeInit", 1)) { MFEM_ABORT("CVodeInit"); }
   }
   else
   {
      /* Call CVodeInit to initialize the integrator memory and specify the
       * user's right hand side function in x'=f(t,x), the inital time t, and
       * the initial dependent variable vector y. */
      flag = WrapCVodeInit(ode_mem, _t, y);
      if (check_flag(&flag, "CVodeInit", 1)) { MFEM_ABORT("CVodeInit"); }
      initialized_sundials=true;
      SetSStolerances(RELTOL,ABSTOL);
   }

   /* Set the pointer to user-defined data */
   flag = CVodeSetUserData(ode_mem, this->f);
   if (check_flag(&flag, "CVodeSetUserData", 1)) { MFEM_ABORT("CVodeSetUserData"); }
}

void CVODESolver::SetSStolerances(realtype reltol, realtype abstol)
{
   int flag=0;
   /* Call CVodeSStolerances to specify the scalar relative tolerance
    * and scalar absolute tolerance */
   flag = CVodeSStolerances(ode_mem, reltol, abstol);
   if (check_flag(&flag, "CVodeSStolerances", 1)) { return; }
   tolerances_set_sundials=true;
}
void CVODESolver::SetLinearSolve()
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
      WrapCVodeInit(ode_mem,t0,y);
      CVodeSetUserData(ode_mem, this->f);
      if (!tolerances_set_sundials)
      {
         SetSStolerances(RELTOL,ABSTOL);
      }
   }

   MFEMLinearCVSolve(ode_mem, PREC_NONE, 100);

}
void CVODESolver::SetStopTime(double tf)
{
   CVodeSetStopTime(ode_mem, tf);
}

void CVODESolver::Step(Vector &x, double &t, double &dt)
{
   int flag=0;
   realtype tout=t+dt;
   TransferNVectorShallow(&x,y);

   //Step
   flag = CVode(ode_mem, tout, y, &t, CV_NORMAL);
   if (check_flag(&flag, "CVode", 1)) { MFEM_ABORT("CVode"); }

   //Record last incremental step size
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

int CVODESolver::check_flag(void *flagvalue, char *funcname, int opt)
{
   int *errflag;
   char str_buffer[80];

   /* Check if SUNDIALS function returned NULL pointer - no memory allocated */

   if (opt == 0 && flagvalue == NULL)
   {
      sprintf(str_buffer,"\nSUNDIALS_ERROR: %s() failed - returned NULL pointer\n\n",
              funcname);
      mfem_error(str_buffer);
      return (1);
   }

   /* Check if flag < 0 */

   else if (opt == 1)
   {
      errflag = (int *) flagvalue;
      if (*errflag < 0)
      {
         sprintf(str_buffer, "\nSUNDIALS_ERROR: %s() failed with flag = %d\n\n",
                 funcname, *errflag);
         mfem_error(str_buffer);
         return (1);
      }
   }

   /* Check if function returned NULL pointer - no memory allocated */

   else if (opt == 2 && flagvalue == NULL)
   {
      sprintf(str_buffer, "\nMEMORY_ERROR: %s() failed - returned NULL pointer\n\n",
              funcname);
      mfem_error(str_buffer);
      return (1);
   }

   return (0);
}

#ifdef MFEM_USE_MPI
CVODEParSolver::CVODEParSolver(MPI_Comm _comm, TimeDependentOperator &_f,
                               Vector &_x, double &_t, int lmm, int iter)
{
   y = NULL;
   f = &_f;

   /* Call CVodeCreate to create the solver memory */
   ode_mem=CVodeCreate(lmm,iter);
   initialized_sundials=false;
   tolerances_set_sundials=false;
   linear_multistep_method_type=lmm;
   solver_iteration_type=iter;
   //set MPI_Comm communicator
   comm=_comm;
   ReInit(_f,_x,_t);
}


void CVODEParSolver::CreateNVector(long int& yin_length, realtype* ydata)
{
   int nprocs, myid;
   long int global_length;
   MPI_Comm_size(comm,&nprocs);
   MPI_Comm_rank(comm,&myid);
   realtype in=yin_length;
   realtype out;
   MPI_Allreduce(&in, &out, 1, PVEC_REAL_MPI_TYPE, MPI_SUM, comm);
   global_length= out;
   y = N_VMake_ParHyp(comm, yin_length, global_length,
                      ydata);   /* Allocate y vector */
}

void CVODEParSolver::CreateNVector(long int& yin_length, Vector* _x)
{
   int nprocs, myid;
   long int global_length;
   MPI_Comm_size(comm,&nprocs);
   MPI_Comm_rank(comm,&myid);

   //Process appropriate sizes for creating y as a new ParHyp NVector
   realtype in=yin_length;
   in=_x->Size();
   realtype out;
   MPI_Allreduce(&in, &out, 1, PVEC_REAL_MPI_TYPE, MPI_SUM, comm);
   global_length= out;
   y = N_VNew_ParHyp(comm, yin_length, global_length);   /* Allocate y vector */
   TransferNVectorShallow(_x,y);
}

void CVODEParSolver::TransferNVectorShallow(Vector* _x, N_Vector &_y)
{
   int nprocs, myid;
   NV_HYPRE_PARVEC_PH(_y)=((HypreParVector*) _x)->StealParVector();
   NV_OWN_PARVEC_PH(_y)=true;
}

void CVODEParSolver::DestroyNVector(N_Vector& _y)
{

   if (NV_OWN_PARVEC_PH(y)==true)
   {
      N_VDestroy_ParHyp(y);   // Free y vector
   }
}

int CVODEParSolver::WrapCVodeInit(void* _ode_mem, double &_t, N_Vector &_y)
{
   return CVodeInit(_ode_mem, sun_f_fun_par, (realtype) _t, _y);
}

#endif
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

   // Create a serial vector
   y = N_VMake_Serial(yin_length,ydata);   /* Allocate y vector */
   if (check_flag((void*)y, "N_VNew_Serial", 0)) { MFEM_ABORT("N_VNew_Serial"); }

}

void ARKODESolver::CreateNVector(long int& yin_length, Vector* _x)
{

   // Create a serial vector
   y = N_VMake_Serial(yin_length,
                      (realtype*) _x->GetData());   /* Allocate y vector */
   if (check_flag((void*)y, "N_VNew_Serial", 0)) { MFEM_ABORT("N_VNew_Serial"); }

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
   //Assumes integrating TimeDependentOperator f explicitly
   //Consider adding a flag to switch between explicit and implicit
   return use_explicit ?
          ARKodeInit(_ode_mem, sun_f_fun, NULL, (realtype) _t, _y) :
          ARKodeInit(_ode_mem, NULL, sun_f_fun, (realtype) _t, _y);
}

int ARKODESolver::WrapARKodeReInit(void* _ode_mem, double &_t, N_Vector &_y)
{
   //Assumes integrating TimeDependentOperator f explicitly
   //Consider adding a flag to switch between explicit and implicit
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
      if (check_flag(&flag, "ARKodeInit", 1)) { MFEM_ABORT("ARKodeInit"); }
   }
   else
   {
      /* Call ARKodeInit to initialize the integrator memory and specify the
       * user's right hand side function in x'=f(t,x), the inital time t, and
       * the initial dependent variable vector y. */
      flag = WrapARKodeInit(ode_mem, t, y);
      if (check_flag(&flag, "ARKodeInit", 1)) { MFEM_ABORT("ARKodeInit"); }
      initialized_sundials=true;
      SetSStolerances(RELTOL,ABSTOL);
   }

   /* Set the pointer to user-defined data */
   flag = ARKodeSetUserData(ode_mem, this->f);
   if (check_flag(&flag, "ARKodeSetUserData", 1)) { MFEM_ABORT("ARKodeSetUserData"); }
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
      if (check_flag(&flag, "ARKodeInit", 1)) { MFEM_ABORT("ARKodeInit"); }
   }
   else
   {
      /* Call ARKodeInit to initialize the integrator memory and specify the
       * user's right hand side function in x'=f(t,x), the inital time t, and
       * the initial dependent variable vector y. */
      flag = WrapARKodeInit(ode_mem, _t, y);
      if (check_flag(&flag, "ARKodeInit", 1)) { MFEM_ABORT("ARKodeInit"); }
      initialized_sundials=true;
      SetSStolerances(RELTOL,ABSTOL);
   }

   /* Set the pointer to user-defined data */
   flag = ARKodeSetUserData(ode_mem, this->f);
   if (check_flag(&flag, "ARKodeSetUserData", 1)) { MFEM_ABORT("ARKodeSetUserData"); }
}

void ARKODESolver::SetSStolerances(realtype reltol, realtype abstol)
{
   int flag=0;
   /* Call ARKodeSStolerances to specify the scalar relative tolerance
    * and scalar absolute tolerance */
   flag = ARKodeSStolerances(ode_mem, reltol, abstol);
   if (check_flag(&flag, "ARKodeSStolerances", 1)) { return; }
   tolerances_set_sundials=true;
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
void ARKODESolver::SetLinearSolve()
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

   MFEMLinearARKSolve(ode_mem, PREC_NONE, 100);


}
void ARKODESolver::Step(Vector &x, double &t, double &dt)
{
   int flag=0;
   realtype tout=t+dt;
   TransferNVectorShallow(&x,y);

   //Step
   flag = ARKode(ode_mem, tout, y, &t, ARK_NORMAL);
   if (check_flag(&flag, "ARKode", 1)) { MFEM_ABORT("ARKode"); }
   //Record last incremental step size
   flag = ARKodeGetLastStep(ode_mem, &dt);
}

ARKODESolver::~ARKODESolver()
{
   // Free the used memory.
   // Clean up and return with successful completion
   DestroyNVector(y);
   if (ode_mem!=NULL)
   {
      ARKodeFree(&ode_mem);   // Free integrator memory
   }
}

int ARKODESolver::check_flag(void *flagvalue, char *funcname, int opt)
{
   int *errflag;
   char str_buffer[80];

   /* Check if SUNDIALS function returned NULL pointer - no memory allocated */

   if (opt == 0 && flagvalue == NULL)
   {
      sprintf(str_buffer,"\nSUNDIALS_ERROR: %s() failed - returned NULL pointer\n\n",
              funcname);
      mfem_error(str_buffer);
      return (1);
   }

   /* Check if flag < 0 */

   else if (opt == 1)
   {
      errflag = (int *) flagvalue;
      if (*errflag < 0)
      {
         sprintf(str_buffer, "\nSUNDIALS_ERROR: %s() failed with flag = %d\n\n",
                 funcname, *errflag);
         mfem_error(str_buffer);
         return (1);
      }
   }

   /* Check if function returned NULL pointer - no memory allocated */

   else if (opt == 2 && flagvalue == NULL)
   {
      sprintf(str_buffer, "\nMEMORY_ERROR: %s() failed - returned NULL pointer\n\n",
              funcname);
      mfem_error(str_buffer);
      return (1);
   }

   return (0);
}

#ifdef MFEM_USE_MPI
ARKODEParSolver::ARKODEParSolver(MPI_Comm _comm, TimeDependentOperator &_f,
                                 Vector &_x, double &_t, int _use_explicit)
{
   y = NULL;
   f = &_f;

   /* Call ARKodeCreate to create the solver memory */
   ode_mem=ARKodeCreate();
   initialized_sundials=false;
   tolerances_set_sundials=false;
   use_explicit=_use_explicit;
   //Set MPI_Comm
   comm=_comm;
   ReInit(_f,_x,_t);
}

void ARKODEParSolver::CreateNVector(long int& yin_length, realtype* ydata)
{
   int nprocs, myid;
   long int global_length;
   MPI_Comm_size(comm,&nprocs);
   MPI_Comm_rank(comm,&myid);
   realtype in=yin_length;
   realtype out;
   MPI_Allreduce(&in, &out, 1, PVEC_REAL_MPI_TYPE, MPI_SUM, comm);
   global_length= out;
   y = N_VMake_ParHyp(comm, yin_length, global_length,
                      ydata);   /* Allocate y vector */
}

void ARKODEParSolver::CreateNVector(long int& yin_length, Vector* _x)
{
   int nprocs, myid;
   long int global_length;
   MPI_Comm_size(comm,&nprocs);
   MPI_Comm_rank(comm,&myid);

   //Process appropriate sizes for creating y as a new ParHyp NVector
   realtype in=yin_length;
   in=_x->Size();
   realtype out;
   MPI_Allreduce(&in, &out, 1, PVEC_REAL_MPI_TYPE, MPI_SUM, comm);
   global_length= out;
   y = N_VNew_ParHyp(comm, yin_length, global_length);   /* Allocate y vector */
   TransferNVectorShallow(_x,y);
}

void ARKODEParSolver::TransferNVectorShallow(Vector* _x, N_Vector &_y)
{
   NV_HYPRE_PARVEC_PH(_y)=((HypreParVector*) _x)->StealParVector();
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
   //Assumes integrating TimeDependentOperator f explicitly
   //Consider adding a flag to switch between explicit and implicit
   return use_explicit ? ARKodeInit(_ode_mem, sun_f_fun_par, NULL, (realtype) _t,
                                    _y) : ARKodeInit(_ode_mem, NULL, sun_f_fun, (realtype) _t, _y);
}

int ARKODEParSolver::WrapARKodeReInit(void* _ode_mem, double &_t, N_Vector &_y)
{
   //Assumes integrating TimeDependentOperator f explicitly
   //Consider adding a flag to switch between explicit and implicit
   return use_explicit ? ARKodeReInit(_ode_mem, sun_f_fun, NULL, (realtype) _t,
                                      _y) : ARKodeInit(_ode_mem, NULL, sun_f_fun, (realtype) _t, _y);
}

#endif


}


int sun_f_fun(realtype t, N_Vector y, N_Vector ydot,void *user_data)
{

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

   // Creates mfem vectors with pointers to the data array in y and in ydot respectively
   // Have not explicitly set as owndata, so allocated size is -size
   mfem::Vector mfem_vector_y((double*) ydata, ylen);
   mfem::Vector mfem_vector_ydot((double*) ydotdata, ydotlen);

   //Apply ydot=f(t,y)
   f->SetTime(t);
   f->Mult(mfem_vector_y,mfem_vector_ydot);

   return (0);
}

#ifdef MFEM_USE_MPI
int sun_f_fun_par(realtype t, N_Vector y, N_Vector ydot,void *user_data)
{

   realtype *ydata, *ydotdata;
   int myid;
   long int ylen, ydotlen;
   HYPRE_Int *col, *col2;
   hypre_ParVector *yparvec, *ydotparvec;
   MPI_Comm comm=NV_COMM_PH(y);
   MPI_Comm_rank(comm,&myid);

   //f is now a pointer of abstract base class type TimeDependentOperator. It points to the TimeDependentOperator in the user_data struct
   mfem::TimeDependentOperator* f = (mfem::TimeDependentOperator*) user_data;

   // Creates mfem HypreParVectors mfem_vector_y and mfem_vector_ydot by using the casting
   // operators in HypreParVector to cast the hypre_ParVector in y and in ydot respectively
   // Have not explicitly set as owndata, so allocated size is -size
   mfem::HypreParVector mfem_vector_y=
      (mfem::HypreParVector) (NV_HYPRE_PARVEC_PH(y));
   mfem::HypreParVector mfem_vector_ydot=
      (mfem::HypreParVector) (NV_HYPRE_PARVEC_PH(ydot));

   //Apply ydot=f(t,y)
   f->SetTime(t);
   f->Mult(mfem_vector_y,mfem_vector_ydot);

   return (0);
}
#endif

/*---------------------------------------------------------------
 MFEMLinearSolve:

 This routine initializes the memory record and sets various
 function fields specific to the linear solver module.
 MFEMLinearSolve first calls the existing lfree routine if this is not
 NULL. It then sets the ark_linit, ark_lsetup, ark_lsolve,
 ark_lfree fields in (*arkode_mem) to be LinearSolveInit,
 LinearSolveSetup, LinearSolve, and LinearSolveFree, respectively.
---------------------------------------------------------------*/
int MFEMLinearCVSolve(void *ode_mem, int pretype, int maxl)
{
   CVodeMem cv_mem;
   int mxl;

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
   cv_mem->cv_lmem;

   return (CVSPILS_SUCCESS);
}

static int WrapLinearCVSolveInit(CVodeMem cv_mem)
{
   cout<<"entered init"<<endl;
   return 0;
}


static int WrapLinearCVSolveSetup(CVodeMem cv_mem, int convfail,
                                  N_Vector ypred, N_Vector fpred,
                                  booleantype *jcurPtr, N_Vector vtemp1,
                                  N_Vector vtemp2, N_Vector vtemp3)
{

   cout<<"----------setup"<<endl;
   cout<<"entered setup at step "<<cv_mem->cv_nst<<endl;
   cout<<"entered setup at linear step "<<cv_mem->cv_nstlp<<endl;
   cout<<"entered setup at time "<<cv_mem->cv_tn<<endl;
   cout<<"entered setup at h "<<cv_mem->cv_h<<endl;

   return 0;
}



static int WrapLinearCVSolve(CVodeMem cv_mem, N_Vector b,
                             N_Vector weight, N_Vector ynow,
                             N_Vector fnow)
{

   cout<<"----------solve"<<endl;
   cout<<"entered solve at step "<<cv_mem->cv_nst<<endl;
   cout<<"entered solve at linear step "<<cv_mem->cv_nstlp<<endl;

   return 0;
}


static void WrapLinearCVSolveFree(CVodeMem cv_mem)
{
   cout<<"entered free"<<endl;
   return;
}

int MFEMLinearARKSolve(void *arkode_mem, int pretype, int maxl)
{
   ARKodeMem ark_mem;
   int mxl;

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
   ark_mem->ark_setupNonNull = 1;
   // forces arkode to call lsetup prior to every time it calls lsolve
   ark_mem->ark_msbp = 0;

   //void* for containing linear solver memory
   ark_mem->ark_lmem;
   
   return (ARKSPILS_SUCCESS);
}
static int WrapLinearARKSolveInit(ARKodeMem ark_mem)
{
   cout<<"Called init"<<endl;
   return 0;
}

//ypred is the predicted y at the current time, fpred is f(t,ypred)
static int WrapLinearARKSolveSetup(ARKodeMem ark_mem, int convfail,
                                   N_Vector ypred, N_Vector fpred,
                                   booleantype *jcurPtr, N_Vector vtemp1,
                                   N_Vector vtemp2, N_Vector vtemp3)
{
   cout<<"Called setup at step "<<ark_mem->ark_nst<<endl;
   cout<<"entered setup at time "<<ark_mem->ark_tn<<endl;
   cout<<"entered setup at h "<<ark_mem->ark_h<<endl;
   if (ark_mem->ark_mass_matrix)
   {
      cout<<"Assume dy/dt=f_I(y,t) for implicit solve"<<endl;
      return -1;
   }
   
   return 0;
}

static int WrapLinearARKSolve(ARKodeMem ark_mem, N_Vector b,
                              N_Vector weight, N_Vector ycur,
                              N_Vector fcur)
{
   cout<<"Called solve at step "<<ark_mem->ark_nst<<endl;
   return 0;
}

static void WrapLinearARKSolveFree(ARKodeMem ark_mem)
{

   return;
}

static int WrapLinearSolveSetup(void* lmem, double tn,
                                   mfem::Vector* ypred, mfem::Vector* fpred)
{

}                                  

static int WrapLinearSolve(void* lmem, double tn, mfem::Vector* b, mfem::Vector* ycur,
                              mfem::Vector* fcur)
{

}
