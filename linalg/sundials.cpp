#include "../config/config.hpp"

#include "../linalg/operator.hpp"
#include "../linalg/ode.hpp"
#include "sundials.hpp"
#include "nvector_parcsr.hpp"
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

#define RELTOL RCONST(1.0e-6)
#define ABSTOL RCONST(1.0e-3)

using namespace std;

namespace mfem
{

CVODESolver::CVODESolver()
{
   y = NULL;
   f = NULL;

   /* Call CVodeCreate to create the solver memory */
   ode_mem=CVodeCreate(CV_ADAMS,CV_FUNCTIONAL);
   initialized_sundials=false;
   tolerances_set_sundials=false;
}

CVODESolver::CVODESolver(TimeDependentOperator &_f, Vector &_x, double&_t)
{
   y = NULL;
   f = NULL;

   /* Call CVodeCreate to create the solver memory */
   ode_mem=CVodeCreate(CV_ADAMS,CV_FUNCTIONAL);
   initialized_sundials=false;
   tolerances_set_sundials=false;
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
   printf("found local length %ld\n",yin_length);
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

void CVODESolver::SetStopTime(double tf)
{
   CVodeSetStopTime(ode_mem, tf);
}

void CVODESolver::Step(Vector &x, double &t, double &dt)
{
   int flag=0;
   realtype tout=t+dt;
   //   printf("starting step\n");
   TransferNVectorShallow(&x,y);
   //   printf("called transfernvectorshallow\n");

   //Step
   flag = CVode(ode_mem, tout, y, &t, CV_NORMAL);
   if (check_flag(&flag, "CVode", 1)) { MFEM_ABORT("CVode"); }
   return;
   flag = CVodeGetLastStep(ode_mem, &dt);
   if (check_flag(&flag, "CVodeGetLastStep", 1)) { return; }
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
                               Vector &_x, double &_t)
{
   y = NULL;
   f = &_f;
   //     printf("entered constructor\n");

   /* Call CVodeCreate to create the solver memory */
   ode_mem=CVodeCreate(CV_ADAMS,CV_FUNCTIONAL);
   //     printf("called CVodeCreate\n");
   initialized_sundials=false;
   tolerances_set_sundials=false;
   comm=_comm;
   ReInit(_f,_x,_t);
   //     printf("called ReInit(_f...\n");
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
   y = N_VMake_ParCsr(comm, yin_length, global_length,
                      ydata);   /* Allocate y vector */
}

void CVODEParSolver::CreateNVector(long int& yin_length, Vector* _x)
{
   int nprocs, myid;
   long int global_length;
   MPI_Comm_size(comm,&nprocs);
   realtype in=yin_length;
   realtype out;
   MPI_Allreduce(&in, &out, 1, PVEC_REAL_MPI_TYPE, MPI_SUM, comm);
   global_length= out;
   y = N_VNew_ParCsr(comm, yin_length, global_length);   /* Allocate y vector */
   TransferNVectorShallow(_x,y);
}

void CVODEParSolver::TransferNVectorShallow(Vector* _x, N_Vector &_y)
{
   NV_DATA_PC(_y)=_x->GetData();
   hypre_ParVectorPartitioning(NV_HYPRE_PARCSR_PC(_y))=
      hypre_ParVectorPartitioning((hypre_ParVector*) (*( (HypreParVector*) (_x))));
}

void CVODEParSolver::DestroyNVector(N_Vector& _y)
{

   if (NV_OWN_DATA_PC(y)==true)
   {
      N_VDestroy_ParCsr(y);   // Free y vector
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

ARKODESolver::ARKODESolver(TimeDependentOperator &_f, Vector &_x, double&_t)
{
   y = NULL;
   f = NULL;

   /* Call ARKodeCreate to create the solver memory */
   ode_mem=ARKodeCreate();
   initialized_sundials=false;
   tolerances_set_sundials=false;
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
   return ARKodeInit(_ode_mem, sun_f_fun, NULL, (realtype) _t, _y);
}

int ARKODESolver::WrapARKodeReInit(void* _ode_mem, double &_t, N_Vector &_y)
{
   return ARKodeReInit(_ode_mem, sun_f_fun, NULL, (realtype) _t, _y);
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
   printf("found local length %ld\n",yin_length);
   // Create an NVector
   CreateNVector(yin_length, &_x);

   if (initialized_sundials)
   {
      /* Call ARKodeReInit to initialize the integrator memory and specify the inital time t,
       * and the initial dependent variable vector y. */
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

void ARKODESolver::Step(Vector &x, double &t, double &dt)
{
   int flag=0;
   realtype tout=t+dt;
   //   printf("starting step\n");
   TransferNVectorShallow(&x,y);
   //   printf("called transfernvectorshallow\n");

   //Step
   flag = ARKode(ode_mem, tout, y, &t, ARK_NORMAL);
   if (check_flag(&flag, "ARKode", 1)) { MFEM_ABORT("ARKode"); }
   return;
   flag = ARKodeGetLastStep(ode_mem, &dt);
   if (check_flag(&flag, "ARKodeGetLastStep", 1)) { return; }
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
                                 Vector &_x, double &_t)
{
   y = NULL;
   f = &_f;
   //     printf("entered constructor\n");

   /* Call ARKodeCreate to create the solver memory */
   ode_mem=ARKodeCreate();
   //     printf("called ARKodeCreate\n");
   initialized_sundials=false;
   tolerances_set_sundials=false;
   comm=_comm;
   ReInit(_f,_x,_t);
   //     printf("called ReInit(_f...\n");
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
   y = N_VMake_ParCsr(comm, yin_length, global_length,
                      ydata);   /* Allocate y vector */
}

void ARKODEParSolver::CreateNVector(long int& yin_length, Vector* _x)
{
   int nprocs, myid;
   long int global_length;
   MPI_Comm_size(comm,&nprocs);
   realtype in=yin_length;
   realtype out;
   MPI_Allreduce(&in, &out, 1, PVEC_REAL_MPI_TYPE, MPI_SUM, comm);
   global_length= out;
   y = N_VNew_ParCsr(comm, yin_length, global_length);   /* Allocate y vector */
   TransferNVectorShallow(_x,y);
}

void ARKODEParSolver::TransferNVectorShallow(Vector* _x, N_Vector &_y)
{
   NV_DATA_PC(_y)=_x->GetData();
   hypre_ParVectorPartitioning(NV_HYPRE_PARCSR_PC(_y))=
      hypre_ParVectorPartitioning((hypre_ParVector*) (*( (HypreParVector*) (_x))));
}

void ARKODEParSolver::DestroyNVector(N_Vector& _y)
{

   if (NV_OWN_DATA_PC(y)==true)
   {
      N_VDestroy_ParCsr(y);   // Free y vector
   }
}

int ARKODEParSolver::WrapARKodeReInit(void* _ode_mem, double &_t, N_Vector &_y)
{
   return ARKodeInit(_ode_mem, sun_f_fun_par, NULL, (realtype) _t, _y);
}

int ARKODEParSolver::WrapARKodeInit(void* _ode_mem, double &_t, N_Vector &_y)
{
   return ARKodeInit(_ode_mem, sun_f_fun_par, NULL, (realtype) _t, _y);
}

#endif


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

#ifdef MFEM_USE_MPI
int sun_f_fun_par(realtype t, N_Vector y, N_Vector ydot,void *user_data)
{

   //using namespace mfem;
   realtype *ydata, *ydotdata;
   long int ylen, ydotlen;
   HYPRE_Int *col, *col2;
   hypre_ParVector *yparvec, *ydotparvec;
   MPI_Comm comm=MPI_COMM_WORLD;

   //ydata is now a pointer to the realtype data array in y
   ydata = NV_DATA_PC(y);
   ylen = NV_LOCLENGTH_PC(y);
   yparvec = NV_HYPRE_PARCSR_PC(y);

   col=hypre_ParVectorPartitioning(yparvec);

   // probably unnecessary, since overwriting ydot as output
   //ydotdata is now a pointer to the realtype data array in ydot
   ydotdata = NV_DATA_PC(ydot);
   ydotlen = NV_LOCLENGTH_PC(ydot);
   ydotparvec = NV_HYPRE_PARCSR_PC(ydot);

   col2=hypre_ParVectorPartitioning(ydotparvec);

   //f is now a pointer of abstract base class type TimeDependentOperator. It points to the TimeDependentOperator in the user_data struct
   mfem::TimeDependentOperator* f = (mfem::TimeDependentOperator*) user_data;

   //if gridfunction information necessary for Mult, keep Vector in userdata
   //  Vector* u = udata->u;
   //  u->SetData((double*) ydata);
   // Creates mfem vectors with pointers to the data array in y and in ydot respectively
   // Have not explicitly set as owndata, so allocated size is -size
   mfem::HypreParVector mfem_vector_y(comm, (HYPRE_Int) NV_GLOBLENGTH_PC(y),
                                      ydata, col);
   mfem::HypreParVector mfem_vector_ydot(comm, (HYPRE_Int) NV_GLOBLENGTH_PC(ydot),
                                         ydotdata, col2);
   f->SetTime(t);
   f->Mult(mfem_vector_y,mfem_vector_ydot);

   return (0);
}
#endif
