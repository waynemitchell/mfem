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
#include <nvector/nvector_parallel.h>  /* serial N_Vector types, fcts., macros */
#include <sundials/sundials_types.h> /* definition of type realtype */
#include <sundials/sundials_math.h>  /* definition of ABS and EXP */

#define RELTOL RCONST(1.0e-6)
#define ABSTOL RCONST(1.0e-3)

using namespace std;

namespace mfem
{

CVODESolver::CVODESolver()
{
   f = NULL;

   PtrToStep=&CVODESolver::SetIC;

   /* Call CVodeCreate to create the solver memory */
   ode_mem=CVodeCreate(CV_ADAMS,CV_FUNCTIONAL);
   step_type=CV_NORMAL;
   parallel = false;
}

CVODESolver::CVODESolver(MPI_Comm _comm)
{
   f = NULL;

   PtrToStep=&CVODESolver::SetIC;

   /* Call CVodeCreate to create the solver memory */
   ode_mem=CVodeCreate(CV_ADAMS,CV_FUNCTIONAL);
   step_type=CV_NORMAL;
   comm=_comm;
   parallel = true;
}

void CVODESolver::CreateNVector()
{
   long int local_size=f->Width(); //local length
   long int yin_length=local_size;
   //intial time
   realtype t = 0.0;
   realtype *yin;
#ifdef MFEM_USE_MPI
   if (parallel)
   {
      int nprocs, extra, myid, ilower, iupper;
      MPI_Comm_size(comm,&nprocs);
      MPI_Comm_rank(comm,&myid);

      realtype in=local_size;
      realtype out;
      MPI_Allreduce(&in, &out, 1, PVEC_REAL_MPI_TYPE, MPI_SUM, comm);
      yin_length= out;


      int id=0;
      while (id < nprocs)
      {
         if (myid == id)
         {
            cout<<"pid="<<myid<<" local_size="<<local_size<<" yin_length="<<yin_length<<endl;
            fflush (stdout);
            //cout.flush();
         }
         id++;
         MPI_Barrier(comm);
      }
      yin= new realtype[local_size];
      // Create a parallel vector
      y = N_VMake_Parallel(comm, local_size, yin_length,
                           yin);   /* Allocate y vector */
      if (NV_DATA_P(y)==NULL)
      {
         cout<<"data in y unintialized"<<endl;
      }
      if (check_flag((void*)y, "N_VMake_Parallel", 0)) { MFEM_ABORT("N_VMake_Parallel"); }
   }
   else
   {
      yin= new realtype[yin_length];
      // Create a serial vector
      y = N_VMake_Serial(yin_length,yin);   /* Allocate y vector */
      if (check_flag((void*)y, "N_VNew_Serial", 0)) { MFEM_ABORT("N_VNew_Serial"); }
   }
#else
   yin= new realtype[yin_length];
   // Create a serial vector
   y = N_VMake_Serial(yin_length,yin);   /* Allocate y vector */
   if (check_flag((void*)y, "N_VNew_Serial", 0)) { MFEM_ABORT("N_VNew_Serial"); }
#endif
}

void CVODESolver::CreateNVector(Vector& x)
{
   cout<<"In setIC"<<endl;
   long int local_size=f->Width(); //local length
   long int yin_length=local_size;
   //intial time
   realtype t = 0.0;
   realtype *yin;
#ifdef MFEM_USE_MPI
   if (parallel)
   {
      int nprocs, extra, myid, ilower, iupper;
      MPI_Comm_size(comm,&nprocs);
      MPI_Comm_rank(comm,&myid);

      /*
      realtype in=local_size;
      realtype out;
      MPI_Allreduce(&in, &out, 1, PVEC_REAL_MPI_TYPE, MPI_SUM, comm);
      yin_length= out;

      local_size = x.Size();
      yin=NV_DATA_P(y);
      if(yin!=NULL)
        delete yin;
      yin=NULL;*/
      NV_DATA_P(y)=x.GetData();

      int id=0;
      while (id < nprocs)
      {
         if (myid == id)
         {
            cout<<"SetIC\tpid="<<myid<<" local_size="<<local_size<<" yin_length="<<yin_length<<endl;
            fflush (stdout);
            //cout.flush();
         }
         id++;
         MPI_Barrier(comm);
      }
      // Create a parallel vector
      //   y = N_VMake_Parallel(comm, local_size ,yin_length, yin);   /* Allocate y vector */
      //   if (check_flag((void*)y, "N_VNew_Parallel", 0)) { MFEM_ABORT("N_VNew_Parallel"); }
   }
   else
   {
      realtype* yin=NV_DATA_S(y);
      if (yin!=NULL)
      {
         delete yin;
      }
      yin=NULL;
      yin=x.GetData();
      // Create a serial vector
      y = N_VMake_Serial(yin_length,yin);   /* Allocate y vector */
      if (check_flag((void*)y, "N_VNew_Serial", 0)) { MFEM_ABORT("N_VNew_Serial"); }
   }
#else
   realtype* yin=NV_DATA_S(y);
   if (yin!=NULL)
   {
      delete yin;
   }
   yin=NULL;
   yin=x.GetData();
   // Create a serial vector
   y = N_VMake_Serial(yin_length,yin);   /* Allocate y vector */
   if (check_flag((void*)y, "N_VNew_Serial", 0)) { MFEM_ABORT("N_VNew_Serial"); }
#endif
}

void CVODESolver::Init(TimeDependentOperator &_f)
{
   //not checking that initial pointers set to NULL:
   f = &_f;
   //intial time
   realtype t = 0.0;
   int flag;

   CreateNVector();

   /* Call CVodeInit to initialize the integrator memory and specify the
    * user's right hand side function in x'=f(t,x), the inital time t, and
    * the initial dependent variable vector y. */
   flag = CVodeInit(ode_mem, sun_f_fun, t, y);
   if (check_flag(&flag, "CVodeInit", 1)) { MFEM_ABORT("CVodeInit"); }

   //Come up with a better way to test whether this has already been set
   SetSStolerances(RELTOL,ABSTOL);
   cout<<"f!=NULL"<<(f!=NULL)<<endl;
   cout<<"&(*f):"<<&(*f)<<endl;
   cout<<"&(_f):"<<&_f<<endl;

   /* Set the pointer to user-defined data */
   flag = CVodeSetUserData(ode_mem, this->f);
   //cout.flush();
#ifdef MFEM_USE_MPI
   MPI_Barrier(comm);
#endif
   if (check_flag(&flag, "CVodeSetUserData", 1)) { MFEM_ABORT("CVodeSetUserData"); }

}

void CVODESolver::Init(TimeDependentOperator &_f, Vector &x, double&t,
                       double&dt)
{
   int flag;
   //not checking that initial pointers set to NULL:
   f = &_f;
   if (_f.Width()!=x.Size())
   {
      mfem_error("Function f takes argument of different size than intial condition");
   }

   CreateNVector(x);

   /* Call CVodeInit to initialize the integrator memory and specify the
    * user's right hand side function in x'=f(t,x), the inital time t, and
    * the initial dependent variable vector y. */
   flag = CVodeInit(ode_mem, sun_f_fun, (realtype) t, y);
   if (check_flag(&flag, "CVodeInit", 1)) { MFEM_ABORT("CVodeInit"); }

   //Come up with a better way to test whether this has already been set
   SetSStolerances(RELTOL,ABSTOL);

   /* Set the pointer to user-defined data */
   flag = CVodeSetUserData(ode_mem, this->f);
   if (check_flag(&flag, "CVodeSetUserData", 1)) { MFEM_ABORT("CVodeSetUserData"); }

   PtrToStep=&CVODESolver::GetY;

}

void CVODESolver::ReInit(TimeDependentOperator &_f)
{
   f = &_f;
   int flag;

   //This addition may need destruction of previous vector data
   CreateNVector();

   //Come up with a better way to test whether this has already been set
   SetSStolerances(RELTOL,ABSTOL);

   /* Set the pointer to user-defined data */
   flag = CVodeSetUserData(ode_mem, this->f);
   if (check_flag(&flag, "CVodeSetUserData", 1)) { MFEM_ABORT("CVodeSetUserData"); }

   PtrToStep=&CVODESolver::SetIC;

}

void CVODESolver::ReInit(TimeDependentOperator &_f, Vector &x, double&t,
                         double&dt)
{
   int flag;
   //not checking that initial pointers set to NULL:
   f = &_f;
   if (_f.Width()!=x.Size())
   {
      mfem_error("Function f takes argument of different size than intial condition");
   }

   CreateNVector(x);

   /* Call CVodeInit to initialize the integrator memory and specify the
    * user's right hand side function in x'=f(t,x), the inital time t, and
    * the initial dependent variable vector y. */
   flag = CVodeReInit(ode_mem, (realtype) t, y);
   if (check_flag(&flag, "CVodeInit", 1)) { MFEM_ABORT("CVodeInit"); }

   //Come up with a better way to test whether this has already been set
   SetSStolerances(RELTOL,ABSTOL);

   /* Set the pointer to user-defined data */
   flag = CVodeSetUserData(ode_mem, this->f);
   if (check_flag(&flag, "CVodeSetUserData", 1)) { MFEM_ABORT("CVodeSetUserData"); }
   PtrToStep=&CVODESolver::GetY;

}

void CVODESolver::SetSStolerances(realtype reltol, realtype abstol)
{
   int flag=0;
   /* Call CVodeSStolerances to specify the scalar relative tolerance
    * and scalar absolute tolerance */
   flag = CVodeSStolerances(ode_mem, reltol, abstol);
   if (check_flag(&flag, "CVodeSStolerances", 1)) { mfem_error("CVodeSStolerances"); }
}

void CVODESolver::SetStopTime(double tf)
{
   CVodeSetStopTime(ode_mem, tf);
}

void CVODESolver::SetIC(Vector &x, double&t, double&dt)
{
   int flag=0;
   CreateNVector(x);
   flag = CVodeReInit(ode_mem, t, y);
   //Come up with a better way to test whether this has already been set
   /*   SetSStolerances(RELTOL,ABSTOL);*/

   /* Set the minimum step size */
   //   flag = CVodeSetMinStep(ode_mem, dt);
   //   if(check_flag(&flag, "CVodeSetMinStep", 1)) return;

   /* Set the maximum step size */

   /*   flag = CVodeSetMaxStep(ode_mem, dt);
      if (check_flag(&flag, "CVodeSetMaxStep", 1)) { return; }
   */
   PtrToStep=&CVODESolver::GetY;
}

void CVODESolver::GetY(Vector &x, double&t, double&dt)
{
#ifdef MFEM_USE_MPI
   if (parallel)
   {
      NV_DATA_P(y)=x.GetData();
   }
   else
   {
      NV_DATA_S(y)=x.GetData();
   }
#else
   NV_DATA_S(y)=x.GetData();
#endif
}

void CVODESolver::Step(Vector &x, double &t, double &dt)
{
   int flag=0;
   realtype tout=t+dt;

   (this->*PtrToStep)(x,t,dt);
   /*   #ifdef MFEM_USE_MPI
      int nprocs, myid;
      MPI_Comm_size(comm,&nprocs);
      MPI_Comm_rank(comm,&myid);

      int id=0;
      while (id < nprocs) {
      if (myid == id) {
          cout<<"pid="<<myid<<" local_size="<<x.Size()<<endl;
          fflush (stdout);
      }
      id++;
      MPI_Barrier(comm);
      }
      MPI_Barrier(comm);
      #endif*/
   //Step
   flag = CVode(ode_mem, tout, y, &t, CV_NORMAL);
   if (check_flag(&flag, "CVode", 1)) { mfem_error("CVode"); }
   flag = CVodeGetLastStep(ode_mem, &dt);
   if (check_flag(&flag, "CVodeGetLastStep", 1)) { mfem_error("CVodeGetLastStep"); }
}

CVODESolver::~CVODESolver()
{
   // Free the used memory.
   // Clean up and return with successful completion
   if (NV_OWN_DATA_S(y)==true)
   {
#ifdef MFEM_USE_MPI
      N_VDestroy_Parallel(y);
#else
      N_VDestroy_Serial(y);   // Free y vector
#endif
   }
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


ARKODESolver::ARKODESolver()
{
   y = NULL;
   f = NULL;

   PtrToStep=&ARKODESolver::SetIC;

   /* Call ARKodeCreate to create the solver memory */
   ode_mem=ARKodeCreate();
   step_type=ARK_NORMAL;
}

ARKODESolver::ARKODESolver(MPI_Comm _comm)
{
   y = NULL;
   f = NULL;

   PtrToStep=&ARKODESolver::SetIC;

   /* Call CVodeCreate to create the solver memory */
   ode_mem=ARKodeCreate();
   step_type=ARK_NORMAL;
   comm=_comm;
}

void ARKODESolver::Init(TimeDependentOperator &_f)
{
   f = &_f;
   long int yin_length=_f.Width(); //assume don't have initial condition in Init
   //intial time
   realtype t = 0.0;
   realtype *yin;
   yin= new realtype[yin_length];
   int flag;


   // Create a serial vector
   y = N_VMake_Serial(yin_length,yin);   /* Allocate y vector */
   if (check_flag((void*)y, "N_VNew_Serial", 0)) { MFEM_ABORT("N_VNew_Serial"); }

   /* Call ARKodeInit to initialize the integrator memory and specify the
    * user's right hand side function in x'=f(t,x), the inital time t, and
    * the initial dependent variable vector y. */
   flag = ARKodeInit(ode_mem, sun_f_fun, NULL, t, y);
   if (check_flag(&flag, "ARKodeInit", 1)) { MFEM_ABORT("ARKodeInit"); }

   /* Call ARKodeSetERKTableNum to compare directly with method integration
    * in MFEM example 9. */
   flag = ARKodeSetERKTableNum(ode_mem, 3);
   if (check_flag(&flag, "ARKodeSetERKTableNum", 1)) { return; }

   /* Set the pointer to user-defined data */
   flag = ARKodeSetUserData(ode_mem, this->f);
   if (check_flag(&flag, "ARKodeSetUserData", 1)) { MFEM_ABORT("ARKodeSetUserData"); }

}

void ARKODESolver::Init(TimeDependentOperator &_f, Vector &x, double&t,
                        double&dt)
{
   //not checking that initial pointers set to NULL:
   f = &_f;
   long int yin_length=_f.Width();
   if (_f.Width()!=x.Size())
   {
      mfem_error("Function f takes argument of different size than intial condition");
   }

   realtype *yin = x.GetData();
   int flag;


   // Create a serial vector
   y = N_VMake_Serial(yin_length,yin);   /* Allocate y vector */
   if (check_flag((void*)y, "N_VNew_Serial", 0)) { MFEM_ABORT("N_VNew_Serial"); }

   /* Call ARKodeInit to initialize the integrator memory and specify the
    * user's right hand side function in x'=f(t,x), the inital time t, and
    * the initial dependent variable vector y. */
   flag = ARKodeInit(ode_mem, sun_f_fun, NULL, (realtype) t, y);
   if (check_flag(&flag, "ARKodeInit", 1)) { MFEM_ABORT("ARKodeInit"); }

   //Come up with a better way to test whether this has already been set
   SetSStolerances(RELTOL,ABSTOL);

   /* Set the pointer to user-defined data */
   flag = ARKodeSetUserData(ode_mem, this->f);
   if (check_flag(&flag, "ARKodeSetUserData", 1)) { MFEM_ABORT("ARKodeSetUserData"); }

   PtrToStep=&ARKODESolver::GetY;

}

void ARKODESolver::ReInit(TimeDependentOperator &_f)
{
   f = &_f;
   int flag;

   //Come up with a better way to test whether this has already been set
   SetSStolerances(RELTOL,ABSTOL);

   /* Set the pointer to user-defined data */
   flag = ARKodeSetUserData(ode_mem, this->f);
   if (check_flag(&flag, "ARKodeSetUserData", 1)) { MFEM_ABORT("ARKodeSetUserData"); }

   PtrToStep=&ARKODESolver::SetIC;

}

void ARKODESolver::ReInit(TimeDependentOperator &_f, Vector &x, double&t,
                          double&dt)
{
   //not checking that initial pointers set to NULL:
   f = &_f;
   long int yin_length=_f.Width();
   if (_f.Width()!=x.Size())
   {
      mfem_error("Function f takes argument of different size than intial condition");
   }

   realtype *yin = x.GetData();
   int flag;

   // Create a serial vector
   y = N_VMake_Serial(yin_length,yin);   /* Allocate y vector */
   if (check_flag((void*)y, "N_VNew_Serial", 0)) { MFEM_ABORT("N_VNew_Serial"); }

   /* Call ARKodeInit to initialize the integrator memory and specify the
    * user's right hand side function in x'=f(t,x), the inital time t, and
    * the initial dependent variable vector y. */
   flag = ARKodeReInit(ode_mem, sun_f_fun, NULL, (realtype) t, y);
   if (check_flag(&flag, "ARKodeInit", 1)) { MFEM_ABORT("ARKodeInit"); }

   /* Set the pointer to user-defined data */
   flag = ARKodeSetUserData(ode_mem, this->f);
   if (check_flag(&flag, "ARKodeSetUserData", 1)) { MFEM_ABORT("ARKodeSetUserData"); }
   PtrToStep=&ARKODESolver::GetY;

}

void ARKODESolver::SetSStolerances(realtype reltol, realtype abstol)
{
   int flag=0;
   /* Call ARKodeSStolerances to specify the scalar relative tolerance
    * and scalar absolute tolerance */
   flag = ARKodeSStolerances(ode_mem, reltol, abstol);
   if (check_flag(&flag, "ARKodeSStolerances", 1)) { return; }
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

   //Come up with a better way to test whether this has already been set
   SetSStolerances(RELTOL,ABSTOL);
   /*
      if (step_type==ARK_ONE_STEP)
      {
         flag = ARKodeSetFixedStep(ode_mem, dt);
         if (check_flag(&flag, "ARKodeSetInitStep", 1)) { return; }
      }
   */
   /* Set the minimum step size */
   //   flag = ARKodeSetMinStep(ode_mem, dt);
   //   if(check_flag(&flag, "ARKodeSetMinStep", 1)) return;
   /* Set the maximum step size */
   /*
      flag = ARKodeSetMaxStep(ode_mem, dt);
      if (check_flag(&flag, "ARKodeSetMaxStep", 1)) { return; }
   */
   PtrToStep=&ARKODESolver::GetY;
}

void ARKODESolver::GetY(Vector &x, double&t, double&dt)
{
   NV_DATA_S(y)=x.GetData();
}

void ARKODESolver::Step(Vector &x, double &t, double &dt)
{
   int flag=0;
   realtype tout=t+dt;

   (this->*PtrToStep)(x,t,dt);

   //Step
   flag = ARKode(ode_mem, tout, y, &t, step_type);
   if (check_flag(&flag, "ARKode", 1)) { MFEM_ABORT("ARKode"); }
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
#ifdef MFEM_USE_MPI
      N_VDestroy_Parallel(y);
#else
      N_VDestroy_Serial(y);   // Free y vector
#endif
   }
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

}


int sun_f_fun(realtype t, N_Vector y, N_Vector ydot,void *user_data)
{

   //using namespace mfem;
   realtype *ydata, *ydotdata;
   long int ylen, ydotlen;
   mfem::Vector *vec_y, *vec_ydot;

#ifdef MFEM_USE_MPI
   if (true) //find a way to pass parallel here
   {
      MPI_Comm comm=NV_COMM_P(y); //find a way to pass comm here
      //ydata is now a pointer to the realtype data array in y
      ydata = NV_DATA_P(y);
      ylen = NV_GLOBLENGTH_P(y);

      // probably unnecessary, since overwriting ydot as output
      //ydotdata is now a pointer to the realtype data array in ydot
      ydotdata = NV_DATA_P(ydot);
      ydotlen = NV_GLOBLENGTH_P(ydot);

      //if gridfunction information necessary for Mult, keep Vector in userdata
      //  Vector* u = udata->u;
      //  u->SetData((double*) ydata);
      // Creates mfem vectors with pointers to the data array in y and in ydot respectively
      // Have not explicitly set as owndata, so allocated size is -size
      mfem::HypreParVector mfem_vector_y(comm, ylen, (double*) ydata, NULL);
      mfem::HypreParVector mfem_vector_ydot(comm, ydotlen, (double*) ydotdata, NULL);
      vec_y=&mfem_vector_y;
      vec_ydot=&mfem_vector_ydot;
   }
   else
   {
      //ydata is now a pointer to the realtype data array in y
      ydata = NV_DATA_S(y);
      ylen = NV_LENGTH_S(y);

      // probably unnecessary, since overwriting ydot as output
      //ydotdata is now a pointer to the realtype data array in ydot
      ydotdata = NV_DATA_S(ydot);
      ydotlen = NV_LENGTH_S(ydot);

      //if gridfunction information necessary for Mult, keep Vector in userdata
      //  Vector* u = udata->u;
      //  u->SetData((double*) ydata);
      // Creates mfem vectors with pointers to the data array in y and in ydot respectively
      // Have not explicitly set as owndata, so allocated size is -size
      mfem::Vector mfem_vector_y((double*) ydata, ylen);
      mfem::Vector mfem_vector_ydot((double*) ydotdata, ydotlen);
      vec_y=&mfem_vector_y;
      vec_ydot=&mfem_vector_ydot;
   }
#else
   //ydata is now a pointer to the realtype data array in y
   ydata = NV_DATA_S(y);
   ylen = NV_LENGTH_S(y);

   // probably unnecessary, since overwriting ydot as output
   //ydotdata is now a pointer to the realtype data array in ydot
   ydotdata = NV_DATA_S(ydot);
   ydotlen = NV_LENGTH_S(ydot);

   //if gridfunction information necessary for Mult, keep Vector in userdata
   //  Vector* u = udata->u;
   //  u->SetData((double*) ydata);
   // Creates mfem vectors with pointers to the data array in y and in ydot respectively
   // Have not explicitly set as owndata, so allocated size is -size
   mfem::Vector mfem_vector_y((double*) ydata, ylen);
   mfem::Vector mfem_vector_ydot((double*) ydotdata, ydotlen);
   vec_y=&mfem_vector_y;
   vec_ydot=&mfem_vector_ydot;
#endif

   //f is now a pointer of abstract base class type TimeDependentOperator. It points to the TimeDependentOperator in the user_data struct
   mfem::TimeDependentOperator* f = (mfem::TimeDependentOperator*) user_data;

   f->SetTime(t);
   f->Mult(*vec_y,*vec_ydot);

   return (0);
}


