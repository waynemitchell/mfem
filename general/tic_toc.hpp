// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.googlecode.com.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#ifndef MFEM_TIC_TOC
#define MFEM_TIC_TOC

#include "../config/config.hpp"

#ifndef MFEM_TIMER_TYPE
#ifndef _WIN32
#define MFEM_TIMER_TYPE 0
#else
#define MFEM_TIMER_TYPE 3
#endif
#endif

#if (MFEM_TIMER_TYPE == 0)
#include <ctime>
#elif (MFEM_TIMER_TYPE == 1)
#include <sys/times.h>
#elif (MFEM_TIMER_TYPE == 2)
#include <time.h>
#if (!defined(CLOCK_MONOTONIC) || !defined(CLOCK_PROCESS_CPUTIME_ID))
#error "CLOCK_MONOTONIC and CLOCK_PROCESS_CPUTIME_ID not defined in <time.h>"
#endif
#elif (MFEM_TIMER_TYPE == 3)
#include <windows.h>
#else
#error "Unknown MFEM_TIMER_TYPE"
#endif

namespace mfem
{

/// Timing object
class StopWatch
{
private:
#if (MFEM_TIMER_TYPE == 0)
   std::clock_t user_time, start_utime;
#elif (MFEM_TIMER_TYPE == 1)
   clock_t real_time, user_time, syst_time;
   clock_t start_rtime, start_utime, start_stime;
   long my_CLK_TCK;
   void Current(clock_t *, clock_t *, clock_t *);
#elif (MFEM_TIMER_TYPE == 2)
   struct timespec real_time, user_time;
   struct timespec start_rtime, start_utime;
   inline void GetRealTime(struct timespec &tp);
   inline void GetUserTime(struct timespec &tp);
#elif (MFEM_TIMER_TYPE == 3)
   LARGE_INTEGER frequency, real_time, start_rtime;
#endif
   short Running;
public:
   StopWatch();
   void Clear();
   void Start();
   void Stop();
   double Resolution();
   double RealTime();
   double UserTime();
   double SystTime();
};


extern StopWatch tic_toc;

/// Start timing
extern void tic();

/// End timing
extern double toc();

}

#endif
