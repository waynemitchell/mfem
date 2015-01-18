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

#include "tic_toc.hpp"

#if (MFEM_TIMER_TYPE == 1)
#include <climits>
#include <unistd.h>
#endif

namespace mfem
{

StopWatch::StopWatch()
{
#if (MFEM_TIMER_TYPE == 0)
   user_time = 0;
#elif (MFEM_TIMER_TYPE == 1)
   my_CLK_TCK = sysconf(_SC_CLK_TCK);
   real_time = user_time = syst_time = 0;
#elif (MFEM_TIMER_TYPE == 2)
   real_time.tv_sec  = user_time.tv_sec  = 0;
   real_time.tv_nsec = user_time.tv_nsec = 0;
#elif (MFEM_TIMER_TYPE == 3)
   QueryPerformanceFrequency(&frequency);
   real_time.QuadPart = 0;
#endif
   Running = 0;
}

#if (MFEM_TIMER_TYPE == 1)
void StopWatch::Current(clock_t *r, clock_t *u, clock_t *s)
{
   struct tms my_tms;

   *r = times(&my_tms);
   *u = my_tms.tms_utime;
   *s = my_tms.tms_stime;
}
#elif (MFEM_TIMER_TYPE == 2)
inline void StopWatch::GetRealTime(struct timespec &tp)
{
   clock_gettime(CLOCK_MONOTONIC, &tp);
}

inline void StopWatch::GetUserTime(struct timespec &tp)
{
   clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &tp);
}
#endif

void StopWatch::Clear()
{
#if (MFEM_TIMER_TYPE == 0)
   user_time = 0;
   if (Running)
      start_utime = std::clock();
#elif (MFEM_TIMER_TYPE == 1)
   real_time = user_time = syst_time = 0;
   if (Running)
      Current(&start_rtime, &start_utime, &start_stime);
#elif (MFEM_TIMER_TYPE == 2)
   real_time.tv_sec  = user_time.tv_sec  = 0;
   real_time.tv_nsec = user_time.tv_nsec = 0;
   if (Running)
   {
      GetRealTime(start_rtime);
      GetUserTime(start_utime);
   }
#elif (MFEM_TIMER_TYPE == 3)
   real_time.QuadPart = 0;
   if (Running)
      QueryPerformanceCounter(&start_rtime);
#endif
}

void StopWatch::Start()
{
   if (Running) return;
#if (MFEM_TIMER_TYPE == 0)
   start_utime = std::clock();
#elif (MFEM_TIMER_TYPE == 1)
   Current(&start_rtime, &start_utime, &start_stime);
#elif (MFEM_TIMER_TYPE == 2)
   GetRealTime(start_rtime);
   GetUserTime(start_utime);
#elif (MFEM_TIMER_TYPE == 3)
   QueryPerformanceCounter(&start_rtime);
#endif
   Running = 1;
}

void StopWatch::Stop()
{
   if (!Running) return;
#if (MFEM_TIMER_TYPE == 0)
   user_time += ( std::clock() - start_utime );
#elif (MFEM_TIMER_TYPE == 1)
   clock_t curr_rtime, curr_utime, curr_stime;
   Current(&curr_rtime, &curr_utime, &curr_stime);
   real_time += ( curr_rtime - start_rtime );
   user_time += ( curr_utime - start_utime );
   syst_time += ( curr_stime - start_stime );
#elif (MFEM_TIMER_TYPE == 2)
   struct timespec curr_rtime, curr_utime;
   GetRealTime(curr_rtime);
   GetUserTime(curr_utime);
   real_time.tv_sec  += ( curr_rtime.tv_sec  - start_rtime.tv_sec  );
   real_time.tv_nsec += ( curr_rtime.tv_nsec - start_rtime.tv_nsec );
   user_time.tv_sec  += ( curr_utime.tv_sec  - start_utime.tv_sec  );
   user_time.tv_nsec += ( curr_utime.tv_nsec - start_utime.tv_nsec );
#elif (MFEM_TIMER_TYPE == 3)
   LARGE_INTEGER curr_rtime;
   QueryPerformanceCounter(&curr_rtime);
   real_time.QuadPart += (curr_rtime.QuadPart - start_rtime.QuadPart);
#endif
   Running = 0;
}

double StopWatch::Resolution()
{
#if (MFEM_TIMER_TYPE == 0)
   return 1.0 / CLOCKS_PER_SEC; // potential resolution
#elif (MFEM_TIMER_TYPE == 1)
   return static_cast<double>(1.) / static_cast<double>( my_CLK_TCK );
#elif (MFEM_TIMER_TYPE == 2)
   // return the resolution of the "real time" clock, CLOCK_MONOTONIC, which may
   // be different from the resolution of the "user time" clock,
   // CLOCK_PROCESS_CPUTIME_ID.
   struct timespec res;
   clock_getres(CLOCK_MONOTONIC, &res);
   return res.tv_sec + 1e-9*res.tv_nsec;
#elif (MFEM_TIMER_TYPE == 3)
   return 1.0 / frequency.QuadPart;
#endif
}

double StopWatch::RealTime()
{
#if (MFEM_TIMER_TYPE == 0)
   return UserTime();
#elif (MFEM_TIMER_TYPE == 1)
   clock_t curr_rtime, curr_utime, curr_stime;
   clock_t rtime = real_time;
   if (Running)
   {
      Current(&curr_rtime, &curr_utime, &curr_stime);
      rtime += (curr_rtime - start_rtime);
   }
   return (double)(rtime) / my_CLK_TCK;
#elif (MFEM_TIMER_TYPE == 2)
   if (Running)
   {
      struct timespec curr_rtime;
      GetRealTime(curr_rtime);
      return ((real_time.tv_sec + (curr_rtime.tv_sec - start_rtime.tv_sec)) +
              1e-9*(real_time.tv_nsec +
                    (curr_rtime.tv_nsec - start_rtime.tv_nsec)));
   }
   else
   {
      return real_time.tv_sec + 1e-9*real_time.tv_nsec;
   }
#elif (MFEM_TIMER_TYPE == 3)
   LARGE_INTEGER curr_rtime, rtime = real_time;
   if (Running)
   {
      QueryPerformanceCounter(&curr_rtime);
      rtime.QuadPart += (curr_rtime.QuadPart - start_rtime.QuadPart);
   }
   return (double)(rtime.QuadPart) / frequency.QuadPart;
#endif
}

double StopWatch::UserTime()
{
#if (MFEM_TIMER_TYPE == 0)
   std::clock_t utime = user_time;
   if (Running)
      utime += (std::clock() - start_utime);
   return (double)(utime) / CLOCKS_PER_SEC;
#elif (MFEM_TIMER_TYPE == 1)
   clock_t curr_rtime, curr_utime, curr_stime;
   clock_t utime = user_time;
   if (Running)
   {
      Current(&curr_rtime, &curr_utime, &curr_stime);
      utime += (curr_utime - start_utime);
   }
   return (double)(utime) / my_CLK_TCK;
#elif (MFEM_TIMER_TYPE == 2)
   if (Running)
   {
      struct timespec curr_utime;
      GetUserTime(curr_utime);
      return ((user_time.tv_sec + (curr_utime.tv_sec - start_utime.tv_sec)) +
              1e-9*(user_time.tv_nsec +
                    (curr_utime.tv_nsec - start_utime.tv_nsec)));
   }
   else
   {
      return user_time.tv_sec + 1e-9*user_time.tv_nsec;
   }
#elif (MFEM_TIMER_TYPE == 3)
   return RealTime();
#endif
}

double StopWatch::SystTime()
{
#if (MFEM_TIMER_TYPE == 1)
   clock_t curr_rtime, curr_utime, curr_stime;
   clock_t stime = syst_time;
   if (Running)
   {
      Current(&curr_rtime, &curr_utime, &curr_stime);
      stime += (curr_stime - start_stime);
   }
   return (double)(stime) / my_CLK_TCK;
#else
   return 0.0;
#endif
}


StopWatch tic_toc;

void tic()
{
   tic_toc.Clear();
   tic_toc.Start();
}

double toc()
{
   return tic_toc.UserTime();
}

}
