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

#include <limits.h>
#include <sys/times.h>
#include <unistd.h>

#include "tic_toc.hpp"

StopWatch::StopWatch()
{
#ifndef MFEM_USE_POSIX_CLOCKS
   my_CLK_TCK = sysconf(_SC_CLK_TCK);
   real_time = user_time = syst_time = 0;
#else
   real_time.tv_sec  = user_time.tv_sec  = 0;
   real_time.tv_nsec = user_time.tv_nsec = 0;
#endif
   Running = 0;
}

#ifndef MFEM_USE_POSIX_CLOCKS
void StopWatch::Current(clock_t *r, clock_t *u, clock_t *s)
{
   struct tms my_tms;

   *r = times(&my_tms);
   *u = my_tms.tms_utime;
   *s = my_tms.tms_stime;
}
#else
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
#ifndef MFEM_USE_POSIX_CLOCKS
   real_time = user_time = syst_time = 0;
   if (Running)
      Current(&start_rtime, &start_utime, &start_stime);
#else
   real_time.tv_sec  = user_time.tv_sec  = 0;
   real_time.tv_nsec = user_time.tv_nsec = 0;
   if (Running)
   {
      GetRealTime(start_rtime);
      GetUserTime(start_utime);
   }
#endif
}

void StopWatch::Start()
{
   if (Running) return;
#ifndef MFEM_USE_POSIX_CLOCKS
   Current(&start_rtime, &start_utime, &start_stime);
#else
   GetRealTime(start_rtime);
   GetUserTime(start_utime);
#endif
   Running = 1;
}

void StopWatch::Stop()
{
   if (!Running) return;

#ifndef MFEM_USE_POSIX_CLOCKS
   clock_t curr_rtime, curr_utime, curr_stime;

   Current(&curr_rtime, &curr_utime, &curr_stime);
   real_time += ( curr_rtime - start_rtime );
   user_time += ( curr_utime - start_utime );
   syst_time += ( curr_stime - start_stime );
#else
   struct timespec curr_rtime, curr_utime;

   GetRealTime(curr_rtime);
   GetUserTime(curr_utime);
   real_time.tv_sec  += ( curr_rtime.tv_sec  - start_rtime.tv_sec  );
   real_time.tv_nsec += ( curr_rtime.tv_nsec - start_rtime.tv_nsec );
   user_time.tv_sec  += ( curr_utime.tv_sec  - start_utime.tv_sec  );
   user_time.tv_nsec += ( curr_utime.tv_nsec - start_utime.tv_nsec );
#endif

   Running = 0;
}

double StopWatch::Resolution()
{
#ifndef MFEM_USE_POSIX_CLOCKS
   return static_cast<double>(1.) / static_cast<double>( my_CLK_TCK );
#else
   struct timespec res;

   // return the resolution of the "real time" clock, CLOCK_MONOTONIC, which may
   // be different from the resolution of the "user time" clock,
   // CLOCK_PROCESS_CPUTIME_ID.
   clock_getres(CLOCK_MONOTONIC, &res);
   return res.tv_sec + 1e-9*res.tv_nsec;
#endif
}

double StopWatch::RealTime()
{
#ifndef MFEM_USE_POSIX_CLOCKS
   clock_t curr_rtime, curr_utime, curr_stime;
   clock_t rtime = real_time;

   if (Running)
   {
      Current(&curr_rtime, &curr_utime, &curr_stime);
      rtime += (curr_rtime - start_rtime);
   }

   return (double)(rtime) / my_CLK_TCK;
#else
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
#endif
}

double StopWatch::UserTime()
{
#ifndef MFEM_USE_POSIX_CLOCKS
   clock_t curr_rtime, curr_utime, curr_stime;
   clock_t utime = user_time;

   if (Running)
   {
      Current(&curr_rtime, &curr_utime, &curr_stime);
      utime += (curr_utime - start_utime);
   }

   return (double)(utime) / my_CLK_TCK;
#else
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
#endif
}

double StopWatch::SystTime()
{
#ifndef MFEM_USE_POSIX_CLOCKS
   clock_t curr_rtime, curr_utime, curr_stime;
   clock_t stime = syst_time;

   if (Running)
   {
      Current(&curr_rtime, &curr_utime, &curr_stime);
      stime += (curr_stime - start_stime);
   }

   return (double)(stime) / my_CLK_TCK;
#else
   return 0.;
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
