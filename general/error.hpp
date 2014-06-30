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

#ifndef MFEM_ERROR
#define MFEM_ERROR

#include <iomanip>
#include <sstream>

void mfem_error (const char *msg = NULL);

// This is nice because it shows the class and method name
#define _MFEM_FUNC_NAME __PRETTY_FUNCTION__
// This one is C99 standard.
//#define _MFEM_FUNC_NAME __func__

// Outputs lots of useful information and aborts.
#define MFEM_ABORT(msg)                                               \
  {                                                                   \
    std::ostringstream s;                                             \
    s << std::setprecision(16);                                       \
    s << std::setiosflags(std::ios_base::scientific);                 \
    s << "MFEM error: " << msg << '\n';                               \
    s << "...at line " << __LINE__;                                   \
    s << " in " << _MFEM_FUNC_NAME << " of file " << __FILE__ << "."; \
    s << std::ends;                                                   \
    mfem_error(s.str().c_str());                                      \
  }

// Does a check, and then outputs lots of useful information if the test fails
#define MFEM_VERIFY(x, msg)                                                 \
  {                                                                         \
    if (!(x)) {                                                             \
      std::ostringstream s;                                                 \
      s << std::setprecision(16);                                           \
      s << std::setiosflags(std::ios_base::scientific);                     \
      s << "Verification failed: (" << #x << ") is false: " << msg << '\n'; \
      s << "...at line " << __LINE__;                                       \
      s << " in " << _MFEM_FUNC_NAME << " of file " << __FILE__ << ".";     \
      s << std::ends;                                                       \
      mfem_error(s.str().c_str());                                          \
    }                                                                       \
  }

// Use this if the only place your variable is used is in ASSERT's
#define MFEM_CONTRACT_VAR(x) if (0 && &x == &x){}

// Now set up some optional checks, but only if the right flags are on
#ifdef MFEM_DEBUG

#define MFEM_ASSERT(x, msg)                                              \
  {                                                                      \
    if (!(x)) {                                                          \
      std::ostringstream s;                                              \
      s << std::setprecision(16);                                        \
      s << std::setiosflags(std::ios_base::scientific);                  \
      s << "Assertion failed: (" << #x << ") is false: " << msg << '\n'; \
      s << "...at line " << __LINE__;                                    \
      s << " in " << _MFEM_FUNC_NAME << " of file " << __FILE__ << ".";  \
      s << std::ends;                                                    \
      mfem_error(s.str().c_str());                                       \
    }                                                                    \
  }

#else

// Get rid of all this code, since we're not checking.
#define MFEM_ASSERT(x, msg)

#endif

#endif
