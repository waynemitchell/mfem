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

// Common error message and abort macro
#define _MFEM_MSG_ABORT(msg)                                            \
   {                                                                    \
      std::ostringstream s;                                             \
      s << std::setprecision(16);                                       \
      s << std::setiosflags(std::ios_base::scientific);                 \
      s << msg << '\n';                                                 \
      s << "...at line " << __LINE__;                                   \
      s << " in " << _MFEM_FUNC_NAME << " of file " << __FILE__ << "."; \
      s << std::ends;                                                   \
      mfem_error(s.str().c_str());                                      \
   }                                                                    \

// Outputs lots of useful information and aborts.
// For all of these functions, "msg" is pushed to an ostream, so you can
// write useful (if complicated) error messages instead of writing
// out to the screen first, then calling abort.  For example:
// MFEM_ABORT( "Unknown geometry type: " << type );
#define MFEM_ABORT(msg) _MFEM_MSG_ABORT("MFEM abort: " << msg);

// Does a check, and then outputs lots of useful information if the test fails
#define MFEM_VERIFY(x, msg)                             \
   if (!(x))                                            \
   {                                                    \
      _MFEM_MSG_ABORT("Verification failed: ("          \
                      << #x << ") is false: " << msg);  \
   }

// Use this if the only place your variable is used is in ASSERT's
// For example, this code snippet:
//   int err = MPI_Reduce(ldata, maxdata, 5, MPI_INT, MPI_MAX, 0, MyComm);
//   MFEM_CONTRACT_VAR(err);
//   MFEM_ASSERT( err == 0, "MPI_Reduce gave an error with length "
//                       << ldata );
#define MFEM_CONTRACT_VAR(x) if (0 && &x == &x){}

// Now set up some optional checks, but only if the right flags are on
#ifdef MFEM_DEBUG

#define MFEM_ASSERT(x, msg)                             \
   if (!(x))                                            \
   {                                                    \
      _MFEM_MSG_ABORT("Assertion failed: ("             \
                      << #x << ") is false: " << msg);  \
   }

#else

// Get rid of all this code, since we're not checking.
#define MFEM_ASSERT(x, msg)

#endif

#endif
