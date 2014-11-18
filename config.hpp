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

#ifndef MFEM_CONFIG_HEADER
#define MFEM_CONFIG_HEADER

// Build the parallel MFEM library.
// Requires an MPI compiler, and the libraries HYPRE and METIS.
// #define MFEM_USE_MPI

// Enable this option if linking with METIS version 5 (parallel MFEM).
// #define MFEM_USE_METIS_5

// Enable debug checks in MFEM.
// #define MFEM_DEBUG

// Use high-resolution POSIX clocks in class StopWatch, link with -lrt.
// #define MFEM_USE_POSIX_CLOCKS

// Use LAPACK routines for various dense linear algebra operations.
// #define MFEM_USE_LAPACK

// Use thread-safe implementation. This comes at the cost of extra memory
// allocation and de-allocation.
// #define MFEM_THREAD_SAFE

// Enable experimental OpenMP support. Requires MFEM_THREAD_SAFE.
// #define MFEM_USE_OPENMP

// Enable MFEM functionality based on the Mesquite library.
// #define MFEM_USE_MESQUITE

// Enable MFEM functionality based on the SuiteSparse library.
// #define MFEM_USE_SUITESPARSE

// Internal MFEM option: enable group/batch allocation for some small objects.
// #define MFEM_USE_MEMALLOC

#endif
