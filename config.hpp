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

// Namespace configuration and macros
#ifdef MFEM_USE_NAMESPACE
#define MFEM_NAMESPACE mfem
#define MFEM_NAMESPACE_BEGIN() namespace MFEM_NAMESPACE {
#define MFEM_NAMESPACE_END() }
#else
#define MFEM_NAMESPACE
#define MFEM_NAMESPACE_BEGIN()
#define MFEM_NAMESPACE_END()
#endif

#endif
