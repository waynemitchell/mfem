// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#ifndef MFEM_TEMPLATE_INTEG
#define MFEM_TEMPLATE_INTEG

#include "config/tconfig.hpp"
#include "tbilinearform.hpp"

namespace mfem
{

template <typename coeff_t, template<int,int,typename> class kernel_t>
class TIntegrator
{
public:
   typedef coeff_t coefficient_type;

   template <int SDim, int Dim, typename complex_t>
   struct kernel { typedef kernel_t<SDim,Dim,complex_t> type; };

   coeff_t coeff;

   TIntegrator(const coefficient_type &c) : coeff(c) { }
};

} // namespace mfem

#endif // MFEM_TEMPLATE_INTEG
