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

#ifndef MFEM_BACKENDS_KERNELS_COEFFICIENT_HPP
#define MFEM_BACKENDS_KERNELS_COEFFICIENT_HPP

#include "../../../config/config.hpp"
#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_KERNELS)

namespace mfem
{

namespace kernels
{

class KernelsIntegrator;

//---[ Coefficient ]------------------
class KernelsCoefficient
{
private:
   SharedPtr<const Engine> engine;
   KernelsIntegrator *integ;
   std::string name;
public:
   KernelsCoefficient(const double value = 1.0);
   KernelsCoefficient(const Engine &e, const double value = 1.0);
   KernelsCoefficient(const Engine &e, const std::string &source);
   KernelsCoefficient(const Engine &e, const char *source);
   ~KernelsCoefficient();
   KernelsCoefficient(const KernelsCoefficient &coeff);
   const Engine &KernelsEngine() const { return *engine; }
   kernels::device GetDevice(int idx = 0) const
   { return engine->GetDevice(idx); }
   KernelsCoefficient& SetName(const std::string &name_);
   void Setup(KernelsIntegrator &integ_);
   bool IsConstant();
   double GetConstantValue();
   Vector Eval();
   void Eval(Vector &quadCoeff);
};

} // namespace mfem::kernels

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_KERNELS)

#endif // MFEM_BACKENDS_KERNELS_COEFFICIENT_HPP
