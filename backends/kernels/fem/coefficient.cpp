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

#include "../../../config/config.hpp"
#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_KERNELS)

#include "../kernels.hpp"

namespace mfem
{

namespace kernels
{

//---[ Coefficient ]------------------
KernelsCoefficient::KernelsCoefficient(const double value) :
   engine(NULL),
   integ(NULL),
   name("COEFF")
{
   push();
   pop();
}
//---[ Coefficient ]------------------
KernelsCoefficient::KernelsCoefficient(const Engine &e, const double value) :
   engine(&e),
   integ(NULL),
   name("COEFF")
{
   push();
   pop();
}

KernelsCoefficient::KernelsCoefficient(const Engine &e,
                                       const std::string &source) :
   engine(&e),
   integ(NULL),
   name("COEFF")
{
   push();
   pop();
}

KernelsCoefficient::KernelsCoefficient(const Engine &e, const char *source) :
   engine(&e),
   integ(NULL),
   name("COEFF")
{
   push();
   pop();
}

KernelsCoefficient::KernelsCoefficient(const KernelsCoefficient &coeff) :
   engine(coeff.engine),
   integ(NULL),
   name(coeff.name)
{
   push();
   pop();
}

KernelsCoefficient::~KernelsCoefficient()
{}

KernelsCoefficient& KernelsCoefficient::SetName(const std::string &name_)
{
   name = name_;
   return *this;
}

void KernelsCoefficient::Setup(KernelsIntegrator &integ_)
{
   push();
   integ = &integ_;
   pop();
}

bool KernelsCoefficient::IsConstant()
{
   assert(false);
   return true;
}

double KernelsCoefficient::GetConstantValue()
{
   if (!IsConstant())
   {
      mfem_error("KernelsCoefficient is not constant");
   }
   assert(false);
   return 1.0;
}

Vector KernelsCoefficient::Eval()
{
   if (integ == NULL)
   {
      mfem_error("KernelsCoefficient requires a Setup() call before Eval()");
   }

   mfem::FiniteElementSpace &fespace = integ->GetTrialFESpace();
   const mfem::IntegrationRule &ir   = integ->GetIntegrationRule();

   const int elements = fespace.GetNE();
   const int numQuad  = ir.GetNPoints();

   Vector quadCoeff(*(new Layout(KernelsEngine(), numQuad * elements)));
   Eval(quadCoeff);
   return quadCoeff;
}

void KernelsCoefficient::Eval(Vector &quadCoeff)
{
   assert(false);
}

} // namespace mfem::kernels

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_KERNELS)
