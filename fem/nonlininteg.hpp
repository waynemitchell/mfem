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

#ifndef MFEM_NONLININTEG
#define MFEM_NONLININTEG

/** The abstract base class NonlinearFormIntegrator is used to express the
    local action of a general nonlinear finite element operator. In addition
    it may provide the capability to assemble the local gradient operator
    and to compute the local energy. */
class NonlinearFormIntegrator
{
public:
   /// Perform the local action of the NonlinearFormIntegrator
   virtual void AssembleElementVector(const FiniteElement &el,
                                      ElementTransformation &Tr,
                                      const Vector &elfun, Vector &elvect) = 0;

   /// Assemble the local gradient matrix
   virtual void AssembleElementGrad(const FiniteElement &el,
                                    ElementTransformation &Tr,
                                    const Vector &elfun, DenseMatrix &elmat);

   /// Compute the local energy
   virtual double GetElementEnergy(const FiniteElement &el,
                                   ElementTransformation &Tr,
                                   const Vector &elfun);

   virtual ~NonlinearFormIntegrator() { }
};

#endif
