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

#include "../config/config.hpp"

#include "pabilininteg.hpp"

namespace mfem {

PAIntegrator::PAIntegrator(Coefficient &q, FiniteElementSpace &f, bool gpu)
{
    Q = &q;
    fes = &f;
    onGPU = gpu;
    fe = fes->GetFE(0);
    tfe = dynamic_cast<const TensorBasisElement*>(fe);

    nElem  = fes->GetNE();
    GeomType = fe->GetGeomType();
    FEOrder = fe->GetOrder();
    nDim    = fe->GetDim();
    nDof   = fe->GetDof();

    ElementTransformation *Trans = fes->GetElementTransformation(0);
    int irorder = 2*fe->GetOrder() + Trans->OrderW();
    ir = &IntRules.Get(GeomType, irorder);
    nQuad = ir->GetNPoints();
    hasTensorBasis = tfe ? true : false;

    if (nDim > 3) 
    {
        mfem_error("AcroIntegrator tensor computations don't support dim > 3.");
    }
}

PAIntegrator::~PAIntegrator()
{

}

}
