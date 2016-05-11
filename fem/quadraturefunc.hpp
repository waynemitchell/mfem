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

/// Implementation of 1-D numerical quadrature generating functions

#ifndef MFEM_QUADRATUREFUNC
#define MFEM_QUADRATUREFUNC

#include "intrules.hpp"

namespace mfem
{
    enum class NumericalQuad1D : int {
            GaussLegendre = 0,
            GaussLobatto = 1,
            OpenEquallySpaced = 2,
            ClosedEquallySpaced = 3
    };

    class QuadratureFunctions1D
    {
    public:
        QuadratureFunctions1D();
        ~QuadratureFunctions1D();

        void GaussLegendre(const int np, IntegrationRule* ir);
        void GaussLobatto(const int np, IntegrationRule *ir);
        void OpenEquallySpaced(const int np, IntegrationRule *ir);
        void ClosedEquallySpaced(const int np, IntegrationRule *ir);
    private:
        void CalculateLagrangeWeights(IntegrationRule *ir);
    };
}

#endif
