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

#ifndef MFEM_BACKENDS_ACRO_BILIN_INTEG_HPP
#define MFEM_BACKENDS_ACRO_BILIN_INTEG_HPP

#include "../../config/config.hpp"
#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_OCCA) && defined(MFEM_USE_ACROTENSOR)

#include "fespace.hpp"
#include "bilinearform.hpp"
#include "coefficient.hpp"

#include "bilininteg.hpp"

#include "AcroTensor.hpp"

namespace mfem
{

namespace acro
{

// TODO Move these into their own sub-backend
// - [ ] Determine a method for a backend-independent coefficient (hard)


class PAIntegrator : public mfem::occa::OccaIntegrator
{
protected:
   const FiniteElement *fe;
   const TensorBasisElement *tfe;
   const IntegrationRule *ir1D;
   Array<int> tDofMap;
   int GeomType;
   int FEOrder;
   bool onGPU;
   int nDim;
   int nDof;
   int nQuad;
   int nDof1D;
   int nQuad1D;

public:
   PAIntegrator(const mfem::occa::Engine &engine);
   virtual ~PAIntegrator() {}
   virtual bool PAIsEnabled() const { return true; }
   int GetExpandedNDOF() { return trialFESpace->GetNE() * nDof; }
   FiniteElementSpace *GetFES() { return trialFESpace; }
   virtual void BatchedPartialAssemble() = 0;
   virtual void BatchedAssembleElementMatrices(DenseTensor &elmats) = 0;
   virtual void Assemble() { BatchedPartialAssemble(); }
   virtual void Setup();
};

class AcroMassIntegrator : public PAIntegrator
{
private:
   ::acro::TensorEngine TE;

   ::acro::Tensor W;            //Integration weights
   ::acro::Tensor B;            //Basis evaluated on the quad points
   ::acro::Tensor D;            //Product of integration weight, physical consts, and element shape info
   ::acro::Tensor M;            //The assembled local mass matrices
   ::acro::Tensor T1, T2, T3;   //Intermediate computations for tensor product partial assembly

public:
   AcroMassIntegrator(const mfem::occa::Engine &engine);
   virtual ~AcroMassIntegrator() {}
   virtual std::string GetName() { return "AcroMassIntegrator"; }
   virtual void BatchedPartialAssemble();
   virtual void BatchedAssembleElementMatrices(DenseTensor &elmats);
   virtual void MultAdd(mfem::occa::Vector &x, mfem::occa::Vector &y);
   virtual void SetupIntegrationRule();
   virtual void Setup();
};

class AcroDiffusionIntegrator : public PAIntegrator
{
private:
   ::acro::TensorEngine TE;

   ::acro::Tensor B, G;         //Basis and dbasis evaluated on the quad points
   ::acro::Tensor W;            //Integration weights
   mfem::Array<::acro::Tensor*> Btil; //Btilde used to compute stiffness matrix
   ::acro::Tensor D;            //Product of integration weight, physical consts, and element shape info
   ::acro::Tensor S;            //The assembled local stiffness matrices
   ::acro::Tensor U, Z, T1, T2; //Intermediate computations for tensor product partial assembly
   ::acro::Tensor X, Y;

   void ComputeBTilde();

public:
   AcroDiffusionIntegrator(const mfem::occa::Engine &engine);
   virtual ~AcroDiffusionIntegrator() {}
   virtual std::string GetName() { return "AcroDiffusionIntegrator"; }
   virtual void BatchedPartialAssemble();
   virtual void BatchedAssembleElementMatrices(DenseTensor &elmats);
   virtual void MultAdd(mfem::occa::Vector &x, mfem::occa::Vector &y);
   virtual void SetupIntegrationRule();
   virtual void Setup();
};


} // namespace mfem::acro

} // namespace mfem


#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_ACROTENSOR) && defined(MFEM_USE_OCCA)
#endif // MFEM_BACKENDS_ACRO_BILIN_INTEG_HPP
