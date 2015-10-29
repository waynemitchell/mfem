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

#include "../../config/config.hpp"

#ifdef MFEM_USE_MPI

#include "pfem_extras.hpp"

using namespace std;

namespace mfem
{

H1_ParFESpace::H1_ParFESpace(ParMesh *m,
			     const int p, const int space_dim, const int type,
			     int vdim, int order)
  : ParFiniteElementSpace(m, new H1_FECollection(p,space_dim,type),vdim,order)
{
   FEC_ = this->FiniteElementSpace::fec;
}

H1_ParFESpace::~H1_ParFESpace()
{
   delete FEC_;
}

ND_ParFESpace::ND_ParFESpace(ParMesh *m, const int p, const int space_dim,
			     int vdim, int order)
  : ParFiniteElementSpace(m, new ND_FECollection(p,space_dim),vdim,order)
{
   FEC_ = this->FiniteElementSpace::fec;
}

ND_ParFESpace::~ND_ParFESpace()
{
   delete FEC_;
}

RT_ParFESpace::RT_ParFESpace(ParMesh *m, const int p, const int space_dim,
			     int vdim, int order)
  : ParFiniteElementSpace(m, new RT_FECollection(p-1,space_dim),vdim,order)
{
   FEC_ = this->FiniteElementSpace::fec;
}

RT_ParFESpace::~RT_ParFESpace()
{
   delete FEC_;
}

L2_ParFESpace::L2_ParFESpace(ParMesh *m, const int p, const int space_dim,
			     int vdim, int order)
  : ParFiniteElementSpace(m, new L2_FECollection(p,space_dim),vdim,order)
{
   FEC_ = this->FiniteElementSpace::fec;
}

L2_ParFESpace::~L2_ParFESpace()
{
   delete FEC_;
}

ParDiscreteInterpolationOperator::~ParDiscreteInterpolationOperator()
{
   if ( pdlo_ != NULL ) delete pdlo_;
   if ( mat_  != NULL ) delete mat_;
}

HYPRE_Int
ParDiscreteInterpolationOperator::Mult(HypreParVector &x, HypreParVector &y,
				       double alpha, double beta)
{
   return mat_->Mult( x, y, alpha, beta);
}

HYPRE_Int
ParDiscreteInterpolationOperator::Mult(HYPRE_ParVector x, HYPRE_ParVector y,
				       double alpha, double beta)
{
   return mat_->Mult( x, y, alpha, beta);
}

HYPRE_Int
ParDiscreteInterpolationOperator::MultTranspose(HypreParVector &x,
						HypreParVector &y,
						double alpha, double beta)
{
   return mat_->MultTranspose( x, y, alpha, beta);
}

void
ParDiscreteInterpolationOperator::Mult(double a, const Vector &x,
				       double b, Vector &y) const
{
   mat_->Mult( a, x, b, y);
}

void
ParDiscreteInterpolationOperator::MultTranspose(double a, const Vector &x,
						double b, Vector &y) const
{
   mat_->MultTranspose( a, x, b, y);
}

void
ParDiscreteInterpolationOperator::Mult(const Vector &x, Vector &y) const
{
   mat_->Mult( x, y);
}

void
ParDiscreteInterpolationOperator::MultTranspose(const Vector &x,
						Vector &y) const
{
   mat_->MultTranspose( x, y);
}

ParDiscreteGradOperator::ParDiscreteGradOperator(ParFiniteElementSpace *dfes,
						 ParFiniteElementSpace *rfes)
{
   pdlo_ = new ParDiscreteLinearOperator(dfes, rfes);
   pdlo_->AddDomainInterpolator(new GradientInterpolator);
   pdlo_->Assemble();
   pdlo_->Finalize();
   mat_ = pdlo_->ParallelAssemble();
}

ParDiscreteCurlOperator::ParDiscreteCurlOperator(ParFiniteElementSpace *dfes,
						 ParFiniteElementSpace *rfes)
{
   pdlo_ = new ParDiscreteLinearOperator(dfes, rfes);
   pdlo_->AddDomainInterpolator(new CurlInterpolator);
   pdlo_->Assemble();
   pdlo_->Finalize();
   mat_ = pdlo_->ParallelAssemble();
}

ParDiscreteDivOperator::ParDiscreteDivOperator(ParFiniteElementSpace *dfes,
					       ParFiniteElementSpace *rfes)
{
   pdlo_ = new ParDiscreteLinearOperator(dfes, rfes);
   pdlo_->AddDomainInterpolator(new DivergenceInterpolator);
   pdlo_->Assemble();
   pdlo_->Finalize();
   mat_ = pdlo_->ParallelAssemble();
}

}

#endif
