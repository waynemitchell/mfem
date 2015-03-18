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

#ifndef MFEM_EP
#define MFEM_EP

#include <cassert>

#include "../config/config.hpp"

extern "C" void
dsptrf_(char *, int *, double *, int *, int *);
extern "C" void
dsptri_(char *, int *, double *, int *, double *,int *);

#ifdef MFEM_USE_MPI

#include <mpi.h>
#include "../linalg/hypre.hpp"
#include "pfespace.hpp"
#include "bilinearform.hpp"

#endif // MFEM_USE_MPI

namespace mfem {

#ifdef MFEM_USE_MPI

class MyHypreParVector : public HypreParVector {
private:
  MPI_Comm comm_;
public:
  MyHypreParVector(MPI_Comm comm, int glob_size, int *col);
  MyHypreParVector(ParFiniteElementSpace *pfes);

  double Norml2();

  double Normlinf();
};

#endif // MFEM_USE_MPI

class EPDoFs
{
private:
  FiniteElementSpace * fes_;
  int nExposedDofs_;
  int nPrivateDofs_;

  Table * expDoFsByElem_;
  int   * priOffset_;

protected:
public:
  EPDoFs(FiniteElementSpace & fes);

  ~EPDoFs();

  inline FiniteElementSpace * FESpace() const { return fes_; }

  inline int GetNDofs()        { return fes_->GetNDofs(); }
  inline int GetNElements()    { return fes_->GetNE(); }
  inline int GetNExposedDofs() { return nExposedDofs_; }
  inline int GetNPrivateDofs() { return nPrivateDofs_; }

  void BuildElementToDofTable();

  void GetElementDofs(const int elem,
		      Array<int> & ExpDoFs);

  void GetElementDofs(const int elem,
		      Array<int> & ExpDoFs,
		      int & PriOffset, int & numPri);

  inline const int * GetPrivateOffsets() const { return priOffset_; }
};

#ifdef MFEM_USE_MPI

class ParEPDoFs : public EPDoFs
{
private:
  ParFiniteElementSpace * pfes_;
  HypreParMatrix        * Pe_;

  int   nParExposedDofs_;
  int * ExposedPart_;
  int * TExposedPart_;

protected:
public:
  ParEPDoFs(ParFiniteElementSpace & pfes);

  ~ParEPDoFs();

  inline ParFiniteElementSpace * PFESpace() const { return pfes_; }

  inline HypreParMatrix * EDof_TrueEDof_Matrix() { return Pe_; }

  inline MPI_Comm GetComm()            { return pfes_->GetComm(); }
  inline int      GetNRanks()          { return pfes_->GetNRanks(); }
  inline int      GetNParExposedDofs() { return nParExposedDofs_; }
  inline int *    GetPartitioning()    { return ExposedPart_; }
  inline int *    GetTPartitioning()   { return TExposedPart_; }
};

#endif // MFEM_USE_MPI

class EPField : protected Vector
{
protected:
  unsigned int numFields_;

private:
  EPDoFs *  epdofs_;

  Vector ** ExposedDoFs_;
  Vector ** PrivateDoFs_;

  void initVectors(const unsigned int num = 1);

public:
  EPField(ParEPDoFs & epdofs);

  ~EPField();

  inline int GetNFields() const { return numFields_; }

  double Norml2();

  EPField & operator-=(const EPField &v);

  void initFromInterleavedVector(const Vector & x);

  const Vector * ExposedDoFs(const unsigned int i = 0) const;

  Vector * ExposedDoFs(const unsigned int i = 0);

  const Vector * PrivateDoFs(const unsigned int i = 0) const;

  Vector * PrivateDoFs(const int unsigned i = 0);
};

#ifdef MFEM_USE_MPI

class ParEPField : public EPField
{
private:
  ParEPDoFs * pepdofs_;
  MyHypreParVector ** ParExposedDoFs_;

  void initVectors(const unsigned int num = 1);

protected:
public:
  ParEPField(ParEPDoFs & pepdofs);

  ~ParEPField();

  void updateParExposedDoFs();

  void updateExposedDoFs();

  double Norml2();

  double Normlinf();

  ParEPField & operator-=(const ParEPField &v);

  void initFromInterleavedVector(const HypreParVector & x);

  const MyHypreParVector * ParExposedDoFs(const unsigned int i=0) const;

  MyHypreParVector * ParExposedDoFs(const unsigned int i=0);

};

#endif // MFEM_USE_MPI

class BlockDiagonalMatrixInverse;

class BlockDiagonalMatrix : public Matrix
{
  friend class BlockDiagonalMatrixInverse;
private:
  DenseMatrix ** blocks_;
  const int    * blockOffsets_;
  int            nBlocks_;

public:
  BlockDiagonalMatrix(const int nBlocks, const int * blockOffsets);

  ~BlockDiagonalMatrix();

  inline void Finalize(int) {}

  inline DenseMatrix * GetBlock(const int i) { return blocks_[i]; }
};

class EPMatrix : public Operator
{
private:
  EPDoFs * epdofsL_;
  EPDoFs * epdofsR_;

  BilinearFormIntegrator * bfi_;

  SparseMatrix  *  Mee_;
  SparseMatrix  *  Mep_;
  SparseMatrix  *  Mpe_;
  SparseMatrix  *  Mrr_;
  DenseMatrix   ** Mpp_;
  DenseMatrixInverse ** MppInv_;

  Vector        *  reducedRHS_;
  Vector        *  vecp_;

protected:
public:
  EPMatrix(EPDoFs & epdofsL, EPDoFs & epdofsR, BilinearFormIntegrator & bfi);

  ~EPMatrix();

  void Assemble();

  void Mult(const EPField & x, EPField & y) const;

  inline SparseMatrix * GetMee() const { return Mee_; }
  inline SparseMatrix * GetMep() const { return Mep_; }
  inline SparseMatrix * GetMpe() const { return Mpe_; }
  inline SparseMatrix * GetMrr() const { return Mrr_; }
  inline DenseMatrix ** GetMpp() const { return Mpp_; }
  inline DenseMatrixInverse ** GetMppInv() const { return MppInv_; }

  void Mult(const Vector & x, Vector & y) const;

  const Vector * ReducedRHS(const EPField & x) const;

  void SolvePrivateDoFs(const Vector & x, EPField & y) const;
};

#ifdef MFEM_USE_MPI

class ParEPMatrix : public EPMatrix
{
private:
  ParEPDoFs      * pepdofsL_;
  ParEPDoFs      * pepdofsR_;

  Operator       * preducedOp_;
  HypreParVector * preducedRHS_;
  Vector         * vec_;
  Vector         * vecp_;

protected:

  class ParReducedOp : public Operator {
  private:
    ParEPDoFs      * pepdofs_;
    HypreParMatrix * ParMrr_;
    HypreParMatrix * Pe_;

  public:
    ParReducedOp(ParEPDoFs * pepdofs, SparseMatrix * Mrr)
      : pepdofs_(pepdofs),
	ParMrr_(NULL),
	Pe_(NULL)
    {
      Operator::width = pepdofs_->GetNParExposedDofs();
      Pe_  = pepdofs_->EDof_TrueEDof_Matrix();

      HypreParMatrix * HypreMrr = new HypreParMatrix(Pe_->GetComm(),
						     Pe_->M(),
						     Pe_->RowPart(),Mrr);

      ParMrr_ = RAP(HypreMrr,Pe_);

      delete HypreMrr;
    }

    ~ParReducedOp()
    {
      if ( ParMrr_ != NULL ) delete ParMrr_;
    }

    inline void Mult(const Vector & x, Vector & y) const
    {
      ParMrr_->Mult(x,y);
    }
  };

public:
  ParEPMatrix(ParEPDoFs & pepdofsL, ParEPDoFs & pepdofsR,
	      BilinearFormIntegrator & bfi);

  ~ParEPMatrix();

  void Assemble();

  void Mult(const ParEPField & x, ParEPField & y) const;

  const Operator * ReducedOperator() const;

  const HypreParVector * ReducedRHS(const ParEPField & x) const;
};

#endif // MFEM_USE_MPI

template<class Solver>
class EPSolver :
public virtual Solver
{
private:
  const EPMatrix * epMat_;
protected:
public:
  EPSolver() : Solver(), epMat_(NULL) {}
  ~EPSolver() {}

  void SetOperator(const EPMatrix & A)
  {
    epMat_ = &A;

    this->Solver::SetOperator(*epMat_->GetMrr());
  }

  void Mult (const EPField & x, EPField & y) const
  {
    assert( epMat_ != NULL  );

    this->Solver::Mult(*epMat_->ReducedRHS(x),*y.ExposedDoFs());

    epMat_->SolvePrivateDoFs(*x.PrivateDoFs(),y);
  }

};

#ifdef MFEM_USE_MPI

template<class Solver>
class ParEPSolver: public EPSolver<Solver> {
private:
  const ParEPMatrix * pepMat_;
protected:
public:

  ParEPSolver(MPI_Comm _comm)
    : EPSolver<Solver>(),
      Solver(_comm),
      pepMat_(NULL) {}

  void SetOperator(const ParEPMatrix & A)
  {
    pepMat_ = &A;

    this->Solver::SetOperator(*pepMat_->ReducedOperator());
  }

  ~ParEPSolver() {}

  void Mult (const ParEPField & x, ParEPField & y) const
  {
    assert( pepMat_ != NULL  );

    this->Solver::Mult(*pepMat_->ReducedRHS(x),*y.ParExposedDoFs());

    y.updateExposedDoFs();

    pepMat_->SolvePrivateDoFs(*x.PrivateDoFs(),y);
  }
};

#endif // MFEM_USE_MPI

}

#endif // MFEM_EP
