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

#ifndef BLOCKOPERATOR_HPP_
#define BLOCKOPERATOR_HPP_

class BlockOperator : public Operator
{
public:

	BlockOperator();
	BlockOperator(int nRowBlocks);
	BlockOperator(int nRowBlocks, int nColBlocks);

	void SetUp(int nRowBlocks, int nColBlocks);
	void SetDiagonalBlock(int iblock, Operator *op, int size);
	void SetBlock(int iRow, int iCol, Operator *op, int size, int width);

	void Finalize();

	/// Operator application
	virtual void Mult (const Vector & x, Vector & y) const;

	/// Action of the transpose operator
	virtual void MultTranspose (const Vector & x, Vector & y) const;

private:
	int nRowBlocks;
	int nColBlocks;
	Array<int> row_offsets;
	Array<int> col_offsets;
	Array2D<Operator *> op;

	mutable BlockVector xblock;
	mutable BlockVector yblock;
	mutable Vector tmp;

};

class BlockDiagonalPreconditioner : public Solver
{
public:

  BlockDiagonalPreconditioner();
  BlockDiagonalPreconditioner(int nRowBlocks);
  
  void SetUp(int nRowBlocks);
  void SetDiagonalBlock(int iblock, Operator *op, int size);
  void SetBlock(int iRow, int iCol, Operator *op, int size, int width);
  void SetOperator(const Operator &op){ }

  void Finalize();

  /// Operator application
  virtual void Mult (const Vector & x, Vector & y) const;
  
  /// Action of the transpose operator
  virtual void MultTranspose (const Vector & x, Vector & y) const;

private:
  int nBlocks;
  Array<int> offsets;
  Array<Operator *> op;
  
  mutable BlockVector xblock;
  mutable BlockVector yblock;
  
};

#endif /* BLOCKOPERATOR_HPP_ */
