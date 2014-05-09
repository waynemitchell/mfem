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

#ifndef MFEM_BLOCKOPERATOR
#define MFEM_BLOCKOPERATOR

//! @class BlockOperator
/**
 * \brief A class to handle Block systems in a matrix-free implementation.
 *
 * Usage:
 * - Use one of the constructors or the SetUp method to define how many row and columns blocks.
 * - Use SetDiagonalBlock or SetBlock to fill the BlockOperator
 * - Call the method Finalize() to check for consistency and generate the BlockOffsets.
 * - Use the method Mult and MultTranspose to apply the operator to a vector.
 *
 */
class BlockOperator : public Operator
{
public:

	//! Default Constructor
	BlockOperator();
	//! Constructor for BlockOperators with the same number of block rows and block columns.
	BlockOperator(int nRowBlocks);
	//! Constructor for general BlockOperators.
	BlockOperator(int nRowBlocks, int nColBlocks);

	//! Set the number of row and columns blocks.
	void SetUp(int nRowBlocks, int nColBlocks);
	//! Add a square block op in the block-entry (iblock, iblock).
	/**
	 * iblock: The block will be inserted in location (iblock, iblock).
	 * op: the Operator to be inserted.
	 * size: the local size (number of rows) of the operator.
	 */
	void SetDiagonalBlock(int iblock, Operator *op, int size);
	//! Add a block op in the block-entry (iblock, jblock).
	/**
	 * irow, icol: The block will be inserted in location (irow, icol).
	 * op: the Operator to be inserted.
	 * size: the local size (number of rows) of the operator.
	 * width: the local width (number of columns) of the operator.
	 */
	void SetBlock(int iRow, int iCol, Operator *op, int size, int width);
	//! Finalize the block structure of the operator.
	/**
	 * You must call this method before calling Mult or MultTranspose
	 */
	void Finalize();

	/// Operator application
	virtual void Mult (const Vector & x, Vector & y) const;

	/// Action of the transpose operator
	virtual void MultTranspose (const Vector & x, Vector & y) const;

private:
	//! Number of block rows
	int nRowBlocks;
	//! Number of block columns
	int nColBlocks;
	//! Row offsets for the starting position of each block
	Array<int> row_offsets;
	//! Column offsets for the starting position of each block
	Array<int> col_offsets;
	//! 2D array that stores each block of the operator.
	Array2D<Operator *> op;

	//! Temporary Vectors used to efficiently apply the Mult and MultTranspose methods.
	mutable BlockVector xblock;
	mutable BlockVector yblock;
	mutable Vector tmp;

};

//! @class BlockDiagonalPreconditioner
/*
 * \brief A class to handle Block diagonal preconditioners in a matrix-free implementation.
 *
 * Usage:
 * - Use the constructors or the SetUp method to define how many blocks.
 * - Use SetDiagonalBlock to fill the BlockOperator
 * - Call the method Finalize() to generate the BlockOffsets.
 * - Use the method Mult and MultTranspose to apply the operator to a vector.
 *
 */
class BlockDiagonalPreconditioner : public Solver
{
public:

  //! Default Constructor
  BlockDiagonalPreconditioner();
  //! Constructor that specifies the number of blocks
  BlockDiagonalPreconditioner(int nRowBlocks);
  //! Set the number of blocks.
  void SetUp(int nRowBlocks);
  //! Add a square block op in the block-entry (iblock, iblock).
  /**
   * iblock: The block will be inserted in location (iblock, iblock).
   * op: the Operator to be inserted.
   * size: the local size (number of rows) of the operator.
   */
  void SetDiagonalBlock(int iblock, Operator *op, int size);
  //! This method is present for the purpose of having the same interface as BlockOperator
  /**
   * This method will raise an error if:
   * - op is not square (i.e. size ~= width)
   * - the block is not on the diagonal (i.e. iRow ~= iCol)
   */
  void SetBlock(int iRow, int iCol, Operator *op, int size, int width);
  //! This method is present since required by the abstract base class Solver
  void SetOperator(const Operator &op){ }
  //! Finalize the block structure of the operator.
  /**
  * You must call this method before calling Mult or MultTranspose
  */
  void Finalize();

  /// Operator application
  virtual void Mult (const Vector & x, Vector & y) const;
  
  /// Action of the transpose operator
  virtual void MultTranspose (const Vector & x, Vector & y) const;

private:
  //! Number of Blocks
  int nBlocks;
  //! Offsets for the starting position of each block
  Array<int> offsets;
  //! 1D array that stores each block of the operator.
  Array<Operator *> op;
  //! Temporary Vectors used to efficiently apply the Mult and MultTranspose methods.
  mutable BlockVector xblock;
  mutable BlockVector yblock;
  
};

#endif /* MFEM_BLOCKOPERATOR */
