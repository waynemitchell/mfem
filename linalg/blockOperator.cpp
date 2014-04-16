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


#include "../general/array.hpp"
#include "operator.hpp"
#include "blockVector.hpp"
#include "blockOperator.hpp"


BlockOperator::BlockOperator():
	nRowBlocks(0),
	nColBlocks(0),
	row_offsets(0),
	col_offsets(0),
	op()
	{ }


BlockOperator::BlockOperator(int nRowBlocks_):
	nRowBlocks(nRowBlocks_),
	nColBlocks(nRowBlocks_),
	row_offsets(nRowBlocks_+1),
	col_offsets(nRowBlocks_+1),
	op(nRowBlocks_, nRowBlocks_)
	{
	  op = static_cast<Operator *>(NULL);
	  row_offsets = 0;
	  col_offsets = 0;
	}


BlockOperator::BlockOperator(int nRowBlocks_, int nColBlocks_):
	nRowBlocks(nRowBlocks_),
	nColBlocks(nColBlocks_),
	row_offsets(nRowBlocks_+1),
	col_offsets(nColBlocks_+1),
	op(nRowBlocks_, nColBlocks_)
	{
	  op = static_cast<Operator *>(NULL);
	  row_offsets = 0;
	  col_offsets = 0;
	}

void BlockOperator::SetUp(int nRowBlocks_, int nColBlocks_)
{
	nRowBlocks = nRowBlocks_;
	nRowBlocks = nColBlocks_;
	row_offsets.SetSize(nRowBlocks+1);
	col_offsets.SetSize(nColBlocks+1);
	op.SetSize(nRowBlocks, nColBlocks);
}

void BlockOperator::SetDiagonalBlock(int iblock, Operator *op, int size)
{
	SetBlock(iblock, iblock, op, size, size);
}


void BlockOperator::SetBlock(int iRow, int iCol, Operator *opt, int size, int width)
{
	op(iRow, iCol) = opt;

	if( row_offsets[iRow+1] > 0 && row_offsets[iRow+1] != size)
		mfem_error("BlockOperator::SetBlock Incompatible Row Size\n");
	row_offsets[iRow+1] = size;

	if( col_offsets[iCol+1] > 0 && col_offsets[iCol+1] != width)
		mfem_error("BlockOperator::SetBlock Incompatible Col Size\n");
	col_offsets[iCol+1] = width;
}

void BlockOperator::Finalize()
{
	row_offsets[0] = 0; col_offsets[0] = 0;
	row_offsets.PartialSum();
	col_offsets.PartialSum();
	size = row_offsets.Last();
}

/// Operator application
void BlockOperator::Mult (const Vector & x, Vector & y) const
{
	if(row_offsets[0] != 0)
		mfem_error("BlockOperator::Mult You need to call Finalize() first");

	yblock.Update(y.GetData(),const_cast<Array<int>&>(row_offsets).GetData(), nRowBlocks);
	xblock.Update(x.GetData(),const_cast<Array<int>&>(col_offsets).GetData(), nColBlocks);

	y = 0.0;
	for (int iRow(0); iRow < nRowBlocks; ++iRow)
	{
		tmp.SetSize(row_offsets[iRow+1] - row_offsets[iRow]);
		for(int jCol(0); jCol < nColBlocks; ++jCol)
		{
			if(op(iRow,jCol))
			{
				op(iRow,jCol)->Mult(xblock.Block(jCol), tmp);
				yblock.Block(iRow) += tmp;
			}
		}
	}
}

/// Action of the transpose operator
void BlockOperator::MultTranspose (const Vector & x, Vector & y) const
{
	if(row_offsets[0] != 0)
		mfem_error("BlockOperator::Mult You need to call Finalize() first");

	y = 0.0;

	xblock.Update(x.GetData(),const_cast<Array<int>&>(row_offsets).GetData(), nRowBlocks);
	yblock.Update(y.GetData(),const_cast<Array<int>&>(col_offsets).GetData(), nColBlocks);

	for (int iRow(0); iRow < nRowBlocks; ++iRow)
	{
		tmp.SetSize(row_offsets[iRow+1] - row_offsets[iRow]);
		for(int jCol(0); jCol < nColBlocks; ++jCol)
		{
			if(op(jCol,iRow))
			{
				op(jCol,iRow)->MultTranspose(xblock.Block(iRow), tmp);
				yblock.Block(jCol) += tmp;
			}
		}
	}

}

//-----------------------------------------------------------------------
BlockDiagonalPreconditioner::BlockDiagonalPreconditioner():
  Solver(),
  nBlocks(0),
  offsets(0),
  op()
{ }

BlockDiagonalPreconditioner::BlockDiagonalPreconditioner(int nRowBlocks):
  Solver(),
  nBlocks(nRowBlocks),
  offsets(nRowBlocks+1),
  op(nRowBlocks)

{
  op = static_cast<Operator *>(NULL);
}

void BlockDiagonalPreconditioner::SetUp(int nRowBlocks)
{
  nBlocks = nRowBlocks;
  op.SetSize(nRowBlocks);
  offsets.SetSize(nRowBlocks+1, -1);
}

void BlockDiagonalPreconditioner::SetDiagonalBlock(int iblock, Operator *opt, int size)
{
  op[iblock] = opt;
  offsets[iblock+1] = size;
}
void BlockDiagonalPreconditioner::SetBlock(int iRow, int iCol, Operator *opt, int size, int width)
{
  if(iRow!=iCol)
    mfem_error("Trying to add offdiagonal block in BlockDiagonalPreconditioner");
  if(size!=width)
    mfem_error("Block is not square in BlockDiagonalPreconditioner");

  op[iRow] = opt;
  offsets[iRow+1] = size;
}

void BlockDiagonalPreconditioner::Finalize()
{
  offsets[0] = 0;
  offsets.PartialSum();
  size = offsets.Last();
}

/// Operator application
void BlockDiagonalPreconditioner::Mult (const Vector & x, Vector & y) const
{
  if(offsets[0] != 0)
    mfem_error("BlockOperator::Mult You need to call Finalize() first");

  y = 0.0;

  yblock.Update(y.GetData(), offsets.GetData(), nBlocks);
  xblock.Update(x.GetData(), offsets.GetData(), nBlocks);

  for(int i(0); i<nBlocks; ++i)
    if(op[i])
      op[i]->Mult(xblock.Block(i), yblock.Block(i));

}

/// Action of the transpose operator
void BlockDiagonalPreconditioner::MultTranspose (const Vector & x, Vector & y) const
{
  if(offsets[0] != 0)
    mfem_error("BlockOperator::Mult You need to call Finalize() first");

  y = 0.0;

  yblock.Update(y.GetData(), offsets.GetData(), nBlocks);
  xblock.Update(x.GetData(), offsets.GetData(), nBlocks);

  for(int i(0); i<nBlocks; ++i)
    (op[i])->MultTranspose(xblock.Block(i), yblock.Block(i));

}
