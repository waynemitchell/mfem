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

//! @class BlockVector
/*
 * \brief A class to handle Vectors in a block fashion
 *
 * All data is contained in Vector::data, while blockVector is just a viewer for this data
 *
 */

#include "../general/array.hpp"
#include "vector.hpp"
#include "blockVector.hpp"

BlockVector::BlockVector():
	Vector(),
	numBlocks(0),
	blockOffsets(NULL)
{

}

//! Standard constructor
BlockVector::BlockVector(const Array<int> & bOffsets):
		Vector(bOffsets.Last()),
		numBlocks(bOffsets.Size()-1),
		blockOffsets(bOffsets.GetData())
{
	Vector::operator=(0.0);
}

//! Copy constructor
BlockVector::BlockVector(const BlockVector & v):
	Vector(v),
	numBlocks(v.numBlocks),
	blockOffsets(v.blockOffsets)
{

}

//! View constructor
BlockVector::BlockVector(double *data, const Array<int> & bOffsets):
		Vector(data, bOffsets.Last()),
		numBlocks(bOffsets.Size()-1),
		blockOffsets(bOffsets.GetData())
{

}

void BlockVector::Update(double *data, const Array<int> & bOffsets)
{
	SetDataAndSize(data, bOffsets.Last());
	numBlocks = bOffsets.Size()-1;
	blockOffsets = bOffsets.GetData();
}

BlockVector & BlockVector::operator=(const BlockVector & original)
{
  if(numBlocks!=original.numBlocks)
    mfem_error("Number of Blocks don't match in BlockVector::operator=");

	for(int i(0); i <= numBlocks; ++i)
	  if(blockOffsets[i]!=original.blockOffsets[i])
	    mfem_error("Size of Blocks don't match in BlockVector::operator=");
	
	for(int i = 0; i < original.size; i++ )
	  data[i] = original.data[i];

	return *this;
}


BlockVector & BlockVector::operator=(double val)
{
  Vector::operator=(val);
  return *this;
}


//! Destructor
BlockVector::~BlockVector()
{

}


Vector & BlockVector::Block(int i)
{
	tmp_block.SetDataAndSize(data+blockOffsets[i], blockOffsets[i+1]-blockOffsets[i]);
	return tmp_block;
}

const Vector &  BlockVector::Block(int i) const
{
	tmp_block.SetDataAndSize(data+blockOffsets[i], blockOffsets[i+1]-blockOffsets[i]);
	return tmp_block;
}

void BlockVector::BlockView(int i, Vector & blockView)
{
	blockView.SetDataAndSize(data+blockOffsets[i], blockOffsets[i+1]-blockOffsets[i]);
}

BlockVector * stride(const Array<const Vector *> & vectors, Array<int> & bOffsets)
{
	int nBlocks(vectors.Size());

	if(bOffsets.Size() == 0)
	{
		bOffsets.SetSize(nBlocks+1);
		bOffsets[0] = 0;

		for(int i(0); i<nBlocks; ++i)
			bOffsets[i+1] = bOffsets[i] + vectors[i]->Size();
	}
#ifdef MFEM_DEBUG
	else
	{
		if(bOffsets.Size()!=nBlocks+1)
			mfem_error("BlockVector * stride #1");

		for(int i(0); i<nBlocks; ++i)
			if(bOffsets[i+1] - bOffsets[i] != vectors[i]->Size())
				mfem_error("BlockVector * stride #2");
	}
#endif


	BlockVector * result = new BlockVector(bOffsets);

	for(int i(0); i<nBlocks; ++i)
		result->Block(i) = *vectors[i];

	return result;
}

BlockVector * stride(const Vector & v1, const Vector & v2, Array<int> & bOffsets)
{
	Array<const Vector *> vectors(2);
	vectors[0] = &v1;
	vectors[1] = &v2;
	return stride(vectors, bOffsets);
}

BlockVector * stride(const Vector & v1, const Vector & v2, const Vector & v3, Array<int> & bOffsets)
{
	Array<const Vector *> vectors(3);
	vectors[0] = &v1;
	vectors[1] = &v2;
	vectors[2] = &v3;
	return stride(vectors, bOffsets);
}
