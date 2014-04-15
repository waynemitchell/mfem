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


#ifndef BLOCKVECTOR_HPP_
#define BLOCKVECTOR_HPP_

// Data type vector



//! @class BlockVector
/*
 * \brief A class to handle Vectors in a block fashion
 *
 */
class BlockVector: public Vector
{

protected:

   //! Number of blocks in the blockVector
   int numBlocks;
   //! Dimension of each block Vector
   int *blockOffsets;
   mutable Vector tmp_block;

private:
	//! Determine whenever the pointer to blockSizes should be deallocated or not
	bool ownBlockOffsets;

public:
   //! empty constructor
   BlockVector();

   //! Standard constructor
   BlockVector(int * bOffsets, int nBlocks);

   //! Copy constructor
   BlockVector(const BlockVector & block);

   //! View constructor
   BlockVector(double *data, int *blockOffsets, int numBlocks);

   BlockVector & operator=(const BlockVector & original);
   BlockVector & operator=(double val);

   //! Destructor
   virtual ~BlockVector();

   //! Get the i-th vector in the block (Thread Unsafe)
   Vector & Block(int i);
   const Vector & Block(int i) const;

   //! Get the i-th vector in the block (Thread safe)
   void BlockView(int i, Vector & blockView);

   void Update(double *data, const int *blockOffsets, int numBlocks);

};

BlockVector * stride(const Array<const Vector *> & vectors);
BlockVector * stride(const Vector & v1, const Vector & v2);
BlockVector * stride(const Vector & v1, const Vector & v2, const Vector & v3);


#endif /* BLOCKVECTOR_HPP_ */
