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
   //! Offset for each block start. (length numBlocks+1)
   /*
    * blockOffsets[i+1] - blockOffsets[i] is the size of block i.
    */
   int *blockOffsets;
   //! temporary Vector used to extract blocks without allocating memory (possibly unsafe).
   mutable Vector tmp_block;

private:
	//! Determine whenever the pointer to blockSizes should be deallocated or not
	bool ownBlockOffsets;

public:
   //! empty constructor
   BlockVector();

   //! Constructor
   /*
    * bOffsets is an array of integers (length nBlocks+1) that tells the offsets of each block start.
    * nBlocks is the number of blocks.
    */
   BlockVector(int * bOffsets, int nBlocks);

   //! Copy constructor
   BlockVector(const BlockVector & block);

   //! View constructor
   /*
    * data is an array of double of length at least blockOffsets[numBlocks] that contain all the values of the monolithic vector.
    * bOffsets is an array of integers (length nBlocks+1) that tells the offsets of each block start.
    * nBlocks is the number of blocks.
    */
   BlockVector(double *data, int *blockOffsets, int numBlocks);

   //! Assignment operator. this and original must have the same block structure.
   BlockVector & operator=(const BlockVector & original);
   //! Set each entry of this equal to val
   BlockVector & operator=(double val);

   //! Destructor
   virtual ~BlockVector();

   //! Get the i-th vector in the block. (Thread Unsafe)
   /*
    * WARNING: Do not call this method with two different inputs in the same instruction or the results will be undefined.
    */
   Vector & Block(int i);
   //! Get the i-th vector in the block (const version). (Thread Unsafe)
   const Vector & Block(int i) const;

   //! Get the i-th vector in the block (Thread safe)
   void BlockView(int i, Vector & blockView);

   //! Update method
   /*
    * data is an array of double of length at least blockOffsets[numBlocks] that contain all the values of the monolithic vector.
    * bOffsets is an array of integers (length nBlocks+1) that tells the offsets of each block start.
    * nBlocks is the number of blocks.
    */
   void Update(double *data, const int *blockOffsets, int numBlocks);

};

//! Stride two or more objects of type Vector in a BlockVector.
BlockVector * stride(const Array<const Vector *> & vectors);
BlockVector * stride(const Vector & v1, const Vector & v2);
BlockVector * stride(const Vector & v1, const Vector & v2, const Vector & v3);


#endif /* BLOCKVECTOR_HPP_ */
