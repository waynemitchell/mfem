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

#ifndef BLOCKMATRIX_HPP_
#define BLOCKMATRIX_HPP_

class BlockMatrix : public SparseRowMatrix {
public:
	BlockMatrix(int nRowBlocks, int nColBlocks);
	void SetBlock(int i, int j, SparseMatrix & mat);
	/// Finalize the matrix (no more blocks allowed)
	virtual void Finalize();
	/// Returns reference to a_{ij}.  Index i, j = 0 .. size-1
	virtual double& Elem (int i, int j);
	/// Returns constant reference to a_{ij}.  Index i, j = 0 .. size-1
	virtual const double& Elem (int i, int j) const;
    /// Returns a pointer to (approximation) of the matrix inverse.
	virtual MatrixInverse * Inverse() const { mfem_error("BlockMatrix::Inverse not implemented \n"); return static_cast<MatrixInverse*>(NULL); }
	int NumRowBlocks() const {return nRowBlocks; }
	int NumColBlocks() const {return nColBlocks; }
	int NumRows() const {return is_filled ? row_offsets[nRowBlocks]:-1;}
	int NumCols() const {return is_filled ? col_offsets[nColBlocks]:-1;}
	int Size() const { return is_filled ? row_offsets[nRowBlocks]:-1;}
	int Width() const {return is_filled ? col_offsets[nColBlocks]:-1;}
	virtual int NumNonZeroElems() const { if(!is_filled) mfem_error("BlockMatrix::NumNonZeroElems()"); return nnz_elem; }
	SparseMatrix & Block(int i, int j);
	const SparseMatrix & Block(int i, int j) const;
	int IsZeroBlock(int i, int j) const {return (Aij(i,j)==NULL) ? 1 : 0; }
	int * GetRowOffsets() { return row_offsets.GetData(); }
	int * GetColOffsets() { return col_offsets.GetData(); }
	const int * GetRowOffsets() const { return row_offsets.GetData(); }
	const int * GetColOffsets() const { return col_offsets.GetData(); }

	int RowSize(const int i) const;
	int GetRow(const int row, Array<int> &cols, Vector &srow) const;
	void EliminateRowCol(Array<int> & ess_bc_dofs, Vector & sol, Vector & rhs);
	void EliminateZeroRows();

	void Mult(const Vector & x, Vector & y) const;
	void AddMult(const Vector & x, Vector & y, const double val = 1.) const;
	void MultTranspose(const Vector & x, Vector & y) const;
	void AddMultTranspose(const Vector & x, Vector & y, const double val = 1.) const;

//	BlockMatrix * ExtractRowAndColumns( const Array<int> & grows, const Array<int> & gcols, Array<int> & colMapper) const;

	SparseMatrix * Monolithic();

	void PrintMatlab(std::ostream & os = std::cout);

	virtual ~BlockMatrix();

	int owns_blocks;

private:

	inline void findGlobalRow(int iglobal, int & iblock, int & iloc) const
	{
		if(!is_filled || iglobal > row_offsets[nRowBlocks])
			mfem_error("BlockMatrix::findGlobalRow");

		for(iblock = 0; iblock < nRowBlocks; ++iblock)
			if(row_offsets[iblock+1] > iglobal)
				break;

		iloc = iglobal - row_offsets[iblock];
	}

	inline void findGlobalCol(int jglobal, int & jblock, int & jloc) const
	{
		if(!is_filled || jglobal > col_offsets[nColBlocks])
			mfem_error("BlockMatrix::findGlobalCol");

		for(jblock = 0; jblock < nColBlocks; ++jblock)
			if(col_offsets[jblock+1] > jglobal)
				break;

		jloc = jglobal - col_offsets[jblock];
	}

	int nRowBlocks;
	int nColBlocks;

	Array<int> row_offsets;
	Array<int> col_offsets;

	int nnz_elem;

	bool is_filled;

	Array2D<SparseMatrix *> Aij;
};

BlockMatrix * Transpose(const BlockMatrix & A);
BlockMatrix * Mult(const BlockMatrix & A, const BlockMatrix & B);

#endif /* BLOCKMATRIX_HPP_ */
