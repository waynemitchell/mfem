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
#include "matrix.hpp"
#include "sparsemat.hpp"
#include "blockVector.hpp"
#include "blockMatrix.hpp"


BlockMatrix::BlockMatrix(int nRowBlocks_, int nColBlocks_):
    SparseRowMatrix(0),
	owns_blocks(false),
	nRowBlocks(nRowBlocks_),
	nColBlocks(nColBlocks_),
	row_offsets(nRowBlocks_+1),
	col_offsets(nColBlocks_+1),
	nnz_elem(0),
	is_filled(false),
	Aij(nRowBlocks_,nColBlocks_)
{
	Aij = (SparseMatrix *)NULL;
}

BlockMatrix::~BlockMatrix()
{
	if(owns_blocks)
		for(SparseMatrix ** it = Aij.GetRow(0); it != Aij.GetRow(0)+(Aij.NumRows()*Aij.NumCols()); ++it)
			delete *it;
}

void BlockMatrix::SetBlock(int i, int j, SparseMatrix & mat)
{
	Aij(i,j) = &mat;
}

SparseMatrix & BlockMatrix::Block(int i, int j)
{
	return *Aij(i,j);
}

void BlockMatrix::Finalize()
{
	if(is_filled)
		return;

	row_offsets[0] = 0;
	for(int irow(0); irow != nRowBlocks; ++irow)
		for(int jcol(0); jcol != nColBlocks; ++jcol)
		{
			if(Aij(irow,jcol) != NULL)
			{
				row_offsets[irow+1] = Aij(irow,jcol)->Size();
				break;
			}
		}

	// do partial sums
	row_offsets.PartialSum();
	size = row_offsets.Last();
	//std::partial_sum(row_offsets.GetData(), row_offsets.GetData()+row_offsets.Size(), row_offsets.GetData());

	col_offsets[0] = 0;
	for(int jcol(0); jcol != nColBlocks; ++jcol)
		for(int irow(0); irow != nRowBlocks; ++irow)
		{
			if(Aij(irow,jcol) != NULL)
			{
				col_offsets[jcol+1] = Aij(irow,jcol)->Width();
				break;
			}
		}

	// do partial sums
	col_offsets.PartialSum();
	//std::partial_sum(col_offsets.GetData(), col_offsets.GetData()+col_offsets.Size(), col_offsets.GetData());

	nnz_elem = 0;
	for(int jcol(0); jcol != nColBlocks; ++jcol)
		for(int irow(0); irow != nRowBlocks; ++irow)
		{
			if(Aij(irow,jcol) )
				nnz_elem+= Aij(irow,jcol)->NumNonZeroElems();

		}

	is_filled = true;

}

double& BlockMatrix::Elem (int i, int j)
{
	int iloc, iblock;
	int jloc, jblock;

	findGlobalRow(i, iblock, iloc);
	findGlobalCol(j, jblock, jloc);

	if( IsZeroBlock(i, j) )
		mfem_error("BlockMatrix::Elem");

	return Aij(iblock, jblock)->Elem(iloc, jloc);
}

const double& BlockMatrix::Elem (int i, int j) const
{
	int iloc, iblock;
	int jloc, jblock;

	findGlobalRow(i, iblock, iloc);
	findGlobalCol(j, jblock, jloc);

	if( IsZeroBlock(i, j) )
		mfem_error("BlockMatrix::Elem");

	return Aij(iblock, jblock)->Elem(iloc, jloc);
}

const SparseMatrix & BlockMatrix::Block(int i, int j) const
{
	return *Aij(i,j);
}

int BlockMatrix::RowSize(const int i) const
{
	if(!is_filled)
		mfem_error("Finalize method should be called before Mult \n");

	int rowsize(0);

	int iblock, iloc;
	findGlobalRow(i, iblock, iloc);

	for(int jblock(0); jblock < nColBlocks; ++jblock)
		if( Aij(iblock,jblock) != NULL )
			rowsize += Aij(iblock,jblock)->RowSize(iloc);

	return rowsize;
}

int BlockMatrix::GetRow(const int row, Array<int> &cols, Vector &srow) const
{
	if(!is_filled)
		mfem_error("Finalize method should be called before Mult \n");

	int iblock, iloc, rowsize;
	findGlobalRow(row, iblock, iloc);
	rowsize = RowSize(row);
	cols.SetSize(rowsize);
	srow.SetSize(rowsize);

	Array<int> bcols;
	Vector bsrow;

	int * it_cols = cols.GetData();
	double *it_srow = srow.GetData();

	for(int jblock(0); jblock < nColBlocks; ++jblock)
		if( Aij(iblock,jblock) != NULL )
		{
			Aij(iblock,jblock)->GetRow(iloc, bcols, bsrow);
			for(int i(0); i < bcols.Size(); ++i)
			{
				*(it_cols++) = bcols[i] + col_offsets[jblock];
				*(it_srow++) = bsrow(i);
			}
		}

	return 0;
}

void BlockMatrix::EliminateRowCol(Array<int> & ess_bc_dofs, Vector & sol, Vector & rhs)
{

	if(nRowBlocks != nColBlocks)
		mfem_error("BlockMatrix::EliminateRowCol: nRowBlocks != nColBlocks");

	for(int iiblock(0); iiblock < nRowBlocks; ++iiblock)
		if(row_offsets[iiblock] != col_offsets[iiblock])
		{
			std::cout << "BlockMatrix::EliminateRowCol: row_offests[" << iiblock << "] != col_offsets["<<iiblock<<"]\n";
			mfem_error();
		}

	//We also have to do the same for each Aij
	Array<int> block_dofs;
	Vector block_sol, block_rhs;

	for(int iiblock(0); iiblock < nRowBlocks; ++iiblock)
	{
		int dsize = row_offsets[iiblock+1] - row_offsets[iiblock];
		block_dofs.MakeRef(ess_bc_dofs.GetData()+row_offsets[iiblock], dsize);
		block_sol.SetDataAndSize(sol.GetData()+row_offsets[iiblock], dsize);
		block_rhs.SetDataAndSize(rhs.GetData()+row_offsets[iiblock], dsize);

		if(Aij(iiblock, iiblock))
		{
			for(int i(0); i < block_dofs.Size(); ++i)
				if(block_dofs[i])
					Aij(iiblock, iiblock)->EliminateRowCol(i,block_sol(i), block_rhs);
		}
		else
		{
			for(int i(0); i < block_dofs.Size(); ++i)
				if(block_dofs[i])
					mfem_error("BlockMatrix::EliminateRowCol: Null diagonal block \n");
		}

		for(int jjblock(0); jjblock < nRowBlocks; ++jjblock)
		{
			if( jjblock != iiblock && Aij(iiblock, jjblock) )
			{
				for(int i(0); i < block_dofs.Size(); ++i)
					if(block_dofs[i])
						Aij(iiblock, jjblock)->EliminateRow(i);
			}
			if( jjblock != iiblock && Aij(jjblock, iiblock) )
			{
				block_rhs.SetDataAndSize(rhs.GetData()+row_offsets[jjblock], row_offsets[jjblock+1] - row_offsets[jjblock]);
				Aij(jjblock, iiblock)->EliminateCols(block_dofs, &block_sol, &block_rhs);
			}
		}
	}
}

void BlockMatrix::EliminateZeroRows()
{
	if(!is_filled)
		mfem_error("BlockMatrix::EliminateZeroRows() #0");

	if(nRowBlocks != nColBlocks)
		mfem_error("BlockMatrix::EliminateZeroRows() #1");

	for(int iblock(0); iblock < nRowBlocks; ++iblock)
	{
		if( Aij(iblock,iblock) )
		{
			double norm;
			for(int i = 0; i < Aij(iblock, iblock)->Size(); ++i)
			{
				norm = 0.;
				for(int jblock = 0; jblock < nColBlocks; ++jblock)
					if(Aij(iblock,jblock))
						norm += Aij(iblock,jblock)->GetRowNorml1(i);

				if(norm < 1e-12)
				{
					for(int jblock = 0; jblock < nColBlocks; ++jblock)
						if(Aij(iblock,jblock))
							Aij(iblock,jblock)->EliminateRow2(i);
				}
			}
		}
		else
		{
			double norm;
			for(int i = 0; i < row_offsets[iblock+1] - row_offsets[iblock]; ++i)
			{
				norm = 0.;
				for(int jblock = 0; jblock < nColBlocks; ++jblock)
					if(Aij(iblock,jblock))
						norm += Aij(iblock,jblock)->GetRowNorml1(i);

				if(norm < 1e-12)
				{
					std::cout<<"i = " << i << "\n";
					std::cout<<"norm = " << norm << "\n";
					mfem_error("BlockMatrix::EliminateZeroRows() #2");
				}
			}
		}
	}
}

void BlockMatrix::Mult(const Vector & x, Vector & y) const
{

	if(x.GetData() == y.GetData())
		mfem_error("Error: x and y can't point to the same datas \n");

	if(!is_filled)
		mfem_error("Finalize method should be called before Mult \n");

	y = 0.;

	Vector xblockview, yblockview;

	for(int iblock(0); iblock != nRowBlocks; ++iblock)
		for(int jblock(0); jblock != nColBlocks; ++jblock)
		{
			if(Aij(iblock, jblock) != NULL )
			{
				xblockview.SetDataAndSize(x.GetData() + col_offsets[jblock], col_offsets[jblock+1]-col_offsets[jblock]);
				yblockview.SetDataAndSize(y.GetData() + row_offsets[iblock], row_offsets[iblock+1]-row_offsets[iblock]);
				Aij(iblock, jblock)->AddMult(xblockview, yblockview);
			}
		}
}

void BlockMatrix::AddMult(const Vector & x, Vector & y, const double val) const
{

	if(x.GetData() == y.GetData())
		mfem_error("Error: x and y can't point to the same datas \n");

	if(!is_filled)
		mfem_error("Finalize method should be called before Mult \n");

	Vector xblockview, yblockview;

	for(int iblock(0); iblock != nRowBlocks; ++iblock)
		for(int jblock(0); jblock != nColBlocks; ++jblock)
		{
			if(Aij(iblock, jblock) != NULL )
			{
				xblockview.SetDataAndSize(x.GetData() + col_offsets[jblock], col_offsets[jblock+1]- col_offsets[jblock]);
				yblockview.SetDataAndSize(y.GetData() + row_offsets[iblock], row_offsets[iblock+1]- row_offsets[iblock]);
				Aij(iblock, jblock)->AddMult(xblockview, yblockview, val);
			}
		}
}

void BlockMatrix::MultTranspose(const Vector & x, Vector & y) const
{

	if(x.GetData() == y.GetData())
		mfem_error("Error: x and y can't point to the same datas \n");

	if(!is_filled)
		mfem_error("Finalize method should be called before Mult \n");

	y = 0.;
	Vector xblockview, yblockview;

	for(int iblock(0); iblock != nRowBlocks; ++iblock)
		for(int jblock(0); jblock != nColBlocks; ++jblock)
		{
			if(Aij(jblock, iblock) != NULL )
			{
				xblockview.SetDataAndSize(x.GetData() + row_offsets[jblock], row_offsets[jblock+1]-row_offsets[jblock]);
				yblockview.SetDataAndSize(y.GetData() + col_offsets[iblock], col_offsets[iblock+1]-col_offsets[iblock]);
				Aij(jblock, iblock)->AddMultTranspose(xblockview, yblockview);
			}
		}
}

void BlockMatrix::AddMultTranspose(const Vector & x, Vector & y, const double val) const
{

	if(x.GetData() == y.GetData())
		mfem_error("Error: x and y can't point to the same datas \n");

	if(!is_filled)
		mfem_error("Finalize method should be called before Mult \n");

	Vector xblockview, yblockview;

	for(int iblock(0); iblock != nRowBlocks; ++iblock)
		for(int jblock(0); jblock != nColBlocks; ++jblock)
		{
			if(Aij(jblock, iblock) != NULL )
			{
				xblockview.SetDataAndSize(x.GetData() + row_offsets[jblock], row_offsets[jblock+1]- row_offsets[jblock]);
				yblockview.SetDataAndSize(y.GetData() + col_offsets[iblock], col_offsets[iblock+1]- col_offsets[iblock]);
				Aij(jblock, iblock)->AddMultTranspose(xblockview, yblockview, val);
			}
		}
}

SparseMatrix * BlockMatrix::Monolithic()
{
	if(!is_filled)
		Finalize();

	int nnz(0);
	for(int irow(0); irow != nRowBlocks; ++irow)
		for(int jcol(0); jcol != nColBlocks; ++jcol)
		{
			if(Aij(irow,jcol) != NULL)
				nnz += Aij(irow,jcol)->NumNonZeroElems();
		}

	int * i_amono = new int[ row_offsets[nRowBlocks]+2 ];
	int * j_amono = new int[ nnz ];
	double * data = new double[ nnz ];

	for (int i = 0; i < row_offsets[nRowBlocks]+2; i++) 
	  i_amono[i] = 0;

	int * i_amono_construction = i_amono+1;

	int * i_it(i_amono_construction);

	for(int iblock(0); iblock != nRowBlocks; ++iblock)
	{
		for(int irow(row_offsets[iblock]); irow < row_offsets[iblock+1]; ++irow)
		{
			int local_row = irow - row_offsets[iblock];
			int ind = i_amono_construction[irow];
			for(int jblock(0); jblock < nColBlocks; ++jblock)
			{
				if(Aij(iblock,jblock) != NULL)
					ind += Aij(iblock, jblock)->GetI()[local_row+1] - Aij(iblock, jblock)->GetI()[local_row];
			}
			i_amono_construction[irow+1] = ind;
		}
	}

	//Fill in the jarray and copy the data
	for(int iblock(0); iblock != nRowBlocks; ++iblock)
	{
		for(int jblock(0); jblock != nColBlocks; ++jblock)
		{
			if(Aij(iblock,jblock) != NULL)
			{
				int nrow = row_offsets[iblock+1]-row_offsets[iblock];
				int * i_aij = Aij(iblock, jblock)->GetI();
				int * j_aij = Aij(iblock, jblock)->GetJ();
				double * data_aij = Aij(iblock, jblock)->GetData();
				i_it = i_amono_construction+row_offsets[iblock];

				int loc_start_index(0);
				int loc_end_index(0);
				int glob_start_index(0);

				int shift(col_offsets[jblock]);
				for(int * i_it_aij(i_aij+1); i_it_aij != i_aij+nrow+1; ++i_it_aij)
				{

					glob_start_index = *i_it;

#ifdef MFEM_DEBUG
					if(glob_start_index > nnz)
					{
						std::cout<<"glob_start_index = " << glob_start_index << "\n";
						std::cout<<"Block:" << iblock << " " << jblock << "\n";
						std::cout<<std::endl;
					}
#endif
					loc_end_index = *(i_it_aij);
					for(int cnt = 0; cnt < loc_end_index-loc_start_index; cnt++) {
					  data[glob_start_index+cnt] = data_aij[loc_start_index+cnt];
					  j_amono[glob_start_index+cnt] = j_aij[loc_start_index+cnt] + shift;
					}
				       
					*i_it += loc_end_index-loc_start_index;
					++i_it;
					loc_start_index = loc_end_index;
				}

			}
		}
	}

	return new SparseMatrix(i_amono, j_amono, data, row_offsets[nRowBlocks], col_offsets[nColBlocks]);
}

void BlockMatrix::PrintMatlab(std::ostream & os)
{
	if(!is_filled)
		mfem_error("BlockMatrix::PrintMatlab please finalize the matrix first");

   Vector row_data;
   Array<int> row_ind;
   os<<"% size " << row_offsets.Last() << " " << col_offsets.Last() << "\n";
   os<<"% Non Zeros " << nnz_elem << "\n";
   int i, j;
   ios::fmtflags old_fmt = os.flags();
   os.setf(ios::scientific);
   int old_prec = os.precision(14);
   for(i = 0; i < row_offsets.Last(); i++)
   {
	   GetRow(i, row_ind, row_data);
       for (j = 0; j < row_ind.Size(); j++)
         os << i+1 << " " << row_ind[j]+1 << " " << row_data[j] << std::endl;
   }

   os.precision(old_prec);
   os.flags(old_fmt);
}

BlockMatrix * Transpose(const BlockMatrix & A)
{
	BlockMatrix * At = new BlockMatrix(A.NumColBlocks(), A.NumRowBlocks());
	At->owns_blocks = 1;

	for(int irowAt(0); irowAt < At->NumRowBlocks(); ++irowAt)
		for(int jcolAt(0); jcolAt < At->NumColBlocks(); ++jcolAt)
			if( !A.IsZeroBlock(jcolAt, irowAt))
				At->SetBlock(irowAt, jcolAt, *Transpose( const_cast<SparseMatrix &>(A.Block(jcolAt, irowAt)) ) );

	At->Finalize();
	return At;
}

BlockMatrix * Mult(const BlockMatrix & A, const BlockMatrix & B)
{
	BlockMatrix * C= new BlockMatrix(A.NumRowBlocks(), B.NumColBlocks() );
	C->owns_blocks = 1;
	Array<SparseMatrix *> CijPieces( A.NumColBlocks() );

	for(int irowC(0); irowC < A.NumRowBlocks(); ++irowC)
		for(int jcolC(0); jcolC < B.NumColBlocks(); ++jcolC)
		{
			CijPieces.SetSize(0, static_cast<SparseMatrix *>(NULL));
			for(int k(0); k < A.NumColBlocks(); ++k)
				if( !A.IsZeroBlock(irowC, k) && !B.IsZeroBlock(k, jcolC))
					CijPieces.Append( Mult(const_cast<SparseMatrix &>(A.Block(irowC, k)), const_cast<SparseMatrix &>(B.Block(k, jcolC) ) ) );

			if( CijPieces.Size() > 1 )
			{
				C->SetBlock(irowC, jcolC, *Add(CijPieces));
				for(SparseMatrix ** it = CijPieces.GetData(); it != CijPieces.GetData()+CijPieces.Size(); ++it)
					delete *it;
			}
			else if(CijPieces.Size() == 1)
			{
				C->SetBlock(irowC, jcolC, *CijPieces[0]);
			}
		}


	C->Finalize();
	return C;
}

