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

// Implementation of class BilinearForm

#include "fem.hpp"
#include <cmath>

#include <fstream>

namespace mfem
{

void BilinearForm::AllocMat()
{
   bool symInt = true;
   if ( fes->GetNPrDofs() != 0 )
   {
     tmp_p = new Vector(fes->GetPrVSize());

     if ( fes->GetOrdering() == Ordering::byNODES && fes->GetVDim() > 1 )
     {
        v1_e = new Vector(fes->GetExVSize());
	v1_p = new Vector(fes->GetPrVSize());
	v2_e = new Vector(fes->GetExVSize());
	v2_p = new Vector(fes->GetPrVSize());
     }
     else
     {
        v1_e = new Vector();
	v1_p = new Vector();
	v2_e = new Vector();
	v2_p = new Vector();
     }

     if ( fbfi.Size() > 0 ) symInt = false;
     if ( dbfi.Size() > 0 )
     {
       for (int k = 1; k < dbfi.Size(); k++)
       {
	 if ( !dbfi[k]->IsSymmetric() )
	 {
	   symInt = false;
	 }
       }
     }
   }

   if (precompute_sparsity == 0 || fes->GetVDim() > 1)
   {
      if ( fes->GetNPrDofs() == 0 )
      {
	 mat = new SparseMatrix(height);
      }
      else
      {
	 mat_ee = new SparseMatrix(fes->GetExVSize());
 	 mat_ep = new SparseMatrix(fes->GetExVSize(),fes->GetPrVSize());
	 if ( !symInt )
	 {
	    mat_pe = new SparseMatrix(fes->GetPrVSize(),fes->GetExVSize());
	 }
	 mat_rr = new SparseMatrix(fes->GetExVSize());
	 mat_pp = new DenseMatrix*[fes->GetNE()];
	 mat_pp_inv = new DenseMatrixInverse*[fes->GetNE()];
	 for (int i=0; i<fes->GetNE(); i++) mat_pp[i] = new DenseMatrix();
      }
      return;
   }

   fes->BuildElementToDofTable();
   const Table &elem_dof = fes->GetElementToDofTable();
   Table dof_dof, dof_prdof, prdof_dof;

   if (fbfi.Size() > 0)
   {
      // the sparsity pattern is defined from the map: face->element->dof
      Table face_dof, dof_face;
      {
         Table *face_elem = fes->GetMesh()->GetFaceToElementTable();
         mfem::Mult(*face_elem, elem_dof, face_dof);
         delete face_elem;
      }
      Transpose(face_dof, dof_face, height);
      mfem::Mult(dof_face, face_dof, dof_dof);
   }
   else
   {
      // the sparsity pattern is defined from the map: element->dof
      Table dof_elem, elem_pr, pr_elem;
      Transpose(elem_dof, dof_elem, fes->GetExVSize());
      mfem::Mult(dof_elem, elem_dof, dof_dof);

      if ( fes->GetNPrDofs() > 0 )
      {
  	 int vdim = fes->GetVDim();
 	 int * pr_offsets = fes->GetPrivateOffsets();
	 int * pr_j = new int[vdim*pr_offsets[fes->GetNE()]];
	 for (int j=0; j<vdim*pr_offsets[fes->GetNE()]; j++) pr_j[j] = j;
	 elem_pr.SetIJ(pr_offsets,pr_j,fes->GetNE());
	 mfem::Mult(dof_elem, elem_pr,dof_prdof);
	 if ( !symInt )
	 {
	    Transpose(elem_pr, pr_elem, fes->GetPrVSize());
	    mfem::Mult(pr_elem,elem_dof, prdof_dof);
	 }
	 elem_pr.LoseData();
	 delete [] pr_j;
      }
   }

   int *I = dof_dof.GetI();
   int *J = dof_dof.GetJ();

   if ( fes->GetNPrDofs() == 0  )
   {
      mat = new SparseMatrix(I, J, NULL, dof_dof.Size(), dof_dof.Size(),
			     true, true, false);
      *mat = 0.0;
   }
   else
   {
      int * I_pr = dof_prdof.GetI();
      int * J_pr = dof_prdof.GetJ();

      mat_ee = new SparseMatrix(I, J, NULL, dof_dof.Size(), dof_dof.Size(),
				true, true, false);
      mat_rr = new SparseMatrix(I, J, NULL, dof_dof.Size(), dof_dof.Size(),
				false, true, false);
      mat_ep = new SparseMatrix(I_pr, J_pr, NULL,
				dof_dof.Size(), fes->GetPrVSize(),
				true, true, false);
      mat_pp = new DenseMatrix*[fes->GetNE()];
      for (int i=0; i<fes->GetNE(); i++) mat_pp[i] = new DenseMatrix();

      mat_pp_inv = new DenseMatrixInverse*[fes->GetNE()];

      *mat_ee = 0.0;
      *mat_rr = 0.0;
      *mat_ep = 0.0;

      dof_prdof.LoseData();

      if ( !symInt )
      {
	 int * I_pr_T = prdof_dof.GetI();
	 int * J_pr_T = prdof_dof.GetJ();

	 mat_pe = new SparseMatrix(I_pr_T, J_pr_T, NULL,
				   fes->GetPrVSize(), dof_dof.Size(),
				   true, true, false);
	 *mat_pe = 0.0;
	 prdof_dof.LoseData();
      }
   }

   dof_dof.LoseData();
}

BilinearForm::BilinearForm (FiniteElementSpace * f)
   : Matrix (f->GetVSize())
{
   fes = f;
   mat = mat_e = NULL;
   mat_ee = mat_ep = mat_pe = mat_rr = NULL;
   mat_pp = NULL;
   mat_pp_inv = NULL;
   tmp_p = NULL;
   v1_e = v1_p = v2_e = v2_p = NULL;
   extern_bfs = 0;
   element_matrices = NULL;
   precompute_sparsity = 0;
}

BilinearForm::BilinearForm (FiniteElementSpace * f, BilinearForm * bf, int ps)
   : Matrix (f->GetVSize())
{
   int i;
   Array<BilinearFormIntegrator*> *bfi;

   fes = f;
   mat_e = NULL;
   mat_ee = mat_ep = mat_pe = mat_rr = NULL;
   mat_pp = NULL;
   mat_pp_inv = NULL;
   tmp_p = NULL;
   v1_e = v1_p = v2_e = v2_p = NULL;
   extern_bfs = 1;
   element_matrices = NULL;
   precompute_sparsity = ps;

   bfi = bf->GetDBFI();
   dbfi.SetSize (bfi->Size());
   for (i = 0; i < bfi->Size(); i++)
   {
      dbfi[i] = (*bfi)[i];
   }

   bfi = bf->GetBBFI();
   bbfi.SetSize (bfi->Size());
   for (i = 0; i < bfi->Size(); i++)
   {
      bbfi[i] = (*bfi)[i];
   }

   bfi = bf->GetFBFI();
   fbfi.SetSize (bfi->Size());
   for (i = 0; i < bfi->Size(); i++)
   {
      fbfi[i] = (*bfi)[i];
   }

   bfi = bf->GetBFBFI();
   bfbfi.SetSize (bfi->Size());
   for (i = 0; i < bfi->Size(); i++)
   {
      bfbfi[i] = (*bfi)[i];
   }

   AllocMat();
}

double& BilinearForm::Elem (int i, int j)
{
   return mat -> Elem(i,j);
}

const double& BilinearForm::Elem (int i, int j) const
{
   return mat -> Elem(i,j);
}

void BilinearForm::Mult (const Vector & x, Vector & y) const
{
   if ( mat != NULL )
   {
      mat -> Mult (x, y);
   }
   else
   {
      // Create temporary vectors for the exposed and private
      // portions of x and y
      this->SplitExposedPrivate(x,v1_e,v1_p);
      this->SplitExposedPrivate(y,v2_e,v2_p);

      // Compute the Exposed portion of the product
      mat_ee->Mult(*v1_e,*v2_e);
      mat_ep->AddMult(*v1_p,*v2_e);

      // Compute the Private portion of the product
      // Begin by multiplying the block diagonal portion element by element

      int vdim = fes->GetVDim();
      int npr  = fes->GetNPrDofs();
      const int * pr_offsets = fes->GetPrivateOffsets();

      Vector v1,v2;

      for (int i=0; i<fes->GetNE(); i++)
      {
	 if ( fes->GetOrdering() == Ordering::byNODES && fes->GetVDim() > 1 )
	 {
	    int size = mat_pp[i]->Size();
	    v1.SetSize(size);
	    v2.SetSize(size);

	    int jj = 0;
	    for (int dj=0; dj<vdim; dj++)
	    {
	       for (int j=pr_offsets[i]; j<pr_offsets[i+1]; j++)
	       {
		  v1(jj) = (*v1_p)(dj*npr+j);
		  jj++;
	       }
	    }

	    mat_pp[i]->Mult(v1,v2);

	    jj = 0;
	    for (int dj=0; dj<vdim; dj++)
	    {
	       for (int j=pr_offsets[i]; j<pr_offsets[i+1]; j++)
	       {
		  (*v2_p)(dj*npr+j) = v2(jj);
		  jj++;
	       }
	    }

	 }
	 else
	 {
	    mat_pp[i]->Mult(&(*v1_p)[vdim*pr_offsets[i]],
			    &(*v2_p)[vdim*pr_offsets[i]]);
	 }
      }

      // Finish by multiplying the off-diagonal block
      if ( mat_pe != NULL )
      {
	 mat_pe->AddMult(*v1_e,*v2_p);
      }
      else
      {
 	 mat_ep->AddMultTranspose(*v1_e,*v2_p);
      }

      if ( fes->GetOrdering() == Ordering::byNODES && fes->GetVDim() > 1 )
	this->MergeExposedPrivate(v2_e,v2_p,y);

   }
}

void
BilinearForm::SplitExposedPrivate(const Vector &x,
				  Vector *x_e, Vector *x_p) const
{
  if ( fes->GetOrdering() == Ordering::byNODES && fes->GetVDim() > 1 )
  {
     int nex = fes->GetNExDofs();
     int npr = fes->GetNPrDofs();
     int vdim = fes->GetVDim();
     if ( x_e != NULL )
        for (int di=0; di<vdim; di++)
	   for (int i=0; i<nex; i++)
	      (*x_e)(nex*di+i) = x((nex+npr)*di+i);
     if ( x_p != NULL )
        for (int di=0; di<vdim; di++)
	   for (int i=0; i<npr; i++)
	      (*x_p)(npr*di+i) = x((nex+npr)*di+nex+i);
  }
  else
  {
     if ( x_e != NULL )
        x_e->SetDataAndSize(const_cast<double*>(&x[0]),fes->GetExVSize());
     if ( x_p != NULL && (fes->GetPrVSize() > 0) )
        x_p->SetDataAndSize(const_cast<double*>(&x[fes->GetExVSize()]),
			    fes->GetPrVSize());
  }
}

void
BilinearForm::MergeExposedPrivate(Vector *x_e, Vector *x_p, Vector &x) const
{
  if ( fes->GetOrdering() == Ordering::byNODES && fes->GetVDim() > 1 )
  {
     int nex = fes->GetNExDofs();
     int npr = fes->GetNPrDofs();
     int vdim = fes->GetVDim();
     if ( x_e != NULL )
        for (int di=0; di<vdim; di++)
	   for (int i=0; i<nex; i++)
	      x((nex+npr)*di+i) = (*x_e)(nex*di+i);
     if ( x_p != NULL )
        for (int di=0; di<vdim; di++)
	   for (int i=0; i<npr; i++)
	      x((nex+npr)*di+nex+i) = (*x_p)(npr*di+i);
  }
}

Vector *
BilinearForm::RHS_R(const Vector & rhs) const
{
   // Create temporary vectors for the exposed and private portions of rhs
   this->SplitExposedPrivate(rhs,v1_e,v1_p);

   return this->RHS_R(*v1_e,*v1_p);
}

Vector *
BilinearForm::RHS_R(const Vector & rhs_e, const Vector & rhs_p) const
{
   int vdim = fes->GetVDim();
   int npr  = fes->GetNPrDofs();
   const int * pr_offsets = fes->GetPrivateOffsets();

   if ( mat_pp_inv != NULL )
   {
     Vector v1,v2;
     for (int i=0; i<fes->GetNE(); i++)
     {
       int size = mat_pp_inv[i]->Size();
       if ( fes->GetOrdering() == Ordering::byNODES && fes->GetVDim() > 1 )
       {
	 v1.SetSize(size);
	 v2.SetSize(size);

	 int jj = 0;
	 for (int dj=0; dj<vdim; dj++)
	 {
	   for (int j=pr_offsets[i]; j<pr_offsets[i+1]; j++)
	   {
	     v1(jj) = (*tmp_p)(dj*npr+j);
	     jj++;
	   }
	 }
       }
       else
       {
	 v1.SetDataAndSize(&rhs_p.GetData()[vdim*pr_offsets[i]],size);
	 v2.SetDataAndSize(&(tmp_p->GetData())[vdim*pr_offsets[i]],size);
       }

       mat_pp_inv[i]->Mult(v1,v2);

       if ( fes->GetOrdering() == Ordering::byNODES && fes->GetVDim() > 1 )
       {
	 int jj = 0;
	 for (int dj=0; dj<vdim; dj++)
	 {
	   for (int j=pr_offsets[i]; j<pr_offsets[i+1]; j++)
	   {
	     (*v2_p)(dj*npr+j) = v2(jj);
	     jj++;
	   }
	 }
       }
     }
   }

   Vector * rhs_r = new Vector(fes->GetExVSize());

   rhs_r->Set(1.0,rhs_e);

   if ( mat_ep != NULL )
   {
     mat_ep->AddMult(*tmp_p,*rhs_r,-1.0);
   }

   return rhs_r;
}

void
BilinearForm::UpdatePrivateDoFs(const Vector &rhs, Vector &sol) const
{
   if ( fes->GetNPrDofs() == 0 ) { return; }

   // Create temporary vectors for the private portion of rhs
   this->SplitExposedPrivate(rhs, NULL, v1_p);

   // Create temporary vectors for the exposed and private portions of sol
   this->SplitExposedPrivate(sol, v2_e, v2_p);

   tmp_p->Set(1.0,(*v1_p));
   if ( mat_pe != NULL )
   {
      mat_pe->AddMult(*v2_e,*tmp_p,-1.0);
   }
   else
   {
     mat_ep->AddMultTranspose(*v2_e,*tmp_p,-1.0);
   }

   int vdim = fes->GetVDim();
   int npr  = fes->GetNPrDofs();
   const int * pr_offsets = fes->GetPrivateOffsets();

   Vector v1,v2;

   for (int i=0; i<fes->GetNE(); i++)
   {
      int size = mat_pp_inv[i]->Size();
      if ( fes->GetOrdering() == Ordering::byNODES && fes->GetVDim() > 1 )
      {
	 v1.SetSize(size);
	 v2.SetSize(size);

	 int jj = 0;
	 for (int dj=0; dj<vdim; dj++)
	 {
	    for (int j=pr_offsets[i]; j<pr_offsets[i+1]; j++)
	    {
	       v1(jj) = (*tmp_p)(dj*npr+j);
	       jj++;
	    }
	 }
      }
      else
      {
 	 v1.SetDataAndSize(&tmp_p->GetData()[vdim*pr_offsets[i]],size);
	 v2.SetDataAndSize(&(*v2_p)[vdim*pr_offsets[i]],size);
      }
      /*
      v1.SetDataAndSize(&tmp_p->GetData()[vdim*pr_offsets[i]],size);
      if ( fes->GetOrdering() == Ordering::byNODES && fes->GetVDim() > 1 )
      {
	 v2.SetSize(size);
      }
      else
      {
	 v2.SetDataAndSize(&(*v2_p)[vdim*pr_offsets[i]],size);
      }
      */
      mat_pp_inv[i]->Mult(v1,v2);

      if ( fes->GetOrdering() == Ordering::byNODES && fes->GetVDim() > 1 )
      {
	int jj = 0;
	for (int dj=0; dj<vdim; dj++)
	{
	   for (int j=pr_offsets[i]; j<pr_offsets[i+1]; j++)
	   {
	     (*v2_p)(dj*npr+j) = v2(jj);
	      jj++;
	   }
	}
      }
   }

   if ( fes->GetOrdering() == Ordering::byNODES && fes->GetVDim() > 1 )
     this->MergeExposedPrivate(NULL,v2_p,sol);
}

MatrixInverse * BilinearForm::Inverse() const
{
   return mat -> Inverse();
}

void BilinearForm::Finalize (int skip_zeros)
{
   if ( mat != NULL )
   {
      mat -> Finalize (skip_zeros);
   }
   if ( mat_ee != NULL )
   {
      mat_ee -> Finalize (skip_zeros);
   }
   if ( mat_ep != NULL )
   {
      mat_ep -> Finalize (skip_zeros);
   }
   if ( mat_rr != NULL )
   {
      mat_rr -> Finalize (skip_zeros);
   }
   if ( mat_pe != NULL )
   {
     mat_pe -> Finalize (skip_zeros);
   }
   if (mat_e)
   {
      mat_e -> Finalize (skip_zeros);
   }
}

void BilinearForm::AddDomainIntegrator (BilinearFormIntegrator * bfi)
{
   dbfi.Append (bfi);
}

void BilinearForm::AddBoundaryIntegrator (BilinearFormIntegrator * bfi)
{
   bbfi.Append (bfi);
}

void BilinearForm::AddInteriorFaceIntegrator (BilinearFormIntegrator * bfi)
{
   fbfi.Append (bfi);
}

void BilinearForm::AddBdrFaceIntegrator (BilinearFormIntegrator * bfi)
{
   bfbfi.Append (bfi);
}

void BilinearForm::ComputeElementMatrix(int i, DenseMatrix &elmat)
{
   if (element_matrices)
   {
      elmat.SetSize(element_matrices->SizeI(), element_matrices->SizeJ());
      elmat = element_matrices->GetData(i);
      return;
   }

   if (dbfi.Size())
   {
      const FiniteElement &fe = *fes->GetFE(i);
      ElementTransformation *eltrans = fes->GetElementTransformation(i);
      dbfi[0]->AssembleElementMatrix(fe, *eltrans, elmat);
      for (int k = 1; k < dbfi.Size(); k++)
      {
         dbfi[k]->AssembleElementMatrix(fe, *eltrans, elemmat);
         elmat += elemmat;
      }
   }
   else
   {
      fes->GetElementVDofs(i, vdofs);
      elmat.SetSize(vdofs.Size());
      elmat = 0.0;
   }
}

void BilinearForm::AssembleElementMatrix(
   int i, const DenseMatrix &elmat, Array<int> &vdofs, int skip_zeros)
{
   if (mat == NULL)
   {
      AllocMat();
   }
   fes->GetElementVDofs(i, vdofs);
   mat->AddSubMatrix(vdofs, vdofs, elmat, skip_zeros);
}

void BilinearForm::Assemble(int skip_zeros)
{
   ElementTransformation *eltrans;
   Mesh *mesh = fes -> GetMesh();

   int i;

   if (mat == NULL && mat_ee == NULL )
   {
      AllocMat();
   }

#ifdef MFEM_USE_OPENMP
   int free_element_matrices = 0;
   if (!element_matrices)
   {
      ComputeElementMatrices();
      free_element_matrices = 1;
   }
#endif

   if (dbfi.Size())
   {
      if ( fes->GetNPrDofs() == 0 )
      {
	 for (i = 0; i < fes -> GetNE(); i++)
	 {
	    if ( 100*(i+1)/fes->GetNE() > 100*i/fes->GetNE() && false )
	       std::cout << 100*(i+1)/fes->GetNE() << "%" << std::endl;
	    fes->GetElementVDofs(i, vdofs);
	    if (element_matrices)
	    {
	       mat->AddSubMatrix(vdofs, vdofs, (*element_matrices)(i),
				 skip_zeros);
	    }
	    else
	    {
	       const FiniteElement &fe = *fes->GetFE(i);
	       eltrans = fes->GetElementTransformation(i);
	       for (int k = 0; k < dbfi.Size(); k++)
	       {
		  dbfi[k]->AssembleElementMatrix(fe, *eltrans, elemmat);
		  mat->AddSubMatrix(vdofs, vdofs, elemmat, skip_zeros);
	       }
	    }
	 }
      }
      else
      {
 	 DenseMatrix mee,mpe,mep,mrr;
	 Vector vpR,veL,vcMpe;

	 *mat_ee = 0.0;
	 *mat_ep = 0.0;
	 *mat_rr = 0.0;
	 if ( mat_pe != NULL )
	   *mat_pe = 0.0;

	 int npr = fes->GetNPrDofs();

	 for (i = 0; i < fes -> GetNE(); i++)
	 {
 	    if ( 100*(i+1)/fes->GetNE() > 100*i/fes->GetNE() && false )
	      std::cout << 100*(i+1)/fes->GetNE() << "%" << std::endl;
	    int vdim = fes->GetVDim();
	    int pr_offset, npri;
	    fes->GetElementVDofs(i, vdofs, pr_offset, npri);

	    if (element_matrices)
	    {
	       mee.CopyMN((*element_matrices)(i),
			  vdofs.Size(),vdofs.Size(),0,0);
	       mep.CopyMN((*element_matrices)(i),
			  vdofs.Size(),vdim*npri,0,vdofs.Size());
	       if ( mat_pe != NULL )
	       {
		  mpe.CopyMN((*element_matrices)(i),
			     vdim*npri,vdofs.Size(),vdofs.Size(),0);
	       }

	       mat_pp[i]->CopyMN((*element_matrices)(i),
				 vdim*npri, vdim*npri,
				 vdofs.Size(), vdofs.Size());
	    }
	    else
	    {
	       const FiniteElement &fe = *fes->GetFE(i);
	       eltrans = fes->GetElementTransformation(i);

	       mee.SetSize(vdofs.Size(),vdofs.Size());
	       mep.SetSize(vdofs.Size(),vdim*npri);
	       mat_pp[i]->SetSize(vdim*npri,vdim*npri);

	       mee = 0.0;
	       mep = 0.0;
	       *mat_pp[i] = 0.0;

	       if ( mat_pe != NULL )
	       {
		  mpe.SetSize(vdim*npri,vdofs.Size());
		  mpe = 0.0;
	       }

	       for (int k = 0; k < dbfi.Size(); k++)
	       {
		  dbfi[k]->AssembleElementMatrix(fe, *eltrans, elemmat);
		  permuteElementMatrix(elemmat,vdim,npri);

		  mee.AddMN(elemmat,
			    vdofs.Size(),vdofs.Size(),0,0);
		  mep.AddMN(elemmat,
			    vdofs.Size(),vdim*npri,0,vdofs.Size());
		  if ( mat_pe != NULL )
		  {
		     mpe.AddMN(elemmat,
			       vdim*npri,vdofs.Size(),vdofs.Size(),0);
		  }
		  // mat_pp[i]->AddMN(elemmat,nvpri,nvpri,
		  //          	      vdofs.Size(),vdofs.Size());

		  if ( fes->GetOrdering() == Ordering::byNODES )
		  {
		    for (int ii=0; ii<vdim*npri; ii++)
		      for (int jj=0; jj<vdim*npri; jj++)
			(*mat_pp[i])(ii,jj)
			  += elemmat(vdofs.Size()+ii,
				     vdofs.Size()+jj);
		  }
		  else
		  {
		    for (int ii=0; ii<npri; ii++)
		      for (int jj=0; jj<npri; jj++)
			for (int di=0; di<vdim; di++)
			  for (int dj=0; dj<vdim; dj++)
			    (*mat_pp[i])(vdim*ii+di,vdim*jj+dj)
			      += elemmat(vdofs.Size()+npri*di+ii,
					 vdofs.Size()+npri*dj+jj);
		  }
	       }
	    }

	    mat_ee->AddSubMatrix(vdofs, vdofs, mee, skip_zeros);

	    if ( fes->GetOrdering() == Ordering::byNODES )
	    {
	      for (int ii=0; ii<vdofs.Size(); ii++)
		for (int dj=0; dj<vdim; dj++)
		  for (int jj=0; jj<npri; jj++)
		    mat_ep->Add(vdofs[ii],npr*dj+pr_offset+jj,
				mep(ii,npri*dj+jj));
	    }
	    else
	    {
	      for (int ii=0; ii<vdofs.Size(); ii++)
		for (int jj=0; jj<npri; jj++)
		  for (int dj=0; dj<vdim; dj++)
		    mat_ep->Add(vdofs[ii],vdim*pr_offset+vdim*jj+dj,
				mep(ii,npri*dj+jj));
	    }

	    if ( mat_pe != NULL )
	    {
	      if ( fes->GetOrdering() == Ordering::byNODES )
	      {
		 for (int di=0; di<vdim; di++)
		   for (int ii=0; ii<npri; ii++)
		     for (int jj=0; jj<vdofs.Size(); jj++)
		       mat_pe->Add(npr*di+pr_offset+ii,vdofs[jj],
				   mpe(npri*di+ii,jj));
	      }
	      else
	      {
		for (int ii=0; ii<npri; ii++)
		  for (int jj=0; jj<vdofs.Size(); jj++)
		    for (int di=0; di<vdim; di++)
		      mat_pe->Add(vdim*pr_offset+vdim*ii+di,vdofs[jj],
				  mpe(npri*di+ii,jj));
	      }
	    }

	    mat_pp_inv[i] = (DenseMatrixInverse*)mat_pp[i]->Inverse();

	    vcMpe.SetSize(vdim*npri);
	    vpR.SetSize(vdim*npri);
	    veL.SetSize(vdofs.Size());
	    mrr.SetSize(vdofs.Size(),vdofs.Size());

	    for (int jj=0; jj<vdofs.Size(); jj++)
	    {
	       if ( fes->GetOrdering() == Ordering::byNODES )
	       {
		  for (int kk=0; kk<vdim*npri; kk++)
		    vcMpe(kk) = mep(jj,kk);
		  mat_pp_inv[i]->Mult(vcMpe,vpR);

		  for (int kk=0; kk<vdim*npri; kk++)
		    vcMpe(kk) = vpR(kk);
	       }
	       else
	       {
		  for (int kk=0; kk<npri; kk++)
		    for (int dk=0; dk<vdim; dk++)
		      vcMpe(vdim*kk+dk) = mep(jj,npri*dk+kk);
		  mat_pp_inv[i]->Mult(vcMpe,vpR);

		  for (int kk=0; kk<npri; kk++)
		    for (int dk=0; dk<vdim; dk++)
		      vcMpe(npri*dk+kk) = vpR(vdim*kk+dk);
	       }
	       mep.Mult(vcMpe,veL);

	       for (int ii=0; ii<vdofs.Size(); ii++)
		  mrr(ii,jj) = -veL(ii);
	    }

	    mrr += mee;
	    mat_rr->AddSubMatrix(vdofs, vdofs, mrr, skip_zeros);
	 }
      }
   }

   if (bbfi.Size())
   {
      for (i = 0; i < fes -> GetNBE(); i++)
      {
         const FiniteElement &be = *fes->GetBE(i);
         fes -> GetBdrElementVDofs (i, vdofs);
         eltrans = fes -> GetBdrElementTransformation (i);
         for (int k=0; k < bbfi.Size(); k++)
         {
            bbfi[k] -> AssembleElementMatrix(be, *eltrans, elemmat);
            mat -> AddSubMatrix (vdofs, vdofs, elemmat, skip_zeros);
         }
      }
   }

   if (fbfi.Size())
   {
      FaceElementTransformations *tr;
      Array<int> vdofs2;

      int nfaces = mesh->GetNumFaces();
      for (i = 0; i < nfaces; i++)
      {
         tr = mesh -> GetInteriorFaceTransformations (i);
         if (tr != NULL)
         {
            fes -> GetElementVDofs (tr -> Elem1No, vdofs);
            fes -> GetElementVDofs (tr -> Elem2No, vdofs2);
            vdofs.Append (vdofs2);
            for (int k = 0; k < fbfi.Size(); k++)
            {
               fbfi[k] -> AssembleFaceMatrix (*fes -> GetFE (tr -> Elem1No),
                                              *fes -> GetFE (tr -> Elem2No),
                                              *tr, elemmat);
               mat -> AddSubMatrix (vdofs, vdofs, elemmat, skip_zeros);
            }
         }
      }
   }

   if (bfbfi.Size())
   {
      FaceElementTransformations *tr;
      const FiniteElement *nfe = NULL;

      for (i = 0; i < fes -> GetNBE(); i++)
      {
         tr = mesh -> GetBdrFaceTransformations (i);
         if (tr != NULL)
         {
            fes -> GetElementVDofs (tr -> Elem1No, vdofs);
            for (int k = 0; k < bfbfi.Size(); k++)
            {
               bfbfi[k] -> AssembleFaceMatrix (*fes -> GetFE (tr -> Elem1No),
                                               *nfe, *tr, elemmat);
               mat -> AddSubMatrix (vdofs, vdofs, elemmat, skip_zeros);
            }
         }
      }
   }

#ifdef MFEM_USE_OPENMP
   if (free_element_matrices)
   {
      FreeElementMatrices();
   }
#endif
}

void BilinearForm::permuteElementMatrix(DenseMatrix &mat, int vdim, int npr)
{
   int h = mat.Height(), w = mat.Width();
   int ndof = h/vdim;
   int nex  = ndof-npr;

   DenseMatrix pmat(h,w);

   pmat = 0.0;
   for (int dj = 0; dj<vdim; dj++)
   {
      for (int di = 0; di<vdim; di++)
      {
         pmat.CopyMN(mat,nex,nex,ndof*di,ndof*dj,nex*di,nex*dj);
         pmat.CopyMN(mat,nex,npr,ndof*di,ndof*dj+nex,nex*di,nex*vdim+npr*dj);
         pmat.CopyMN(mat,npr,nex,ndof*di+nex,ndof*dj,nex*vdim+npr*di,nex*dj);
         pmat.CopyMN(mat,npr,npr,ndof*di+nex,ndof*dj+nex,
		     nex*vdim+npr*di,nex*vdim+npr*dj);
      }
   }

   mat = pmat;
}

void BilinearForm::ConformingAssemble()
{
   // Do not remove zero entries to preserve the symmetric structure of the
   // matrix which in turn will give rise to symmetric structure in the new
   // matrix. This ensures that subsequent calls to EliminateRowCol will work
   // correctly.
   Finalize(0);

   SparseMatrix *P = fes->GetConformingProlongation();
   if (!P) { return; } // assume conforming mesh

   SparseMatrix *R = Transpose(*P);
   SparseMatrix *RA = mfem::Mult(*R, *mat);
   delete mat;
   if (mat_e)
   {
      SparseMatrix *RAe = mfem::Mult(*R, *mat_e);
      delete mat_e;
      mat_e = RAe;
   }
   delete R;
   mat = mfem::Mult(*RA, *P);
   delete RA;
   if (mat_e)
   {
      SparseMatrix *RAeP = mfem::Mult(*mat_e, *P);
      delete mat_e;
      mat_e = RAeP;
   }

   height = mat->Height();
   width = mat->Width();
}

void BilinearForm::ComputeElementMatrices()
{
   if (element_matrices || dbfi.Size() == 0 || fes->GetNE() == 0)
   {
      return;
   }

   int num_elements = fes->GetNE();
   int num_dofs_per_el = fes->GetFE(0)->GetDof() * fes->GetVDim();

   element_matrices = new DenseTensor(num_dofs_per_el, num_dofs_per_el,
                                      num_elements);

   DenseMatrix tmp;
   IsoparametricTransformation eltrans;

#ifdef MFEM_USE_OPENMP
   #pragma omp parallel for private(tmp,eltrans)
#endif
   for (int i = 0; i < num_elements; i++)
   {
      DenseMatrix elmat(element_matrices->GetData(i),
                        num_dofs_per_el, num_dofs_per_el);
      const FiniteElement &fe = *fes->GetFE(i);
#ifdef MFEM_DEBUG
      if (num_dofs_per_el != fe.GetDof()*fes->GetVDim())
         mfem_error("BilinearForm::ComputeElementMatrices:"
                    " all elements must have same number of dofs");
#endif
      fes->GetElementTransformation(i, &eltrans);

      dbfi[0]->AssembleElementMatrix(fe, eltrans, elmat);
      for (int k = 1; k < dbfi.Size(); k++)
      {
         // note: some integrators may not be thread-safe
         dbfi[k]->AssembleElementMatrix(fe, eltrans, tmp);
         elmat += tmp;
      }
      elmat.ClearExternalData();
   }
}

void BilinearForm::EliminateEssentialBC (
   Array<int> &bdr_attr_is_ess, Vector &sol, Vector &rhs, int d )
{
   Array<int> ess_dofs, conf_ess_dofs;
   fes->GetEssentialVDofs(bdr_attr_is_ess, ess_dofs);
   if (fes->GetConformingProlongation() == NULL)
   {
      EliminateEssentialBCFromDofs(ess_dofs, sol, rhs, d);
   }
   else
   {
      fes->ConvertToConformingVDofs(ess_dofs, conf_ess_dofs);
      EliminateEssentialBCFromDofs(conf_ess_dofs, sol, rhs, d);
   }
}

void BilinearForm::EliminateVDofs (
   Array<int> &vdofs, Vector &sol, Vector &rhs, int d)
{
   for (int i = 0; i < vdofs.Size(); i++)
   {
      int vdof = vdofs[i];
      if ( vdof >= 0 )
      {
         mat -> EliminateRowCol (vdof, sol(vdof), rhs, d);
      }
      else
      {
         mat -> EliminateRowCol (-1-vdof, sol(-1-vdof), rhs, d);
      }
   }
}

void BilinearForm::EliminateVDofs(Array<int> &vdofs, int d)
{
   if (mat_e == NULL)
   {
      mat_e = new SparseMatrix(height);
   }

   for (int i = 0; i < vdofs.Size(); i++)
   {
      int vdof = vdofs[i];
      if ( vdof >= 0 )
      {
         mat -> EliminateRowCol (vdof, *mat_e, d);
      }
      else
      {
         mat -> EliminateRowCol (-1-vdof, *mat_e, d);
      }
   }
}

void BilinearForm::EliminateVDofsInRHS(
   Array<int> &vdofs, const Vector &x, Vector &b)
{
   mat_e->AddMult(x, b, -1.);
   mat->PartMult(vdofs, x, b);
}

void BilinearForm::EliminateEssentialBC (Array<int> &bdr_attr_is_ess, int d)
{
   Array<int> ess_dofs, conf_ess_dofs;
   fes->GetEssentialVDofs(bdr_attr_is_ess, ess_dofs);
   if (fes->GetConformingProlongation() == NULL)
   {
      EliminateEssentialBCFromDofs(ess_dofs, d);
   }
   else
   {
      fes->ConvertToConformingVDofs(ess_dofs, conf_ess_dofs);
      EliminateEssentialBCFromDofs(conf_ess_dofs, d);
   }
}

void BilinearForm::EliminateEssentialBCFromDofs (
   Array<int> &ess_dofs, Vector &sol, Vector &rhs, int d )
{
   MFEM_ASSERT(ess_dofs.Size() == height, "incorrect dof Array size");
   MFEM_ASSERT(sol.Size() == height, "incorrect sol Vector size");
   MFEM_ASSERT(rhs.Size() == height, "incorrect rhs Vector size");

   for (int i = 0; i < ess_dofs.Size(); i++)
      if (ess_dofs[i] < 0)
      {
         mat -> EliminateRowCol (i, sol(i), rhs, d);
      }
}

void BilinearForm::EliminateEssentialBCFromDofs (Array<int> &ess_dofs, int d)
{
   MFEM_ASSERT(ess_dofs.Size() == height, "incorrect dof Array size");

   for (int i = 0; i < ess_dofs.Size(); i++)
      if (ess_dofs[i] < 0)
      {
         mat -> EliminateRowCol (i, d);
      }
}

void BilinearForm::Update(FiniteElementSpace *nfes)
{
   if (nfes) { fes = nfes; }

   if ( mat_e != NULL ) delete mat_e;
   if ( mat != NULL ) delete mat;
   if ( mat_ee != NULL ) delete mat_ee;
   if ( mat_ep != NULL ) delete mat_ep;
   if ( mat_pe != NULL ) delete mat_pe;
   if ( mat_rr != NULL ) delete mat_rr;

   if ( mat_pp != NULL )
   {
     for (int i=0; i<fes->GetNE(); i++)
     {
        if ( mat_pp[i] != NULL ) delete mat_pp[i];
     }
     delete [] mat_pp;
   }
   if ( mat_pp_inv != NULL )
   {
     for (int i=0; i<fes->GetNE(); i++)
     {
        if ( mat_pp_inv[i] != NULL ) delete mat_pp_inv[i];
     }
     delete [] mat_pp_inv;
   }

   FreeElementMatrices();

   height = width = fes->GetVSize();

   mat = mat_e = mat_ee = mat_ep = mat_pe = mat_rr = NULL;
}

BilinearForm::~BilinearForm()
{
   if ( mat_e != NULL )delete mat_e;
   if ( mat != NULL ) delete mat;
   delete element_matrices;

   if ( mat_ee != NULL ) delete mat_ee;
   if ( mat_ep != NULL ) delete mat_ep;
   if ( mat_pe != NULL ) delete mat_pe;
   if ( mat_rr != NULL ) delete mat_rr;
   if ( mat_pp != NULL )
   {
     for (int i=0; i<fes->GetNE(); i++)
     {
        if ( mat_pp[i] != NULL ) delete mat_pp[i];
     }
     delete [] mat_pp;
   }
   if ( mat_pp_inv != NULL )
   {
     for (int i=0; i<fes->GetNE(); i++)
     {
        if ( mat_pp_inv[i] != NULL ) delete mat_pp_inv[i];
     }
     delete [] mat_pp_inv;
   }
   if ( tmp_p != NULL ) delete tmp_p;

   if ( v1_e != NULL ) delete v1_e;
   if ( v1_p != NULL ) delete v1_p;
   if ( v2_e != NULL ) delete v2_e;
   if ( v2_p != NULL ) delete v2_p;

   if (!extern_bfs)
   {
      int k;
      for (k=0; k < dbfi.Size(); k++) { delete dbfi[k]; }
      for (k=0; k < bbfi.Size(); k++) { delete bbfi[k]; }
      for (k=0; k < fbfi.Size(); k++) { delete fbfi[k]; }
      for (k=0; k < bfbfi.Size(); k++) { delete bfbfi[k]; }
   }
}

void MixedBilinearForm::AllocMat()
{
  if ( trial_fes->GetNPrDofs() == 0 || test_fes->GetNPrDofs() == 0 )
  {
    mat = new SparseMatrix(height, width);
  }
  else
  {
    mat_ee = new SparseMatrix(test_fes->GetExVSize(),
			      trial_fes->GetExVSize());
    mat_ep = new SparseMatrix(test_fes->GetExVSize(),
			      trial_fes->GetPrVSize());
    mat_pe = new SparseMatrix(test_fes->GetPrVSize(),
			      trial_fes->GetExVSize());
    mat_pp = new DenseMatrix*[test_fes->GetNE()];
    for (int i=0; i<test_fes->GetNE(); i++) mat_pp[i] = new DenseMatrix();

    if ( trial_fes->GetOrdering() == Ordering::byNODES &&
	 trial_fes->GetVDim() > 1 )
    {
      v1_e = new Vector(trial_fes->GetExVSize());
      v1_p = new Vector(trial_fes->GetPrVSize());
    }
    else
    {
      v1_e = new Vector();
      v1_p = new Vector();
    }
    if ( test_fes->GetOrdering() == Ordering::byNODES &&
	 test_fes->GetVDim() > 1 )
    {
      v2_e = new Vector(test_fes->GetExVSize());
      v2_p = new Vector(test_fes->GetPrVSize());
    }
    else
    {
      v2_e = new Vector();
      v2_p = new Vector();
    }
  }
}

void MixedBilinearForm::permuteElementMatrix(DenseMatrix &mat,
					     int vdim_r, int npr_r,
					     int vdim_c, int npr_c)
{
   int h = mat.Height(), w = mat.Width();

   int ndof_r = h/vdim_r;
   int ndof_c = w/vdim_c;

   int nex_r  = ndof_r-npr_r;
   int nex_c  = ndof_c-npr_c;

   DenseMatrix pmat(h,w);

   pmat = 0.0;
   for (int dj = 0; dj<vdim_c; dj++)
   {
      for (int di = 0; di<vdim_r; di++)
      {
         pmat.CopyMN(mat,nex_r,nex_c,
		     ndof_r*di,ndof_c*dj,
		     nex_r*di,nex_c*dj);
         pmat.CopyMN(mat,nex_r,npr_c,
		     ndof_r*di,ndof_c*dj+nex_c,
		     nex_r*di,nex_c*vdim_c+npr_c*dj);
         pmat.CopyMN(mat,npr_r,nex_c,
		     ndof_r*di+nex_r,ndof_c*dj,
		     nex_r*vdim_r+npr_r*di,nex_c*dj);
         pmat.CopyMN(mat,npr_r,npr_c,
		     ndof_r*di+nex_r,ndof_c*dj+nex_c,
		     nex_r*vdim_r+npr_r*di,nex_c*vdim_c+npr_c*dj);
      }
   }

   mat = pmat;
}

MixedBilinearForm::MixedBilinearForm (FiniteElementSpace *tr_fes,
                                      FiniteElementSpace *te_fes)
   : Matrix(te_fes->GetVSize(), tr_fes->GetVSize())
{
   trial_fes = tr_fes;
   test_fes = te_fes;
   mat = NULL;

   mat_ee = NULL;
   mat_ep = NULL;
   mat_pe = NULL;
   mat_pp = NULL;

   v1_e = v1_p = v2_e = v2_p = NULL;
}

double & MixedBilinearForm::Elem (int i, int j)
{
   if ( mat != NULL )
   {
     return (*mat)(i, j);
   }
}

const double & MixedBilinearForm::Elem (int i, int j) const
{
   if ( mat != NULL )
   {
     return (*mat)(i, j);
   }
}

void MixedBilinearForm::Mult (const Vector & x, Vector & y) const
{
   if ( mat != NULL )
   {
     mat -> Mult (x, y);
   }
   else
   {
      // Create temporary vectors for the exposed and private
      // portions of x and y
      this->SplitExposedPrivate(*trial_fes,x,v1_e,v1_p);
      this->SplitExposedPrivate(*test_fes,y,v2_e,v2_p);

      // Compute the Exposed portion of the product
      mat_ee->Mult(*v1_e,*v2_e);
      mat_ep->AddMult(*v1_p,*v2_e);

      // Compute the Private portion of the product
      // Begin by multiplying the block diagonal portion element by element

      int vdim_r = test_fes->GetVDim();
      int vdim_c = trial_fes->GetVDim();
      int npr_r  = test_fes->GetNPrDofs();
      int npr_c  = trial_fes->GetNPrDofs();
      const int * pr_offsets_r = test_fes->GetPrivateOffsets();
      const int * pr_offsets_c = trial_fes->GetPrivateOffsets();

      Vector v1,v2;

      for (int i=0; i<test_fes->GetNE(); i++)
      {
	 int size_r = mat_pp[i]->NumRows();
 	 int size_c = mat_pp[i]->NumCols();

	 // Setup the private portion of the local domain space vector
	 if ( trial_fes->GetOrdering() == Ordering::byNODES &&
	      trial_fes->GetVDim() > 1 )
	 {
	    v1.SetSize(size_c);

	    int jj = 0;
	    for (int dj=0; dj<vdim_c; dj++)
	    {
	       for (int j=pr_offsets_c[i]; j<pr_offsets_c[i+1]; j++)
	       {
		  v1(jj) = (*v1_p)(dj*npr_c+j);
		  jj++;
	       }
	    }
	 }
	 else
	 {
 	    v1.SetDataAndSize(&(*v1_p)[vdim_c*pr_offsets_c[i]],size_c);
	 }

	 // Setup the private portion of the local range space vector
	 if ( test_fes->GetOrdering() == Ordering::byNODES &&
	      test_fes->GetVDim() > 1 )
	 {
	    v2.SetSize(size_r);
	 }
	 else
	 {
 	    v2.SetDataAndSize(&(*v2_p)[vdim_r*pr_offsets_r[i]],size_r);
	 }

	 // Perform the private portion of the multiplication
	 mat_pp[i]->Mult(v1,v2);

	 // Commit the private portion of the local range space vector
	 if ( test_fes->GetOrdering() == Ordering::byNODES &&
	      test_fes->GetVDim() > 1 )
	 {
	    int jj = 0;
	    for (int dj=0; dj<vdim_r; dj++)
	    {
	       for (int j=pr_offsets_r[i]; j<pr_offsets_r[i+1]; j++)
	       {
		  (*v2_p)(dj*npr_r+j) = v2(jj);
		  jj++;
	       }
	    }
	 }
      }

      // Finish by multiplying the off-diagonal block
      mat_pe->AddMult(*v1_e,*v2_p);

      // Merge the exposed and private portions back into the result vector
      if ( test_fes->GetOrdering() == Ordering::byNODES &&
	   test_fes->GetVDim() > 1 )
	this->MergeExposedPrivate(*test_fes,v2_e,v2_p,y);

   }
}

void MixedBilinearForm::AddMult (const Vector & x, Vector & y,
                                 const double a) const
{
   if ( mat != NULL )
   {
     mat -> AddMult (x, y, a);
   }
   else
   {
      // Create temporary vectors for the exposed and private
      // portions of x and y
      this->SplitExposedPrivate(*trial_fes,x,v1_e,v1_p);
      this->SplitExposedPrivate(*test_fes,y,v2_e,v2_p);

      // Compute the Exposed portion of the product
      mat_ee->AddMult(*v1_e, *v2_e, a);
      mat_ep->AddMult(*v1_p, *v2_e, a);

      // Compute the Private portion of the product
      // Begin by multiplying the block diagonal portion element by element

      int vdim_r = test_fes->GetVDim();
      int vdim_c = trial_fes->GetVDim();
      int npr_r  = test_fes->GetNPrDofs();
      int npr_c  = trial_fes->GetNPrDofs();
      const int * pr_offsets_r = test_fes->GetPrivateOffsets();
      const int * pr_offsets_c = trial_fes->GetPrivateOffsets();

      Vector v1,v2;

      for (int i=0; i<test_fes->GetNE(); i++)
      {
	 int size_r = mat_pp[i]->NumRows();
 	 int size_c = mat_pp[i]->NumCols();

	 // Setup the private portion of the local domain space vector
	 if ( trial_fes->GetOrdering() == Ordering::byNODES &&
	      trial_fes->GetVDim() > 1 )
	 {
	    v1.SetSize(size_c);

	    int jj = 0;
	    for (int dj=0; dj<vdim_c; dj++)
	    {
	       for (int j=pr_offsets_c[i]; j<pr_offsets_c[i+1]; j++)
	       {
		  v1(jj) = (*v1_p)(dj*npr_c+j);
		  jj++;
	       }
	    }
	 }
	 else
	 {
 	    v1.SetDataAndSize(&(*v1_p)[vdim_c*pr_offsets_c[i]],size_c);
	 }

	 // Setup the private portion of the local range space vector
	 if ( test_fes->GetOrdering() == Ordering::byNODES &&
	      test_fes->GetVDim() > 1 )
	 {
	    v2.SetSize(size_r);

	    int jj = 0;
	    for (int dj=0; dj<vdim_r; dj++)
	    {
	       for (int j=pr_offsets_r[i]; j<pr_offsets_r[i+1]; j++)
	       {
		  v2(jj) = (*v2_p)(dj*npr_r+j);
		  jj++;
	       }
	    }
	 }
	 else
	 {
 	    v2.SetDataAndSize(&(*v2_p)[vdim_r*pr_offsets_r[i]],size_r);
	 }

	 // Perform the private portion of the multiplication
	 mat_pp[i]->AddMult(v1, v2, a);

	 // Commit the private portion of the local range space vector
	 if ( test_fes->GetOrdering() == Ordering::byNODES &&
	      test_fes->GetVDim() > 1 )
	 {
	    int jj = 0;
	    for (int dj=0; dj<vdim_r; dj++)
	    {
	       for (int j=pr_offsets_r[i]; j<pr_offsets_r[i+1]; j++)
	       {
		  (*v2_p)(dj*npr_r+j) = v2(jj);
		  jj++;
	       }
	    }
	 }
      }

      // Finish by multiplying the off-diagonal block
      mat_pe->AddMult(*v1_e, *v2_p, a);

      // Merge the exposed and private portions back into the result vector
      if ( test_fes->GetOrdering() == Ordering::byNODES &&
	   test_fes->GetVDim() > 1 )
	this->MergeExposedPrivate(*test_fes,v2_e,v2_p,y);

   }
}

void MixedBilinearForm::AddMultTranspose (const Vector & x, Vector & y,
                                          const double a) const
{
   if ( mat != NULL )
   {
     mat -> AddMultTranspose (x, y, a);
   }
   else
   {
      // Create temporary vectors for the exposed and private
      // portions of x and y
      this->SplitExposedPrivate(*test_fes,x,v2_e,v2_p);
      this->SplitExposedPrivate(*trial_fes,y,v1_e,v1_p);

      // Compute the Exposed portion of the product
      mat_ee->AddMultTranspose(*v2_e, *v1_e, a);
      mat_pe->AddMultTranspose(*v2_p, *v1_e, a);

      // Compute the Private portion of the product
      // Begin by multiplying the block diagonal portion element by element

      int vdim_r = test_fes->GetVDim();
      int vdim_c = trial_fes->GetVDim();
      int npr_r  = test_fes->GetNPrDofs();
      int npr_c  = trial_fes->GetNPrDofs();
      const int * pr_offsets_r = test_fes->GetPrivateOffsets();
      const int * pr_offsets_c = trial_fes->GetPrivateOffsets();

      Vector v1,v2;

      for (int i=0; i<test_fes->GetNE(); i++)
      {
	 int size_r = mat_pp[i]->NumRows();
 	 int size_c = mat_pp[i]->NumCols();

	 // Setup the private portion of the local domain space vector
	 if ( trial_fes->GetOrdering() == Ordering::byNODES &&
	      trial_fes->GetVDim() > 1 )
	 {
	    v1.SetSize(size_c);

	    int jj = 0;
	    for (int dj=0; dj<vdim_c; dj++)
	    {
	       for (int j=pr_offsets_c[i]; j<pr_offsets_c[i+1]; j++)
	       {
		  v1(jj) = (*v1_p)(dj*npr_c+j);
		  jj++;
	       }
	    }
	 }
	 else
	 {
 	    v1.SetDataAndSize(&(*v1_p)[vdim_c*pr_offsets_c[i]],size_c);
	 }

	 // Setup the private portion of the local range space vector
	 if ( test_fes->GetOrdering() == Ordering::byNODES &&
	      test_fes->GetVDim() > 1 )
	 {
	    v2.SetSize(size_r);

	    int jj = 0;
	    for (int dj=0; dj<vdim_r; dj++)
	    {
	       for (int j=pr_offsets_r[i]; j<pr_offsets_r[i+1]; j++)
	       {
		  v2(jj) = (*v2_p)(dj*npr_r+j);
		  jj++;
	       }
	    }
	 }
	 else
	 {
 	    v2.SetDataAndSize(&(*v2_p)[vdim_r*pr_offsets_r[i]],size_r);
	 }

	 // Perform the private portion of the multiplication
	 mat_pp[i]->AddMultTranspose(v2, v1, a);

	 // Commit the private portion of the local range space vector
	 if ( trial_fes->GetOrdering() == Ordering::byNODES &&
	      trial_fes->GetVDim() > 1 )
	 {
	    int jj = 0;
	    for (int dj=0; dj<vdim_c; dj++)
	    {
	       for (int j=pr_offsets_c[i]; j<pr_offsets_c[i+1]; j++)
	       {
		  (*v1_p)(dj*npr_c+j) = v1(jj);
		  jj++;
	       }
	    }
	 }
      }

      // Finish by multiplying the off-diagonal block
      mat_ep->AddMultTranspose(*v2_e, *v1_p, a);

      // Merge the exposed and private portions back into the result vector
      if ( trial_fes->GetOrdering() == Ordering::byNODES &&
	   trial_fes->GetVDim() > 1 )
	this->MergeExposedPrivate(*trial_fes,v1_e,v1_p,y);

   }
}

void
MixedBilinearForm::SplitExposedPrivate(const FiniteElementSpace &fes,
				       const Vector &x,
				       Vector *x_e, Vector *x_p) const
{
  if ( fes.GetOrdering() == Ordering::byNODES && fes.GetVDim() > 1 )
  {
     int nex = fes.GetNExDofs();
     int npr = fes.GetNPrDofs();
     int vdim = fes.GetVDim();
     if ( x_e != NULL )
        for (int di=0; di<vdim; di++)
	   for (int i=0; i<nex; i++)
	      (*x_e)(nex*di+i) = x((nex+npr)*di+i);
     if ( x_p != NULL )
        for (int di=0; di<vdim; di++)
	   for (int i=0; i<npr; i++)
	      (*x_p)(npr*di+i) = x((nex+npr)*di+nex+i);
  }
  else
  {
     if ( x_e != NULL )
        x_e->SetDataAndSize(const_cast<double*>(&x[0]),fes.GetExVSize());
     if ( x_p != NULL && (fes.GetPrVSize() > 0) )
        x_p->SetDataAndSize(const_cast<double*>(&x[fes.GetExVSize()]),
			    fes.GetPrVSize());
  }
}

void
MixedBilinearForm::MergeExposedPrivate(const FiniteElementSpace &fes,
				       Vector *x_e, Vector *x_p,
				       Vector &x) const
{
  if ( fes.GetOrdering() == Ordering::byNODES && fes.GetVDim() > 1 )
  {
     int nex = fes.GetNExDofs();
     int npr = fes.GetNPrDofs();
     int vdim = fes.GetVDim();
     if ( x_e != NULL )
        for (int di=0; di<vdim; di++)
	   for (int i=0; i<nex; i++)
	      x((nex+npr)*di+i) = (*x_e)(nex*di+i);
     if ( x_p != NULL )
        for (int di=0; di<vdim; di++)
	   for (int i=0; i<npr; i++)
	      x((nex+npr)*di+nex+i) = (*x_p)(npr*di+i);
  }
}

MatrixInverse * MixedBilinearForm::Inverse() const
{
   return mat -> Inverse ();
}

void MixedBilinearForm::Finalize (int skip_zeros)
{
   if ( mat != NULL )
     mat -> Finalize (skip_zeros);
   if ( mat_ee != NULL )
     mat_ee -> Finalize (skip_zeros);
   if ( mat_ep != NULL )
     mat_ep -> Finalize (skip_zeros);
   if ( mat_pe != NULL )
     mat_pe -> Finalize (skip_zeros);
}

void MixedBilinearForm::GetBlocks(Array2D<SparseMatrix *> &blocks) const
{
   if (trial_fes->GetOrdering() != Ordering::byNODES ||
       test_fes->GetOrdering() != Ordering::byNODES)
      mfem_error("MixedBilinearForm::GetBlocks :\n"
                 " Both trial and test spaces must use Ordering::byNODES!");

   blocks.SetSize(test_fes->GetVDim(), trial_fes->GetVDim());

   mat->GetBlocks(blocks);
}

void MixedBilinearForm::GetBlocksReduced(Array2D<SparseMatrix *> &blocks) const
{
   if (trial_fes->GetOrdering() != Ordering::byNODES ||
       test_fes->GetOrdering() != Ordering::byNODES)
      mfem_error("MixedBilinearForm::GetBlocksReduced :\n"
                 " Both trial and test spaces must use Ordering::byNODES!");

   blocks.SetSize(test_fes->GetVDim(), trial_fes->GetVDim());

   if ( test_fes->GetNPrDofs() == 0 && trial_fes->GetNPrDofs() == 0 )
   {
      mat->GetBlocks(blocks);
   }
   else
   {
      mat_ee->GetBlocks(blocks);
   }
}

void MixedBilinearForm::AddDomainIntegrator (BilinearFormIntegrator * bfi)
{
   dom.Append (bfi);
}

void MixedBilinearForm::AddBoundaryIntegrator (BilinearFormIntegrator * bfi)
{
   bdr.Append (bfi);
}

void MixedBilinearForm::AddTraceFaceIntegrator (BilinearFormIntegrator * bfi)
{
   skt.Append (bfi);
}

void MixedBilinearForm::Assemble (int skip_zeros)
{
   int i, k;
   Array<int> tr_vdofs, te_vdofs;
   ElementTransformation *eltrans;
   DenseMatrix elemmat;

   Mesh *mesh = test_fes -> GetMesh();

   if (mat == NULL && mat_ee == NULL )
   {
     AllocMat();
   }

   if (dom.Size())
   {
      if ( test_fes->GetNPrDofs() == 0 || trial_fes->GetNPrDofs() == 0 )
      {
	 for (i = 0; i < test_fes -> GetNE(); i++)
	 {
	    trial_fes -> GetElementVDofs (i, tr_vdofs);
	    test_fes  -> GetElementVDofs (i, te_vdofs);
	    eltrans = test_fes -> GetElementTransformation (i);
	    for (k = 0; k < dom.Size(); k++)
	    {
	       dom[k] -> AssembleElementMatrix2 (*trial_fes -> GetFE(i),
						 *test_fes  -> GetFE(i),
						 *eltrans, elemmat);
	       mat -> AddSubMatrix (te_vdofs, tr_vdofs, elemmat, skip_zeros);
	    }
	 }
      }
      else
      {
 	 DenseMatrix mee,mpe,mep;

	 *mat_ee = 0.0;
	 *mat_ep = 0.0;
	 *mat_pe = 0.0;

	 int npr_r = test_fes->GetNPrDofs();
	 int npr_c = trial_fes->GetNPrDofs();

	 for (i = 0; i < test_fes -> GetNE(); i++)
	 {
	    int vdim_r = test_fes->GetVDim();
	    int vdim_c = trial_fes->GetVDim();
	    int pr_offset_r, npri_r;
	    int pr_offset_c, npri_c;
	    test_fes->GetElementVDofs(i, te_vdofs, pr_offset_r, npri_r);
	    trial_fes->GetElementVDofs(i, tr_vdofs, pr_offset_c, npri_c);

	    const FiniteElement &te_fe = *test_fes->GetFE(i);
	    const FiniteElement &tr_fe = *trial_fes->GetFE(i);
	    eltrans = test_fes->GetElementTransformation(i);

	    mee.SetSize(te_vdofs.Size(),tr_vdofs.Size());
	    mep.SetSize(te_vdofs.Size(),vdim_c*npri_c);
	    mpe.SetSize(vdim_r*npri_r,tr_vdofs.Size());
	    mat_pp[i]->SetSize(vdim_r*npri_r,vdim_c*npri_c);

	    mee = 0.0;
	    mep = 0.0;
	    mpe = 0.0;
	    *mat_pp[i] = 0.0;

	    for (int k = 0; k < dom.Size(); k++)
	    {
	      dom[k]->AssembleElementMatrix2(tr_fe, te_fe,
					     *eltrans, elemmat);
	      permuteElementMatrix(elemmat,vdim_r,npri_r,vdim_c,npri_c);

	      mee.AddMN(elemmat,
			te_vdofs.Size(),tr_vdofs.Size(),0,0);
	      mep.AddMN(elemmat,
			te_vdofs.Size(),vdim_c*npri_c,0,tr_vdofs.Size());
	      mpe.AddMN(elemmat,
			vdim_r*npri_r,tr_vdofs.Size(),te_vdofs.Size(),0);

	      if ( test_fes->GetOrdering() == Ordering::byNODES &&
		   trial_fes->GetOrdering() == Ordering::byNODES )
	      {
		for (int ii=0; ii<vdim_r*npri_r; ii++)
		  for (int jj=0; jj<vdim_c*npri_c; jj++)
		    (*mat_pp[i])(ii,jj)
		      += elemmat(te_vdofs.Size()+ii,
				 tr_vdofs.Size()+jj);
	      }
	      else if ( test_fes->GetOrdering() == Ordering::byNODES &&
			trial_fes->GetOrdering() == Ordering::byVDIM )
	      {
		for (int ii=0; ii<vdim_r*npri_r; ii++)
		  for (int jj=0; jj<npri_c; jj++)
		    for (int dj=0; dj<vdim_c; dj++)
		      (*mat_pp[i])(ii,vdim_c*jj+dj)
			+= elemmat(te_vdofs.Size()+ii,
				   tr_vdofs.Size()+npri_c*dj+jj);
	      }
	      else if ( test_fes->GetOrdering() == Ordering::byVDIM &&
			trial_fes->GetOrdering() == Ordering::byNODES )
	      {
		for (int ii=0; ii<npri_r; ii++)
		  for (int di=0; di<vdim_r; di++)
		    for (int jj=0; jj<vdim_c*npri_c; jj++)
		      (*mat_pp[i])(vdim_r*ii+di,jj)
			+= elemmat(te_vdofs.Size()+npri_r*di+ii,
				   tr_vdofs.Size()+jj);
	      }
	      else
	      {
		for (int ii=0; ii<npri_r; ii++)
		  for (int jj=0; jj<npri_c; jj++)
		    for (int di=0; di<vdim_r; di++)
		      for (int dj=0; dj<vdim_c; dj++)
			(*mat_pp[i])(vdim_r*ii+di,vdim_c*jj+dj)
			  += elemmat(te_vdofs.Size()+npri_r*di+ii,
				     tr_vdofs.Size()+npri_c*dj+jj);
	      }
	    }

	    mat_ee->AddSubMatrix(te_vdofs, tr_vdofs, mee, skip_zeros);

	    if ( trial_fes->GetOrdering() == Ordering::byNODES )
	    {
	      for (int ii=0; ii<te_vdofs.Size(); ii++)
		for (int dj=0; dj<vdim_c; dj++)
		  for (int jj=0; jj<npri_c; jj++)
		    mat_ep->Add(te_vdofs[ii],npr_c*dj+pr_offset_c+jj,
				mep(ii,npri_c*dj+jj));
	    }
	    else
	    {
	      for (int ii=0; ii<te_vdofs.Size(); ii++)
		for (int jj=0; jj<npri_c; jj++)
		  for (int dj=0; dj<vdim_c; dj++)
		    mat_ep->Add(te_vdofs[ii],vdim_c*pr_offset_c+vdim_c*jj+dj,
				mep(ii,npri_c*dj+jj));
	    }

	    if ( test_fes->GetOrdering() == Ordering::byNODES )
	    {
	      for (int di=0; di<vdim_r; di++)
		for (int ii=0; ii<npri_r; ii++)
		  for (int jj=0; jj<tr_vdofs.Size(); jj++)
		    mat_pe->Add(npr_r*di+pr_offset_r+ii,tr_vdofs[jj],
				mpe(npri_r*di+ii,jj));
	    }
	    else
	    {
	      for (int ii=0; ii<npri_r; ii++)
		for (int jj=0; jj<tr_vdofs.Size(); jj++)
		  for (int di=0; di<vdim_r; di++)
		    mat_pe->Add(vdim_r*pr_offset_r+vdim_r*ii+di,tr_vdofs[jj],
				mpe(npri_r*di+ii,jj));
	    }
	 }
      }
   }

   if (bdr.Size())
   {
      for (i = 0; i < test_fes -> GetNBE(); i++)
      {
         trial_fes -> GetBdrElementVDofs (i, tr_vdofs);
         test_fes  -> GetBdrElementVDofs (i, te_vdofs);
         eltrans = test_fes -> GetBdrElementTransformation (i);
         for (k = 0; k < bdr.Size(); k++)
         {
            bdr[k] -> AssembleElementMatrix2 (*trial_fes -> GetBE(i),
                                              *test_fes  -> GetBE(i),
                                              *eltrans, elemmat);
            mat -> AddSubMatrix (te_vdofs, tr_vdofs, elemmat, skip_zeros);
         }
      }
   }

   if (skt.Size())
   {
      FaceElementTransformations *ftr;
      Array<int> te_vdofs2;
      const FiniteElement *trial_face_fe, *test_fe1, *test_fe2;

      int nfaces = mesh->GetNumFaces();
      for (i = 0; i < nfaces; i++)
      {
         ftr = mesh->GetFaceElementTransformations(i);
         trial_fes->GetFaceVDofs(i, tr_vdofs);
         test_fes->GetElementVDofs(ftr->Elem1No, te_vdofs);
         trial_face_fe = trial_fes->GetFaceElement(i);
         test_fe1 = test_fes->GetFE(ftr->Elem1No);
         if (ftr->Elem2No >= 0)
         {
            test_fes->GetElementVDofs(ftr->Elem2No, te_vdofs2);
            te_vdofs.Append(te_vdofs2);
            test_fe2 = test_fes->GetFE(ftr->Elem2No);
         }
         else
         {
            test_fe2 = NULL;
         }
         for (int k = 0; k < skt.Size(); k++)
         {
            skt[k]->AssembleFaceMatrix(*trial_face_fe, *test_fe1, *test_fe2,
                                       *ftr, elemmat);
            mat->AddSubMatrix(te_vdofs, tr_vdofs, elemmat, skip_zeros);
         }
      }
   }
}

void MixedBilinearForm::ConformingAssemble()
{
   Finalize();

   SparseMatrix *P2 = test_fes->GetConformingProlongation();
   if (P2)
   {
      SparseMatrix *R = Transpose(*P2);
      SparseMatrix *RA = mfem::Mult(*R, *mat);
      delete R;
      delete mat;
      mat = RA;
   }

   SparseMatrix *P1 = trial_fes->GetConformingProlongation();
   if (P1)
   {
      SparseMatrix *RAP = mfem::Mult(*mat, *P1);
      delete mat;
      mat = RAP;
   }

   height = mat->Height();
   width = mat->Width();
}

void MixedBilinearForm::EliminateTrialDofs (
   Array<int> &bdr_attr_is_ess, Vector &sol, Vector &rhs )
{
   int i, j, k;
   Array<int> tr_vdofs, cols_marker (trial_fes -> GetVSize());

   cols_marker = 0;
   for (i = 0; i < trial_fes -> GetNBE(); i++)
      if (bdr_attr_is_ess[trial_fes -> GetBdrAttribute (i)-1])
      {
         trial_fes -> GetBdrElementVDofs (i, tr_vdofs);
         for (j = 0; j < tr_vdofs.Size(); j++)
         {
            if ( (k = tr_vdofs[j]) < 0 )
            {
               k = -1-k;
            }
            cols_marker[k] = 1;
         }
      }
   mat -> EliminateCols (cols_marker, &sol, &rhs);
}

void MixedBilinearForm::EliminateEssentialBCFromTrialDofs (
   Array<int> &marked_vdofs, Vector &sol, Vector &rhs)
{
   mat -> EliminateCols (marked_vdofs, &sol, &rhs);
}

void MixedBilinearForm::EliminateTestDofs (Array<int> &bdr_attr_is_ess)
{
   int i, j, k;
   Array<int> te_vdofs;

   for (i = 0; i < test_fes -> GetNBE(); i++)
      if (bdr_attr_is_ess[test_fes -> GetBdrAttribute (i)-1])
      {
         test_fes -> GetBdrElementVDofs (i, te_vdofs);
         for (j = 0; j < te_vdofs.Size(); j++)
         {
            if ( (k = te_vdofs[j]) < 0 )
            {
               k = -1-k;
            }
            mat -> EliminateRow (k);
         }
      }
}

void MixedBilinearForm::Update()
{
   delete mat;
   mat = NULL;
   height = test_fes->GetVSize();
   width = trial_fes->GetVSize();
}

MixedBilinearForm::~MixedBilinearForm()
{
   int i;

   if (mat) { delete mat; }

   if ( mat_ee != NULL ) delete mat_ee;
   if ( mat_ep != NULL ) delete mat_ep;
   if ( mat_pe != NULL ) delete mat_pe;
   if ( mat_pp != NULL )
   {
     for (int i=0; i<test_fes->GetNE(); i++)
     {
        if ( mat_pp[i] != NULL ) delete mat_pp[i];
     }
     delete [] mat_pp;
   }

   if ( v1_e != NULL ) delete v1_e;
   if ( v1_p != NULL ) delete v1_p;
   if ( v2_e != NULL ) delete v2_e;
   if ( v2_p != NULL ) delete v2_p;

   for (i = 0; i < dom.Size(); i++) { delete dom[i]; }
   for (i = 0; i < bdr.Size(); i++) { delete bdr[i]; }
   for (i = 0; i < skt.Size(); i++) { delete skt[i]; }
}


void DiscreteLinearOperator::Assemble(int skip_zeros)
{
   Array<int> dom_vdofs, ran_vdofs;
   ElementTransformation *T;
   const FiniteElement *dom_fe, *ran_fe;
   DenseMatrix totelmat, elmat;

   if (mat == NULL && mat_ee == NULL)
   {
      AllocMat();
   }

   if (dom.Size() > 0)
   {
      if ( test_fes->GetNPrDofs() == 0 || trial_fes->GetNPrDofs() == 0 )
      {
	 for (int i = 0; i < test_fes->GetNE(); i++)
	 {
	    trial_fes->GetElementVDofs(i, dom_vdofs);
	    test_fes->GetElementVDofs(i, ran_vdofs);
	    T = test_fes->GetElementTransformation(i);
	    dom_fe = trial_fes->GetFE(i);
	    ran_fe = test_fes->GetFE(i);

	    dom[0]->AssembleElementMatrix2(*dom_fe, *ran_fe, *T, totelmat);
	    for (int j = 1; j < dom.Size(); j++)
	    {
	       dom[j]->AssembleElementMatrix2(*dom_fe, *ran_fe, *T, elmat);
	       totelmat += elmat;
	    }
	    mat->SetSubMatrix(ran_vdofs, dom_vdofs, totelmat, skip_zeros);
	 }
      }
      else
      {
 	 DenseMatrix mee,mpe,mep;

	 *mat_ee = 0.0;
	 *mat_ep = 0.0;
	 *mat_pe = 0.0;

	 int npr_r = test_fes->GetNPrDofs();
	 int npr_c = trial_fes->GetNPrDofs();

	 for (int i = 0; i < test_fes -> GetNE(); i++)
	 {
	    int vdim_r = test_fes->GetVDim();
	    int vdim_c = trial_fes->GetVDim();
	    int pr_offset_r, npri_r;
	    int pr_offset_c, npri_c;
	    test_fes->GetElementVDofs(i, ran_vdofs, pr_offset_r, npri_r);
	    trial_fes->GetElementVDofs(i, dom_vdofs, pr_offset_c, npri_c);

	    T = test_fes->GetElementTransformation(i);
	    dom_fe = trial_fes->GetFE(i);
	    ran_fe = test_fes->GetFE(i);

	    mee.SetSize(ran_vdofs.Size(),dom_vdofs.Size());
	    mep.SetSize(ran_vdofs.Size(),vdim_c*npri_c);
	    mpe.SetSize(vdim_r*npri_r,dom_vdofs.Size());
	    mat_pp[i]->SetSize(vdim_r*npri_r,vdim_c*npri_c);

	    mee = 0.0;
	    mep = 0.0;
	    mpe = 0.0;
	    *mat_pp[i] = 0.0;

	    for (int k = 0; k < dom.Size(); k++)
	    {
	      dom[k]->AssembleElementMatrix2(*dom_fe, *ran_fe, *T, totelmat);
	      permuteElementMatrix(totelmat,vdim_r,npri_r,vdim_c,npri_c);

	      mee.AddMN(totelmat,
			ran_vdofs.Size(),dom_vdofs.Size(),0,0);
	      mep.AddMN(totelmat,
			ran_vdofs.Size(),vdim_c*npri_c,0,dom_vdofs.Size());
	      mpe.AddMN(totelmat,
			vdim_r*npri_r,dom_vdofs.Size(),ran_vdofs.Size(),0);

	      if ( test_fes->GetOrdering() == Ordering::byNODES &&
		   trial_fes->GetOrdering() == Ordering::byNODES )
	      {
		for (int ii=0; ii<vdim_r*npri_r; ii++)
		  for (int jj=0; jj<vdim_c*npri_c; jj++)
		    (*mat_pp[i])(ii,jj)
		      += totelmat(ran_vdofs.Size()+ii,
				  dom_vdofs.Size()+jj);
	      }
	      else if ( test_fes->GetOrdering() == Ordering::byNODES &&
			trial_fes->GetOrdering() == Ordering::byVDIM )
	      {
		for (int ii=0; ii<vdim_r*npri_r; ii++)
		  for (int jj=0; jj<npri_c; jj++)
		    for (int dj=0; dj<vdim_c; dj++)
		      (*mat_pp[i])(ii,vdim_c*jj+dj)
			+= totelmat(ran_vdofs.Size()+ii,
				    dom_vdofs.Size()+npri_c*dj+jj);
	      }
	      else if ( test_fes->GetOrdering() == Ordering::byVDIM &&
			trial_fes->GetOrdering() == Ordering::byNODES )
	      {
		for (int ii=0; ii<npri_r; ii++)
		  for (int jj=0; jj<vdim_c*npri_c; jj++)
		    for (int di=0; di<vdim_r; di++)
		      (*mat_pp[i])(vdim_r*ii+di,jj)
			+= totelmat(ran_vdofs.Size()+npri_r*di+ii,
				    dom_vdofs.Size()+jj);
	      }
	      else
	      {
		for (int ii=0; ii<npri_r; ii++)
		  for (int jj=0; jj<npri_c; jj++)
		    for (int di=0; di<vdim_r; di++)
		      for (int dj=0; dj<vdim_c; dj++)
			(*mat_pp[i])(vdim_r*ii+di,vdim_c*jj+dj)
			  += totelmat(ran_vdofs.Size()+npri_r*di+ii,
				      dom_vdofs.Size()+npri_c*dj+jj);
	      }
	    }

	    mat_ee->SetSubMatrix(ran_vdofs, dom_vdofs, mee, skip_zeros);

	    if ( trial_fes->GetOrdering() == Ordering::byNODES )
	    {
	      for (int ii=0; ii<ran_vdofs.Size(); ii++)
		for (int dj=0; dj<vdim_c; dj++)
		  for (int jj=0; jj<npri_c; jj++)
		    mat_ep->Set(ran_vdofs[ii],npr_c*dj+pr_offset_c+jj,
				mep(ii,npri_c*dj+jj));
	    }
	    else
	    {
	      for (int ii=0; ii<ran_vdofs.Size(); ii++)
		for (int jj=0; jj<npri_c; jj++)
		  for (int dj=0; dj<vdim_c; dj++)
		    mat_ep->Set(ran_vdofs[ii],vdim_c*pr_offset_c+vdim_c*jj+dj,
				mep(ii,npri_c*dj+jj));
	    }

	    if ( test_fes->GetOrdering() == Ordering::byNODES )
	    {
	      for (int di=0; di<vdim_r; di++)
		for (int ii=0; ii<npri_r; ii++)
		  for (int jj=0; jj<dom_vdofs.Size(); jj++)
		    mat_pe->Set(npr_r*di+pr_offset_r+ii,dom_vdofs[jj],
				mpe(npri_r*di+ii,jj));
	    }
	    else
	    {
	      for (int ii=0; ii<npri_r; ii++)
		for (int jj=0; jj<dom_vdofs.Size(); jj++)
		  for (int di=0; di<vdim_r; di++)
		    mat_pe->Set(vdim_r*pr_offset_r+vdim_r*ii+di,dom_vdofs[jj],
				mpe(npri_r*di+ii,jj));
	    }
	 }
      }
   }
}


} // namespace mfem;
