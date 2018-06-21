// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

//This file contains useful functions to compute fluxes for DG methods.


#ifndef MFEM_DGFACEFUNC
#define MFEM_DGFACEFUNC
#include "dalg.hpp"

using std::vector;
using std::pair;

namespace mfem
{

/**
*	Returns a Change a basis matrix, associated to an integer. decrypts the integer encrypted by Permutation3D.
*/
void GetChangeOfBasis(const int permutation, IntMatrix& P);

/**
*	Returns the indices, face ID, and number of rotations, of the two element sharing a face.
*  The number of rotations is relative to the element 1, so nb_rot1 is always 0.
*/
void GetFaceInfo(const Mesh* mesh, const int face,
						int& ind_elt1, int& ind_elt2,
						int& face_id1, int& face_id2,
						int& nb_rot1, int& nb_rot2);

/**
*	Returns an integer identifying the permutation to apply to be in structured-
*  like configuration.
*/
void GetPermutation(const int dim, const int face_id1, const int face_id2, const int orientation, int& perm1, int& perm2);

/**
*	Returns the indices of a quadrature point on the face of an hex element relative to the index of the quadrature
*  point on the reference face.
*/
int GetFaceQuadIndex(const int dim, const int face_id, const int orientation, const int qind, const int quads, Tensor<1,int>& ind_f);

/**
*	Returns the indices of a quadrature point on the element relative to the index of the quadrature
*  point on the reference face.
*/
const int GetGlobalQuadIndex(const int dim, const int face_id, const int quads, Tensor<1,int>& ind_f);

}

#endif // MFEM_DGFACEFUNC