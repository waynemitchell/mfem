// Copyright (c) 2010,  Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory.
// This file is part of the MFEM library.  See file COPYRIGHT for details.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.


#include "mesh_headers.hpp"

Segment::Segment( const int *ind, int attr ) : Element(Geometry::SEGMENT) {
   attribute = attr;
   for (int i=0; i<2; i++)
      indices[i] = ind[i];
}

Segment::Segment( int ind1, int ind2, int attr ) : Element(Geometry::SEGMENT) {
   attribute  = attr;
   indices[0] = ind1;
   indices[1] = ind2;
}

void Segment::SetVertices(const int *ind){
   indices[0] = ind[0];
   indices[1] = ind[1];
}

void Segment::GetVertices( Array<int> &v ) const {
   v.SetSize( 2 );
   for (int i=0; i<2; i++)
      v[i] = indices[i];
}

Linear1DFiniteElement SegmentFE;
