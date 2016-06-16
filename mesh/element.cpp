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

#include "mesh_headers.hpp"

namespace mfem
{
int NumOfIndices[6] = {1, 2, 3, 4, 4, 8};

Element::Element(int bg, int *_indices, size_t indices_count, 
      int *_attribute)
{ 
      base_geom = bg; 
      init(_indices, indices_count, _attribute); 
}

void Element::SetVertices(const int *ind)
{
   int i, n, *v;

   n = GetNVertices();
   v = GetVertices();

   for (i = 0; i < n; i++)
   {
      v[i] = ind[i];
   }
}

void Element::init(int *_indices, size_t indices_count, 
      int *_attribute)
{
   if (_indices == NULL && indices_count > 0) {
      indices = new int[indices_count];
      attribute = -1;
      //printf("new self alloc at %p\n", indices);
      self_alloc = true;
   }
   else {
      indices = _indices;
      ptr_attribute = _attribute;
      self_alloc = false;
   }
}

}
