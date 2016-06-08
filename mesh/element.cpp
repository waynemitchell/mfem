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

void Element::init_indices(int *alloc, size_t count) {
   self_alloc = false;
   if (alloc) {
      indices = alloc;
   }
   else if (count > 0) {
      indices = new int[count];
      //printf("new self alloc at %p\n", indices);
      self_alloc = true;
   }
   else {
      indices = NULL;
   }
}

}
