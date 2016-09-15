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


#include "vertex.hpp"

namespace mfem
{
#ifdef CPP11
Vertex::Vertex (double *xx, int dim) {
   for (int i = 0; i < dim; i++)
   {
      coord[i] = xx[i];
   }
}
#else
Vertex* Vertex::newVertex(double *xx, int dim) { 
   Vertex *v = new Vertex();
   for (int i = 0; i < dim; i++)
   {
      v->coord[i] = xx[i];
   }
   return v;
}

Vertex* Vertex::newVertex(double x, double y) { 
   Vertex *v = new Vertex(); 
   v->coord[0] = x; 
   v->coord[1] = y; 
   v->coord[2] = 0.; 
   return v; 
}

Vertex* Vertex::newVertex(double x, double y, double z) { 
   Vertex *v = new Vertex();
   v->coord[0] = x; 
   v->coord[1] = y; 
   v->coord[2] = z; 
   return v;
}
#endif
}
