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

#ifndef MFEM_VERTEX
#define MFEM_VERTEX

#include "../config/config.hpp"
#include <stdio.h>
#include <string.h>

namespace mfem
{

/// Data type for vertex
struct Vertex
{
   double coord[3];

   // only in C++11 can we have user-defined constructors for POD type
#ifdef CPP11
   Vertex(double *xx, int dim);
   Vertex(double x, double y) { coord[0] = x; coord[1] = y; coord[2] = 0.; }
   Vertex(double x, double y, double z) { coord[0] = x; coord[1] = y; coord[2] = z; }
#else
   static Vertex* newVertex(double *xx, int dim);
   static Vertex* newVertex(double x, double y);
   static Vertex* newVertex(double x, double y, double z);
   inline void operator=(const Vertex *v) { memcpy(coord, v->coord, 3 * sizeof(double)); }
#endif

   /// Returns pointer to the coordinates of the vertex.
   inline double * operator() () const { return (double*)coord; }

   /// Returns the i'th coordinate of the vertex.
   inline double & operator() (int i) { return coord[i]; }

   /// Returns the i'th coordinate of the vertex.
   inline const double & operator() (int i) const { return coord[i]; }

   void SetCoords(const double *p, size_t dim) { memcpy(coord, p, dim * sizeof(double)); }
   void SetCoords(const double *p) { coord[0] = p[0]; coord[1] = p[1]; coord[2] = p[2]; }
   void SetCoords(double x, double y, double z) { coord[0] = x; coord[1] = y; coord[2] = z; }
};

}

#endif
