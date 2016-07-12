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

namespace mfem
{

/// Data type for vertex
class Vertex
{
protected:
   // this is awful 
   union {
      double coord[3];
      double *coord_ptr;
   };
   bool is_external;
   void init(double *_data = 0);

public:
   Vertex() { printf("called\n"); }

   Vertex(double *_data) { init(_data); }

   Vertex (double *xx, int dim, double *_data = 0);
   Vertex( double x, double y, double *_data = 0) { init(_data); coord[0] = x; coord[1] = y; coord[2] = 0.; }
   Vertex( double x, double y, double z, double *_data = 0)
   { init(_data); coord[0] = x; coord[1] = y; coord[2] = z; }

   /// Returns pointer to the coordinates of the vertex.
   inline double * operator() () const { if (is_external) return coord_ptr; else return (double*)coord; }

   /// Returns the i'th coordinate of the vertex.
   inline double & operator() (int i) { return is_external ? coord_ptr[i] : coord[i]; }

   /// Returns the i'th coordinate of the vertex.
   inline const double & operator() (int i) const { if (is_external) return coord_ptr[i]; else return coord[i]; }//is_external ? coord_ptr[i] : coord[i]; }

   void SetCoordPtr(double *_coord) {
      coord_ptr = _coord;
      is_external = true;
   }

   void SetCoords(const double *p)
   { if (!is_external) { coord[0] = p[0]; coord[1] = p[1]; coord[2] = p[2]; }
     else { coord_ptr[0] = p[0]; coord_ptr[1] = p[1]; coord_ptr[2] = p[2]; }
   }

   ~Vertex() { }
};

}

#endif
