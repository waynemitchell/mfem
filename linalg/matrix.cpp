// Copyright (c) 2010,  Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory.
// This file is part of the MFEM library.  See file COPYRIGHT for details.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

/*
  Implementation of class matrix
*/

#include <iostream>
#include <iomanip>

#include "matrix.hpp"

void Matrix::Print (ostream & out, int width)
{
   // output flags = scientific + show sign
   out << setiosflags(ios::scientific | ios::showpos);
   for (int i = 0; i < size; i++)
   {
      out << "[row " << i << "]\n";
      for (int j = 0; j < size; j++)
      {
         out << Elem(i,j) << " ";
         if ( !((j+1) % width) )
            out << endl;
      }
      out << endl;
   }
   out << endl;
}
