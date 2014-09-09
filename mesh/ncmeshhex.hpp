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

#ifndef MFEM_NCMESHHEX
#define MFEM_NCMESHHEX


class HashTable
{
   // get
};

struct Hashed2
{
   int id;
   int p1, p2;
   Hashed2* next;
};


class NCMeshHex
{
   struct Vertex : public Hashed2
   {
      double pos[3];
   };

   /*struct Edge : public Hashed2
   {
   };

   struct Face : public Hashed4
   {
   };*/

   struct Element
   {
      bool refined;
      union {
         Vertex* vertex[8];
         Element* child[2];
      };
   };

   Array<Element> elements;
   HashTable<Vertex> vertices;
   //HashTable<Edge> edges;
   //HashTable<Face> faces;

   struct Dependency
   {
      int dof;
      double coef;
   };

   typedef Array<Dependency> DepList;
   DepList* v_dep;
   DepList* e_dep;
   DepList* f_dep;

};


#endif
